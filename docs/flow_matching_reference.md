# Conditional Flow Matching — implementation reference

Code-level notes distilled from the two canonical repos plus this project's own
flow bridge. Written so an agent can implement/extend flow matching here without
re-deriving the math. **Read the "Small details that bite" section last — it is
where bugs live.**

Sources:
- `atong01/conditional-flow-matching` (TorchCFM) — the OT-CFM reference library.
- `facebookresearch/flow_matching` (Lipman et al. 2024) — the `ProbPath`/scheduler design.
- This repo: `src/training/utils/sampling.py` (`flow_sample`), `src/training/train/train_diff2d.py`
  (`flow_loss`, `diffusion_in_channels`), `src/tests/test_flow_bridge.py`.

---

## 0. The one-paragraph mental model

Flow matching trains a network `v_θ(x_t, t)` to predict a **velocity** (a vector
field). At inference you start from a sample of the source distribution and
integrate the ODE `dx/dt = v_θ(x_t, t)` from the source time to the target time.
Training is **simulation-free**: you never integrate during training. You pick a
conditional path `x_t` between a source sample `x_0` and a target sample `x_1`,
compute the path's *analytic* time-derivative `u_t` (the target velocity), and
regress `v_θ` onto it with plain MSE. That's the whole objective:

```
t ~ U[0,1];  x_t = interpolate(x_0, x_1, t);  u_t = d/dt x_t
loss = || v_θ(x_t, t) - u_t ||²        # mean over batch & dims, UNWEIGHTED
```

No betas, no SNR weighting, no noise-prediction reparam. If you find yourself
adding Min-SNR weighting to a flow loss, stop — see §6.

---

## 1. Time convention (pick one and never deviate)

Both reference libs use: **`t=0` → source (`x_0`, usually noise), `t=1` → target
(`x_1`, data).**

- TorchCFM: `μ_t = (1-t)·x_0 + t·x_1`.
- FAIR `CondOTProbPath`: `x_t = α_t·x_1 + σ_t·x_0` with `α_t = t`, `σ_t = 1-t`
  → identical: `x_t = t·x_1 + (1-t)·x_0`.

⚠️ This repo's **bridge uses the OPPOSITE convention** (see §5): its trajectory
parameter `tau=0` is the *target* (AC) and `tau=1` is the *source* (NAC). Do not
mix them in one function. When porting code from either library, the first thing
to check is which endpoint `t=0` denotes.

---

## 2. Canonical CFM (noise → data) — the four variants

All inherit a base class. Per-sample `t ~ torch.rand(bs).type_as(x0)`. Broadcasting
of `t` to `x`'s shape is done by a `pad_t_like_x` helper that reshapes `(bs,)` →
`(bs, 1, 1, ...)`. `ε = torch.randn_like(x)`.

`sample_location_and_conditional_flow(x0, x1, t=None)` returns `(t, x_t, u_t)`:
```python
if t is None: t = torch.rand(x0.shape[0]).type_as(x0)
eps        = torch.randn_like(x0)
mu_t       = compute_mu_t(x0, x1, t)
sigma_t    = compute_sigma_t(t)
x_t        = mu_t + sigma_t * eps          # sample_xt
u_t        = compute_conditional_flow(x0, x1, t, x_t)
return t, x_t, u_t
```

| Variant | `μ_t` | `σ_t` | target `u_t` |
|---|---|---|---|
| **Base `ConditionalFlowMatcher`** | `(1-t)x0 + t·x1` | `σ` (const) | `x1 - x0` |
| **`ExactOptimalTransport…`** | same as base | `σ` | `x1 - x0`, but `(x0,x1)` are **reordered by an OT plan first** (§4) |
| **`TargetConditionalFlowMatcher`** (Lipman OT path) | `t·x1` (ignores x0) | `1-(1-σ)t` | `(x1-(1-σ)x_t)/(1-(1-σ)t)` |
| **`SchrödingerBridge…`** | `(1-t)x0+t·x1` | `σ·√(t(1-t))` | `((1-2t)/(2t(1-t)+1e-8))·(x_t-μ_t) + (x1-x0)`; OT plan uses entropic reg `2σ²` |
| **`VariancePreserving…`** (trig, Albergo) | `cos(πt/2)x0 + sin(πt/2)x1` | `σ` | `(π/2)(cos(πt/2)x1 - sin(πt/2)x0)` |

Notes:
- **`σ` is the path width**, not a diffusion noise schedule. `σ=0` gives a
  deterministic straight line (pure rectified flow / I-CFM). Small `σ` (e.g. 0.0–0.1)
  is common for images. Schrödinger-bridge variant *requires* `σ>0` (raises if ≤0,
  warns if <1e-3) because `σ_t` and the `(1-2t)/(2t(1-t))` term degenerate.
- The base and ExactOT variants are the workhorses. ExactOT differs from base
  **only** by the OT reordering of the batch — the formulas are identical.
- `u_t = x1 - x0` is **constant in `t`** for the base/OT variants. This is why the
  oracle test (§5) shows step-count invariance: a constant field telescopes, so 1
  Euler step == 8 Euler steps.

### FAIR `AffineProbPath` framing (equivalent, often cleaner to extend)
`x_t = α_t·x_1 + σ_t·x_0`, `dx_t = α̇_t·x_1 + σ̇_t·x_0`. The scheduler supplies
`(α_t, σ_t, α̇_t, σ̇_t)`. `sample()` returns a `PathSample(x_t, dx_t, x_0, x_1, t)`
where **`dx_t` is the regression target** (the velocity). `CondOTProbPath` plugs in
`α_t=t, σ_t=1-t` → `dx_t = x_1 - x_0`. The library also ships exact reparam
converters you'll want if the net predicts something other than velocity:
`velocity_to_epsilon`, `epsilon_to_velocity`, `velocity_to_target`,
`target_to_velocity`, `epsilon_to_target`, `target_to_epsilon`. Predict velocity by
default; the converters let you train an ε- or x1-head and convert at sampling time.

---

## 3. Training loop (canonical, verbatim shape)

```python
FM = ConditionalFlowMatcher(sigma=0.0)          # or ExactOptimalTransport…(sigma=0.0)
# x1 = data batch; x0 = torch.randn_like(x1)    # source = noise
t, x_t, u_t = FM.sample_location_and_conditional_flow(x0, x1)
v_t = model(x_t, t)                              # model signature is (x, t)
loss = torch.mean((v_t - u_t) ** 2)              # plain MSE, no weighting
loss.backward()
```

Model-call convention: **`model(x_t, t)`** with `t` shape `(bs,)`. In MLP toy
examples people instead do `model(torch.cat([x_t, t[:, None]], dim=-1))`; for a
UNet, `t` goes through a timestep embedding exactly like a diffusion UNet — so an
existing diffusion UNet is reusable as-is, just reinterpret its output as velocity.
`t` is **continuous in [0,1]**, not an integer index — make sure the time embedding
accepts floats (scale by 1000 if reusing a discrete-timestep sinusoidal embedding).

---

## 4. OT minibatch coupling (`OTPlanSampler`) — the detail that makes "OT-CFM"

Plain CFM pairs `x0[i]` with `x1[i]` arbitrarily. OT-CFM re-pairs them within the
minibatch to minimize transport cost → straighter paths, lower-variance targets,
fewer sampling steps needed.

```python
M   = torch.cdist(x0_flat, x1_flat) ** 2     # squared euclidean cost
pi  = pot.emd(uniform, uniform, M)           # "exact"  (POT library)
# draw paired indices from the (normalized) plan as a flat categorical
p   = pi.flatten() / pi.sum()
idx = np.random.choice(len(p), size=batch, p=p, replace=True)
i, j = idx // n, idx % n
x0, x1 = x0[i], x1[j]                         # reordered, THEN feed to base CFM
```

- Methods: `"exact"` (EMD), `"sinkhorn"` (entropic, needs `reg`), `"unbalanced"`,
  `"partial"`. SB-CFM uses `reg=2σ²`.
- `replace=True` is the default — the same target can pair with multiple sources.
- There is a deterministic `sample_plan_with_scipy` (linear assignment) for
  variance-reduced, repeatable matching.
- Optional cost normalization (divide `M` by its max) for numerical stability; falls
  back to a uniform plan if the solver returns NaNs.
- **Coupling is per-minibatch**, so OT quality improves with batch size. It's an
  approximation of the global OT map, not the real thing.

---

## 5. Sampling / inference (the ODE integration)

Integrate `dx/dt = v_θ(x_t, t)` from source-time to target-time.

Reference libs use adaptive solvers (`torchdyn.NeuralODE`, or FAIR's `ODESolver`
wrapping `torchdiffeq.odeint`) over `time_grid = torch.linspace(0, 1, steps)`.
A fixed-step Euler integrator is fine and is what this repo uses:

```python
x = x0                                  # source sample
ts = torch.linspace(0, 1, num_steps+1)
for k in range(num_steps):
    dt = ts[k+1] - ts[k]
    x = x + dt * v_θ(x, ts[k].expand(bs))   # forward Euler
```

- **Sampling is deterministic** when `σ=0` / velocity field is fixed — no `randn`
  in the loop. The repo test asserts bit-identical repeat runs.
- Step count: straight-path (OT/rectified) models need very few steps (often 1–8);
  curved paths (trig/VP, or a poorly-rectified field) need more.

---

## 6. THIS REPO's flow bridge — read carefully, it diverges from canonical CFM

The repo does **NAC→AC PET translation** as a *data-to-data Schrödinger/rectified
bridge* (I2SB-style), **not** noise→data. Implemented in `flow_loss`
(`train_diff2d.py`), `flow_sample` (`sampling.py`), tests in `test_flow_bridge.py`.

Key facts (all locked by tests — do not "fix" them):

1. **Endpoints are two real PET volumes, not noise.** Linear path with the repo's
   own convention:
   ```
   x_tau = (1-tau)·AC + tau·NAC ,   tau = t / (T-1)
   ```
   So `tau=0 → AC` (target), `tau=1 → NAC` (source). **Opposite** of canonical §1.
   `tau = timesteps / (T-1)` — note the **`T-1` divisor** (endpoints land exactly on
   `tau∈{0,1}`); requires `num_train_timesteps ≥ 2` (raises otherwise).

2. **Constant target velocity `v = NAC − AC`** (sign matters; the test pins
   `target == NAC-AC`, and predicting `−v` gives loss `(−1−1)²=4`).

3. **Sampling starts from `x_init = NAC`** and integrates to recover AC. With the
   oracle velocity it recovers AC exactly at *any* step count (constant field).

4. **⚠️ Do NOT concatenate the source (NAC) into the model input.** This is the
   single most important detail and the opposite of the epsilon-diffusion path:
   - epsilon mode: `in_channels = 2C` (concat `noisy_AC_latent` + `NAC_latent`).
   - flow mode: `in_channels = C` — feed **only `x_t`**.
   - Helper: `diffusion_in_channels(C, is_flow=True) == C`, `…(…, is_flow=False) == 2C`.
   - Why: along the line, `AC = (x_t − tau·NAC)/(1−tau)`. If NAC is also an input,
     the network can solve for AC by algebra for `tau<1`, the velocity loss
     collapses to a no-op, and generation fails. I2SB avoids this by construction.
     Both `flow_loss` and `flow_sample` must feed `C` channels — tests assert
     `last_in_channels == C` in both.

5. **Loss is plain unweighted MSE.** No `snr_gamma` / Min-SNR (the term is inert /
   absent in flow mode — `flow_loss` has no `snr_gamma` parameter, asserted).

6. **Model selection metric differs:** flow selects `best.pt` on the **honest
   rollout L1** (actually integrate and compare to AC), epsilon selects on the
   velocity/MSE `loss`. `_selection_metric(vm, is_flow=True) → vm["l1"]`.

7. **Resume across a prediction-type switch resets `best_val` to `inf`** (the
   metric scale changes between epsilon-loss and flow-L1), via `_resume_best_val`.

8. The epsilon (DDIM) path is untouched and still starts from noise — flow and
   epsilon coexist behind a `prediction_type` flag.

### 6a. Why the constant velocity does NOT make this trivially one-step

The obvious objection: the target velocity `v = NAC − AC` is **constant in tau**, so a
perfectly learned field telescopes to `x_final = NAC − (NAC−AC) = AC` in a *single* Euler
step, for any step count. Note this is equally true of canonical rectified flow — there
too the *conditional* target `x_1 − x_0` is constant in `t`. The constancy is not what
distinguishes the two.

What distinguishes them is **what the model can actually learn**. The net sees only
`(x_t, t)`, never the pair, so the MSE-optimal output is the *marginal* field

```
u(x_t, t) = E[ x_1 − x_0 | x_t ]
```

and path curvature IS the breadth of that posterior:

- **Canonical FM (noise→data):** the coupling is *independent* — a random noise sample is
  paired with a random image. Near `t≈0`, `x_t` is nearly pure noise and says almost
  nothing about the endpoint, so `E[x_1|x_t]` is close to the dataset mean: one Euler step
  yields a blurry average, not a sample. The field is strongly curved, hence many steps —
  and hence *reflow*, which re-couples `(x_0,x_1)` along learned trajectories precisely to
  straighten it.
- **This bridge (NAC→AC):** the coupling is *fixed and physical* — each NAC volume is
  paired with the one true AC volume of the same scan, and the trajectory starts at NAC
  itself. `E[AC | x_tau]` is narrow from the first step, so the marginal field is close to
  the conditional one, few steps suffice, and no reflow stage is needed. The observed
  step-count invariance (s16 ≈ s32, ΔSSIM ≤ 0.002) is exactly that signature.

**Narrow is not degenerate**, and that matters: AC is *not* a deterministic function of
NAC (the attenuation map lives in the CT, which the model never sees), so the model does
what conditional-mean regression always does — it regresses toward the middle. That is the
mechanism behind the measured `reg_slope ≈ 0.86` and the ~10% under-estimation of
high-uptake tissue (`hot_band_rel_error`), and it is the same phenomenon as canonical FM's
one-step blur, merely much milder because the posterior is narrow.

Consequence to state honestly: this model behaves closer to a deterministic residual
regressor wrapped in an ODE than to a generative sampler. The decisive cheap experiment —
evaluate at 1/2/4 steps; if s2 ≈ s16 the ODE is doing little — plus a direct one-shot
`NAC−AC` regression baseline, is still **not run** (no such baseline exists in the repo;
`prediction_type` accepts only `epsilon` and `flow`).

### 6b. Loss shaping (2026-08-06): RFPP's premetric and U-shaped tau

Where the perceptual term belongs on a bridge, and where this repo already agreed with the
literature. RFPP, *Improving the Training of Rectified Flows* (Lee et al., NeurIPS 2024,
arXiv 2405.20320, [github.com/sangyun884/rfpp](https://github.com/sangyun884/rfpp)):

```
m_lp-hub = (1−t)·m_huber(z−x, v_theta)  +  LPIPS(x, x_t − t·v_theta(x_t,t))
```

- The perceptual term is applied to the **analytic endpoint estimate** `x_t − t·v`, at
  **all** `t`, ungated. That is exactly `flow_ac_estimate_latent` /
  `flow_x0_decode_in_graph` in this repo, and RFPP's time convention matches ours
  (`t=0` → target data, `t=1` → source). `lucidrains/rectified-flow-pytorch`
  (`PseudoHuberLossWithLPIPS`) independently does the same:
  `pred_data = noised + pred_flow*(1-t)`.
- **Why ungated is correct, and gating is actively wrong.** `d(x_tau − tau·v)/dv = −tau`,
  so a loss on the estimate has a gradient intrinsically scaled by `tau`: it vanishes at
  `tau→0` (where the estimate is trivially correct) and peaks at the NAC end (where the
  model must supply the whole residual). Applying the *epsilon* path's
  `perceptual_active_frac` gate (`keep = t < 0.7T`) would retain precisely the
  low-gradient half. `perceptual_term` therefore skips the gate when `is_flow`.
  Side effect: uniform-tau averaging dilutes the term ~2x, so its effective weight is
  about half the nominal `perceptual_weight`.
- **`flow_loss_weighting: rfpp`** applies the `(1−tau)` factor to the regression term, so
  regression dominates near the AC target and the perceptual term (whose gradient grows
  like `tau`) takes over near the NAC source — a hand-off rather than a flat sum.
- **`flow_tau_dist: ushaped`** samples `tau` from `p(u) ∝ 2·cosh(a·(u − 1/2))`, `a≈4`
  (RFPP report −28% FID vs uniform on CIFAR-10; the loss is large at both ends and small
  in the middle). **Beware the paper's literal formula** `exp(a·u)+exp(−a·u)`: on
  `u∈[0,1]` that is `2·cosh(a·u)`, which is *monotonically increasing*, not U-shaped. The
  `(u − 1/2)` shift is required for the stated "both ends" motivation — the implementation
  and its test pin this.
- All three knobs default OFF (`uniform` / `none`), reproducing prior behaviour exactly.
- **Perceptual backends.** `"vgg"` is 2.5D (a 2D backbone run slice-wise; no volumetric
  receptive field). `"lpl"` (*Boosting Latent Diffusion with Perceptual Objectives*, arXiv
  2411.04873) instead compares the frozen AE **decoder's own intermediate features**,
  consuming *latents* rather than decoded images: natively volumetric, no external
  weights, and measured ~2.4x faster / ~2.8x lighter on peak VRAM because the decoder
  forward stops at the deepest tap and never runs the full-resolution tail.
- Prior evidence to keep expectations calibrated: the 2D VGG perceptual arm moved best
  rollout val L1 by ~1% (`diff2d_flow2` 0.01251 → `diff2d_flow_perc` 0.01236) and test
  SSIM by +0.001 — within noise. RFPP optimized FID; the decisive metrics here are
  quantitative.

---

## 7. Small details that bite (checklist)

- **Which endpoint is `t=0`?** Canonical: noise. This repo's bridge: AC (target).
  Getting this backwards flips the velocity sign → model learns the negation.
- **`t` is a float in [0,1]**, fed per-sample shape `(bs,)`, broadcast via reshape to
  `(bs,1,1,...)`. Don't pass an int timestep index to a flow loss.
- **Velocity target, not noise.** MSE is against `u_t`/`dx_t`, never against `ε`.
  No `1/σ_t` or SNR reweighting — unweighted mean. (Reparam to ε/x1 only via the
  FAIR converters if you deliberately train a non-velocity head.)
- **`σ` is path width, not a noise schedule.** `σ=0` → deterministic straight path.
  Schrödinger-bridge needs `σ>0` and has a `1/(2t(1-t))` term guarded by `+1e-8`.
- **OT coupling reorders the batch before computing the (still constant) target**;
  it's the only thing separating OT-CFM from vanilla CFM. Quality scales with batch
  size; it's a minibatch approximation.
- **Bridge: never concat the source into the input** (§6.4). This is the failure
  mode that silently collapses training.
- **`tau = t/(T-1)`** in the bridge (endpoints exact); needs `T≥2`.
- **Sampling is RNG-free** for a fixed velocity field — guard tests with
  `torch.equal` on repeat runs.
- **Reuse the diffusion UNet** for velocity directly; just confirm its timestep
  embedding accepts continuous floats (scale by ~1000 if it's a discrete sinusoidal
  embedding) and that `in_channels` matches the mode (C for flow, 2C for epsilon).
- **Step-count invariance is a correctness signal for straight paths**: with the
  oracle constant velocity, 1 step == N steps. If your sampler fails this, the
  integrator or the time grid is wrong.
```
