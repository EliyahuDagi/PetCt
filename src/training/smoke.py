"""Smoke test + latency estimator for the whole training pipeline.

Runs each stage's real ``main()`` for a few steps on the selected device, wiring
the checkpoint chain into an isolated work dir so nothing real is clobbered:

    ae2d -> ae3d (inflated 3D AE) -> diff2d -> ft3d

For each stage it records pass/fail, steady-state ms/step (median of the
``wall_time`` deltas the train scripts already write to ``metrics.jsonl``), and
peak GPU memory (captured in-process). It then extrapolates a full-run ETA from
``--plan_epochs`` x ``--plan_steps_per_epoch`` and writes ``--report`` (JSON) plus
a readable table to stdout. The GUI spawns this in WSL and renders the report.

Because it runs the actual stage code paths, it doubles as an end-to-end
component check: imports, CUDA, data loading, model build, train step, checkpoint
save and the cross-stage inflation are all exercised. A stage that raises is
reported as failed (with the error) and the run continues to the next stage.

After training, it also runs the inference CLI (``src.training.infer``) for each
task whose checkpoints were produced, so the encode/sample/decode paths and the
2D/3D AE selection are checked too. Pass ``--skip_infer`` to train-only.
"""

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from statistics import median

import torch

from src.training.utils.metrics import read_metrics

STAGES = ["ae2d", "ae3d", "diff2d", "ft3d"]


def _as_roots(data_dir):
    """Normalize ``args.data_dir`` (str or list) to a list of root strings."""
    return [data_dir] if isinstance(data_dir, str) else list(data_dir)


def _stage_argv(stage, args, work_dir):
    """Build the argv for a stage's main(), wiring the checkpoint chain."""
    save_dir = os.path.join(work_dir, stage)
    common = [
        "--data_dir", *_as_roots(args.data_dir),
        "--patient_index", str(args.patient_index),
        "--device", args.device,
        "--epochs", "1",
        "--steps_per_epoch", str(args.steps),
        "--val_every", "0",          # skip mid-run val; final val still saves best.pt
        "--val_batches", "1",
        "--save_dir", save_dir,
    ]
    ae2d_ckpt = os.path.join(work_dir, "ae2d", "best.pt")
    ae3d_ckpt = os.path.join(work_dir, "ae3d", "best.pt")
    diff2d_ckpt = os.path.join(work_dir, "diff2d", "best.pt")
    if stage == "ae2d":
        return common + ["--batch_size", str(args.batch2d), "--slice_size", str(args.size)]
    if stage == "ae3d":
        argv = common + ["--batch_size", str(args.batch3d), "--crop_size", str(args.crop3d), "--ae2d_ckpt", ae2d_ckpt]
        if getattr(args, "extra_levels", 0):
            argv += ["--extra_levels", str(args.extra_levels)]
        return argv
    if stage == "diff2d":
        return common + ["--batch_size", str(args.batch2d), "--latent_size", str(args.size), "--ae_ckpt", ae2d_ckpt]
    if stage == "ft3d":
        return common + ["--batch_size", str(args.batch3d), "--latent_size", str(args.crop3d),
                         "--ae_ckpt", ae3d_ckpt, "--inflate_from", diff2d_ckpt]
    raise ValueError(stage)


def _stage_ckpts(work_dir):
    return {s: os.path.join(work_dir, s, "best.pt") for s in STAGES}


# Checkpoints each task's inference depends on (besides its own).
INFER_PREREQS = {
    "ae2d": ["ae2d"],
    "ae3d": ["ae3d"],
    "diff2d": ["ae2d", "diff2d"],
    "ft3d": ["ae3d", "ft3d"],
}


def _infer_argv(task, args, work_dir):
    """Build argv for ``infer.main()``: 3D tasks use the 3D AE + crop size."""
    ckpts = _stage_ckpts(work_dir)
    ae_ckpt = ckpts["ae3d"] if task in ("ae3d", "ft3d") else ckpts["ae2d"]
    size = args.crop3d if task in ("ae3d", "ft3d") else args.size
    argv = [
        "--task", task,
        "--data_dir", *_as_roots(args.data_dir),
        "--patient_index", str(args.patient_index),
        "--device", args.device,
        "--slice", "0",
        "--ae_ckpt", ae_ckpt,
        "--size", str(size),
        "--out", os.path.join(work_dir, "infer", task),
    ]
    if task in ("diff2d", "ft3d"):
        argv += ["--diff_ckpt", ckpts[task], "--ddim_steps", "2"]
    return argv


def _run_infer(task, infer_main, argv):
    """Run one task's inference; verify pred.npy is written. Never raises."""
    result = {"stage": task, "ok": False, "error": None}
    out_dir = argv[argv.index("--out") + 1]
    saved_argv = sys.argv
    try:
        sys.argv = ["smoke_infer_" + task] + argv
        infer_main()
        if not os.path.exists(os.path.join(out_dir, "pred.npy")):
            raise FileNotFoundError("inference produced no pred.npy")
        result["ok"] = True
    except Exception as e:  # noqa: BLE001 - smoke test must report, not crash
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc().splitlines()[-4:]
    finally:
        sys.argv = saved_argv
    return result


def _ms_per_step(save_dir):
    """Median ms/step from consecutive train-row wall_time deltas (first dropped
    to exclude warmup / cudnn autotune). Returns None if not enough rows."""
    rows = [r for r in read_metrics(os.path.join(save_dir, "metrics.jsonl")) if r.get("phase") == "train"]
    times = [r["wall_time"] for r in rows if "wall_time" in r]
    if len(times) < 3:
        return None
    deltas = [(b - a) * 1000.0 for a, b in zip(times[:-1], times[1:])]
    return float(median(deltas[1:] if len(deltas) > 1 else deltas))


def _run_stage(stage, main_fn, argv, device_is_cuda):
    """Run one stage; return a result dict (never raises)."""
    save_dir = argv[argv.index("--save_dir") + 1]
    result = {"stage": stage, "ok": False, "ms_per_step": None, "peak_mem_mb": None,
              "steps": None, "error": None}
    saved_argv = sys.argv
    if device_is_cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.time()
    try:
        sys.argv = ["smoke_" + stage] + argv
        main_fn()
        if device_is_cuda:
            torch.cuda.synchronize()
        result["ok"] = True
        result["wall_seconds"] = round(time.time() - t0, 3)
        result["ms_per_step"] = _ms_per_step(save_dir)
        if device_is_cuda:
            result["peak_mem_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
    except Exception as e:  # noqa: BLE001 - smoke test must report, not crash
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc().splitlines()[-4:]
    finally:
        sys.argv = saved_argv
    return result


def _precheck(args):
    """Light environment + data check before running any stage."""
    checks = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    try:
        from src.training.dataset_index import enumerate_patients
        patients = enumerate_patients(args.data_dir, missing_ok=True)
        checks["data_patients"] = len(patients)
        checks["data_ok"] = len(patients) > 0
    except Exception as e:  # noqa: BLE001
        checks["data_ok"] = False
        checks["data_error"] = f"{type(e).__name__}: {e}"
    return checks


def _print_report(report):
    print("\n===== SMOKE TEST REPORT =====")
    c = report["checks"]
    print(f"torch={c['torch']}  cuda={c['cuda_available']}  device={c.get('device_name')}")
    print(f"data: patients={c.get('data_patients')} ok={c.get('data_ok')}")
    print(f"plan: epochs={report['plan']['epochs']} x steps/epoch={report['plan']['steps_per_epoch']}")
    print(f"{'stage':8} {'ok':4} {'ms/step':>10} {'peak MB':>9} {'ETA':>12}")
    for s in report["stages"]:
        eta = _fmt_eta(s.get("eta_seconds"))
        ms = f"{s['ms_per_step']:.1f}" if s.get("ms_per_step") is not None else "-"
        mem = f"{s['peak_mem_mb']:.0f}" if s.get("peak_mem_mb") is not None else "-"
        print(f"{s['stage']:8} {('OK' if s['ok'] else 'FAIL'):4} {ms:>10} {mem:>9} {eta:>12}")
        if s.get("error"):
            print(f"         error: {s['error']}")
    print(f"{'TOTAL':8} {'':4} {'':>10} {'':>9} {_fmt_eta(report.get('eta_total_seconds')):>12}")
    print("(ETA = train-step time only; excludes validation/checkpoint overhead)")
    if report.get("inference"):
        print("inference (encode/sample/decode CLI):")
        for r in report["inference"]:
            line = f"  {r['stage']:8} {'OK' if r['ok'] else 'FAIL'}"
            if r.get("error"):
                line += f"   {r['error']}"
            print(line)
    print()


def _fmt_eta(seconds):
    if seconds is None:
        return "-"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, nargs="+", help="One or more dataset roots / patient folders")
    parser.add_argument("--patient_index", type=int, default=0, help="Global patient index across all roots")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=6, help="Timed train steps per stage")
    parser.add_argument("--size", type=int, default=128, help="2D slice/latent size (ae2d, diff2d)")
    parser.add_argument("--crop3d", type=int, default=64, help="3D cube size (ae3d, ft3d)")
    parser.add_argument("--extra_levels", type=int, default=0,
                        help="Extra 3D AE downsample level(s) for ae3d (B2 config: crop3d 128 + extra_levels 1).")
    parser.add_argument("--batch2d", type=int, default=4)
    parser.add_argument("--batch3d", type=int, default=1)
    parser.add_argument("--plan_epochs", type=int, default=1, help="Epochs to extrapolate ETA for")
    parser.add_argument("--plan_steps_per_epoch", type=int, default=50)
    parser.add_argument("--stages", default=",".join(STAGES), help="Comma-separated subset of stages")
    parser.add_argument("--work_dir", default=os.path.join("outputs", "smoke"))
    parser.add_argument("--report", default=os.path.join("outputs", "smoke_report.json"))
    parser.add_argument("--keep", action="store_true", help="Keep the work dir (checkpoints) after the run")
    parser.add_argument("--skip_infer", action="store_true", help="Train-only; skip the inference CLI check")
    args = parser.parse_args()

    stages = [s for s in args.stages.split(",") if s in STAGES]
    device_is_cuda = args.device == "cuda" and torch.cuda.is_available()

    checks = _precheck(args)
    print("Running smoke test on %s ..." % (checks.get("device_name") or args.device))

    # Import stage mains lazily so an import error is reported, not fatal.
    main_fns = {}
    import_error = None
    try:
        from src.training.train import train_ae2d, train_ae3d, train_diff2d, train_ft3d
        main_fns = {"ae2d": train_ae2d.main, "ae3d": train_ae3d.main,
                    "diff2d": train_diff2d.main, "ft3d": train_ft3d.main}
    except Exception as e:  # noqa: BLE001
        import_error = f"{type(e).__name__}: {e}"

    stage_results = []
    plan_factor = max(1, args.plan_epochs) * max(1, args.plan_steps_per_epoch)
    for stage in stages:
        if import_error:
            stage_results.append({"stage": stage, "ok": False, "error": "import failed: " + import_error})
            continue
        print(f"-- {stage} --")
        res = _run_stage(stage, main_fns[stage], _stage_argv(stage, args, args.work_dir), device_is_cuda)
        res["steps"] = args.steps
        if res.get("ms_per_step") is not None:
            res["eta_seconds"] = round(res["ms_per_step"] / 1000.0 * plan_factor, 1)
        stage_results.append(res)

    # Inference phase: exercise the infer CLI for every task whose checkpoints
    # exist (the encode/sample/decode paths + 2D vs 3D AE selection).
    infer_results = []
    if not args.skip_infer and not import_error:
        try:
            from src.training import infer as infer_mod
        except Exception as e:  # noqa: BLE001
            infer_results.append({"stage": "infer_import", "ok": False, "error": f"{type(e).__name__}: {e}"})
            infer_mod = None
        if infer_mod is not None:
            ckpts = _stage_ckpts(args.work_dir)
            for task in stages:
                need = INFER_PREREQS[task]
                if not all(os.path.exists(ckpts[c]) for c in need):
                    infer_results.append({"stage": task, "ok": False,
                                          "error": "skipped: prerequisite checkpoint(s) missing"})
                    continue
                print(f"-- infer {task} --")
                infer_results.append(_run_infer(task, infer_mod.main, _infer_argv(task, args, args.work_dir)))

    etas = [s["eta_seconds"] for s in stage_results if s.get("eta_seconds") is not None]
    report = {
        "checks": checks if not import_error else {**checks, "import_error": import_error},
        "plan": {"epochs": args.plan_epochs, "steps_per_epoch": args.plan_steps_per_epoch},
        "stages": stage_results,
        "inference": infer_results,
        "eta_total_seconds": round(sum(etas), 1) if etas else None,
        "all_ok": (
            all(s.get("ok") for s in stage_results)
            and all(r.get("ok") for r in infer_results)
            and not import_error
        ),
    }

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    _print_report(report)

    if not args.keep:
        shutil.rmtree(args.work_dir, ignore_errors=True)
    print("Report written to %s" % args.report)
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
