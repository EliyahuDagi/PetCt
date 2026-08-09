"""Tabulate ae2d embedding-size ablation results: reconstruction fidelity vs latent.

Reads the LAST ``phase == "val"`` row from each
``outputs/ae2d_ablation/lc{N}/metrics.jsonl`` (those rows already carry the
recon-fidelity metrics the trainer logged: recon_l1, psnr, ssim, nrmse, mae) and
prints an aligned table sorted by latent_channels, then writes the same data to
``outputs/ae2d_ablation/report.json`` and ``report.md``.

The goal is to read off, at a glance, the SMALLEST embedding with no large recon
loss. Columns are ordered so the headline fidelity metrics (recon_l1, psnr, ssim)
are prominent. We flag the "knee": the row beyond which growing N to the next grid
value buys < KNEE_PSNR_DB PSNR -- i.e. diminishing returns from a bigger latent.

Latent geometry assumption: slice_size 128 with the default 3-level
block_out_channels ([64,128,256] -> /4) gives a latent_channels x 32 x 32 code.
    compression = 16384 / (latent_channels * 1024)
This is stdlib-only / torch-free; run on the Windows host or in WSL.
"""

import json
import os

# Repo-relative paths resolved from this file so cwd does not matter.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
ABLATION_ROOT = os.path.join(_REPO, "outputs", "ae2d_ablation")

# Embedding grid (kept in sync with gen_ae2d_ablation_configs.LATENT_CHANNELS_GRID).
LATENT_CHANNELS_GRID = [2, 4, 8, 16]

# Latent spatial size under the stated assumption (slice 128, /4 downsample).
LATENT_HW = 32
INPUT_ELEMS = 128 * 128 * 1  # 16384

# Knee threshold: < this PSNR gain (dB) from the next-larger N == diminishing return.
KNEE_PSNR_DB = 0.5

# Metric columns to surface, in display order (headline fidelity first).
METRIC_KEYS = ["recon_l1", "psnr", "ssim", "nrmse", "mae"]


def _last_val_row(metrics_path):
    """Return the last JSON object with phase == 'val', or None if absent."""
    if not os.path.isfile(metrics_path):
        return None
    last = None
    with open(metrics_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("phase") == "val":
                last = row
    return last


def _compression(n):
    return INPUT_ELEMS / float(n * LATENT_HW * LATENT_HW)


def collect():
    """Gather one result dict per latent_channels point that has a val row."""
    results = []
    for n in sorted(LATENT_CHANNELS_GRID):
        metrics_path = os.path.join(ABLATION_ROOT, "lc{}".format(n), "metrics.jsonl")
        row = _last_val_row(metrics_path)
        if row is None:
            print("WARN: no val row for latent_channels={} ({})".format(n, metrics_path))
            continue
        entry = {
            "latent_channels": n,
            "latent": "{}x{}x{}".format(n, LATENT_HW, LATENT_HW),
            "compression": round(_compression(n), 3),
        }
        for k in METRIC_KEYS:
            entry[k] = row.get(k)
        results.append(entry)
    return results


def _flag_knees(results):
    """Mark each row where the NEXT-larger N gains < KNEE_PSNR_DB PSNR.

    Such a row is the smaller, cheaper embedding past which a bigger latent barely
    improves reconstruction -- the candidate "good enough" operating point.
    """
    for i, r in enumerate(results):
        r["knee"] = False
        if i + 1 >= len(results):
            continue
        cur, nxt = r.get("psnr"), results[i + 1].get("psnr")
        if isinstance(cur, (int, float)) and isinstance(nxt, (int, float)):
            if (nxt - cur) < KNEE_PSNR_DB:
                r["knee"] = True
    return results


def _fmt(v):
    if isinstance(v, float):
        return "{:.4f}".format(v)
    if v is None:
        return "-"
    return str(v)


def _table_rows(results):
    """Build header + aligned string rows (returns list[list[str]])."""
    header = ["lc", "latent", "compress", "recon_l1", "psnr", "ssim", "nrmse", "mae", "knee"]
    rows = [header]
    for r in results:
        rows.append([
            str(r["latent_channels"]),
            r["latent"],
            "{:.2f}x".format(r["compression"]),
            _fmt(r.get("recon_l1")),
            _fmt(r.get("psnr")),
            _fmt(r.get("ssim")),
            _fmt(r.get("nrmse")),
            _fmt(r.get("mae")),
            "<-- knee" if r.get("knee") else "",
        ])
    return rows


def _render_aligned(rows):
    widths = [max(len(r[c]) for r in rows) for c in range(len(rows[0]))]
    lines = []
    for ri, r in enumerate(rows):
        lines.append("  ".join(cell.ljust(widths[c]) for c, cell in enumerate(r)))
        if ri == 0:
            lines.append("  ".join("-" * widths[c] for c in range(len(r))))
    return "\n".join(lines)


def _render_markdown(rows):
    out = []
    out.append("| " + " | ".join(rows[0]) + " |")
    out.append("| " + " | ".join("---" for _ in rows[0]) + " |")
    for r in rows[1:]:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def main():
    results = _flag_knees(collect())
    rows = _table_rows(results)

    print()
    print("ae2d embedding-size ablation -- reconstruction fidelity vs latent")
    print("(latent assumes slice_size 128 & default 3-level block_out_channels -> 32x32)")
    print()
    print(_render_aligned(rows))
    print()
    knees = [r["latent_channels"] for r in results if r.get("knee")]
    if knees:
        print("Knee (next-larger N gains < {:.2f} dB PSNR): latent_channels={}".format(
            KNEE_PSNR_DB, ", ".join(str(k) for k in knees)))
        print("-> smallest embedding with no large recon loss is around the flagged row(s).")
    else:
        print("No knee detected: PSNR still climbing across the grid (try larger N).")

    os.makedirs(ABLATION_ROOT, exist_ok=True)
    json_path = os.path.join(ABLATION_ROOT, "report.json")
    md_path = os.path.join(ABLATION_ROOT, "report.md")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({"knee_psnr_db": KNEE_PSNR_DB, "results": results}, fh, indent=2)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("# ae2d embedding-size ablation\n\n")
        fh.write("Reconstruction fidelity vs latent embedding size (last val row per run). "
                 "Latent assumes slice_size 128 & default 3-level block_out_channels (-> 32x32).\n\n")
        fh.write(_render_markdown(rows) + "\n\n")
        if knees:
            fh.write("**Knee** (next-larger N gains < {:.2f} dB PSNR): latent_channels={}\n".format(
                KNEE_PSNR_DB, ", ".join(str(k) for k in knees)))
    print()
    print("wrote {}".format(json_path))
    print("wrote {}".format(md_path))


if __name__ == "__main__":
    main()
