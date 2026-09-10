"""FOLD THE PACKED VOLUMES INTO ONE SELF-CONTAINED HTML FILE.

``scripts/make_viewer_data.py`` writes one png per volume plus a manifest. This reads them
back, encodes each png as a data url, and drops the whole lot into
``docs/_viewer_template.html`` in place of the ``__DATA__`` marker. The result is a single
file with no external references at all -- no server, no sibling folder, no network -- so it
can be mailed or copied to a colleague and opened by double-clicking it.

The cost of that is size: base64 is 4 bytes for every 3, so three patients land at roughly
12 MB. That is the price of "hand it over as one file".

    ~/petct/.venv/bin/python scripts/make_viewer_html.py
"""

import argparse
import base64
import json
import os

TITLES = {"median": "Typical patient", "best": "Best patient",
          "worst": "Worst patient", "p25": "Lower quarter",
          "lung": "Lung case", "bladder": "Bladder case",
          "uterine": "Uterine case"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="docs/viewer_data")
    ap.add_argument("--template", default="docs/_viewer_template.html")
    ap.add_argument("--out", default="docs/nac_to_ac_viewer.html")
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(args.data, "manifest.json")))
    payload = []
    for m in manifest:
        imgs = {}
        for name, f in m["files"].items():
            with open(os.path.join(args.data, f["file"]), "rb") as fh:
                raw = fh.read()
            if len(raw) != f["bytes"]:
                raise SystemExit("%s is %d bytes, manifest says %d -- stale data directory"
                                 % (f["file"], len(raw), f["bytes"]))
            imgs[name] = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
        payload.append({
            "tag": m["tag"], "title": TITLES.get(m["tag"], m["tag"]),
            "label": m["label"], "collection": m["collection"],
            "z": m["z"], "h": m["h"], "w": m["w"], "cols": m["cols"],
            "psnr": m["psnr"], "ssim": m["ssim"], "voxel_r2": m["voxel_r2"],
            "slope": m["slope"], "hot": m["hot"], "imgs": imgs,
        })

    # Keep the tabs in a fixed, meaningful order rather than whatever the packer emitted.
    rank = {"best": 0, "median": 1, "lung": 2, "bladder": 3, "uterine": 4,
            "p25": 5, "worst": 6}
    payload.sort(key=lambda p: rank.get(p["tag"], 9))

    with open(args.template, "r", encoding="utf-8") as fh:
        html = fh.read()
    if "__DATA__" not in html:
        raise SystemExit("the template has no __DATA__ marker")
    # separators drops the spaces json.dumps would otherwise add between every one of the
    # ~11000 numbers; </ is escaped so a base64 run can never close the script tag early.
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<" + chr(92) + "/")
    html = html.replace("__DATA__", blob)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(html)

    print("%s  %.2f MB" % (args.out, os.path.getsize(args.out) / 1e6))
    for p in payload:
        print("  %-16s %-16s z=%3d  %5.2f dB" % (p["title"], p["label"][:16], p["z"], p["psnr"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
