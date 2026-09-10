"""BUILD THE HANDOVER FOLDER: the report, the viewer, and the configs the report links to.

The two html files are the deliverable; everything they need travels with them. The report
is rebuilt from the markdown with ``--standalone`` (the copy in ``docs/`` is a body fragment
and would open in quirks mode with the wrong encoding), and the three config files it links
to are copied in beside it so the links still resolve once the folder leaves the repository.

Run it after changing the report markdown, the viewer, or the configs -- it is the whole
package, not a patch, so it is always safe to re-run:

    python scripts/make_handover.py

    --out       where to build          (default handover/nac_to_ac_report)
    --no-zip    skip the archive
"""

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The report links to these by relative path; they move into configs/ inside the package.
CONFIGS = ["ae3d_slab.yaml", "ft3d_slab_flow.yaml", "ft3d_slab_flow_ae3d.yaml"]

README = """CT-free PET correction - predicting corrected PET from non-corrected PET
========================================================================
Eli Dagi.  Handover package, {date}.

WHAT TO OPEN
------------
1. PROJECT_REPORT.html   - start here.  Double-click it; it opens in any
                           browser.  It is the whole write-up: the problem,
                           the method, the numbers, what is and is not
                           claimed, the data and its split, and how to
                           reproduce it.
2. nac_to_ac_viewer.html - the pictures.  Input / prediction / truth side by
                           side for {n_patients} held-out patients, every slice, in all
                           three planes.  Drag or use the arrow keys to move
                           through slices.  Section 6 of the report links
                           straight to it.

Everything is inside these two files - images included.  No install, no
server, no internet needed.  Copy the folder anywhere and it still works.
(With internet the report picks up nicer web fonts; without, it falls back
to system fonts and reads the same.)

THE PATIENTS IN THE VIEWER
--------------------------
{patient_lines}
Three are picked by rank so the file cannot flatter the model - the best,
the median and the worst of the 41 test patients.  The other three are
ordinary cases at three different cancer sites, all within about a decibel
of the 26.9 dB test mean.

THE RESULT IN ONE LINE
----------------------
Over 41 held-out patients: the non-corrected input scores 16.6 dB PSNR and
a negative variance explained; the model turns that into 26.9 dB and 0.77,
and lifts the hot band from reading 24% low to 7% low.  Encoding and
decoding the true image scores 45.1 dB, so the remaining gap is the
generative step, not the compressor.  Details in sections 3 and 4; the data
and its train/validation/test split are in section 11.

ALSO IN HERE
------------
configs/  the three YAML files that define the two training stages of the
          main experiment.  The report links to them from section 9.
            ae3d_slab.yaml            the 3D autoencoder
            ft3d_slab_flow.yaml       the flow bridge, 2D-autoencoder chain
            ft3d_slab_flow_ae3d.yaml  the flow bridge, 3D-autoencoder chain
                                      (this is the main experiment)

Code, data and checkpoints are not in this package - the report says in
section 9 which scripts produce each result.
"""


def report_date(md_path):
    """The report states its own date in the summary table; keep the package in step."""
    text = open(md_path, encoding="utf-8").read()
    m = re.search(r"\|\s*Report date\s*\|\s*([0-9]{4}-[0-9]{2}-[0-9]{2})\s*\|", text)
    return m.group(1) if m else "undated"


def viewer_patients(viewer_html):
    """Read back what the built viewer actually holds, rather than assuming."""
    m = re.search(r"const PATIENTS\s*=\s*(\[.*?\]);\n", viewer_html, re.S)
    if not m:
        raise SystemExit("cannot find the patient list in the viewer -- did the template change?")
    return json.loads(m.group(1))


def check_selfcontained(name, text, expect_doctype=True):
    """A page that reaches for a file we are not shipping is a broken handover."""
    outside = [u for u in re.findall(r'(?:src|href)="([^"]+)"', text)
               if not u.startswith(("data:", "#", "http://", "https://"))]
    problems = []
    for u in outside:
        if not os.path.exists(os.path.join(os.path.dirname(name), u)):
            problems.append("links to a file that is not in the package: " + u)
    if expect_doctype and not text.lstrip().lower().startswith("<!doctype html>"):
        problems.append("no doctype")
    if "meta charset" not in text:
        problems.append("no charset -- minus signs will come out as mojibake")
    for b in re.findall(r"base64,([A-Za-z0-9+/=]+)", text):
        try:
            base64.b64decode(b, validate=True)
        except Exception:
            problems.append("a broken embedded image")
            break
    if problems:
        raise SystemExit("%s: %s" % (os.path.basename(name), "; ".join(problems)))
    return len(re.findall(r"base64,", text)), outside


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "handover", "nac_to_ac_report"))
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()

    docs = os.path.join(ROOT, "docs")
    out = os.path.abspath(args.out)
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "configs"))

    # 1. the report, as a document that can be opened from disk rather than a fragment
    report = os.path.join(out, "PROJECT_REPORT.html")
    subprocess.check_call([sys.executable, os.path.join(ROOT, "scripts", "build_report_html.py"),
                           report, "--standalone"], cwd=ROOT)
    text = open(report, encoding="utf-8").read()
    for name in CONFIGS:
        text = text.replace('"../src/training/configs/%s"' % name, '"configs/%s"' % name)
        shutil.copy2(os.path.join(ROOT, "src", "training", "configs", name),
                     os.path.join(out, "configs", name))
    open(report, "w", encoding="utf-8", newline="\n").write(text)

    # 2. the viewer is already a complete standalone document
    viewer = os.path.join(out, "nac_to_ac_viewer.html")
    shutil.copy2(os.path.join(docs, "nac_to_ac_viewer.html"), viewer)
    patients = viewer_patients(open(viewer, encoding="utf-8").read())

    # 3. the readme, describing whatever actually went in
    lines = "".join("  %-16s %-11s %-21s %5.1f dB\n"
                    % (p["title"], p["label"][:11], p["collection"][:21], p["psnr"])
                    for p in patients)
    open(os.path.join(out, "README.txt"), "w", encoding="utf-8", newline="\r\n").write(
        README.format(date=report_date(os.path.join(docs, "PROJECT_REPORT.md")),
                      n_patients=len(patients), patient_lines=lines))

    n_report, _ = check_selfcontained(report, open(report, encoding="utf-8").read())
    n_viewer, _ = check_selfcontained(viewer, open(viewer, encoding="utf-8").read())
    print("report: %d figures embedded, %d config links rewritten" % (n_report, len(CONFIGS)))
    print("viewer: %d patients, %d images embedded" % (len(patients), n_viewer))

    total = 0
    for dirpath, _, files in os.walk(out):
        for f in sorted(files):
            p = os.path.join(dirpath, f)
            total += os.path.getsize(p)
            print("  %8.2f MB  %s" % (os.path.getsize(p) / 1e6, os.path.relpath(p, out)))

    if not args.no_zip:
        z = shutil.make_archive(out, "zip", root_dir=os.path.dirname(out),
                                base_dir=os.path.basename(out))
        print("folder %.1f MB -> %s (%.1f MB)" % (total / 1e6, z, os.path.getsize(z) / 1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
