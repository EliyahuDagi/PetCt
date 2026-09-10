"""Convert docs/PROJECT_REPORT.md into one self-contained HTML page (figures inlined as data URIs).

Run from the repository root:  python scripts/build_report_html.py docs/PROJECT_REPORT.html
Add --standalone to wrap the output in a full <html> document. The plain output is a body
fragment, which is what a host supplying its own <head> wants; a file meant to be opened by
double-clicking needs the wrapper, or the browser reads it in quirks mode and guesses the
encoding, turning every minus sign in the report into mojibake.
Handles the Markdown subset the report uses: ATX headings, paragraphs, pipe tables, fenced code,
images with an italic "*Figure N.*" caption paragraph, flat bullet / numbered lists, horizontal
rules, and inline **bold**, *italic*, `code`, [links](url).
"""
import base64
import html
import os
import re
import sys

ROOT = os.getcwd()
MD_PATH = os.path.join(ROOT, "docs", "PROJECT_REPORT.md")
STANDALONE = "--standalone" in sys.argv[1:]
_args = [a for a in sys.argv[1:] if a != "--standalone"]
OUT = _args[0] if _args else "report.html"


def esc(s):
    return html.escape(s, quote=False)


def slug(text):
    s = re.sub(r"[^\w\s.-]", "", text.lower())
    s = re.sub(r"[\s.]+", "-", s).strip("-")
    return s or "section"


def inline(s):
    s = esc(s)
    codes = []

    def _code(m):
        codes.append(m.group(1))
        return "\x00%d\x00" % (len(codes) - 1)

    s = re.sub(r"`([^`]+)`", _code, s)
    s = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>', s)
    s = re.sub(r"\*\*(?=\S)(.+?)(?<=\S)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*])\*(?=[^\s*])(.+?)(?<=[^\s*])\*(?![\w*])", r"<em>\1</em>", s)
    for i, c in enumerate(codes):
        s = s.replace("\x00%d\x00" % i, "<code>%s</code>" % c)
    return s


NUM_RE = re.compile(r"^[\s+\-−±≈~]*[\d.,]+(\s*(dB|M|MB|GB|%|×))?(\s*[±/]\s*[\d.,]+)*\s*$|^[+\-−]?\d[\d.,]*\s*/\s*\d[\d.,]*.*$")


def split_row(row):
    row = row.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|"):
        row = row[:-1]
    row = re.sub(r"`[^`]*`", lambda m: m.group(0).replace("|", "\x01"), row)
    cells = [c.replace("\x01", "|").strip() for c in row.split("|")]
    return cells


def img_data_uri(rel):
    path = os.path.join(ROOT, "docs", rel)
    with open(path, "rb") as fh:
        b = base64.b64encode(fh.read()).decode("ascii")
    return "data:image/png;base64," + b


lines = open(MD_PATH, encoding="utf-8").read().split("\n")
out = []
nav = []  # (level, text, id)
i = 0
n = len(lines)
skipping_toc = False
fig_count = 0

while i < n:
    line = lines[i]
    stripped = line.strip()

    if skipping_toc:
        if stripped == "---":
            skipping_toc = False
        i += 1
        continue

    if not stripped:
        i += 1
        continue

    # fenced code
    if stripped.startswith("```"):
        lang = stripped[3:].strip()
        buf = []
        i += 1
        while i < n and not lines[i].strip().startswith("```"):
            buf.append(lines[i])
            i += 1
        i += 1
        cls = ' class="lang-%s"' % esc(lang) if lang else ""
        out.append("<pre%s><code>%s</code></pre>" % (cls, esc("\n".join(buf))))
        continue

    # headings
    m = re.match(r"^(#{1,4})\s+(.*)$", stripped)
    if m:
        level = len(m.group(1))
        text = m.group(2).strip()
        if level == 2 and text.lower().startswith("table of contents"):
            skipping_toc = True
            i += 1
            continue
        hid = slug(text)
        if level in (2, 3):
            nav.append((level, text, hid))
        out.append('<h%d id="%s">%s</h%d>' % (level, hid, inline(text), level))
        i += 1
        continue

    # horizontal rule
    if re.match(r"^-{3,}$", stripped):
        out.append("<hr>")
        i += 1
        continue

    # image (+ optional caption paragraph)
    m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", stripped)
    if m:
        alt, src = m.group(1), m.group(2)
        fig_count += 1
        j = i + 1
        while j < n and not lines[j].strip():
            j += 1
        caption = ""
        if j < n and lines[j].strip().startswith("*Figure"):
            cap = lines[j].strip()
            if cap.startswith("*") and cap.endswith("*"):
                cap = cap[1:-1]
            cm = re.match(r"^(Figure \d+\.)\s*(.*)$", cap)
            if cm:
                caption = '<figcaption><span class="fig-label">%s</span> %s</figcaption>' % (esc(cm.group(1)), inline(cm.group(2)))
            else:
                caption = "<figcaption>%s</figcaption>" % inline(cap)
            i = j + 1
        else:
            i += 1
        out.append('<figure><img src="%s" alt="%s" loading="lazy">%s</figure>' % (img_data_uri(src), esc(alt), caption))
        continue

    # table
    if stripped.startswith("|"):
        rows = []
        while i < n and lines[i].strip().startswith("|"):
            rows.append(lines[i])
            i += 1
        cells = [split_row(r) for r in rows]
        header = cells[0]
        body = [r for r in cells[1:] if not all(re.match(r"^:?-{2,}:?$", c) or c == "" for c in r)]
        ncol = max(len(r) for r in cells)
        html_rows = ['<div class="table-wrap"><table>']
        if any(h for h in header):
            html_rows.append("<thead><tr>" + "".join("<th>%s</th>" % inline(h) for h in header + [""] * (ncol - len(header))) + "</tr></thead>")
        html_rows.append("<tbody>")
        for r in body:
            r = r + [""] * (ncol - len(r))
            tds = []
            for c in r:
                cls = ' class="num"' if c and NUM_RE.match(c) else ""
                tds.append("<td%s>%s</td>" % (cls, inline(c)))
            html_rows.append("<tr>" + "".join(tds) + "</tr>")
        html_rows.append("</tbody></table></div>")
        out.append("\n".join(html_rows))
        continue

    # lists
    if re.match(r"^(-|\d+\.)\s+", stripped):
        ordered = bool(re.match(r"^\d+\.", stripped))
        items = []
        while i < n and re.match(r"^(-|\d+\.)\s+", lines[i].strip()):
            item = re.sub(r"^(-|\d+\.)\s+", "", lines[i].strip())
            i += 1
            # continuation lines (indented, non-list)
            while i < n and lines[i].strip() and not re.match(r"^(-|\d+\.)\s+", lines[i].strip()) and lines[i].startswith("  ") and not lines[i].strip().startswith("|"):
                item += " " + lines[i].strip()
                i += 1
            items.append("<li>%s</li>" % inline(item))
        tag = "ol" if ordered else "ul"
        out.append("<%s>%s</%s>" % (tag, "".join(items), tag))
        continue

    # paragraph
    buf = [stripped]
    i += 1
    while i < n and lines[i].strip() and not re.match(r"^(#{1,4}\s|```|\||!\[|-\s|\d+\.\s|-{3,}$)", lines[i].strip()):
        buf.append(lines[i].strip())
        i += 1
    text = " ".join(buf)
    cls = ""
    if re.match(r"^\*[A-Z][^*]*\*\s", text) and not text.startswith("*Figure"):
        cls = ' class="lead-in"'
    out.append("<p%s>%s</p>" % (cls, inline(text)))

# --- navigation ---
nav_html = ['<nav class="toc" aria-label="Contents"><p class="toc-title">Contents</p><ol>']
open_sub = False
for level, text, hid in nav:
    if level == 2:
        if open_sub:
            nav_html.append("</ol></li>")
            open_sub = False
        nav_html.append('<li><a href="#%s">%s</a>' % (hid, inline(text)))
        nav_html.append("<ol>")
        open_sub = True
    else:
        nav_html.append('<li><a href="#%s">%s</a></li>' % (hid, inline(text)))
if open_sub:
    nav_html.append("</ol></li>")
nav_html.append("</ol></nav>")

CSS = r"""
:root{
  --paper:#F5F7F6; --surface:#FFFFFF; --ink:#1B2228; --muted:#5A6670; --rule:#D5DCDA; --rule-soft:#E6EBE9;
  --accent:#1F6F8B; --accent-ink:#17546A; --accent-soft:#E2EEF3; --hot:#B9531B;
  --code-bg:#EDF1F0; --th-bg:#EEF2F1; --shadow:0 1px 2px rgba(20,40,50,.06);
  color-scheme:light;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --paper:#11171B; --surface:#171E23; --ink:#E4E9E7; --muted:#9BA7AE; --rule:#2C363C; --rule-soft:#222B31;
    --accent:#6DB8D2; --accent-ink:#8FCBE0; --accent-soft:#17303A; --hot:#E08A55;
    --code-bg:#1D262C; --th-bg:#1C252B; --shadow:none; color-scheme:dark;
  }
}
:root[data-theme="dark"]{
  --paper:#11171B; --surface:#171E23; --ink:#E4E9E7; --muted:#9BA7AE; --rule:#2C363C; --rule-soft:#222B31;
  --accent:#6DB8D2; --accent-ink:#8FCBE0; --accent-soft:#17303A; --hot:#E08A55;
  --code-bg:#1D262C; --th-bg:#1C252B; --shadow:none; color-scheme:dark;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
@media (prefers-reduced-motion: reduce){html{scroll-behavior:auto}}
body{margin:0;background:var(--paper);color:var(--ink);font-family:"Public Sans",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;font-size:17px;line-height:1.62;-webkit-font-smoothing:antialiased}
.page{display:grid;grid-template-columns:1fr;gap:0;max-width:1240px;margin:0 auto;padding:0 20px 80px}
@media (min-width:1180px){.page{grid-template-columns:250px minmax(0,1fr);gap:56px}}
main{min-width:0;max-width:960px}
/* header */
.masthead{padding:56px 0 28px;border-bottom:1px solid var(--rule);margin-bottom:36px}
.masthead .kicker{font-family:"JetBrains Mono",ui-monospace,Menlo,Consolas,monospace;font-size:.78rem;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);margin:0 0 14px}
h1{font-family:"Newsreader","Iowan Old Style",Georgia,serif;font-weight:500;font-size:clamp(2rem,3.4vw,2.7rem);line-height:1.15;letter-spacing:-.01em;margin:0 0 18px;text-wrap:balance;max-width:24ch}
h2{font-family:"Newsreader","Iowan Old Style",Georgia,serif;font-weight:500;font-size:1.75rem;line-height:1.2;margin:64px 0 18px;padding-top:18px;border-top:2px solid var(--ink);text-wrap:balance}
h3{font-family:"Newsreader","Iowan Old Style",Georgia,serif;font-weight:500;font-size:1.32rem;line-height:1.25;margin:44px 0 12px;text-wrap:balance}
h4{font-family:"Public Sans",system-ui,sans-serif;font-weight:600;font-size:1.02rem;line-height:1.35;margin:34px 0 10px;color:var(--accent-ink);text-wrap:balance}
p,li{max-width:72ch}
p{margin:0 0 1.05em}
ul,ol{margin:0 0 1.2em;padding-left:1.4em}
li{margin:.35em 0}
li::marker{color:var(--muted)}
a{color:var(--accent-ink);text-decoration:underline;text-decoration-color:var(--rule);text-underline-offset:3px}
a:hover{text-decoration-color:var(--accent)}
a:focus-visible,button:focus-visible,summary:focus-visible{outline:2px solid var(--accent);outline-offset:3px;border-radius:2px}
strong{font-weight:600}
hr{border:0;border-top:1px solid var(--rule);margin:48px 0}
.lead-in em:first-child{color:var(--accent-ink)}
/* code */
code,pre{font-family:"JetBrains Mono",ui-monospace,Menlo,Consolas,monospace;font-size:.86em}
code{background:var(--code-bg);padding:.1em .38em;border-radius:3px;word-break:break-word}
pre{background:var(--code-bg);border:1px solid var(--rule-soft);border-radius:4px;padding:14px 16px;overflow-x:auto;line-height:1.5;margin:0 0 1.3em;max-width:100%}
pre code{background:none;padding:0;font-size:.84rem}
/* tables */
.table-wrap{overflow-x:auto;margin:0 0 1.5em;border:1px solid var(--rule);border-radius:4px;background:var(--surface);box-shadow:var(--shadow)}
table{border-collapse:collapse;width:100%;font-size:.9rem;line-height:1.4}
th,td{padding:8px 12px;text-align:left;vertical-align:top;border-bottom:1px solid var(--rule-soft)}
th{background:var(--th-bg);font-weight:600;font-size:.8rem;letter-spacing:.02em;color:var(--muted);white-space:nowrap}
tbody tr:last-child td{border-bottom:0}
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
td code,th code{font-size:.82em}
/* figures */
figure{margin:28px 0 36px;background:var(--surface);border:1px solid var(--rule);border-radius:4px;padding:14px 14px 12px;box-shadow:var(--shadow)}
figure img{display:block;width:100%;height:auto;max-width:100%;border-radius:2px}
figcaption{margin-top:12px;font-size:.9rem;line-height:1.5;color:var(--muted);max-width:none}
.fig-label{font-family:"JetBrains Mono",ui-monospace,Menlo,Consolas,monospace;font-size:.78rem;letter-spacing:.06em;text-transform:uppercase;color:var(--hot);margin-right:4px}
/* toc */
.toc{display:none}
@media (min-width:1180px){
  .toc{display:block;position:sticky;top:0;align-self:start;max-height:100vh;overflow-y:auto;padding:56px 0 40px;font-size:.84rem;line-height:1.45;scrollbar-width:thin}
}
.toc-title{font-family:"JetBrains Mono",ui-monospace,Menlo,Consolas,monospace;font-size:.72rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin:0 0 12px}
.toc ol{list-style:none;margin:0;padding:0}
.toc>ol>li{margin:0 0 10px}
.toc>ol>li>a{font-weight:600;color:var(--ink);text-decoration:none}
.toc ol ol{margin:4px 0 0 0;padding-left:12px;border-left:1px solid var(--rule)}
.toc ol ol li{margin:3px 0}
.toc ol ol a{color:var(--muted);text-decoration:none}
.toc a:hover{color:var(--accent)}
.toc-mobile{margin:0 0 32px;border:1px solid var(--rule);border-radius:4px;background:var(--surface)}
.toc-mobile summary{cursor:pointer;padding:12px 16px;font-weight:600;font-size:.92rem}
.toc-mobile .toc{display:block;position:static;padding:0 16px 16px;max-height:none}
@media (min-width:1180px){.toc-mobile{display:none}}
/* meta table in the masthead keeps its own quieter look */
.masthead .table-wrap{border:0;box-shadow:none;background:transparent;margin:0}
.masthead table{font-size:.92rem}
.masthead th{display:none}
.masthead td:first-child{color:var(--muted);white-space:nowrap;width:1%;padding-left:0}
.masthead td{border-bottom:1px solid var(--rule-soft)}
.masthead .intro{margin-top:22px;color:var(--muted);font-size:.98rem}
@media print{
  body{font-size:11pt;background:#fff;color:#000}
  .toc,.toc-mobile{display:none!important}
  .page{display:block;padding:0}
  figure,.table-wrap,pre{break-inside:avoid;box-shadow:none}
  h2{break-before:page;border-top:0;margin-top:0}
  a{color:inherit;text-decoration:none}
}
"""

# --- split the masthead (everything before the first <hr>) so it can be styled apart ---
body_html = "\n".join(out)
first_hr = body_html.find("<hr>")
head_part, rest = body_html[:first_hr], body_html[first_hr + 4:]
# the first <h1> is the title; the bold "Project report" line becomes the kicker; the intro paragraph gets a class
head_part = head_part.replace("<p><strong>Project report</strong></p>", "", 1)
head_part = re.sub(r"<h1 id=\"[^\"]+\">", '<p class="kicker">Project report · 2026-09-02</p><h1>', head_part, count=1)
head_part = head_part.replace("<p>This report is written for", '<p class="intro">This report is written for', 1)

doc = """<title>CT-Free PET Correction</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&family=Public+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>%s</style>
<div class="page">
%s
<main>
<details class="toc-mobile"><summary>Contents</summary>%s</details>
<header class="masthead">
%s
</header>
%s
</main>
</div>
""" % (CSS, "\n".join(nav_html), "\n".join(nav_html), head_part, rest)


# A body fragment by default. --standalone closes it into a real document so the file can be
# opened straight from disk: the charset is the part that matters, not the doctype.
HTML_OPEN = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
"""
BODY_OPEN = """</head>
<body>
<div class="page">"""
HTML_CLOSE = """</body>
</html>
"""

if STANDALONE:
    doc = HTML_OPEN + doc.replace('<div class="page">', BODY_OPEN, 1) + HTML_CLOSE

open(OUT, "w", encoding="utf-8").write(doc)
print("wrote", OUT, "bytes", os.path.getsize(OUT), "figures", fig_count,
      "nav entries", len(nav), "standalone" if STANDALONE else "fragment")
