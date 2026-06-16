"""Build docs/EVALUATION_REPORT.docx from the markdown (one-command rebuild).

pandoc(=pypandoc) converts the markdown → docx, then the docx styles are
post-processed so the result looks like a finished report:
  - figures and their captions are centered (CaptionedFigure / ImageCaption),
  - all tables get visible gridlines + a shaded, bold header row.

Run after editing docs/EVALUATION_REPORT.md:
    uv run --with pypandoc-binary python examples/build_report_docx.py
"""

from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

import pypandoc

DOCS = Path("docs")
MD = DOCS / "EVALUATION_REPORT.md"
DOCX = DOCS / "EVALUATION_REPORT.docx"

BORDER = ('<w:tblBorders>'
          '<w:top w:val="single" w:sz="6" w:space="0" w:color="7F7F7F"/>'
          '<w:left w:val="single" w:sz="6" w:space="0" w:color="7F7F7F"/>'
          '<w:bottom w:val="single" w:sz="6" w:space="0" w:color="7F7F7F"/>'
          '<w:right w:val="single" w:sz="6" w:space="0" w:color="7F7F7F"/>'
          '<w:insideH w:val="single" w:sz="6" w:space="0" w:color="BFBFBF"/>'
          '<w:insideV w:val="single" w:sz="6" w:space="0" w:color="BFBFBF"/>'
          '</w:tblBorders>')
SHD = '<w:shd w:val="clear" w:color="auto" w:fill="D9E2F3"/>'


def _center_style(xml: str, sid: str) -> str:
    pat = re.compile(r'(<w:style[^>]*w:styleId="' + sid + r'"[^>]*>)(.*?)(</w:style>)', re.S)
    m = pat.search(xml)
    if not m:
        return xml
    head, body, tail = m.groups()
    if '<w:jc ' in body:
        body = re.sub(r'<w:jc w:val="[^"]*"/>', '<w:jc w:val="center"/>', body)
    elif '<w:pPr>' in body:
        body = body.replace('</w:pPr>', '<w:jc w:val="center"/></w:pPr>', 1)
    else:
        nm = re.search(r'(<w:name[^>]*/>)', body)
        ins = '<w:pPr><w:jc w:val="center"/></w:pPr>'
        body = body.replace(nm.group(1), nm.group(1) + ins, 1) if nm else ins + body
    return xml[:m.start()] + head + body + tail + xml[m.end():]


def _border_tables(docxml: str) -> str:
    """Inject gridlines into every table's tblPr (the 'Table' style is referenced
    but not defined, so per-table direct borders are what actually render)."""
    def fix(m):
        pr = m.group(0)
        if '<w:tblBorders>' in pr:
            return pr
        if '<w:tblLook' in pr:           # tblBorders must precede tblLook
            return pr.replace('<w:tblLook', BORDER + '<w:tblLook', 1)
        return pr.replace('</w:tblPr>', BORDER + '</w:tblPr>', 1)
    return re.sub(r'<w:tblPr>.*?</w:tblPr>', fix, docxml, flags=re.S)


def _shade_headers(docxml: str) -> str:
    """Shade each table's first (header) row for readability."""
    def add_shd(cm):
        tc = cm.group(0)
        if '<w:shd' in tc:
            return tc
        if '<w:tcPr>' in tc:
            return tc.replace('</w:tcPr>', SHD + '</w:tcPr>', 1)
        return tc.replace('<w:tc>', '<w:tc><w:tcPr>' + SHD + '</w:tcPr>', 1)

    def fix_tbl(m):
        tbl = m.group(0)
        rm = re.search(r'<w:tr\b.*?</w:tr>', tbl, re.S)
        if not rm:
            return tbl
        row = rm.group(0)
        new_row = re.sub(r'<w:tc>.*?</w:tc>', add_shd, row, flags=re.S)
        return tbl.replace(row, new_row, 1)
    return re.sub(r'<w:tbl>.*?</w:tbl>', fix_tbl, docxml, flags=re.S)


def main() -> None:
    pypandoc.convert_file(
        str(MD), 'docx', outputfile=str(DOCX),
        extra_args=['--toc', '--toc-depth=2', '--resource-path=docs',
                    '--metadata', 'title=다중 카메라 3D 스켈레톤 포즈 추정 시스템 성능평가 보고서',
                    '--metadata', 'author=기술 성능평가 (초안)',
                    '--metadata', 'date=2026-06-16'])

    zin = zipfile.ZipFile(str(DOCX), 'r')
    items = {n: zin.read(n) for n in zin.namelist()}
    zin.close()

    styles = items['word/styles.xml'].decode('utf-8')
    for sid in ('CaptionedFigure', 'ImageCaption'):
        styles = _center_style(styles, sid)
    items['word/styles.xml'] = styles.encode('utf-8')

    doc = items['word/document.xml'].decode('utf-8')
    doc = _border_tables(doc)
    doc = _shade_headers(doc)
    items['word/document.xml'] = doc.encode('utf-8')

    tmp = DOCS / '_tmp.docx'
    zout = zipfile.ZipFile(str(tmp), 'w', zipfile.ZIP_DEFLATED)
    for n, d in items.items():
        zout.writestr(n, d)
    zout.close()
    shutil.move(str(tmp), str(DOCX))

    media = len([n for n in items if 'media/' in n])
    print(f"built {DOCX} | images={media} | figures+captions centered | tables bordered")


if __name__ == "__main__":
    main()
