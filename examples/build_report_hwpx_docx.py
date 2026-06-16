"""Build a hwpx-targeted docx: no footnotes, no paragraph styles for structure.

Differences vs build_report_docx.py (for the Hancom hwpx target):
  - footnotes/endnotes removed (term notes live inline + in 부록 C),
  - ATX headings flattened to bold, manually-numbered paragraphs (no Heading
    styles); hierarchy comes from the number scheme (1 / 1.1 / 가. / ○ / -) and
    direct font sizing, not from named styles,
  - blockquotes flattened to plain paragraphs,
  - tables get DIRECT gridlines + header shading, figures/captions DIRECT-centered.

Produces docs/EVALUATION_REPORT_hwp.docx; convert that to hwpx with Hancom.

    uv run --with pypandoc-binary python examples/build_report_hwpx_docx.py
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import pypandoc

DOCS = Path("docs")
MD = DOCS / "EVALUATION_REPORT.md"
TMP_MD = DOCS / "_hwp_src.md"
DOCX = DOCS / "EVALUATION_REPORT_hwp.docx"

BORDER = ('<w:tblBorders>'
          '<w:top w:val="single" w:sz="6" w:space="0" w:color="7F7F7F"/>'
          '<w:left w:val="single" w:sz="6" w:space="0" w:color="7F7F7F"/>'
          '<w:bottom w:val="single" w:sz="6" w:space="0" w:color="7F7F7F"/>'
          '<w:right w:val="single" w:sz="6" w:space="0" w:color="7F7F7F"/>'
          '<w:insideH w:val="single" w:sz="6" w:space="0" w:color="BFBFBF"/>'
          '<w:insideV w:val="single" w:sz="6" w:space="0" w:color="BFBFBF"/>'
          '</w:tblBorders>')
SHD = '<w:shd w:val="clear" w:color="auto" w:fill="D9E2F3"/>'


def preprocess_markdown() -> None:
    """Strip footnotes, flatten headings → bold paragraphs, flatten blockquotes."""
    lines = MD.read_text(encoding="utf-8").splitlines()
    kept = [ln for ln in lines if not re.match(r'^\[\^[^\]]+\]:', ln)]  # drop fn defs
    text = "\n".join(kept)
    text = re.sub(r'\[\^[^\]]+\]', '', text)                            # drop fn refs
    text = re.sub(r'^#{1,6}\s+(.*)$', lambda m: f"**{m.group(1).strip()}**",
                  text, flags=re.M)                                     # headings → bold
    text = re.sub(r'^>\s?', '', text, flags=re.M)                       # blockquotes → plain
    TMP_MD.write_text(text, encoding="utf-8")


def _para_text(p: str) -> str:
    return "".join(re.findall(r'<w:t[^>]*>(.*?)</w:t>', p, re.S))


def _set_size(p: str, half_pt: int) -> str:
    """Direct font size on every run of paragraph p (visual hierarchy, no styles)."""
    def fix_run(m):
        run = m.group(0)
        if '<w:sz ' in run:
            return run
        sz = f'<w:sz w:val="{half_pt}"/><w:szCs w:val="{half_pt}"/>'
        if '<w:rPr>' in run:
            return run.replace('<w:rPr>', '<w:rPr>' + sz, 1)
        return run.replace('<w:r>', '<w:r><w:rPr>' + sz + '</w:rPr>', 1)
    return re.sub(r'<w:r>.*?</w:r>', fix_run, p, flags=re.S)


def _size_headings(doc: str) -> str:
    """Bump section-number paragraphs to give hierarchy via direct sizing."""
    def fix_p(m):
        p = m.group(0)
        t = _para_text(p).strip()
        if re.match(r'^\d+\.\s', t):                 # 1. 2. ...  (level 1)
            return _set_size(p, 32)
        if re.match(r'^\d+\.\d+(\.\d+)?\s', t) or t.startswith("부록"):  # 1.1 / 4.3.1 / 부록
            return _set_size(p, 26)
        return p
    return re.sub(r'<w:p\b.*?</w:p>', fix_p, doc, flags=re.S)


def _border_tables(doc: str) -> str:
    def fix(m):
        pr = m.group(0)
        if '<w:tblBorders>' in pr:
            return pr
        if '<w:tblLook' in pr:
            return pr.replace('<w:tblLook', BORDER + '<w:tblLook', 1)
        return pr.replace('</w:tblPr>', BORDER + '</w:tblPr>', 1)
    return re.sub(r'<w:tblPr>.*?</w:tblPr>', fix, doc, flags=re.S)


def _shade_headers(doc: str) -> str:
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
        return tbl.replace(row, re.sub(r'<w:tc>.*?</w:tc>', add_shd, row, flags=re.S), 1)
    return re.sub(r'<w:tbl>.*?</w:tbl>', fix_tbl, doc, flags=re.S)


def _center_figures(doc: str) -> str:
    """DIRECT-center any paragraph holding an image or a figure caption."""
    def fix_p(m):
        p = m.group(0)
        is_fig = '<w:drawing>' in p or '[그림' in _para_text(p)
        if not is_fig or 'w:jc w:val="center"' in p:
            return p
        if '<w:pPr>' in p:
            return p.replace('<w:pPr>', '<w:pPr><w:jc w:val="center"/>', 1)
        return p.replace('<w:p>', '<w:p><w:pPr><w:jc w:val="center"/></w:pPr>', 1)
    return re.sub(r'<w:p\b.*?</w:p>', fix_p, doc, flags=re.S)


def main() -> None:
    preprocess_markdown()
    pypandoc.convert_file(str(TMP_MD), 'docx', outputfile=str(DOCX),
                          extra_args=['--resource-path=docs'])

    zin = zipfile.ZipFile(str(DOCX), 'r')
    items = {n: zin.read(n) for n in zin.namelist()}
    zin.close()

    doc = items['word/document.xml'].decode('utf-8')
    doc = _border_tables(doc)
    doc = _shade_headers(doc)
    doc = _center_figures(doc)
    doc = _size_headings(doc)
    items['word/document.xml'] = doc.encode('utf-8')

    tmp = DOCS / '_tmp_hwp.docx'
    zout = zipfile.ZipFile(str(tmp), 'w', zipfile.ZIP_DEFLATED)
    for n, d in items.items():
        zout.writestr(n, d)
    zout.close()
    tmp.replace(DOCX)
    TMP_MD.unlink(missing_ok=True)

    has_fn = 'footnoteReference' in doc or 'endnoteReference' in doc
    print(f"built {DOCX} | footnotes_present={has_fn} | tables bordered | figures centered")


if __name__ == "__main__":
    main()
