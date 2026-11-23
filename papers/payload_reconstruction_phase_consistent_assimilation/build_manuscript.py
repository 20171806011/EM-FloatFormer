"""Build the editable manuscript DOCX from the unchanged Markdown and figures.

Requires Pandoc on PATH. The reference document is generated programmatically.
"""
from pathlib import Path
import argparse
import shutil
import subprocess
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT

WORK = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, default=WORK / "build",
                    help="Build directory (default: build/ beside this script).")
args = parser.parse_args()
OUT = args.output_dir.expanduser().resolve()
OUT.mkdir(parents=True, exist_ok=True)
REFERENCE = OUT / "reference.docx"
TARGET = OUT / "EM_FloatFormer_Manuscript.docx"
if shutil.which("pandoc") is None:
    parser.error("Pandoc must be installed and available on PATH.")

doc = Document()
sec = doc.sections[0]
sec.page_width = Inches(8.5)
sec.page_height = Inches(11)
sec.top_margin = Inches(0.78)
sec.bottom_margin = Inches(0.78)
sec.left_margin = Inches(1)
sec.right_margin = Inches(1)
sec.footer_distance = Inches(0.35)
styles = doc.styles
for name in ["Normal", "Body Text", "First Paragraph", "Compact"]:
    try:
        st = styles[name]
    except KeyError:
        st = styles.add_style(name, 1)
    st.font.name = "Times New Roman"
    st.font.size = Pt(11)
    st.font.color.rgb = RGBColor(0, 0, 0)
    pf = st.paragraph_format
    pf.space_after = Pt(5)
    pf.line_spacing = 1.10
    pf.widow_control = True
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
for name, size in [("Title", 20), ("Heading 1", 14), ("Heading 2", 12)]:
    st = styles[name]
    st.font.name = "Times New Roman"
    st.font.size = Pt(size)
    st.font.color.rgb = RGBColor(0, 0, 0)
    st.font.bold = True
    pf = st.paragraph_format
    pf.space_before = Pt(13 if name != "Title" else 0)
    pf.space_after = Pt(7 if name != "Title" else 15)
    pf.keep_with_next = True
    pf.keep_together = True
    pf.widow_control = True
    pf.line_spacing = 1.03
    pf.alignment = WD_ALIGN_PARAGRAPH.CENTER if name == "Title" else WD_ALIGN_PARAGRAPH.LEFT
for name in ["Caption", "Image Caption"]:
    try:
        st = styles[name]
    except KeyError:
        st = styles.add_style(name, 1)
    st.font.name = "Times New Roman"
    st.font.size = Pt(10)
    st.font.italic = False
    st.font.bold = False
    st.font.color.rgb = RGBColor(0, 0, 0)
    st.paragraph_format.space_after = Pt(9)
    st.paragraph_format.line_spacing = 1.05
    st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
if "Hyperlink" not in styles:
    styles.add_style("Hyperlink", 2)
styles["Hyperlink"].font.color.rgb = RGBColor.from_string("204B70")
footer = sec.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = footer.add_run()
run.font.name = "Times New Roman"
run.font.size = Pt(9)
fld = OxmlElement("w:fldSimple")
fld.set(qn("w:instr"), "PAGE")
run._r.addnext(fld)
doc.save(REFERENCE)

subprocess.run(
    ["pandoc", str(WORK / "manuscript.md"), "--from=markdown+tex_math_dollars-implicit_figures",
     "--to=docx", "--standalone", "--reference-doc", str(REFERENCE),
     "--resource-path", str(WORK), "--output", str(TARGET)],
    check=True
)
doc = Document(TARGET)
delimiter_order = {"begChr": 0, "sepChr": 1, "endChr": 2, "grow": 3, "shp": 4, "ctrlPr": 5}
for pr in doc._element.xpath(".//m:dPr"):
    children = list(pr)
    for child in children:
        pr.remove(child)
    for child in sorted(children, key=lambda el: delimiter_order.get(el.tag.rsplit("}", 1)[-1], 99)):
        pr.append(child)
for st in doc.styles:
    for el in st._element.xpath(".//w:pBdr"):
        el.getparent().remove(el)
    for rf in st._element.xpath(".//w:rFonts"):
        for attr in list(rf.attrib):
            if "theme" in attr.lower():
                del rf.attrib[attr]
for el in doc._element.xpath(".//w:pBdr"):
    el.getparent().remove(el)
doc.core_properties.title = "Payload Aided Reconstruction of Dynamic Multipath Channels with Phase Consistent Codeword Assimilation"
doc.core_properties.subject = "Semi-blind channel reconstruction and selective codeword assimilation"
doc.core_properties.author = ""
doc.core_properties.last_modified_by = ""
doc.core_properties.keywords = "channel estimation; coded reception; multipath; phase; covariance"
doc.core_properties.comments = ""
for p in doc.paragraphs:
    text = p.text.strip()
    previous = p._p.getprevious()
    if previous is not None and previous.tag == qn("w:tbl"):
        p.paragraph_format.space_before = Pt(7)
    if text.startswith("Figure ") or text.startswith("Table "):
        p.style = doc.styles["Caption"]
        p.paragraph_format.keep_together = True
        if text.startswith("Table "):
            p.paragraph_format.keep_with_next = True
    if p._p.xpath(".//m:oMathPara"):
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(7)
        p.paragraph_format.keep_together = True
    if p._p.xpath(".//w:drawing"):
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.space_before = Pt(7)
        p.paragraph_format.space_after = Pt(4)
    if text.startswith("[") and text[1:2].isdigit():
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.05
        p.paragraph_format.keep_together = True
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
        for run in p.runs:
            run.font.size = Pt(10)
    if text == "References":
        p.paragraph_format.page_break_before = True
    if text == "Appendix A. Conditional gain integration":
        p.paragraph_format.page_break_before = True
    if text == "Appendix B. Admission parameters and computational scope":
        p.paragraph_format.page_break_before = True
    if text == "1. Introduction":
        p.paragraph_format.page_break_before = True
    if text.startswith("Algorithm 1."):
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.space_before = Pt(8)

for ti, table in enumerate(doc.tables):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    widths = [0.4, 1.55, 2.75, 1.8] if ti == 0 else [2.7, 0.95, 0.95, 0.95, 0.95]
    for col, width in zip(table.columns, widths):
        col.width = Inches(width)
    pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ["top", "left", "bottom", "right", "insideH", "insideV"]:
        b = OxmlElement("w:" + edge)
        b.set(qn("w:val"), "single")
        b.set(qn("w:sz"), "4")
        b.set(qn("w:color"), "D9D9D9")
        borders.append(b)
    old_borders = pr.find(qn("w:tblBorders"))
    if old_borders is not None:
        pr.remove(old_borders)
    pr.append(borders)
    for ri, row in enumerate(table.rows):
        trpr = row._tr.get_or_add_trPr()
        cant = OxmlElement("w:cantSplit")
        trpr.append(cant)
        if ri == 0:
            repeat = OxmlElement("w:tblHeader")
            trpr.append(repeat)
        for ci, cell in enumerate(row.cells):
            cell.width = Inches(widths[ci])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            tcpr = cell._tc.get_or_add_tcPr()
            mar = OxmlElement("w:tcMar")
            for edge, size in [("top", 70), ("bottom", 70), ("left", 90), ("right", 90)]:
                el = OxmlElement("w:" + edge)
                el.set(qn("w:w"), str(size))
                el.set(qn("w:type"), "dxa")
                mar.append(el)
            tcpr.append(mar)
            if ri == 0:
                shade = OxmlElement("w:shd")
                shade.set(qn("w:fill"), "F1F3F5")
                tcpr.append(shade)
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(2)
                p.paragraph_format.space_before = Pt(2)
                p.paragraph_format.line_spacing = 1.03
                p.paragraph_format.keep_with_next = False
                p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT if ci == 0 or ti == 0 else WD_ALIGN_PARAGRAPH.RIGHT
                for run in p.runs:
                    run.font.name = "Times New Roman"
                    run.font.size = Pt(10)
                    if ri == 0:
                        run.font.bold = True
doc.save(TARGET)
print(TARGET)
print("Paragraphs:", len(doc.paragraphs), "Tables:", len(doc.tables), "Images:", len(doc.inline_shapes))
