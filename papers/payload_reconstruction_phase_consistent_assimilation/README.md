# Payload Aided Reconstruction of Dynamic Multipath Channels with Phase Consistent Codeword Assimilation

This directory contains the English manuscript, its final PDF and editable Word document, publication figures, recorded numerical results, and portable build scripts.

**[Read the paper (PDF)](EM_FloatFormer_Manuscript.pdf)** · **[Download the editable manuscript (DOCX)](EM_FloatFormer_Manuscript.docx)** · **[Read the Markdown source](manuscript.md)**

The paper studies unknown-payload covariance reconstruction and selective codeword assimilation for a coded cyclic-prefix single-carrier receiver with OFDM-derived history. Its two separately frozen synthetic studies use a known three-path model at 9 dB. The manuscript provides the signal model, mathematical arguments, receiver procedure, experimental protocol, results, and limitations.

Intended repository location: `papers/payload_reconstruction_phase_consistent_assimilation/`.

## Contents

| File | Purpose |
|:--|:--|
| `EM_FloatFormer_Manuscript.pdf` | Final 20-page paper. |
| `EM_FloatFormer_Manuscript.docx` | Final editable paper with native Word equations. |
| `manuscript.md` | Complete manuscript source, including tables, equations, captions, and references. |
| `figures/figure1_study_a_reconstruction.{png,pdf}` | Study A regime errors and common-failure recovery. |
| `figures/figure2_study_b_matched_bler.{png,pdf}` | Study B matched receiver comparison. |
| `figures/figure3_study_b_fold_geometry.{png,pdf}` | Study B residual errors and fold geometry. |
| `empirical_plot_data.json` | Recorded integer counts, rates, fold results, and related aggregate data. |
| `study_provenance.json` | Frozen commits, cohort sizes, and original source URLs. |
| `references.json` | Structured bibliography for all ten manuscript references, plus one separately identified uncited literature entry. |
| `gf2_check.json` | Recorded generator-projection rank and sign-mask syndrome checks. |
| `make_empirical_figures.py` | Generates the three figures from the bundled numerical data. |
| `build_manuscript.py` | Builds the DOCX from Markdown and the bundled figures. |
| `requirements.txt` | Python package versions used by the build scripts. |
| `SHA256SUMS` | SHA-256 hashes of every packaged file except this manifest itself. |

The figure PNGs are 220 dpi; the figure PDFs retain vector graphics and text. The final PDF, DOCX, manuscript source, and six figure files are exact copies of the completed manuscript assets.

## Frozen experimental sources

| Study | Frozen source |
|:--|:--|
| Study A — reconstruction and selection | [Commit `9e9d93bcbf6345b8645314b223cd954b2f31e63a`](https://github.com/20171806011/EM-FloatFormer/tree/9e9d93bcbf6345b8645314b223cd954b2f31e63a/p10/payload_reconstruction) |
| Study B — complete receiver | [Report commit `307d62fd89206442960fa0cebdcdc6a0530db366`](https://github.com/20171806011/EM-FloatFormer/tree/307d62fd89206442960fa0cebdcdc6a0530db366/p11) |
| Study B code freeze | `cec54bfd7835f0b586cde2863174830c2d619838` |
| Study B manifest freeze | `cb6b2bbb33608fe26a50ddb6e18555cb35df6dd4` |

Study A contains 12,288 core target blocks, with 4,096 per mechanism. Study B contains 24,576 core target blocks, with 8,192 per mechanism and 16 paired cluster aggregates. These cohorts remain separate in the data and figures. The plotting and document scripts use recorded results and do not rerun receiver training or channel simulations. Full simulation code, per-link records, and experiment protocols remain in the pinned study directories above.

## Verify the archived files

Run from this directory before regenerating outputs:

```bash
sha256sum -c SHA256SUMS
```

On macOS, the equivalent command is `shasum -a 256 -c SHA256SUMS`.

## Build environment

Use Python 3.12 and install the declared Python dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

DOCX generation additionally requires **Pandoc** on `PATH`. The completed document was built with Python 3.12.14, Pandoc 3.1.11.1, python-docx 1.2.0, NumPy 2.3.5, and Matplotlib 3.10.8. Figure text uses DejaVu Sans. Document body text uses Times New Roman; native equations require a compatible math font and Word/LibreOffice renderer. Font availability and renderer versions can affect pagination.

The scripts locate their inputs relative to their own files and can be invoked from another working directory. They have no machine-specific paths and need no network access once dependencies are installed.

## Regenerate the figures

```bash
python make_empirical_figures.py --output-dir build/figures
```

This creates all three PNG/PDF pairs under `build/figures/`, using the same filenames as the bundled assets. To regenerate the source figures in `figures/`, omit `--output-dir`. The plot script draws descriptive fractions and does not add confidence intervals or recompute the paper's statistical tests.

## Build the editable manuscript

```bash
python build_manuscript.py --output-dir build
```

The script reads `manuscript.md` and `figures/`, creates its reference formatting programmatically, and writes `build/EM_FloatFormer_Manuscript.docx`. The generated `build/reference.docx` is an intermediate formatting document. The build preserves native equations, table formatting, figure captions, and the native-equation delimiter-order correction used for the final manuscript.

To export a newly built DOCX to PDF, use Microsoft Word or a compatible LibreOffice installation. A headless LibreOffice command is:

```bash
libreoffice --headless --convert-to pdf --outdir build build/EM_FloatFormer_Manuscript.docx
```

Some installations name the executable `soffice`. The included final PDF was rendered using LibreOfficeDev 26.8.0.0.alpha0. The hash manifest identifies the committed files; regenerated documents and figure PDFs may differ in creation metadata, font rendering, or pagination across environments.
