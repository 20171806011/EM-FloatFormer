"""Build the Chinese research report from NN.md source files using Pandoc and XeLaTeX.

Requires Python, Pandoc, TeX Live/XeLaTeX, Noto Serif/Sans CJK SC, Noto Serif,
Latin Modern Math and the LaTeX packages listed in layout.tex.
"""
from pathlib import Path
import argparse
import os
import shutil
import subprocess


def run(command, work, env, logfile):
    with logfile.open("w") as log:
        status = subprocess.run(command, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT)
    if status.returncode:
        raise RuntimeError("\n".join(logfile.read_text(errors="replace").splitlines()[-32:]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pages", type=Path)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--flow", action="store_true", help="Allow the next section to start on the same page.")
    args = parser.parse_args()
    layout = Path(__file__).resolve().parent
    cache = layout / "build"
    cache.mkdir(exist_ok=True)
    sources = sorted(args.pages.glob("[0-9][0-9].md"))
    if not sources:
        parser.error("expected at least one NN.md file")
    separator = "\n\n" if args.flow else "\n\n\\clearpage\n\n"
    combined = cache / "combined.md"
    combined.write_text(separator.join(p.read_text().strip() for p in sources) + "\n")
    env = os.environ.copy()
    if Path("/usr/share/texlive/texmf-dist").is_dir():
        env["TEXMF"] = "{" + str(cache) + ",/etc/texmf,/usr/share/texmf,/usr/share/texlive/texmf-dist}"
        env["TEXFORMATS"] = str(cache) + ":"
    if not (cache / "xelatex.fmt").exists():
        run(["xetex","-ini","-etex","-jobname=xelatex","-interaction=nonstopmode","-halt-on-error","xelatex.ini"],cache,env,cache/"init.out")
    run(["pandoc",str(combined),"--from=markdown+tex_math_single_backslash+tex_math_dollars","-t","latex","--template",str(layout/"layout.tex"),"--lua-filter",str(layout/"widths.lua"),"--no-highlight","--wrap=none","-o","report.tex"],cache,env,cache/"pandoc.out")
    for i in range(2):
        run(["xelatex","-interaction=nonstopmode","-halt-on-error","-file-line-error","report.tex"],cache,env,cache/f"pass{i}.out")
    args.pdf.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(cache/"report.pdf",args.pdf)
    print(f"Rendered {len(sources)} sections to {args.pdf}")


if __name__ == "__main__":
    main()
