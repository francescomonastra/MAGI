#!/usr/bin/env bash
# Build the MAGI user manual to PDF.
#
# TeX Live was installed into the user's home (no sudo, no system TeX
# required) with the upstream install-tl script, scheme-small, plus
# titlesec/tcolorbox/environ/trimspaces pulled in afterwards via tlmgr.
# To reproduce that install on another machine:
#
#   curl -sSL -o install-tl-unx.tar.gz \
#     https://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz
#   tar xzf install-tl-unx.tar.gz && cd install-tl-*
#   ./install-tl -profile /path/to/texlive.profile -no-interaction
#   tlmgr install titlesec tcolorbox environ trimspaces
#
# Note conda-forge's texlive-core does NOT work: the 2026 build ships the
# binaries with no LaTeX macro packages and no format files.
set -euo pipefail

TEXBIN="${TEXBIN:-$HOME/texlive/2026/bin/universal-darwin}"
if [ -d "$TEXBIN" ]; then
    export PATH="$TEXBIN:$PATH"
fi

command -v pdflatex >/dev/null 2>&1 || {
    echo "error: pdflatex not on PATH (looked in $TEXBIN)" >&2
    exit 1
}

cd "$(dirname "$0")"

# Drop any stale .bbl first. It is a build product, and pass 1 below reads
# it before bibtex gets a chance to regenerate it - so a .bbl left over from
# a broken .bib fails the build even after the .bib is fixed.
rm -f magi_manual.bbl

# Two pdflatex passes around bibtex: the first writes the .aux the
# bibliography is built from, the last resolves the citation numbers and
# the table of contents.
pdflatex -interaction=nonstopmode -halt-on-error magi_manual.tex
bibtex magi_manual || echo "warning: bibtex reported problems (see magi_manual.blg)"
pdflatex -interaction=nonstopmode -halt-on-error magi_manual.tex
pdflatex -interaction=nonstopmode -halt-on-error magi_manual.tex

# Keep the directory reviewable; the PDF and the .bbl are the outputs
# worth looking at.
rm -f magi_manual.aux magi_manual.log magi_manual.out magi_manual.toc \
      magi_manual.blg

echo
echo "built: $(pwd)/magi_manual.pdf"
