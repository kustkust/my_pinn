pdflatex %1.tex
makeindex %1.nlo -s nomencl.ist -o %1.nls
bibtex %1
pdflatex %1.tex
pdflatex %1.tex
