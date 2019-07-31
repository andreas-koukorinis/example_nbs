# LaTeX Makefile
# Fixed targets:
.PHONY: clean

all: main.pdf documentation.pdf

clean:
	rm *.pdf *.aux *.blg *.out *.bbl *.log *.nls *.nlo *.ilg *.bcf *.toc

main.pdf: main.tex src/*.tex
	pdflatex -shell-escape main
	makeindex main.nlo -s nomencl.ist -o main.nls
	pdflatex -shell-escape main
	bibtex main
	pdflatex -shell-escape main
	pdflatex -shell-escape main

documentation.pdf: documentation.tex
	pdflatex -shell-escape documentation
