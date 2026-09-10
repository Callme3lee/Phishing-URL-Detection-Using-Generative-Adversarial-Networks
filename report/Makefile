.PHONY: all clean

all:
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
	cp main.pdf Graduation_Report.pdf

clean:
	latexmk -C
