all:
	pdflatex obsdwarf
	pdflatex obsdwarf
	bibtex obsdwarf
	pdflatex obsdwarf

tar:
	git archive --format=tar --output=obsdwarf.tar master

tar-submit:
	tar -cvf obsdwarf_submit.tar obsdwarf.pdf obsdwarf.tex dwarfdraft.bib obsdwarf.bbl mn2e.cls figures
