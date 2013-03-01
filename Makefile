all:
	pdflatex obsdwarf
	pdflatex obsdwarf
	bibtex obsdwarf
	pdflatex obsdwarf

tar:
	git archive --format=tar --output=obsdwarf.tar master
