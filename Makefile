build-pdf: 
	pandoc -N -V block-headings --citeproc --toc --toc-depth=4 --reference-links -o thomas_chaigneau_report.pdf --from=markdown+tex_math_single_backslash+tex_math_dollars --to=latex --pdf-engine=xelatex --template=eisvogel content/index.md content/report/01_declaration.md content/report/02_introduction.md content/report/03_sota.md content/report/04_mkrich.md content/report/05_conclusion.md content/report/06_glossary.md content/report/07_appendix.md

build-sota:
	pandoc -N -V block-headings --citeproc --toc --toc-depth=4 --reference-links -o thomas_chaigneau_sota.pdf --from=markdown+tex_math_single_backslash+tex_math_dollars --to=latex --pdf-engine=xelatex --template=eisvogel content/index_sota.md content/report/03_sota.md

build-diapo:
	marp content/diapo_pco.md  -o content/diapo_pco.html

build-pptx:
	marp content/diapo_pco.md -o content/diapo_pco.pptx