#
# Makefile for ML lecture slides
#
# Costas Andreopoulos <constantinos.andreopoulos@cern.ch>
#

MODULE  = ML
ACYEAR  = 202425
VERSION = vrs_01

part_01   : NAME = ${MODULE}_${ACYEAR}-slides-${VERSION}-part_01
part_02   : NAME = ${MODULE}_${ACYEAR}-slides-${VERSION}-part_02
part_03   : NAME = ${MODULE}_${ACYEAR}-slides-${VERSION}-part_03
part_04   : NAME = ${MODULE}_${ACYEAR}-slides-${VERSION}-part_04
part_05   : NAME = ${MODULE}_${ACYEAR}-slides-${VERSION}-part_05
part_06   : NAME = ${MODULE}_${ACYEAR}-slides-${VERSION}-part_06
part_07   : NAME = ${MODULE}_${ACYEAR}-slides-${VERSION}-part_07
part_08   : NAME = ${MODULE}_${ACYEAR}-slides-${VERSION}-part_08
part_09   : NAME = ${MODULE}_${ACYEAR}-slides-${VERSION}-part_09
part_10   : NAME = ${MODULE}_${ACYEAR}-slides-${VERSION}-part_10
all       : NAME = ${MODULE}_${ACYEAR}-slides-${VERSION}
test      : NAME = ${MODULE}_${ACYEAR}-slides-${VERSION}

part_01   : tex_start add_part01 tex_end pdf 
part_02   : tex_start add_part02 tex_end pdf 
part_03   : tex_start add_part03 tex_end pdf 
part_04   : tex_start add_part04 tex_end pdf 
part_05   : tex_start add_part05 tex_end pdf 
part_06   : tex_start add_part06 tex_end pdf 
part_07   : tex_start add_part07 tex_end pdf 
part_08   : tex_start add_part08 tex_end pdf 
part_09   : tex_start add_part09 tex_end pdf 
part_10   : tex_start add_part10 tex_end pdf 

all       : 	tex_start \
		add_front_matter \
		add_module_info \
		add_part01 \
		add_part02 \
		add_part03 \
		add_part04 \
		add_part05 \
		add_part06 \
		add_part07 \
		add_part08 \
		add_part09 \
		add_part10 \
		add_back_matter \
  		tex_end \
		pdf	       

each :		part_01 \
		part_02 \
		part_03 \
		part_04 \
		part_05 \
		part_06 \
		part_07 \
		part_08 \
		part_09 \
		part_10 \

test :  	tex_start \
		add_front_matter \
		add_module_info \
		add_part01 \
		add_part02 \
		add_back_matter \
		tex_end \
		pdf	       

tex_start:
	echo '\documentclass{beamer}'         > tmp.tex
	echo '\input{settings/all.tex}'      >> tmp.tex
	echo '\\begin{document}'             >> tmp.tex

tex_end:
	echo '\end{document}' >> tmp.tex

add_front_matter: FORCE
	echo '\input{common/cover.tex}'      >> tmp.tex
	echo '\input{common/about.tex}'      >> tmp.tex
	echo '\input{common/toc.tex}'        >> tmp.tex

add_back_matter: FORCE
	echo '\input{common/bibliography.tex}' >> tmp.tex
	echo '\input{common/glossary.tex}'     >> tmp.tex
	echo '\input{common/index.tex}'        >> tmp.tex

add_module_info: FORCE
	echo '\input{parts/00/main.tex}'  >> tmp.tex

add_part01: FORCE
	echo '\input{parts/01/main.tex}'  >> tmp.tex
add_part02: FORCE
	echo '\input{parts/02/main.tex}'  >> tmp.tex
add_part03: FORCE
	echo '\input{parts/03/main.tex}'  >> tmp.tex
add_part04: FORCE
	echo '\input{parts/04/main.tex}'  >> tmp.tex
add_part05: FORCE
	echo '\input{parts/05/main.tex}'  >> tmp.tex
add_part06: FORCE
	echo '\input{parts/06/main.tex}'  >> tmp.tex
add_part07: FORCE
	echo '\input{parts/07/main.tex}'  >> tmp.tex
add_part08: FORCE
	echo '\input{parts/08/main.tex}'  >> tmp.tex
add_part09: FORCE
	echo '\input{parts/09/main.tex}'  >> tmp.tex
add_part10: FORCE
	echo '\input{parts/10/main.tex}'  >> tmp.tex

pdf: FORCE
	pdflatex tmp.tex
	biber tmp
	makeglossaries tmp
	makeindex tmp
	pdflatex tmp.tex
	mv tmp.pdf ${NAME}.pdf

clean: FORCE
	rm *nav
	rm *aux
	rm *log
	rm *out
	rm *snm
	rm *toc
	rm tmp.*

FORCE:
