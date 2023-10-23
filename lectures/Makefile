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
part_11   : NAME = ${MODULE}_${ACYEAR}-slides-${VERSION}-part_11
all       : NAME = ${MODULE}_${ACYEAR}-slides-${VERSION}
dev       : NAME = ${MODULE}_${ACYEAR}-slides-${VERSION}

part_01   : tex_start add_introduction       tex_end pdf 
part_02   : tex_start add_shallow_nets       tex_end pdf 
part_03   : tex_start add_deep_nets          tex_end pdf 
part_04   : tex_start add_training           tex_end pdf 
part_05   : tex_start add_convolutional_nets tex_end pdf 
part_06   : tex_start add_part06 tex_end pdf 
part_07   : tex_start add_part07 tex_end pdf 
part_08   : tex_start add_part08 tex_end pdf 
part_09   : tex_start add_part09 tex_end pdf 
part_10   : tex_start add_part10 tex_end pdf 
part_11   : tex_start add_part11 tex_end pdf 

all: 	tex_start \
		add_front_matter \
		add_glossaries \
		add_module_info \
		add_introduction \
		add_shallow_nets \
		add_deep_nets \
		add_training \
		add_convolutional_nets \
		add_back_matter \
	  	tex_end \
		pdf	       

dev: 	tex_start \
		add_front_matter \
		add_glossaries \
		add_training \
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
	echo '\input{front/main.tex}' >> tmp.tex

add_glossaries: FORCE
	echo '\input{glossaries/main.tex}' >> tmp.tex

add_back_matter: FORCE
	echo '\input{back/main.tex}' >> tmp.tex

add_module_info: FORCE
	echo '\input{parts/00/main.tex}' >> tmp.tex

# part 1
add_introduction: FORCE
	echo '\input{parts/introduction/main.tex}' >> tmp.tex

# part 2
add_part02: FORCE
	echo '\input{parts/shallow_nets/main.tex}' >> tmp.tex

# part 3
add_part03: FORCE
	echo '\input{parts/deep_nets/main.tex}'  >> tmp.tex

# part 4
add_training: FORCE
	echo '\input{parts/training/main.tex}' >> tmp.tex

# part 5
add_part06: FORCE
	echo '\input{parts/convolutional_nets/main.tex}'  >> tmp.tex

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
add_part11: FORCE
	echo '\input{parts/11/main.tex}'  >> tmp.tex

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
