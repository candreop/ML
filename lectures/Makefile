#
# Makefile for ML lecture slides
#
# Costas Andreopoulos <constantinos.andreopoulos@cern.ch>
#

MODULE  = ML
ACYEAR  = 202425
VERSION = vrs_01

part_01   : NAME = ${MODULE}_${ACYEAR}-slides-part_01-${VERSION}
part_02   : NAME = ${MODULE}_${ACYEAR}-slides-part_02-${VERSION}
part_03   : NAME = ${MODULE}_${ACYEAR}-slides-part_03-${VERSION}
part_04   : NAME = ${MODULE}_${ACYEAR}-slides-part_04-${VERSION}
part_05   : NAME = ${MODULE}_${ACYEAR}-slides-part_05-${VERSION}
part_06   : NAME = ${MODULE}_${ACYEAR}-slides-part_06-${VERSION}
part_07   : NAME = ${MODULE}_${ACYEAR}-slides-part_07-${VERSION}
part_08   : NAME = ${MODULE}_${ACYEAR}-slides-part_08-${VERSION}
part_09   : NAME = ${MODULE}_${ACYEAR}-slides-part_09-${VERSION}
part_10   : NAME = ${MODULE}_${ACYEAR}-slides-part_10-${VERSION}
all       : NAME = ${MODULE}_${ACYEAR}-slides-${VERSION}

part_01   : tex_start add_part01 tex_end pdf clean
part_02   : tex_start add_part02 tex_end pdf clean
part_03   : tex_start add_part03 tex_end pdf clean
part_04   : tex_start add_part04 tex_end pdf clean
part_05   : tex_start add_part05 tex_end pdf clean
part_06   : tex_start add_part06 tex_end pdf clean
part_07   : tex_start add_part07 tex_end pdf clean
part_08   : tex_start add_part08 tex_end pdf clean
part_09   : tex_start add_part09 tex_end pdf clean
part_10   : tex_start add_part10 tex_end pdf clean
all       : tex_start  \
	    add_front  \
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
  	    tex_end    \
	    pdf	       \
	    clean

each :      part_01 \
	    part_02 \
	    part_03 \
	    part_04 \
	    part_05 \
	    part_06 \
	    part_07 \
	    part_08 \
	    part_09 \
	    part_10 \

tex_start:
	echo '\documentclass{beamer}'         > tmp.tex
	echo '\input{settings.tex}'          >> tmp.tex
	echo '\\begin{document}'             >> tmp.tex

tex_end:
	echo '\end{document}' >> tmp.tex

add_front: FORCE
	echo '\input{front_matter/cover.tex}' >> tmp.tex
	echo '\input{front_matter/toc.tex}'   >> tmp.tex

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
