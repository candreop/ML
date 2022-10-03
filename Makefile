#
# Makefile for ML lecture slides
#
# Costas Andreopoulos <constantinos.andreopoulos@cern.ch>
#

MODULE  = ML
ACYEAR  = 202324
VERSION = v01

lec_01   : NAME = ${MODULE}_${ACYEAR}-Slides-${VERSION}\ \(Lecture\ 01\)
lec_02   : NAME = ${MODULE}_${ACYEAR}-Slides-${VERSION}\ \(Lecture\ 02\)
lec_03   : NAME = ${MODULE}_${ACYEAR}-Slides-${VERSION}\ \(Lecture\ 03\)
lec_04   : NAME = ${MODULE}_${ACYEAR}-Slides-${VERSION}\ \(Lecture\ 04\)
lec_05   : NAME = ${MODULE}_${ACYEAR}-Slides-${VERSION}\ \(Lecture\ 05\)
lec_06   : NAME = ${MODULE}_${ACYEAR}-Slides-${VERSION}\ \(Lecture\ 06\)
lec_07   : NAME = ${MODULE}_${ACYEAR}-Slides-${VERSION}\ \(Lecture\ 07\)
lec_08   : NAME = ${MODULE}_${ACYEAR}-Slides-${VERSION}\ \(Lecture\ 08\)
lec_09   : NAME = ${MODULE}_${ACYEAR}-Slides-${VERSION}\ \(Lecture\ 09\)
lec_10   : NAME = ${MODULE}_${ACYEAR}-Slides-${VERSION}\ \(Lecture\ 10\)
lec_11   : NAME = ${MODULE}_${ACYEAR}-Slides-${VERSION}\ \(Lecture\ 11\)
lec_12   : NAME = ${MODULE}_${ACYEAR}-Slides-${VERSION}\ \(Lecture\ 12\)
lec_13   : NAME = ${MODULE}_${ACYEAR}-Slides-${VERSION}\ \(Lecture\ 13\)
lec_14   : NAME = ${MODULE}_${ACYEAR}-Slides-${VERSION}\ \(Lecture\ 14\)
lec_15   : NAME = ${MODULE}_${ACYEAR}-Slides-${VERSION}\ \(Lecture\ 15\)
lec_16   : NAME = ${MODULE}_${ACYEAR}-Slides-${VERSION}\ \(Lecture\ 16\)
all_lec  : NAME = ${MODULE}_${ACYEAR}-Slides-${VERSION}

lec_01   : tex_start add_lec01 tex_end pdf
lec_02   : tex_start add_lec02 tex_end pdf
lec_03   : tex_start add_lec03 tex_end pdf
lec_04   : tex_start add_lec04 tex_end pdf
lec_05   : tex_start add_lec05 tex_end pdf
lec_06   : tex_start add_lec06 tex_end pdf
lec_07   : tex_start add_lec07 tex_end pdf
lec_08   : tex_start add_lec08 tex_end pdf
lec_09   : tex_start add_lec09 tex_end pdf
lec_10   : tex_start add_lec10 tex_end pdf
lec_11   : tex_start add_lec11 tex_end pdf
lec_12   : tex_start add_lec12 tex_end pdf
lec_13   : tex_start add_lec13 tex_end pdf
lec_14   : tex_start add_lec14 tex_end pdf
lec_15   : tex_start add_lec15 tex_end pdf
lec_16   : tex_start add_lec16 tex_end pdf

all_lec  : tex_start \
	   add_front \
	   add_lec01 \
	   add_lec02 \
	   add_lec03 \
	   add_lec04 \
	   add_lec05 \
	   add_lec06 \
	   add_lec07 \
	   add_lec08 \
	   add_lec09 \
	   add_lec10 \
	   add_lec11 \
	   add_lec12 \
	   add_lec13 \
	   add_lec14 \
	   add_lec15 \
	   add_lec16 \
	   tex_end \
	   pdf

each :     lec_01 \
	   lec_02 \
	   lec_03 \
	   lec_04 \
	   lec_05 \
	   lec_06 \
	   lec_07 \
	   lec_08 \
	   lec_09 \
	   lec_10 \
	   lec_11 \
	   lec_12 \
	   lec_13 \
	   lec_14 \
	   lec_15 \
	   lec_16

tex_start:
	echo '\documentclass{beamer}' > main_tmp.tex
	echo '\input{lectures/tex/common/settings.tex}' >> main_tmp.tex
	echo '\\begin{document}' >> main_tmp.tex

tex_end:
	echo '\end{document}' >> main_tmp.tex

add_front: FORCE
	echo '\input{lectures/tex/front/cover.tex}'  >> main_tmp.tex
	echo '\input{lectures/tex/front/toc.tex}'    >> main_tmp.tex

add_lec01: FORCE
	echo '\input{lectures/tex/01/main.tex}'  >> main_tmp.tex
add_lec02: FORCE
	echo '\input{lectures/tex/02/main.tex}'  >> main_tmp.tex
add_lec03: FORCE
	echo '\input{lectures/tex/03/main.tex}'  >> main_tmp.tex
add_lec04: FORCE
	echo '\input{lectures/tex/04/main.tex}'  >> main_tmp.tex
add_lec05: FORCE
	echo '\input{lectures/tex/05/main.tex}'  >> main_tmp.tex
add_lec06: FORCE
	echo '\input{lectures/tex/06/main.tex}'  >> main_tmp.tex
add_lec07: FORCE
	echo '\input{lectures/tex/07/main.tex}'  >> main_tmp.tex
add_lec08: FORCE
	echo '\input{lectures/tex/08/main.tex}'  >> main_tmp.tex
add_lec09: FORCE
	echo '\input{lectures/tex/09/main.tex}'  >> main_tmp.tex
add_lec10: FORCE
	echo '\input{lectures/tex/10/main.tex}'  >> main_tmp.tex
add_lec11: FORCE
	echo '\input{lectures/tex/11/main.tex}'  >> main_tmp.tex
add_lec12: FORCE
	echo '\input{lectures/tex/12/main.tex}'  >> main_tmp.tex
add_lec13: FORCE
	echo '\input{lectures/tex/13/main.tex}'  >> main_tmp.tex
add_lec14: FORCE
	echo '\input{lectures/tex/14/main.tex}'  >> main_tmp.tex
add_lec15: FORCE
	echo '\input{lectures/tex/15/main.tex}'  >> main_tmp.tex
add_lec16: FORCE
	echo '\input{lectures/tex/16/main.tex}'  >> main_tmp.tex

pdf: FORCE
	pdflatex main_tmp.tex
	mv main_tmp.pdf ${NAME}.pdf

clean: FORCE
	rm *nav
	rm *aux
	rm *log
	rm *out
	rm *snm
	rm *toc
	rm main_tmp.*

FORCE:
