#
# Makefile for ML lecture slides
#
# Costas Andreopoulos <constantinos.andreopoulos@cern.ch>
#

MODULE  = ML
ACYEAR  = 202223
VERSION = v01

01   : NAME = ${MODULE}_${ACYEAR}-Slides-${VERSION}\ \(Lecture\ 01\)
02   : NAME = ${MODULE}_${ACYEAR}-Slides-${VERSION}\ \(Lecture\ 02\)
03   : NAME = ${MODULE}_${ACYEAR}-Slides-${VERSION}\ \(Lecture\ 03\)
04   : NAME = ${MODULE}_${ACYEAR}-Slides-${VERSION}\ \(Lecture\ 04\)
05   : NAME = ${MODULE}_${ACYEAR}-Slides-${VERSION}\ \(Lecture\ 05\)
06   : NAME = ${MODULE}_${ACYEAR}-Slides-${VERSION}\ \(Lecture\ 06\)
07   : NAME = ${MODULE}_${ACYEAR}-Slides-${VERSION}\ \(Lecture\ 07\)
08   : NAME = ${MODULE}_${ACYEAR}-Slides-${VERSION}\ \(Lecture\ 08\)
09   : NAME = ${MODULE}_${ACYEAR}-Slides-${VERSION}\ \(Lecture\ 09\)
10   : NAME = ${MODULE}_${ACYEAR}-Slides-${VERSION}\ \(Lecture\ 10\)
11   : NAME = ${MODULE}_${ACYEAR}-Slides-${VERSION}\ \(Lecture\ 11\)
12   : NAME = ${MODULE}_${ACYEAR}-Slides-${VERSION}\ \(Lecture\ 12\)
all  : NAME = ${MODULE}_${ACYEAR}-Slides-${VERSION}

01   : tex_start add_lec01 tex_end pdf
02   : tex_start add_lec02 tex_end pdf
03   : tex_start add_lec03 tex_end pdf
04   : tex_start add_lec04 tex_end pdf
05   : tex_start add_lec05 tex_end pdf
06   : tex_start add_lec06 tex_end pdf
07   : tex_start add_lec07 tex_end pdf
08   : tex_start add_lec08 tex_end pdf
09   : tex_start add_lec09 tex_end pdf
10   : tex_start add_lec10 tex_end pdf
11   : tex_start add_lec11 tex_end pdf
12   : tex_start add_lec12 tex_end pdf
all  : tex_start \
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
	tex_end \
	pdf
each : 01 02 03 04 05 06 07 08 09 10 11 12

tex_start:
	echo '\documentclass{beamer}'           >  main_tmp.tex
	echo '\input{src/global/settings.tex}'  >> main_tmp.tex
	echo '\\begin{document}'                >> main_tmp.tex

tex_end:
	echo '\end{document}' >> main_tmp.tex

add_front: FORCE
	echo '\input{src/global/cover.tex}'     >> main_tmp.tex
	echo '\input{src/global/toc.tex}'       >> main_tmp.tex
add_lec01: FORCE
	echo '\input{src/lecture_01/main.tex}'  >> main_tmp.tex
add_lec02: FORCE
	echo '\input{src/lecture_02/main.tex}'  >> main_tmp.tex
add_lec03: FORCE
	echo '\input{src/lecture_03/main.tex}'  >> main_tmp.tex
add_lec04: FORCE
	echo '\input{src/lecture_04/main.tex}'  >> main_tmp.tex
add_lec05: FORCE
	echo '\input{src/lecture_05/main.tex}'  >> main_tmp.tex
add_lec06: FORCE
	echo '\input{src/lecture_06/main.tex}'  >> main_tmp.tex
add_lec07: FORCE
	echo '\input{src/lecture_07/main.tex}'  >> main_tmp.tex
add_lec08: FORCE
	echo '\input{src/lecture_08/main.tex}'  >> main_tmp.tex
add_lec09: FORCE
	echo '\input{src/lecture_09/main.tex}'  >> main_tmp.tex
add_lec10: FORCE
	echo '\input{src/lecture_10/main.tex}'  >> main_tmp.tex
add_lec11: FORCE
	echo '\input{src/lecture_11/main.tex}'  >> main_tmp.tex
add_lec12: FORCE
	echo '\input{src/lecture_12/main.tex}'  >> main_tmp.tex

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
