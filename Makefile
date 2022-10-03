#
# Makefile for ML lecture slides
#
# Costas Andreopoulos <constantinos.andreopoulos@cern.ch>
#

MODULE  = ML
ACYEAR  = 202223
VERSION = v01

s01   : NAME = ${MODULE}_${ACYEAR}-Slides-${VERSION}\ \(Lecture\ 01\)
s02   : NAME = ${MODULE}_${ACYEAR}-Slides-${VERSION}\ \(Lecture\ 02\)
s03   : NAME = ${MODULE}_${ACYEAR}-Slides-${VERSION}\ \(Lecture\ 03\)
s04   : NAME = ${MODULE}_${ACYEAR}-Slides-${VERSION}\ \(Lecture\ 04\)
s05   : NAME = ${MODULE}_${ACYEAR}-Slides-${VERSION}\ \(Lecture\ 05\)
s06   : NAME = ${MODULE}_${ACYEAR}-Slides-${VERSION}\ \(Lecture\ 06\)
s07   : NAME = ${MODULE}_${ACYEAR}-Slides-${VERSION}\ \(Lecture\ 07\)
s08   : NAME = ${MODULE}_${ACYEAR}-Slides-${VERSION}\ \(Lecture\ 08\)
s09   : NAME = ${MODULE}_${ACYEAR}-Slides-${VERSION}\ \(Lecture\ 09\)
s10   : NAME = ${MODULE}_${ACYEAR}-Slides-${VERSION}\ \(Lecture\ 10\)
s11   : NAME = ${MODULE}_${ACYEAR}-Slides-${VERSION}\ \(Lecture\ 11\)
s12   : NAME = ${MODULE}_${ACYEAR}-Slides-${VERSION}\ \(Lecture\ 12\)
sall  : NAME = ${MODULE}_${ACYEAR}-Slides-${VERSION}

s01   : tex_start add_tex01 tex_end pdf
s02   : tex_start add_tex02 tex_end pdf
s03   : tex_start add_tex03 tex_end pdf
s04   : tex_start add_tex04 tex_end pdf
s05   : tex_start add_tex05 tex_end pdf
s06   : tex_start add_tex06 tex_end pdf
s07   : tex_start add_tex07 tex_end pdf
s08   : tex_start add_tex08 tex_end pdf
s09   : tex_start add_tex09 tex_end pdf
s10   : tex_start add_tex10 tex_end pdf
s11   : tex_start add_tex11 tex_end pdf
s12   : tex_start add_tex12 tex_end pdf
sall  : tex_start add_tex00 add_tex01 add_tex02 add_tex03 add_tex04 add_tex05 add_tex06 add_tex07 add_tex08 add_tex09 add_tex10 add_tex11 add_tex12 tex_end pdf
seach : s01 s02 s03 s04 s05 s06 s07 s08 s09 s10 s11 s12

tex_start:
	echo '\documentclass{beamer}'           >  main_tmp.tex
	echo '\input{src/global/settings.tex}'  >> main_tmp.tex
	echo '\\begin{document}'                >> main_tmp.tex

tex_end:
	echo '\end{document}' >> main_tmp.tex

add_tex01: FORCE
	echo '\input{src/lecture_01/main.tex}'  >> main_tmp.tex
add_tex02: FORCE
	echo '\input{src/lecture_02/main.tex}'  >> main_tmp.tex
add_tex03: FORCE
	echo '\input{src/lecture_03/main.tex}'  >> main_tmp.tex
add_tex04: FORCE
	echo '\input{src/lecture_04/main.tex}'  >> main_tmp.tex
add_tex05: FORCE
	echo '\input{src/lecture_05/main.tex}'  >> main_tmp.tex
add_tex06: FORCE
	echo '\input{src/lecture_06/main.tex}'  >> main_tmp.tex
add_tex07: FORCE
	echo '\input{src/lecture_07/main.tex}'  >> main_tmp.tex
add_tex08: FORCE
	echo '\input{src/lecture_08/main.tex}'  >> main_tmp.tex
add_tex09: FORCE
	echo '\input{src/lecture_09/main.tex}'  >> main_tmp.tex
add_tex10: FORCE
	echo '\input{src/lecture_10/main.tex}'  >> main_tmp.tex
add_tex11: FORCE
	echo '\input{src/lecture_11/main.tex}'  >> main_tmp.tex
add_tex12: FORCE
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
