@echo off

REM Ajouter MiKTeX au PATH
set PATH=%PATH%;C:\Users\Alexandre\AppData\Local\Programs\MiKTeX\miktex\bin\x64

echo ========================================
echo Nettoyage et compilation du manuscrit...
echo ========================================
echo.

echo [0/6] Nettoyage des fichiers temporaires...
del /Q /S *.aux *.bbl *.lbl *.loa *.loe *.lof *.log *.maf *.mlf* *.mlt* *.mtc* *.toc *.blg *.out *.synctex.gz 2>nul
if exist sommaire.pdf (
    echo Suppression de l'ancien PDF...
    del /F sommaire.pdf 2>nul
)
echo.

echo [1/6] Premiere passe pdflatex...
pdflatex -interaction=nonstopmode sommaire

echo.
echo [2/6] Bibliographie principale...
bibtex sommaire

echo.
echo [3/6] Bibliographie web...
bibtex web

echo.
echo [4/6] Bibliographie personnelle...
bibtex mine

echo.
echo [5/6] Deuxieme passe pdflatex...
pdflatex -interaction=nonstopmode sommaire

echo.
echo [6/6] Troisieme passe pdflatex...
pdflatex -interaction=nonstopmode sommaire

echo.
echo [7/7] Nettoyage final des fichiers temporaires...
del /Q /S *.aux *.bbl *.lbl *.loa *.loe *.lof *.log *.maf *.mlf* *.mlt* *.mtc* *.toc *.blg *.out *.synctex.gz 2>nul
echo.
echo ========================================
echo Compilation terminee !
echo Le PDF sommaire.pdf a ete genere.
echo Fichiers temporaires nettoyes.
echo ========================================
