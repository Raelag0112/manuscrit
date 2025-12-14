@echo off
echo ========================================
echo Compilation de presentation.tex
echo ========================================

:: Première passe
echo.
echo [1/3] Premiere passe pdflatex...
pdflatex -interaction=nonstopmode presentation.tex

:: Deuxième passe (pour les références)
echo.
echo [2/3] Deuxieme passe pdflatex...
pdflatex -interaction=nonstopmode presentation.tex

:: Troisième passe (pour la table des matières)
echo.
echo [3/3] Troisieme passe pdflatex...
pdflatex -interaction=nonstopmode presentation.tex

:: Nettoyage des fichiers auxiliaires
echo.
echo Nettoyage des fichiers auxiliaires...
del /q presentation.aux 2>nul
del /q presentation.log 2>nul
del /q presentation.nav 2>nul
del /q presentation.out 2>nul
del /q presentation.snm 2>nul
del /q presentation.toc 2>nul
del /q presentation.vrb 2>nul

echo.
echo ========================================
echo Compilation terminee !
echo Fichier genere : presentation.pdf
echo ========================================

pause
