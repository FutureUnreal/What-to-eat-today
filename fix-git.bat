@echo off
echo Fix Git Repository...
echo.

echo [1/6] Backup current changes...
if exist .git (
    git add .
    git stash push -m "backup before git fix"
    echo Changes backed up to stash
) else (
    echo No Git repository found
)

echo.
echo [2/6] Remove corrupted .git directory...
if exist .git (
    rmdir /s /q .git
    echo .git directory removed
)

echo.
echo [3/6] Reinitialize Git repository...
git init
git branch -M main

echo.
echo [4/6] Add remote repository...
git remote add origin https://github.com/FutureUnreal/What-to-eat-today.git

echo.
echo [5/6] Add all files...
git add .

echo.
echo [6/6] Create initial commit...
git commit -m "Reinitialize Git repository"

echo.
echo Git repository fixed successfully!
echo.
echo Now you can push to remote repository:
echo git push -u origin main --force
echo.
pause
