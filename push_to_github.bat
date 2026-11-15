@echo off
cd /d "%~dp0"
set "PATH=%LOCALAPPDATA%\Programs\Git\cmd;%PATH%"
echo ========================================================
echo Pushing project files to GitHub...
echo Repository: https://github.com/Gannu2006/customer_segmentation_analysis-
echo ========================================================
echo.
git.exe status
echo.
echo Starting git push...
git.exe push -u origin main --force
echo.
if %ERRORLEVEL% equ 0 (
    echo ========================================================
    echo SUCCESS! All files have been pushed to your GitHub!
    echo ========================================================
) else (
    echo ========================================================
    echo Push encountered an issue. See message above.
    echo ========================================================
)
echo.
pause
