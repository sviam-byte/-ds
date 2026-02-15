@echo off
setlocal

REM Переходим в папку, где лежит bat
cd /d %~dp0

echo === Time Series Analysis Tool ===
echo.

REM Проверка Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python не найден. Установи Python и добавь в PATH.
    pause
    exit /b
)

REM Если есть requirements — ставим
if exist requirements.txt (
    echo Установка зависимостей...
    REM Обновляем pip и wheel, но НЕ тянем самый новый setuptools:
    REM nolds (0.5.2) использует pkg_resources, который удалён в setuptools>=81.
    python -m pip install --upgrade pip wheel
    python -m pip install "setuptools<81"
    REM Важно для Python 3.13: ставим бинарные колёса (иначе numpy/scipy полезут собираться из исходников)
    python -m pip install --upgrade --prefer-binary -r requirements.txt
)

echo.
echo Запуск GUI...
echo.

python interfaces/gui.py

echo.
echo GUI завершился.
pause
