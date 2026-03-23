@echo off
echo Creating virtual environment...
python -m venv .venv
echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo.
echo Installing Python requirements...
pip install -r requirements.txt
echo.
echo Compiling C++ binaries...
g++ -O2 -std=c++17 -o kmeans_serial kmeans_serial.cpp
g++ -O2 -std=c++17 -fopenmp -o kmeans_parallel kmeans_parallel.cpp
echo.
echo Setup complete. You can now run the benchmark with: .venv\Scripts\python.exe benchmark.py
pause