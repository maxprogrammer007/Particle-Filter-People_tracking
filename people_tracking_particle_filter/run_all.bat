@echo off
echo.
echo 🚀 Starting Full People Tracking Pipeline with NSGA-II + Deep Particle Filter + GPU
echo.

REM Step 1: Run Evaluation on Current Settings
echo [1/4] Evaluating Tracker...
python evaluation.py

REM Step 2: Run NSGA-II Optimization
echo [2/4] Running NSGA-II Optimization...
python nsga_optimization.py

REM Step 3: Plot Pareto Front
echo [3/4] Plotting Pareto Front...
python plot_pareto.py

REM Step 4: Run Main Tracking with Best Configuration
echo [4/4] Running Tracker with Current Settings (manual config from config.py)...
python main.py

echo.
echo ✅ Pipeline Completed!
pause
