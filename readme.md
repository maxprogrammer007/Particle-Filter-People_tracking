
# Particle Filter‑Based People Tracking with NSGA‑II Optimization

This project implements a **real‑time people tracking system** inspired by the paper  
“A Reliable People Tracking in Nuclear Power Plant Control Room Monitoring System Using Particle Filter.”  
It uses HOG for person detection, classic particle filters with color‑histogram observation,  
and NSGA‑II to automatically tune tracking parameters for best accuracy, stability, and speed.

---

## 🚀 Key Features

- **Person Detection**: HOG‑based human detector (OpenCV).  
- **Particle Filter Tracking**:
  - Predicts object motion with Gaussian noise.
  - Updates particle weights via color‑histogram similarity.
  - Resamples to focus on high‑likelihood hypotheses.
- **NSGA‑II Parameter Optimization**:
  - Evolves `(num_particles, motion_noise, patch_size)`
  - Balances three objectives:  
    1. **Tracking Accuracy** (MOTA ↑)  
    2. **ID‑Switch Count** (↓)  
    3. **Frame Rate** (FPS ↑)
- **Automated Evaluation** in `evaluation.py` for MOTA, ID‑switches, FPS.  
- **Pareto‑Front Visualization** in `plot_pareto.py`.  

---

## 📂 Folder Structure

```plaintext
people_tracking_particle_filter/
├── background_subtraction.py      # legacy (optional), not used by default
├── blob_detection.py              # HOG human detection
├── particle_filter.py             # Particle + ParticleFilter classes
├── histogram_model.py             # Color‑histogram extractor
├── tracker_manager.py             # Manages one PF per detected person
├── utils.py                       # Drawing helpers (particles, centers)
├── main.py                        # Runs the tracker with fixed params
├── evaluation.py                  # run_tracking_evaluation() wrapper
├── nsga_optimization.py           # NSGA‑II optimizer (uses DEAP)
├── plot_pareto.py                 # Pareto front plotting (matplotlib)
├── config.py                      # Default parameter ranges (optional)
├── sample_videos/
│   └── control_room_test.mp4      # Test video file
└── README.md                      # Project description (this file)
```

---

## 🛠️ How It Works

1. **Detect People**  
   Each frame is scanned with OpenCV’s HOG‑based people detector.

2. **Initialize Particle Filters**  
   For every new detection, a `ParticleFilter` is created at the person’s center, using default or optimized parameters.

3. **Predict & Update**  
   - **Predict**: Particles move by adding random Gaussian noise.  
   - **Update**: Each particle’s patch color histogram is compared to the target’s original histogram using the Bhattacharyya distance.  
   - **Resample**: Particles with lower weight are discarded; high‑weight ones are duplicated.

4. **Visualize & Save**  
   - Green dots for particles; blue circles for estimated centers.  
   - Video is shown live and saved to `output_tracking.avi`.

5. **Automated Evaluation** (`evaluation.py`)  
   Runs the tracker end‑to‑end, returning three metrics:  
   - **MOTA** (Multiple Object Tracking Accuracy)  
   - **ID‑Switch Count**  
   - **FPS**

6. **NSGA‑II Optimization** (`nsga_optimization.py`)  
   Uses DEAP to evolve `(num_particles, motion_noise, patch_size)` over generations, optimizing the three objectives simultaneously.

7. **Pareto‑Front Visualization** (`plot_pareto.py`)  
   Creates a 2D/3D scatter of the final population’s fitness values, showing the trade‑offs between accuracy, stability, and speed.

---

## ⚙️ Requirements

- Python 3.8+  
- [OpenCV](https://pypi.org/project/opencv-python/)  
- `numpy`  
- `deap` (for NSGA‑II)  
- `matplotlib` (for Pareto plotting)

Install via:

```bash
pip install opencv-python numpy deap matplotlib
```

---

## 📽️ Usage

1. **Run the basic tracker** (with default params):
   ```bash
   python main.py
   ```

2. **Run NSGA‑II optimization**:
   ```bash
   python nsga_optimization.py
   ```
   This will print out the Pareto‑optimal configurations.

3. **Plot Pareto front** (in a Python REPL or separate script):
   ```python
   from nsga_optimization import run_nsga
   from plot_pareto import plot_pareto

   pop, log, hof = run_nsga(generations=10, pop_size=20)
   plot_pareto(pop)
   ```

---

## 📈 Improvements Over Basic Version

| Improvement            | Description                                                        |
|------------------------|--------------------------------------------------------------------|
| NSGA‑II Optimization   | Automated tuning of PF parameters for multi‑objective balance      |
| Automated Evaluation   | `evaluation.py` wraps MOTA, ID‑switch, FPS measurement             |
| Pareto Visualization   | `plot_pareto.py` shows trade‑offs between accuracy, stability, speed |

---

## 🎯 Future Scope

- **Multi‑camera tracking**  
- **Memory‑augmented ReID** for long occlusions  
- **Sensor fusion** (RGB + depth/thermal)  
- **Uncertainty‑aware PF** with adaptive re‑initialization  

---

# 🙌 Credits

Based on:  
_A Reliable People Tracking in Nuclear Power Plant Control Room Monitoring System Using Particle Filter_ (IEEE)  
Extended with HOG detection, NSGA‑II optimization, and automated evaluation/visualization.


