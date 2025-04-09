


# Particle Filter-Based People Tracking System

This project implements a **real-time people tracking system** based on the research paper:  
**"A Reliable People Tracking in Nuclear Power Plant Control Room Monitoring System Using Particle Filter"**

The system detects humans and tracks them over time using **Particle Filters** and **Color Histogram Matching**.

---

## 🚀 Key Features

- **Real-Time People Detection** using HOG-based human detector (OpenCV).
- **Particle Filter Tracking**:
  - Predicts the next position of each detected person.
  - Updates based on color histogram similarity.
  - Resamples particles to maintain robust tracking.
- **Improved Foreground Detection** *(optional old version with background subtraction available)*.
- **Automatic Output Saving**:
  - Saves the tracked video as `output_tracking.avi`.
- **Auto Video Looping**:
  - Automatically restarts the video when it ends for continuous demonstration.

---

## 📂 Folder Structure

```plaintext
people_tracking_particle_filter/
├── background_subtraction.py      # (Old, optional - now replaced with HOG detection)
├── blob_detection.py               # Human detection using HOG descriptor
├── particle_filter.py              # Particle Filter classes (Particle + ParticleFilter)
├── histogram_model.py              # Color histogram extraction and comparison
├── tracker_manager.py              # Manages multiple trackers (one per detected person)
├── utils.py                        # Drawing particles and tracking centers
├── main.py                         # Main script to run tracking
├── config.py                       # Configuration parameters (if needed)
├── README.md                        # Project description (this file)
└── sample_videos/
    └── control_room_test.mp4       # Test video file
```



## 🛠️ How It Works

1. **Detect Humans**:  
   Using OpenCV’s pre-trained **HOG People Detector**.

2. **Initialize Particle Filters**:  
   For each detected person, a **Particle Filter** is initialized at the center.

3. **Particle Prediction**:  
   Particles move slightly around the predicted position based on a simple motion model.

4. **Histogram Matching**:  
   Each particle is weighted by comparing its color histogram to the target person's appearance.

5. **Resampling**:  
   Particles with better matching survive, guiding the tracker.

6. **Tracking Visualization**:  
   Particles (green dots) and estimated centers (blue circles) are drawn over each person.

7. **Save Output Video**:  
   The final tracking output is saved automatically as `output_tracking.avi`.

---

## ⚙️ Requirements

- Python 3.8+
- OpenCV (`opencv-python`, `opencv-contrib-python`)
- Numpy

Install with:

```bash
pip install opencv-python opencv-contrib-python numpy
```

---

## 📈 Improvements Over Basic Version

| Improvement | Description |
|:---|:---|
| **HOG People Detection** | Reliable detection of real humans instead of just moving blobs. |
| **Auto Restart Video** | Restarts video automatically when finished. |
| **Output Video Saving** | Saves the tracking visualization into an `.avi` file. |
| **Better Particle Management** | Improved motion prediction and tracking accuracy. |

---

## 📽️ How to Run

1. Place your test video inside `sample_videos/` folder.
2. Update the video name in `main.py` if needed.
3. Run the tracking system:

```bash
python main.py
```

4. Press `ESC` to exit.
5. Find your saved output in the project folder as `output_tracking.avi`.

---

## 🎯 Future Scope (Optional Extensions)

- Replace HOG detector with more accurate lightweight deep models (e.g., YOLOv4-tiny).
- Multi-camera tracking (extend to multiple camera feeds).
- Add Re-Identification models for robust occlusion handling.
- Improve real-time performance with GPU acceleration (CUDA).

---

# 🙌 Credits

- Based on the research paper:  
  **"A Reliable People Tracking in Nuclear Power Plant Control Room Monitoring System Using Particle Filter"**
- Extended with real human detection for practical implementation.

---
