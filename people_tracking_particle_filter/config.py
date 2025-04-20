# config.py

# --- Tracking Parameters ---
NUM_PARTICLES = 75             # Number of particles per tracker
MOTION_NOISE = 5.0             # Standard deviation for particle motion
PATCH_SIZE = 20                # Size of patch around each particle (WxH)
MOTION_STD_DEV = 15            # (Optional legacy value, used if needed)

# --- Detection Parameters ---
BLOB_MIN_AREA = 500            # Minimum area to consider a detection valid
HISTOGRAM_BINS = 32            # Number of bins in color histogram (if using it)

# --- Feature Selection ---
USE_DEEP_FEATURES = True       # True = Deep Particle Filter, False = Color Histogram

# --- Video Configuration ---
VIDEO_PATH = "C:\\Users\\abhin\\OneDrive\\Documents\\GitHub\\Particle-Filter-People_tracking\\people_tracking_particle_filter\\sample_videos\\test_video.mp4"   # Input video path
OUTPUT_PATH = "C:\\Users\\abhin\\OneDrive\\Documents\\GitHub\\Particle-Filter-People_tracking\\output_tracking.avi"           # Output tracking result path

# --- NSGA-II Optimization Parameters ---
NSGA_GENERATIONS = 10          # Number of generations to evolve
NSGA_POP_SIZE = 20             # Population size per generation
