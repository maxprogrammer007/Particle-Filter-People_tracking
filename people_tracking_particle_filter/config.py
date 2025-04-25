# config.py

USE_DEEP_FEATURES = True
VIDEO_PATH = "C:\\Users\\abhin\\OneDrive\\Documents\\GitHub\\Particle-Filter-People_tracking\\people_tracking_particle_filter\\sample_videos\\test_video.mp4"
OUTPUT_PATH = "output_tracking.avi"
NSGA_GENERATIONS = 4
NSGA_POP_SIZE = 6

# Default fallback if best_config.txt not found
FALLBACK_CONFIG = {
    "NUM_PARTICLES": 75,
    "MOTION_NOISE": 5.0,
    "PATCH_SIZE": 20
}

def load_best_config():
    try:
        with open("C:\\Users\\abhin\\OneDrive\\Documents\\GitHub\\Particle-Filter-People_tracking\\people_tracking_particle_filter\\best_config.txt", "r") as f:
            lines = f.readlines()
            np = int(lines[1].split(":")[1].strip())
            noise = float(lines[2].split(":")[1].strip())
            patch = int(lines[3].split(":")[1].strip())
            return {
                "NUM_PARTICLES": np,
                "MOTION_NOISE": noise,
                "PATCH_SIZE": patch
            }
    except Exception as e:
        print("[WARN] Using fallback config due to:", e)
        return FALLBACK_CONFIG

# Load best config if exists
params = load_best_config()
NUM_PARTICLES = params["NUM_PARTICLES"]
MOTION_NOISE = params["MOTION_NOISE"]
PATCH_SIZE = params["PATCH_SIZE"]


# Parameter ranges for optimization
PARTICLE_RANGE = (30, 150)
NOISE_RANGE = (1.0, 10.0)
PATCH_RANGE = (10, 40)
