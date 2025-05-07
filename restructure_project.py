import os
import shutil

# Current and target directories
old_dir = "people_tracking_particle_filter"
new_dir = "people_tracking_project"

# Ensure fresh new structure
if os.path.exists(new_dir):
    print(f"[!] {new_dir} already exists. Please remove or rename it before proceeding.")
    exit(1)

# Create base structure
folders = [
    "detectors",
    "trackers",
    "optimization",
    "evaluation",
    "utils",
    "sample_videos",
    "results"
]
os.makedirs(new_dir)
for folder in folders:
    os.makedirs(os.path.join(new_dir, folder))

# Move & cleanup files
move_map = {
    "evaluation.py": "evaluation/evaluation.py",
    "plot_pareto.py": "optimization/plot_pareto.py",
    "run_plot_pareto.py": "run_plot_pareto.py",
    "nsga_optimization.py": "optimization/nsga_optimizer.py",
    "config.py": "optimization/config.py",
    "utils.py": "utils/video_io.py",
    "sample_videos/test_video.mp4": "sample_videos/test_video.mp4",
    "output_tracking.avi": "results/output_tracking.avi",
    "README.md": "README.md",
    "main.py": "main.py"
}

# Files to discard (legacy particle filter components)
discard_files = [
    "background_subtraction.py",
    "blob_detection.py",
    "histogram_model.py",
    "particle_filter.py",
    "tracker_manager.py"
]

# Move files
for src, dst in move_map.items():
    src_path = os.path.join(old_dir, src)
    dst_path = os.path.join(new_dir, dst)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.move(src_path, dst_path)
    print(f"[+] Moved {src} → {dst}")

# Remove legacy files
for file in discard_files:
    path = os.path.join(old_dir, file)
    if os.path.exists(path):
        os.remove(path)
        print(f"[x] Removed legacy file: {file}")

# Move nsga_config.yaml if present
yaml_path = os.path.join(old_dir, "configs", "nsga_config.yaml")
if os.path.exists(yaml_path):
    shutil.move(yaml_path, os.path.join(new_dir, "optimization", "config.yaml"))
    print(f"[+] Moved nsga_config.yaml → optimization/config.yaml")

# Final cleanup
if os.path.exists(os.path.join(old_dir, "__pycache__")):
    shutil.rmtree(os.path.join(old_dir, "__pycache__"))

# Optionally remove old directory
if not os.listdir(old_dir):
    os.rmdir(old_dir)
    print(f"[✓] Cleaned up empty old directory: {old_dir}")

print("\n✅ Restructuring complete! You can now begin fresh from `people_tracking_project/`.")
