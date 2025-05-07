# optimization/qpso_optimizer.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import yaml
from tqdm import tqdm
from random import uniform
from evaluation.evaluation import evaluate_pipeline
import csv



def load_config(path="people_tracking_project\\optimization\\config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def sample_config(bounds):
    config = {}
    for k, v in bounds.items():
        if isinstance(v, list):
            config[k] = round(uniform(v[0], v[1]), 2)
        else:
            config[k] = v
    return config

def quantize_config(config, allowed_values):
    quantized = {}
    for key, value in config.items():
        vals = allowed_values.get(key)
        if vals:
            quantized[key] = min(vals, key=lambda x: abs(x - value))
        else:
            quantized[key] = value
    return quantized


import csv, os

def log_to_csv(config, mota, idf1, fps, path="people_tracking_project\\results\\optimization_log.csv"):
    file_exists = os.path.exists(path)
    with open(path, 'a', newline='') as f:
        fieldnames = list(config.keys()) + ['MOTA', 'IDF1', 'FPS']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        row = dict(config)
        row.update({'MOTA': mota, 'IDF1': idf1, 'FPS': fps})
        writer.writerow(row)


def qpso_optimize(bounds, allowed_values, num_particles=20, generations=15):
    global_best = None
    global_best_score = (-1, -1, float("inf"))  # MOTA, IDF1, 1/FPS

    particles = [sample_config(bounds) for _ in range(num_particles)]

    for g in range(generations):
        print(f"\n[Generation {g+1}]")
        for i in tqdm(range(num_particles)):
            quantized = quantize_config(particles[i], allowed_values)
            mota, idf1, fps = evaluate_pipeline(quantized)
            log_to_csv(quantized, mota, idf1, fps)
            score = (mota, idf1, 1 / fps)

            if score > global_best_score:
                global_best_score = score
                global_best = quantized

        # Update particles (naive QPSO-like update)
        for i in range(num_particles):
            particles[i] = {
                k: round((particles[i][k] + global_best[k]) / 2 + uniform(-0.1, 0.1), 3)
                for k in bounds
            }

    print("\n[✓] Best Config Found:")
    print(global_best)
    print("Score (MOTA, IDF1, 1/FPS):", global_best_score)

    return global_best

if __name__ == "__main__":
    config_space = load_config()
    bounds = {}
    allowed_values = {}

    for section, opts in config_space.items():
        for k, v in opts.items():
            bounds[k] = [min(v), max(v)]
            allowed_values[k] = v

    qpso_optimize(bounds, allowed_values)
