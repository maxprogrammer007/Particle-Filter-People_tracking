import numpy as np
import random
import torch
import time
import os
from deap import base, creator, tools
from evaluation import run_tracking_evaluation
from config import VIDEO_PATH, PARTICLE_RANGE, NOISE_RANGE, PATCH_RANGE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output log
os.makedirs("logs", exist_ok=True)
log_path = f"logs/qpso_log.txt"

# --- DEAP Setup ---
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("num_particles", random.randint, *PARTICLE_RANGE)
toolbox.register("motion_noise", random.uniform, *NOISE_RANGE)
toolbox.register("patch_size", random.randint, *PATCH_RANGE)
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.num_particles, toolbox.motion_noise, toolbox.patch_size), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# --- Fitness Evaluation ---
def evaluate(ind):
    num_particles = int(np.clip(round(ind[0]), *PARTICLE_RANGE))
    motion_noise = float(ind[1])
    patch_size = int(np.clip(round(ind[2]), *PATCH_RANGE))
    start = time.time()
    mota, idsw, fps = run_tracking_evaluation(
        VIDEO_PATH, num_particles, motion_noise, patch_size, max_frames=100
    )
    elapsed = time.time() - start
    print(f"✅ [QPSO] Eval: MOTA={mota:.3f}, IDSW={idsw}, FPS={fps:.2f}, Time={elapsed:.1f}s")
    return mota, idsw, fps

toolbox.register("evaluate", evaluate)

# --- QPSO Core ---
def run_qpso(pop_size=20, max_gen=10, alpha=0.75):
    pop = toolbox.population(n=pop_size)

    # Initialize pbest & fitness
    pbest = [ind[:] for ind in pop]
    pbest_fit = [toolbox.evaluate(ind) for ind in pbest]
    for ind, fit in zip(pop, pbest_fit):
        ind.fitness.values = fit

    # Initialize gbest
    best_idx = np.argmax([fit[0] for fit in pbest_fit])
    gbest = pbest[best_idx][:]
    gbest_fit = pbest_fit[best_idx]

    log_lines = []

    for gen in range(max_gen):
        print(f"\n📘 GEN {gen}: Best MOTA={gbest_fit[0]:.3f}, FPS={gbest_fit[2]:.2f}")
        log_lines.append(f"GEN {gen} => BEST: MOTA={gbest_fit[0]:.3f}, FPS={gbest_fit[2]:.2f}")

        mbest = np.mean([ind for ind in pbest], axis=0)

        for i, ind in enumerate(pop):
            for d in range(3):
                u = random.random()
                direction = 1 if random.random() < 0.5 else -1
                ind[d] = (
                    pbest[i][d]
                    + direction * alpha * abs(mbest[d] - pbest[i][d]) * np.log(1 / u)
                )

            # Quantize and clamp
            ind[0] = int(np.clip(round(ind[0]), *PARTICLE_RANGE))
            ind[1] = float(np.clip(ind[1], *NOISE_RANGE))
            ind[2] = int(np.clip(round(ind[2]), *PATCH_RANGE))

            # Anti-convergence: random perturbation
            if random.random() < 0.15:
                j = random.randint(0, 2)
                ind[j] += random.uniform(-5, 5)

            fit = toolbox.evaluate(ind)
            ind.fitness.values = fit

            # Update pbest
            if fit[0] > pbest_fit[i][0]:
                pbest[i] = ind[:]
                pbest_fit[i] = fit

            # Update gbest
            if fit[0] > gbest_fit[0]:
                gbest = ind[:]
                gbest_fit = fit

    print(f"\n🏆 FINAL QPSO BEST: {gbest}, Fitness: {gbest_fit}")
    log_lines.append(f"\n🏆 FINAL: {gbest}, Fitness: {gbest_fit}")

    # Save log
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    return gbest, gbest_fit
