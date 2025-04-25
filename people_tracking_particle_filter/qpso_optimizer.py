import numpy as np
import random
import torch
import time
from deap import base, creator, tools
from evaluation import run_tracking_evaluation
from config import VIDEO_PATH, PARTICLE_RANGE, NOISE_RANGE, PATCH_RANGE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DEAP Setup ---
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, 1.0))  # MOTA ↑, IDSW ↓, FPS ↑
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
    print(f"✅ [QPSO] Eval -> MOTA={mota:.3f}, IDSW={idsw}, FPS={fps:.2f} | Time: {time.time()-start:.1f}s")
    return mota, idsw, fps

toolbox.register("evaluate", evaluate)

# --- QPSO Core ---
def run_qpso(pop_size=20, max_gen=10, alpha=0.75, beta=0.85):
    pop = toolbox.population(n=pop_size)
    global_best = None
    global_best_fit = None

    # Initialize personal bests
    pbest = pop[:]
    pbest_fit = [toolbox.evaluate(ind) for ind in pbest]
    for ind, fit in zip(pbest, pbest_fit):
        ind.fitness.values = fit

    global_best_idx = np.argmax([fit[0] for fit in pbest_fit])
    global_best = pbest[global_best_idx][:]
    global_best_fit = pbest_fit[global_best_idx]

    for gen in range(max_gen):
        print(f"\n📘 [GEN {gen}] Best MOTA={global_best_fit[0]:.3f}, FPS={global_best_fit[2]:.2f}")

        mbest = np.mean([ind for ind in pop], axis=0)
        for i, ind in enumerate(pop):
            for d in range(3):  # dimension-wise
                u = random.random()
                direction = 1 if random.random() < 0.5 else -1
                p = pbest[i][d]
                m = mbest[d]
                ind[d] = p + direction * alpha * abs(m - ind[d]) * np.log(1 / u)

            # Quantize
            ind[0] = int(np.clip(round(ind[0]), *PARTICLE_RANGE))
            ind[1] = float(np.clip(ind[1], *NOISE_RANGE))
            ind[2] = int(np.clip(round(ind[2]), *PATCH_RANGE))

            # Anti-Convergence Kick
            if random.random() < 0.1:
                ind[random.randint(0, 2)] += random.uniform(-3, 3)

            # Evaluate
            fit = toolbox.evaluate(ind)
            ind.fitness.values = fit

            # Update pbest
            if fit[0] > pbest_fit[i][0]:
                pbest[i] = ind[:]
                pbest_fit[i] = fit

            # Update gbest
            if fit[0] > global_best_fit[0]:
                global_best = ind[:]
                global_best_fit = fit

    print(f"\n🏆 [QPSO] Best Found: {global_best}, Fitness: {global_best_fit}")
    return global_best, global_best_fit
