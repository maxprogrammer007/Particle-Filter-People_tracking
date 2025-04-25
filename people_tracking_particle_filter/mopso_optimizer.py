import numpy as np
import random
import time
import torch
from deap import base, creator, tools
from evaluation import run_tracking_evaluation
from config import VIDEO_PATH, PARTICLE_RANGE, NOISE_RANGE, PATCH_RANGE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] MOPSO running on device: {device}")

# DEAP setup
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, 1.0))  # MOTA↑, IDSW↓, FPS↑
creator.create("Particle", list, fitness=creator.FitnessMulti)

def random_position():
    return [
        random.randint(*PARTICLE_RANGE),
        random.uniform(*NOISE_RANGE),
        random.randint(*PATCH_RANGE)
    ]

def random_velocity():
    return [
        random.uniform(-10, 10),
        random.uniform(-2.5, 2.5),
        random.uniform(-5, 5)
    ]

def clip_position(p):
    p[0] = int(np.clip(round(p[0]), *PARTICLE_RANGE))
    p[1] = float(np.clip(p[1], *NOISE_RANGE))
    p[2] = int(np.clip(round(p[2]), *PATCH_RANGE))
    return p

toolbox = base.Toolbox()
toolbox.register("particle", tools.initIterate, creator.Particle, random_position)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)

def evaluate(ind):
    start = time.time()
    mota, idsw, fps = run_tracking_evaluation(
        VIDEO_PATH, ind[0], ind[1], ind[2], max_frames=100
    )
    print(f"✅ [MOPSO] Eval -> MOTA={mota:.3f}, IDSW={idsw}, FPS={fps:.2f} | Time: {time.time() - start:.1f}s")
    return mota, idsw, fps

toolbox.register("evaluate", evaluate)

def crowding_sort(archive, k):
    fronts = tools.sortNondominated(archive, len(archive), first_front_only=True)
    return tools.selNSGA2(fronts[0], k) if len(fronts[0]) > k else fronts[0]

def run_mopso(pop_size=8, generations=3, inertia=0.5, phi_p=1.5, phi_g=1.5):
    pop = toolbox.population(n=pop_size)
    velocities = [random_velocity() for _ in range(pop_size)]

    # Initialize personal bests
    pbest = [ind[:] for ind in pop]
    pbest_fit = [toolbox.evaluate(ind) for ind in pbest]
    for ind, fit in zip(pbest, pbest_fit):
        ind.fitness.values = fit

    archive = pbest[:]
    archive_fit = [ind.fitness.values for ind in archive]

    global_best = max(archive, key=lambda ind: ind.fitness.values[0])

    for gen in range(generations):
        print(f"\n📘 GEN {gen} | Archive Size: {len(archive)}")

        for i, ind in enumerate(pop):
            leader = random.choice(crowding_sort(archive, k=pop_size))

            for d in range(3):
                r_p, r_g = random.random(), random.random()
                velocities[i][d] = (
                    inertia * velocities[i][d]
                    + phi_p * r_p * (pbest[i][d] - ind[d])
                    + phi_g * r_g * (leader[d] - ind[d])
                )
                velocities[i][d] = np.clip(velocities[i][d], -10, 10)
                ind[d] += velocities[i][d]

            # Clip and reassign to preserve fitness object
            clipped = clip_position(ind)
            ind[:] = clipped

            fit = toolbox.evaluate(ind)
            ind.fitness.values = fit

            # Personal best update
            if fit[0] > pbest_fit[i][0]:
                pbest[i] = ind[:]
                pbest_fit[i] = fit

            # Archive update
            archive.append(ind[:])

        # Prune using non-dominated sorting and crowding distance
        archive = crowding_sort(archive, k=pop_size)

        # Track current global best
        current_best = max(archive, key=lambda ind: ind.fitness.values[0])
        if current_best.fitness.values[0] > global_best.fitness.values[0]:
            global_best = current_best

        print(f"📊 Best So Far: MOTA={global_best.fitness.values[0]:.3f}, FPS={global_best.fitness.values[2]:.2f}")

    print(f"\n🏆 FINAL MOPSO BEST: {global_best}, Fitness: {global_best.fitness.values}")
    return global_best, global_best.fitness.values
