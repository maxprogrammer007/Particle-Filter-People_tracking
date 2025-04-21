from deap import base, creator, tools, algorithms
import random
import time
import numpy as np
from evaluation import run_tracking_evaluation
from config import VIDEO_PATH, NSGA_GENERATIONS, NSGA_POP_SIZE
from plot_pareto import plot_pareto



# --- DEAP setup ---
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

# Parameter ranges
PARTICLE_RANGE = (30, 80)       # 🧠 Lower upper bound for faster eval
NOISE_RANGE = (1.0, 8.0)        # Optional: limit extreme values
PATCH_RANGE = (10, 30)          # Large patches slow deep feature extraction


toolbox.register("num_particles", random.randint, *PARTICLE_RANGE)
toolbox.register("motion_noise", random.uniform, *NOISE_RANGE)
toolbox.register("patch_size", random.randint, *PATCH_RANGE)

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.num_particles, toolbox.motion_noise, toolbox.patch_size), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# --- Evaluation Function ---
def evaluate(individual):
    # Cast values to correct types
    num_particles = int(np.clip(round(individual[0]), *PARTICLE_RANGE))
    motion_noise = float(individual[1])
    patch_size = int(np.clip(round(individual[2]), *PATCH_RANGE))

    print(f"🔍 Evaluating: Particles={num_particles}, Noise={motion_noise:.2f}, Patch={patch_size}")
    start = time.time()

    # Fast mode: only process 100 frames inside evaluation
    mota, id_switches, fps = run_tracking_evaluation(
        VIDEO_PATH,
        num_particles=num_particles,
        motion_noise=motion_noise,
        patch_size=patch_size,
        max_frames=100  # ⚡ LIMIT to 100 frames
    )

    elapsed = time.time() - start
    print(f"✅ Result: MOTA={mota}, ID Switches={id_switches}, FPS={fps:.2f} | Time: {elapsed:.1f}s")
    return mota, id_switches, fps

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=5, indpb=0.2)
toolbox.register("select", tools.selNSGA2)

# --- NSGA-II Main Function ---
def run_nsga(generations=NSGA_GENERATIONS, pop_size=NSGA_POP_SIZE):
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda fits: tuple(sum(f) / len(f) for f in zip(*fits)))
    stats.register("min", lambda fits: tuple(min(f) for f in zip(*fits)))
    stats.register("max", lambda fits: tuple(max(f) for f in zip(*fits)))

    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, mu=pop_size, lambda_=pop_size,
                                         cxpb=0.6, mutpb=0.3, ngen=generations,
                                         stats=stats, halloffame=hof, verbose=True)

    best_ind = hof[0]
    print("\n🏆 Best Configuration:", best_ind)

    # Save best config
    with open("best_config.txt", "w") as f:
        f.write("Best Configuration:\n")
        f.write(f"Particles: {int(round(best_ind[0]))}\n")
        f.write(f"Noise: {best_ind[1]:.2f}\n")
        f.write(f"Patch Size: {int(round(best_ind[2]))}\n")
        f.write(f"Fitness (MOTA, ID Switches, FPS): {best_ind.fitness.values}\n")

    # Plot results
    plot_pareto(pop)
    return pop, log, hof

if __name__ == "__main__":
    run_nsga(generations=2, pop_size=4)
