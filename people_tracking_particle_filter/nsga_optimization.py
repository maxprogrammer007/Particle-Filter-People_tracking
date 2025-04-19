# nsga_optimization.py
from deap import base, creator, tools, algorithms
import random
from evaluation import run_tracking_evaluation

# DEAP setup
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, 1.0))  # Maximize MOTA, minimize ID switches, maximize FPS
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

# Parameter ranges
PARTICLE_RANGE = (30, 150)
NOISE_RANGE = (1.0, 10.0)
PATCH_RANGE = (10, 40)

# Register individuals
toolbox.register("num_particles", random.randint, *PARTICLE_RANGE)
toolbox.register("motion_noise", random.uniform, *NOISE_RANGE)
toolbox.register("patch_size", random.randint, *PATCH_RANGE)

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.num_particles, toolbox.motion_noise, toolbox.patch_size), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Evaluation
VIDEO_PATH = "sample_videos/test_video.mp4"  # update path if needed

def evaluate(individual):
    num_particles, motion_noise, patch_size = individual
    mota, id_switches, fps = run_tracking_evaluation(
        VIDEO_PATH,
        num_particles=num_particles,
        motion_noise=motion_noise,
        patch_size=patch_size
    )
    return mota, id_switches, fps

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=5, indpb=0.2)
toolbox.register("select", tools.selNSGA2)

def run_nsga(generations=10, pop_size=10):
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda fits: tuple(sum(f) / len(f) for f in zip(*fits)))
    stats.register("min", lambda fits: tuple(min(f) for f in zip(*fits)))
    stats.register("max", lambda fits: tuple(max(f) for f in zip(*fits)))

    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, mu=pop_size, lambda_=pop_size,
                                         cxpb=0.6, mutpb=0.3, ngen=generations,
                                         stats=stats, halloffame=hof, verbose=True)

    print("\nBest Configuration:", hof[0])
    return pop, log, hof

if __name__ == "__main__":
    run_nsga(generations=5, pop_size=6)  # You can increase for real runs