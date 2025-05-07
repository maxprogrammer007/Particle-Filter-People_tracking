# optimization/nsga_optimizer.py

from deap import base, creator, tools, algorithms
import random
import numpy as np
import yaml
from evaluation.evaluation import evaluate_pipeline

def load_config(path="people_tracking_project\optimization\config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def make_individual(config_bounds):
    return [random.uniform(*config_bounds[key]) for key in config_bounds]

def decode_individual(ind, keys, allowed_values):
    decoded = {}
    for i, key in enumerate(keys):
        decoded[key] = min(allowed_values[key], key=lambda x: abs(x - ind[i]))
    return decoded

def main():
    config_space = load_config()
    bounds = {}
    allowed_values = {}
    for section, opts in config_space.items():
        for k, v in opts.items():
            bounds[k] = [min(v), max(v)]
            allowed_values[k] = v

    keys = list(bounds.keys())

    creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, -1.0))  # Maximize MOTA, IDF1; Minimize 1/FPS
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", make_individual, bounds)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        config = decode_individual(individual, keys, allowed_values)
        mota, idf1, fps = evaluate_pipeline(config)
        return mota, idf1, 1 / fps

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=[b[0] for b in bounds.values()],
                     up=[b[1] for b in bounds.values()], eta=20.0)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=[b[0] for b in bounds.values()],
                     up=[b[1] for b in bounds.values()], eta=20.0, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=30)
    algorithms.eaMuPlusLambda(pop, toolbox, mu=30, lambda_=60, cxpb=0.6, mutpb=0.3, ngen=15,
                              stats=None, halloffame=None, verbose=True)

    best = tools.selBest(pop, 1)[0]
    best_config = decode_individual(best, keys, allowed_values)
    print("\n[✓] Best NSGA-II Config Found:")
    print(best_config)

if __name__ == "__main__":
    main()
