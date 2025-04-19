# plot_pareto.py
import matplotlib.pyplot as plt

def plot_pareto(population):
    mota, id_switches, fps = zip(*[ind.fitness.values for ind in population])
    fig, ax = plt.subplots()
    sc = ax.scatter(mota, fps, c=id_switches, cmap='plasma', s=100, edgecolors='k')
    ax.set_xlabel('MOTA (Tracking Accuracy)')
    ax.set_ylabel('FPS (Performance)')
    ax.set_title('Pareto Front - NSGA-II Optimization')
    plt.colorbar(sc, label='ID Switches (Lower is Better)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage after optimization:
# from nsga_optimization import run_nsga
# pop, log, hof = run_nsga()
# plot_pareto(pop)
