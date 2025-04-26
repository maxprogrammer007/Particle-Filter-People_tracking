# mopso_plot_pareto.py

import matplotlib.pyplot as plt

def plot_mopso_pareto(archive):
    """
    Plot MOPSO Pareto front from the archive.
    Each particle's fitness contains (MOTA, IDSW, FPS)
    """
    mota = [ind.fitness.values[0] for ind in archive]
    fps = [ind.fitness.values[2] for ind in archive]

    plt.figure(figsize=(8,6))
    plt.scatter(fps, mota, c='blue', s=50, edgecolors='k')
    plt.title("MOPSO Pareto Front")
    plt.xlabel("FPS (Higher is Better)")
    plt.ylabel("MOTA (Higher is Better)")
    plt.grid(True)
    plt.savefig("figure_mopso.png")
    plt.show()
