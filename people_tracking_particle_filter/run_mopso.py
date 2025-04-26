from mopso_optimizer import run_mopso
 # Use small values to test first

from mopso_plot_pareto import plot_mopso_pareto

# after run_mopso
best, best_fit = run_mopso(pop_size=4, generations=2)
plot_mopso_pareto(archive)  # archive = final archive list you have
