from qpso_optimizer import run_qpso

if __name__ == "__main__":
    best_config, best_fit = run_qpso(pop_size=4, max_gen=2)  # Adjust for final runs
