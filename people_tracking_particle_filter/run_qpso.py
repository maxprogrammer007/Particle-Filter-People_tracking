#!/usr/bin/env python3
import yaml
from qpso_optimizer import run_qpso

def main():
    # 1) Load QPSO settings
    with open("C:\\Users\\abhin\\OneDrive\\Documents\\GitHub\\Particle-Filter-People_tracking\\people_tracking_particle_filter\\configs\\qpso_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    qpso_cfg = cfg["qpso"]

    pop_size = qpso_cfg["pop_size"]
    max_gen  = qpso_cfg["num_iter"]
    alpha    = qpso_cfg.get("alpha", 0.75)

    # 2) Run the optimizer
    run_qpso(pop_size=5, max_gen=2, alpha=alpha)

if __name__ == "__main__":
    main()
