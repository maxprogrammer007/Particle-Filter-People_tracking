import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1) Collect your results into a DataFrame
data = {
    "MOTA": [0.834, 0.857, 0.850, 0.869, 0.839,
             0.848, 0.856, 0.850, 0.870, 0.840,
             0.853, 0.847, 0.851, 0.867, 0.850],
    "FPS":  [1.20, 0.98, 1.51, 1.29, 1.63,
             1.53, 1.03, 0.99, 1.14, 1.63,
             1.47, 1.53, 1.50, 1.35, 1.51]
}
df = pd.DataFrame(data)

# 2) Identify Pareto‐optimal points
def is_pareto(df):
    pts = df[['FPS','MOTA']].values
    is_optimal = np.ones(len(pts), dtype=bool)
    for i, p in enumerate(pts):
        if is_optimal[i]:
            # any point strictly better (>= in both, > in one) will dominate
            is_optimal &= ~((pts[:,0] >= p[0]) & (pts[:,1] >= p[1]) & ((pts[:,0] > p[0]) | (pts[:,1] > p[1])))
            is_optimal[i] = True
    return is_optimal

pareto_mask = is_pareto(df)
pareto_df = df[pareto_mask]

# 3) Plot all points and highlight Pareto frontier
plt.figure(figsize=(6,4))
plt.scatter(df["FPS"], df["MOTA"], label="All configs")
plt.scatter(pareto_df["FPS"], pareto_df["MOTA"], color="red", label="Pareto frontier")
plt.xlabel("FPS")
plt.ylabel("MOTA")
plt.title("Trade‐off: FPS vs MOTA")
plt.legend()
plt.grid(True)
plt.show()

# 4) Print the Pareto‐optimal configs
print("Pareto‐optimal configs:")
print(pareto_df.sort_values("FPS"))
