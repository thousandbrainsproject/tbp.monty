from pathlib import Path
from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt

save_dir = Path(__file__).parent

rng = np.random.default_rng(12345)
points = rng.random((30, 2))
hull = ConvexHull(points)

# Naive approach: Plot all points without convex hull
plt.plot(points[:, 0], points[:, 1], 'o', color='blue')

# Hypothesized point at (-0.1, 1.1) in big green dot
plt.plot(-0.1, 1.1, 'go', markersize=10)
plt.text(-0.05, 1.05, 'Hypothesis', fontsize=12)
plt.title('Naive approach: Compare to all points')

plt.savefig(save_dir / 'convex_hull_example_naive.png')

# Convex hull approach: Plot only the convex hull
plt.plot(points[:, 0], points[:, 1], 'o', color='blue')
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

plt.plot(-0.1, 1.1, 'go', markersize=10)
plt.text(-0.05, 1.05, 'Hypothesis', fontsize=12)
plt.title('Convex hull approach: Compare to "edge" points')

plt.savefig(save_dir / 'convex_hull_example_convex_hull.png')