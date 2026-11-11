"""
3D animation of the simulation. Run sim.py first (creates trajectories.npz).

    python animate.py        -> figs/flight.gif
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sim import BUILDINGS, D_SAFE as SAFE_SEP, REGION, separation

import os, subprocess, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # always work in this folder
os.makedirs("figs", exist_ok=True)
if not os.path.exists("trajectories.npz"):             # create it by running sim.py
    subprocess.run([sys.executable, "sim.py"], check=True)

data = np.load("trajectories.npz")
t, pos = data["t"], data["pos"]            # pos: (n_drones, N, 3)
step = 5                                    # use every 10th sample -> smaller file
t, pos = t[::step], pos[:, ::step]
sep = np.min(list(separation(pos).values()), axis=0)   # min pairwise sep at each frame

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim(0, REGION[0]); ax.set_ylim(0, REGION[1]); ax.set_zlim(0, REGION[2])
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")

for b in BUILDINGS:                         # static buildings
    x0, y0, z0, x1, y1, z1 = b
    v = np.array([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
                  [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]])
    faces = [[v[k] for k in f] for f in
             [(0,1,2,3),(4,5,6,7),(0,1,5,4),(2,3,7,6),(1,2,6,5),(0,3,7,4)]]
    ax.add_collection3d(Poly3DCollection(faces, facecolors="grey", edgecolors="k", alpha=0.35))

colors = ["tab:blue", "tab:orange", "tab:green"]
trails = [ax.plot([], [], [], color=c, lw=1.5)[0] for c in colors]
dots = [ax.plot([], [], [], "o", color=c, ms=8)[0] for c in colors]
label = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)


def update(k):
    for i in range(len(pos)):
        trails[i].set_data(pos[i, :k + 1, 0], pos[i, :k + 1, 1])
        trails[i].set_3d_properties(pos[i, :k + 1, 2])
        dots[i].set_data(pos[i, k:k + 1, 0], pos[i, k:k + 1, 1])
        dots[i].set_3d_properties(pos[i, k:k + 1, 2])
        dots[i].set_color("red" if sep[k] < SAFE_SEP else colors[i])   # red = too close
    label.set_text(f"t = {t[k]:5.1f} s   min separation = {sep[k]:4.2f} m")
    ax.view_init(elev=25, azim=-60 + 0.3 * k)                           # slow orbit
    return trails + dots


anim = FuncAnimation(fig, update, frames=len(t), interval=50, blit=False)
anim.save("figs/flight.gif", writer=PillowWriter(fps=20), dpi=90)
print("saved figs/flight.gif")
