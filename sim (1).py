"""
3-drone simulation (Member 5), built from the team's models.

  Member 1: flight region, two box buildings, d_ij >= 5 m
  Member 2: C2 piecewise-cubic paths p_i(t), t = tau/T in [0, 2], T = 10 s
  Member 3: arc length / area / volume (re-checked here; coverage efficiency computed here)
  Member 4: r'' = a_ref + kd(v_ref - v) + kp(r_ref - r) + repulsion + wind,  |r''| <= a_max

Run:  python sim.py   ->  results.csv, trajectories.npz, figs/*.png, console report
"""
import os
import numpy as np
from scipy.integrate import solve_ivp

os.makedirs("figs", exist_ok=True)

# ------------------------------------------------------------ parameters
T = 10.0                 # s per unit of t (Member 2)
TAU_END = 20.0           # s, t in [0, 2]
DT = 0.02                # s, output sampling step
D_SAFE = 5.0             # m, drone-drone (Member 1/4)
D_SAFE_OBS = 3.0         # m, building (Member 4)
RHO0, ETA = 5.0, 5.0     # repulsion influence range (m) and gain (Member 4)
KP, KD = 4.0, 4.0        # tracking gains (Member 4)
A_MAX = 8.0              # m/s^2 (Member 4)
WIND_W = np.array([0.5, 0.0, 0.0])   # m/s (Member 4, optional)
WIND_C = 0.2                         # 1/s (Member 4, optional)
R_SENSE = 10.0           # m, ASSUMED ground footprint radius for the coverage metric
GRID = 1.0               # m, coverage grid cell size

REGION = np.array([100.0, 100.0, 50.0])
# (xmin, ymin, zmin, xmax, ymax, zmax)
BUILDINGS = np.array([[25.0, 20.0, 0.0, 40.0, 45.0, 25.0],
                      [60.0, 55.0, 0.0, 75.0, 80.0, 35.0]])

# Waypoints P0, P1, P2 at t = 0, 1, 2 (Member 2, Section 3)
WAYPOINTS = np.array([
    [[0, 0, 10], [20, 60, 30], [100, 100, 10]],     # drone 1
    [[0, 100, 15], [50, 30, 35], [100, 0, 15]],     # drone 2
    [[50, 0, 20], [85, 50, 40], [50, 100, 40]],     # drone 3
], dtype=float)
N_DRONES = len(WAYPOINTS)


# ----------------------------------------- Member 2: C2 piecewise cubics
def build_coefficients(W):
    """Member 2, Section 6: coefficients a + b u + c u^2 + d u^3 on each segment (u = t or t-1)."""
    coef = np.zeros((N_DRONES, 2, 4, 3))
    for i in range(N_DRONES):
        s0, s1, s2 = W[i]
        d1 = (s2 - 2 * s1 + s0) / 4
        b1 = (s1 - s0) - d1
        coef[i, 0] = [s0, b1, 0 * s0, d1]
        coef[i, 1] = [s1, b1 + 3 * d1, 3 * d1, -d1]
    return coef


COEF = build_coefficients(WAYPOINTS)


def ref(tau):
    """Reference position, velocity, acceleration in physical units. Each (n_drones, N, 3)."""
    tau = np.atleast_1d(np.asarray(tau, float))
    t = np.clip(tau / T, 0, 2)
    seg = (t > 1).astype(int)
    u = (t - seg)[None, :, None]
    C = COEF[:, seg]                                   # (n, N, 4, 3)
    a, b, c, d = C[:, :, 0], C[:, :, 1], C[:, :, 2], C[:, :, 3]
    r = a + b * u + c * u**2 + d * u**3
    v = (b + 2 * c * u + 3 * d * u**2) / T             # chain rule: p'(t)/T
    acc = (2 * c + 6 * d * u) / T**2                   # p''(t)/T^2
    return r, v, acc


# ------------------------------------------------------------- geometry
def box_delta(P, box):
    """Member 4: distance delta from point(s) P to a box (0 inside) and gradient n = grad(delta)."""
    P = np.atleast_2d(P)
    lo, hi = box[:3], box[3:]
    c, h = (lo + hi) / 2, (hi - lo) / 2
    q = np.abs(P - c) - h
    qp = np.maximum(q, 0.0)
    delta = np.linalg.norm(qp, axis=1)
    n = np.sign(P - c) * qp / np.where(delta > 0, delta, 1.0)[:, None]
    return delta, n


def clearance(P):
    """Min distance from points P (N,3) to any building (0 means inside/touching)."""
    return np.min([box_delta(P, b)[0] for b in BUILDINGS], axis=0)


def inside_building(P):
    out = np.zeros(len(P), bool)
    for b in BUILDINGS:
        out |= ((P > b[:3]) & (P < b[3:])).all(axis=1)
    return out


def separation(pos):
    n = len(pos)
    return {(i, j): np.linalg.norm(pos[i] - pos[j], axis=1)
            for i in range(n) for j in range(i + 1, n)}


# ------------------------------------------------- Member 4: ODE model
def repulsion(d, ds, n):
    rho = d - ds
    if rho >= RHO0:
        return 0.0 * n
    rho = max(rho, 1e-3)                               # barrier; avoid division by zero
    return ETA * (1 / rho - 1 / RHO0) / rho**2 * n


def make_rhs(avoid, wind):
    def rhs(tau, y):
        r, v = y[:9].reshape(3, 3), y[9:].reshape(3, 3)
        rr, vr, ar = (q[:, 0] for q in ref(tau))
        acc = ar + KD * (vr - v) + KP * (rr - r)
        if avoid:
            for i in range(N_DRONES):
                for j in range(N_DRONES):
                    if i != j:
                        dv = r[i] - r[j]
                        d = np.linalg.norm(dv)
                        acc[i] += repulsion(d, D_SAFE, dv / d)
                for b in BUILDINGS:
                    dl, n = box_delta(r[i], b)
                    acc[i] += repulsion(dl[0], D_SAFE_OBS, n[0])
        if wind:
            acc += WIND_C * (WIND_W - v)
        norms = np.linalg.norm(acc, axis=1, keepdims=True)
        acc = acc * np.minimum(1.0, A_MAX / np.maximum(norms, 1e-12))   # |r''| <= a_max
        return np.concatenate([v.ravel(), acc.ravel()])
    return rhs


def simulate(avoid, wind, t_eval, dr0=None):
    r0, v0, _ = ref(0.0)
    r0, v0 = r0[:, 0].copy(), v0[:, 0]
    if dr0 is not None:
        r0 += dr0
    y0 = np.concatenate([r0.ravel(), v0.ravel()])
    sol = solve_ivp(make_rhs(avoid, wind), (0, t_eval[-1]), y0, t_eval=t_eval,
                    method="RK45", rtol=1e-7, atol=1e-9, max_step=0.05)
    pos = sol.y[:9].T.reshape(-1, 3, 3).transpose(1, 0, 2)
    vel = sol.y[9:].T.reshape(-1, 3, 3).transpose(1, 0, 2)
    return pos, vel


# ---------------------------------------------------------------- metrics
def coverage_mask(p):
    xs = np.arange(GRID / 2, REGION[0], GRID)
    ys = np.arange(GRID / 2, REGION[1], GRID)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    cov = np.zeros(X.shape, bool)
    for q in p[::5]:
        cov |= (X - q[0]) ** 2 + (Y - q[1]) ** 2 <= R_SENSE**2
    foot = np.zeros(X.shape, bool)
    for b in BUILDINGS:
        foot |= (X >= b[0]) & (X <= b[3]) & (Y >= b[1]) & (Y <= b[4])
    return cov & ~foot


def metrics(pos, vel, t, label, ref_pos):
    dt = t[1] - t[0]
    sep = separation(pos)
    min_sep = min(s.min() for s in sep.values())
    min_clear = min(clearance(p).min() for p in pos)
    rows = {
        "case": label,
        "inside_flight_region": bool(all(((p >= -1e-3) & (p <= REGION + 1e-3)).all() for p in pos)),
        "min_separation_m": min_sep,
        "min_building_dist_m": min_clear,
        "separation_ok(>=5)": bool(min_sep >= D_SAFE),
        "building_margin_ok(>=3)": bool(min_clear >= D_SAFE_OBS),
        "entered_building": bool(any(inside_building(p).any() for p in pos)),
    }
    masks = []
    for i in range(N_DRONES):
        a = np.gradient(vel[i], dt, axis=0)
        j = np.gradient(a, dt, axis=0)
        err = np.linalg.norm(pos[i] - ref_pos[i], axis=1)
        L = float(np.sum(np.linalg.norm(np.diff(pos[i], axis=0), axis=1)))
        m = coverage_mask(pos[i]); masks.append(m)
        area = m.sum() * GRID**2
        k = f"d{i+1}"
        rows[f"{k}_length_m"] = L
        rows[f"{k}_vmax"] = np.linalg.norm(vel[i], axis=1).max()
        rows[f"{k}_amax"] = np.linalg.norm(a, axis=1).max()
        rows[f"{k}_jerk_max"] = np.linalg.norm(j, axis=1).max()
        rows[f"{k}_track_err_max_m"] = err.max()
        rows[f"{k}_track_err_rms_m"] = np.sqrt(np.mean(err**2))
        rows[f"{k}_coverage_m2"] = area
        rows[f"{k}_efficiency_m2_per_m"] = area / L
    rows["swarm_coverage_m2"] = np.logical_or.reduce(masks).sum() * GRID**2
    return rows


# ------------------------------------------------- check teammates' numbers
def verify(t, ref_pos, ref_vel):
    print("\n=== CHECK OF TEAM NUMBERS (their value -> recomputed) ===")
    tt = np.linspace(0, 2, 200001)
    r, _, _ = ref(tt * T)
    # Member 2 arc lengths (Member 3 integrals)
    claimed = [(68.38, 92.51, 160.89), (88.79, 62.88, 151.67), (65.98, 65.98, 131.95)]
    for i in range(N_DRONES):
        seg = np.linalg.norm(np.diff(r[i], axis=0), axis=1)
        L1 = seg[:100000].sum(); L2 = seg[100000:].sum()
        print(f"Drone {i+1} arc length: seg1 {claimed[i][0]} -> {L1:.2f}, seg2 {claimed[i][1]} -> {L2:.2f}, total {claimed[i][2]} -> {L1+L2:.2f}")
    # Member 2 min distances
    claim_b = {0: (13.78, 4.35), 1: (7.42, 19.65), 2: (10.00, 5.47)}
    for i in range(N_DRONES):
        d1 = box_delta(r[i], BUILDINGS[0])[0].min(); d2 = box_delta(r[i], BUILDINGS[1])[0].min()
        print(f"Drone {i+1} min dist to B1: {claim_b[i][0]} -> {d1:.2f};  to B2: {claim_b[i][1]} -> {d2:.2f}")
    claim_s = {(0, 1): 31.30, (0, 2): 43.18, (1, 2): 32.92}
    for (i, j), s in separation(r).items():
        print(f"min separation D{i+1}-D{j+1}: {claim_s[(i, j)]} -> {s.min():.2f}")
    # Member 4 example A
    dl = box_delta(r[0], BUILDINGS[1])[0]
    k = dl.argmin()
    print(f"Member 4 ex. A (Drone 1 vs B2): t=1.547, r=(58.37,83.68,22.66), delta=4.03 -> "
          f"t={tt[k]:.3f}, r=({r[0][k][0]:.2f},{r[0][k][1]:.2f},{r[0][k][2]:.2f}), delta={dl[k]:.2f}")
    rho = dl[k] - D_SAFE_OBS
    print(f"   repulsion magnitude 3.63 -> {ETA*(1/rho-1/RHO0)/rho**2:.2f} m/s^2 (using this delta)")
    # speeds / accelerations quoted by Member 2
    sp = np.linalg.norm(ref_vel[0], axis=1)
    print(f"Drone 1 peak speed 10.61 m/s -> {sp.max():.2f};  speed at tau=10 s 7.07 -> {sp[np.argmin(abs(t-10))]:.2f}")
    # Member 4 example B: tracking error (1+2 tau) exp(-2 tau)
    dr0 = np.zeros((3, 3)); dr0[0, 0] = 1.0
    pos, _ = simulate(False, False, t, dr0=dr0)
    e_num = pos[0][:, 0] - ref_pos[0][:, 0]
    e_th = (1 + 2 * t) * np.exp(-2 * t)
    print(f"Member 4 ex. B tracking error e(tau)=(1+2tau)e^(-2tau): max |numerical - analytic| = {np.abs(e_num - e_th).max():.2e} m")


# ------------------------------------------------------------------ plots
def _boxes(ax):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    for b in BUILDINGS:
        x0, y0, z0, x1, y1, z1 = b
        v = np.array([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
                      [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]])
        faces = [[v[k] for k in f] for f in
                 [(0,1,2,3),(4,5,6,7),(0,1,5,4),(2,3,7,6),(1,2,6,5),(0,3,7,4)]]
        ax.add_collection3d(Poly3DCollection(faces, facecolors="grey", edgecolors="k", alpha=0.35))


def plot_3d(cases, fname):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection="3d")
    _boxes(ax)
    for i in range(N_DRONES):
        c = f"C{i}"
        ax.plot(*cases["ref"][i].T, "--", color=c, lw=1, alpha=0.7)
        ax.plot(*cases["wind+avoid"][i].T, color=c, lw=2, label=f"Drone {i+1}")
        ax.scatter(*cases["ref"][i][0], color=c, s=30); ax.scatter(*cases["ref"][i][-1], color=c, marker="*", s=70)
    ax.set_xlim(0, REGION[0]); ax.set_ylim(0, REGION[1]); ax.set_zlim(0, REGION[2])
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.set_title("Planned (dashed) vs flown with avoidance + wind (solid)"); ax.legend()
    fig.tight_layout(); fig.savefig(fname, dpi=200); plt.close(fig)


def plot_safety(cases, t, fname):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for name, pos in cases.items():
        a1.plot(t, np.min(list(separation(pos).values()), axis=0), label=name)
        a2.plot(t, np.min([clearance(p) for p in pos], axis=0), label=name)
    a1.axhline(D_SAFE, color="r", ls="--"); a1.set_ylabel("min drone-drone distance (m)"); a1.legend()
    a2.axhline(D_SAFE_OBS, color="r", ls="--"); a2.set_ylabel("min building distance (m)"); a2.set_xlabel("t (s)")
    fig.tight_layout(); fig.savefig(fname, dpi=200); plt.close(fig)


def plot_tracking(cases, t, ref_pos, fname):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, pos in cases.items():
        if name != "ref":
            ax.plot(t, np.max([np.linalg.norm(pos[i] - ref_pos[i], axis=1) for i in range(N_DRONES)], axis=0), label=name)
    ax.set_xlabel("t (s)"); ax.set_ylabel("max tracking error over drones (m)"); ax.legend()
    fig.tight_layout(); fig.savefig(fname, dpi=200); plt.close(fig)


# ------------------------------------------------------------------- main
if __name__ == "__main__":
    import csv
    t = np.arange(0, TAU_END + DT / 2, DT)
    ref_pos, ref_vel, _ = ref(t)
    cases = {"ref": ref_pos}
    vels = {"ref": ref_vel}
    for name, (av, wi) in {"avoid": (True, False), "wind": (False, True), "wind+avoid": (True, True)}.items():
        cases[name], vels[name] = simulate(av, wi, t)

    rows = [metrics(cases[k], vels[k], t, k, ref_pos) for k in cases]
    with open("results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    np.savez("trajectories.npz", t=t, pos=cases["wind+avoid"], **{f"pos_{k.replace('+','_')}": v for k, v in cases.items()})

    plot_3d(cases, "figs/traj_3d.png"); plot_safety(cases, t, "figs/safety.png"); plot_tracking(cases, t, ref_pos, "figs/tracking.png")

    print(f"{'metric':28s}" + "".join(f"{r['case']:>14s}" for r in rows))
    for k in rows[0]:
        if k != "case":
            print(f"{k:28s}" + "".join(f"{str(r[k]):>14s}" if isinstance(r[k], (bool, np.bool_)) else f"{r[k]:>14.4g}" for r in rows))
    verify(t, ref_pos, ref_vel)
