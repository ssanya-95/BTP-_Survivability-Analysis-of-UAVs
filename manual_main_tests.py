# manual_main_tests.py
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import config
from entities import UCAV, ThreatRadar, Target
from strategies import GradientOptimizer
from matplotlib.patches import Polygon
from matplotlib.transforms import Affine2D


# ============================================================
# 1) Scenario Library (hand-made test cases)
# ============================================================
def build_scenarios():
    """
    Returns a dict: name -> scenario_config
    Each scenario defines:
      - ucav_start: (x, y, heading_deg)
      - target: (x, y)
      - radars: list of (x, y)
    """
    scenarios = {}

    # --- Tier 1: Sanity Checks ---
    scenarios["baseline_no_threat"] = {
        "ucav_start": (0, 0, 0),
        "target": (45000, 0),
        "radars": []
    }

    scenarios["single_radar_on_path"] = {
        "ucav_start": (0, 0, 0),
        "target": (45000, 0),
        "radars": [(20000, 0)]
    }

    scenarios["single_radar_offset"] = {
        "ucav_start": (0, 0, 0),
        "target": (45000, 0),
        "radars": [(20000, 7000)]
    }

    # --- Tier 2: Your classic iron gate ---
    scenarios["iron_gate_2_radars"] = {
        "ucav_start": (0, 0, 0),
        "target": (45000, 0),
        "radars": [(20000, 8000), (20000, -8000)]
    }

    # --- Geometry torture: overlap tight ---
    scenarios["tight_gate_overlap"] = {
        "ucav_start": (0, 0, 0),
        "target": (45000, 0),
        "radars": [(20000, 5000), (20000, -5000)]  # even tighter gap
    }

    # --- Radars are off-center barrier ---
    scenarios["radar_wall_5_emitters"] = {
        "ucav_start": (0, 0, 0),
        "target": (45000, 0),
        "radars": [
            (20000, -15000),
            (20000, -7000),
            (20000, 0),
            (20000, 7000),
            (20000, 15000),
        ]
    }

    # --- Cluster near objective (endgame stress) ---
    scenarios["cluster_near_target"] = {
        "ucav_start": (0, 0, 0),
        "target": (45000, 0),
        "radars": [(38000, 0), (40000, 8000), (40000, -8000)]
    }

    # --- Objective slightly off axis ---
    scenarios["target_off_axis"] = {
        "ucav_start": (0, 0, 0),
        "target": (45000, 6000),
        "radars": [(20000, 0), (28000, 9000)]
    }

    # --- Start with bad heading (tests heading normalization + turn logic) ---
    scenarios["start_heading_wrong_way"] = {
        "ucav_start": (0, 0, 170),  # almost facing backward
        "target": (45000, 0),
        "radars": [(20000, 8000), (20000, -8000)]
    }

    # --- Start near a radar (panic handling) ---
    scenarios["start_inside_detect_zone"] = {
        "ucav_start": (15000, 2000, 0),
        "target": (45000, 0),
        "radars": [(20000, 0)]
    }

    return scenarios


# ============================================================
# 2) Core Simulator Runner
# ============================================================
def run_scenario(name, scenario, seed=0, verbose=True, do_plot=True):
    np.random.seed(seed)

    ucav_x, ucav_y, ucav_h = scenario["ucav_start"]
    tx, ty = scenario["target"]
    radar_positions = scenario["radars"]

    ucav = UCAV(x=ucav_x, y=ucav_y, heading_deg=ucav_h)
    target = Target(x=tx, y=ty)
    radars = [ThreatRadar(x=rx, y=ry) for (rx, ry) in radar_positions]

    brain = GradientOptimizer()

    t = 0.0
    status = "TIMEOUT"
    last_state = None

    if verbose:
        print("\n" + "=" * 85)
        print(f"SCENARIO: {name}")
        print(f"UCAV start: ({ucav_x:.0f}, {ucav_y:.0f}), heading={ucav_h:.1f} deg")
        print(f"TARGET:     ({tx:.0f}, {ty:.0f})")
        print(f"RADARS:     {len(radars)}")
        print(f"MAX TIME:   {config.MAX_TIME}s  |  DT: {config.DT}s")
        print("-" * 85)
        print(f"{'TIME':<8} | {'STATE':<10} | {'CLOSEST RADAR':<22} | {'ACTION'}")
        print("-" * 85)

    while t < config.MAX_TIME:
        # 1) Determine current overall state (worst state among radars)
        current_state = "SAFE"
        closest_dist = float("inf")
        closest_radar_idx = None

        threat_levels = {"HIT": 5, "LOCKED": 4, "TRACKED": 3, "DETECTED": 2, "SAFE": 1}
        highest_priority = 0

        for i, radar in enumerate(radars):
            s = radar.get_perceived_state(ucav)
            d = np.linalg.norm(radar.pos - ucav.pos)

            if d < closest_dist:
                closest_dist = d
                closest_radar_idx = i

            p = threat_levels.get(s, 0)
            if p > highest_priority:
                highest_priority = p
                current_state = s

        ucav.state = current_state

        # 2) Strategy step
        control = brain.get_optimal_control(ucav, radars, target)

        # 3) Logging (state change or every 10 seconds)
        if verbose:
            should_print = False
            if current_state != last_state:
                should_print = True
            if int(t) % 10 == 0 and abs(t - int(t)) < config.DT / 2:
                should_print = True

            if should_print:
                jam_status = control.get("jam", "OFF")
                chaff = "+CHAFF" if control.get("chaff", False) else ""

                if closest_radar_idx is None:
                    radar_str = "None"
                    dist_str = ""
                else:
                    radar_str = f"Radar {chr(65 + closest_radar_idx)}"
                    dist_str = f"({closest_dist/1000:4.1f}km)"

                print(f"{t:5.1f}s   | {current_state:<10} | {radar_str:<8} {dist_str:<12} | {jam_status} {chaff}")
                last_state = current_state

        # 4) Physics step
        ucav.move(config.DT, control)

        # 5) Terminal checks
        if target.check_destroyed(ucav.pos):
            if verbose:
                print(f"\n[{t:.1f}s] ✅ TARGET DESTROYED! Mission Success.")
            status = "SUCCESS"
            break

        if current_state == "HIT":
            if verbose:
                print(f"\n[{t:.1f}s] 💥 UCAV SHOT DOWN. Mission Failure.")
            status = "FAILURE"
            break

        t += config.DT

    if status == "TIMEOUT" and verbose:
        print(f"\n[{t:.1f}s] ⏳ Mission TIMEOUT. (Likely stuck/looping)")

    if do_plot:
        plot_results(ucav, radars, target, status, title=name)

    return status


# ============================================================
# 3) Plotting (reused + generalized)
# ============================================================

def _draw_plane_glyph(ax, x, y, heading_rad, scale=120, color="green", alpha=0.75):
    """
    Draw a small plane-like silhouette at (x,y), oriented along heading_rad.
    Uses a simple polygon: nose + wings + tail.
    """

    # A simple "jet" shape in local coordinates (pointing +X direction)
    # You can tweak these numbers for aesthetics.
    pts = np.array([
        [ 2.2,  0.0],   # Nose
        [ 1.0,  0.35],  # Upper fuselage
        [ 0.4,  0.85],  # Wing upper tip
        [ 0.2,  0.30],  # Wing root upper
        [-1.0,  0.30],  # Tail upper
        [-1.4,  0.55],  # Tail fin upper
        [-1.2,  0.00],  # Tail center
        [-1.4, -0.55],  # Tail fin lower
        [-1.0, -0.30],  # Tail lower
        [ 0.2, -0.30],  # Wing root lower
        [ 0.4, -0.85],  # Wing lower tip
        [ 1.0, -0.35],  # Lower fuselage
    ])

    # Scale it to meters-ish (your plot uses meters)
    pts = pts*2 * scale

    # Rotate + translate
    transform = Affine2D().rotate(heading_rad).translate(x, y) + ax.transData

    poly = Polygon(
        pts,
        closed=True,
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        transform=transform,
        linewidth=1.5
    )
    ax.add_patch(poly)

def plot_parallel_plane_markers(ax, xs, ys, step=40, offset=600, scale=110):
    """
    Places plane silhouettes along the path (every `step` points),
    with a perpendicular offset so it forms a nice parallel line.
    """
    for i in range(0, len(xs) - 2, step):
        x0, y0 = xs[i], ys[i]
        x1, y1 = xs[i + 1], ys[i + 1]

        dx = x1 - x0
        dy = y1 - y0
        norm = (dx*dx + dy*dy) ** 0.5
        if norm < 1e-6:
            continue

        # Direction unit vector
        ux, uy = dx / norm, dy / norm

        # Perpendicular unit vector (left)
        px, py = -uy, ux

        # Offset point to create parallel line
        xo = x0 + offset * px
        yo = y0 + offset * py

        # Heading in radians
        heading_rad = np.arctan2(uy, ux)

        _draw_plane_glyph(ax, xo, yo, heading_rad, scale=scale, color="blue", alpha=0.65)


def plot_results(ucav, radars, target, status, title="Scenario"):
    fig, ax = plt.subplots(figsize=(12, 10))

    # Zones
    for i, radar in enumerate(radars):
        lbl_s = "Search Vol (~25km)" if i == 0 else "_"
        lbl_n = f"NEZ ({config.RADAR_KEEP_OUT_RADIUS/1000:.0f}km)" if i == 0 else "_"
        lbl_l = "High Danger (Lock)" if i == 0 else "_"

        det_circle = patches.Circle(radar.pos, radius=25000, color="yellow", alpha=0.10, label=lbl_s)
        nez_circle = patches.Circle(radar.pos, radius=config.RADAR_KEEP_OUT_RADIUS, color="orange", alpha=0.20, label=lbl_n)
        lock_circle = patches.Circle(radar.pos, radius=8000, color="red", alpha=0.15, label=lbl_l)

        ax.add_patch(det_circle)
        ax.add_patch(nez_circle)
        ax.add_patch(lock_circle)

        ax.plot(radar.pos[0], radar.pos[1], "rx", markersize=12, markeredgewidth=3)
        ax.text(radar.pos[0], radar.pos[1] + 1000, f"Radar {chr(65+i)}", color="red", fontweight="bold")

    # Target
    ax.plot(target.pos[0], target.pos[1], "g*", markersize=18, label="Objective")

    # UCAV path
    history = np.array(ucav.history, dtype=object)
    if len(history) > 0:
        xs = history[:, 0].astype(float)
        ys = history[:, 1].astype(float)
        states = history[:, 2]
        # Add proper plane silhouettes parallel to the route
        plot_parallel_plane_markers(ax, xs, ys, step=30, offset=600, scale=110)

        colors = {
            "SAFE": "green",
            "DETECTED": "orange",
            "TRACKED": "darkorange",
            "LOCKED": "red",
            "ENGAGED": "purple",
            "HIT": "black",
        }

        for i in range(len(xs) - 1):
            ax.plot(xs[i:i+2], ys[i:i+2], color=colors.get(states[i], "blue"), linewidth=2)
    # ------------------------------------------------------------
    # Missile launch arrow (from UCAV end position to target)
    # ------------------------------------------------------------
    if status == "SUCCESS":
        end_pos = ucav.pos  # final UCAV position
        ax.annotate(
            "",  # no text
            xy=(target.pos[0], target.pos[1]),
            xytext=(end_pos[0], end_pos[1]),
            arrowprops=dict(
                arrowstyle="->",
                lw=3,
                color="green",
                alpha=0.9
            )
        )

    # optional label near arrow
    mid_x = 0.5 * (end_pos[0] + target.pos[0])
    mid_y = 0.5 * (end_pos[1] + target.pos[1])
    ax.text(mid_x, mid_y + 800, "MISSILE LAUNCHED", color="green", fontweight="bold")

    arrowprops=dict(
        arrowstyle="Simple,head_length=12,head_width=10,tail_width=1.5",
        color="green",
        alpha=0.9
    )

    ax.set_title(f"{title}  |  Result: {status}")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_aspect("equal")

    # Legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color="r", marker="x", linestyle="None", markersize=10)]
    handles, labels = ax.get_legend_handles_labels()
    handles.append(custom_lines[0])
    labels.append("Threat Emitters")
    ax.legend(handles, labels, loc="upper right")

    plt.show()


# ============================================================
# 4) CLI Entry
# ============================================================
def main():
    scenarios = build_scenarios()

    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="iron_gate_2_radars",
                        help=f"Scenario name. Options: {list(scenarios.keys())}")
    parser.add_argument("--all", action="store_true", help="Run all scenarios sequentially")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for any stochastic behavior")
    parser.add_argument("--no_plot", action="store_true", help="Disable plotting")
    args = parser.parse_args()

    if args.all:
        results = {}
        for name, sc in scenarios.items():
            status = run_scenario(
                name, sc,
                seed=args.seed,
                verbose=True,
                do_plot=(not args.no_plot)
            )
            results[name] = status

        print("\n" + "=" * 85)
        print("SUMMARY RESULTS")
        print("=" * 85)
        for k, v in results.items():
            print(f"{k:<28} -> {v}")

    else:
        if args.scenario not in scenarios:
            raise ValueError(f"Unknown scenario '{args.scenario}'. Use one of: {list(scenarios.keys())}")

        run_scenario(
            args.scenario,
            scenarios[args.scenario],
            seed=args.seed,
            verbose=True,
            do_plot=(not args.no_plot)
        )


if __name__ == "__main__":
    main()

# python3 manual_main_tests.py --scenario baseline_no_threat
# python3 manual_main_tests.py --all

