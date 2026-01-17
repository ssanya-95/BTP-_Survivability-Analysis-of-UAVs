import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import config
from entities import UCAV, ThreatRadar, Target
from strategies import GradientOptimizer

def run_simulation():
    # ==========================================
    # 1. SETUP THE MISSION (STRESS TEST)
    # ==========================================
    print("Initializing High-Stress Mission...")
    
    # 1. UCAV: Start at Origin, flying East (0 deg)
    # Flying East makes mental visualization of coordinates easier.
    my_ucav = UCAV(x=0, y=0, heading_deg=0)
    
    # 2. TARGET: 40km away, directly East
    # This creates a long flight path that passes the threat.
    mission_target = Target(x=40000, y=0)
    
    # 3. RADAR: Placed to force a reaction ("The Gauntlet")
    # Position: (20000, 2000)
    # Logic: It is halfway to the target (20km) but only offset by 2km North.
    # Result: The UCAV is flying straight into the 12km NEZ (No Escape Zone).
    # It MUST curve South (Right) or notch significantly to survive.
    enemy_radar = ThreatRadar(x=20000, y=2000)
    threats = [enemy_radar]
    
    # Initialize the Brain
    brain = GradientOptimizer()
    
    # Data Logging for Plots
    time_log = []
    
    # State tracking for console output (avoid spamming)
    last_state = "START"
    
    # ==========================================
    # 2. SIMULATION LOOP
    # ==========================================
    t = 0.0
    mission_status = "TIMEOUT" # Default result
    
    print(f"Starting Sim: Max Time {config.MAX_TIME}s, dt={config.DT}s")
    print("-" * 60)
    print(f"{'TIME':<8} | {'STATE':<10} | {'DIST (km)':<10} | {'HEADING':<8} | {'ACTION'}")
    print("-" * 60)
    
    while t < config.MAX_TIME:
        # A. REFEREE UPDATE (Adversary Logic)
        # The Radar checks physics to see if it detects the UCAV
        current_state = enemy_radar.get_perceived_state(my_ucav)
        my_ucav.state = current_state
        
        # B. OPTIMIZER UPDATE (The Brain)
        # UCAV decides on control input based on state
        control = brain.get_optimal_control(my_ucav, threats, mission_target)
        
        # C. LOGGING (Console Intelligence)
        # Only print if state changes OR every 10 seconds (heartbeat)
        if current_state != last_state or (int(t) % 10 == 0 and abs(t - int(t)) < config.DT/2):
            dist = np.linalg.norm(enemy_radar.pos - my_ucav.pos)
            
            # Decipher Action for display
            jam_status = "JAM:" + control.get('jam', 'OFF')
            chaff_status = "+CHAFF" if control.get('chaff', False) else ""
            action_str = f"{jam_status} {chaff_status}"
            
            print(f"{t:5.1f}s   | {current_state:<10} | {dist/1000:5.1f} km   | {np.degrees(my_ucav.heading):5.1f}°  | {action_str}")
            
            last_state = current_state

        # D. PHYSICS UPDATE (The Body)
        # Apply the control to move the aircraft
        my_ucav.move(config.DT, control)
        
        # E. MISSION CHECK
        # 1. Check if Target is Destroyed (using updated Weapon Range logic if applied)
        if mission_target.check_destroyed(my_ucav.pos):
            print("-" * 60)
            print(f"[{t:.1f}s] TARGET DESTROYED! Mission Success.")
            mission_status = "SUCCESS"
            break
            
        # 2. Check if UCAV is Shot Down
        if current_state == "HIT": 
            print("-" * 60)
            print(f"[{t:.1f}s] UCAV SHOT DOWN. Mission Failure.")
            mission_status = "FAILURE"
            break
            
        # F. RECORD HISTORY
        time_log.append(t)
        t += config.DT

    # ==========================================
    # 3. VISUALIZATION
    # ==========================================
    plot_results(my_ucav, enemy_radar, mission_target, mission_status)

def plot_results(ucav, radar, target, status):
    print("\nGenerating Trajectory Analysis...")
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 1. Plot Radar Zones
    det_circle = patches.Circle(radar.pos, radius=25000, color='yellow', alpha=0.1, label='Search Vol (~25km)')
    nez_circle = patches.Circle(radar.pos, radius=config.RADAR_KEEP_OUT_RADIUS, color='orange', alpha=0.2, label=f'NEZ ({config.RADAR_KEEP_OUT_RADIUS/1000:.0f}km)')
    lock_circle = patches.Circle(radar.pos, radius=8000, color='red', alpha=0.15, label='High Danger (Lock)')
    
    ax.add_patch(det_circle)
    ax.add_patch(nez_circle)
    ax.add_patch(lock_circle)
    
    # Plot Radar Location
    ax.plot(radar.pos[0], radar.pos[1], 'rx', markersize=12, markeredgewidth=3, label='Threat Radar')
    
    # 2. Plot the Target
    ax.plot(target.pos[0], target.pos[1], 'g*', markersize=18, label='Target Objective')
    
    # 3. Plot the UCAV Path
    history = np.array(ucav.history)
    if len(history) > 0:
        # Convert strings to floats here
        xs = history[:, 0].astype(float)
        ys = history[:, 1].astype(float)
        states = history[:, 2]
        
        # Plot segments
        colors = {
            'SAFE': 'green', 
            'DETECTED': 'orange', 
            'TRACKED': 'darkorange', 
            'LOCKED': 'red', 
            'ENGAGED': 'purple', 
            'HIT': 'black'
        }
        
        for i in range(len(xs) - 1):
            c = colors.get(states[i], 'blue')
            ax.plot(xs[i:i+2], ys[i:i+2], color=c, linewidth=2)
            
        # --- THE FIX: Use xs/ys (floats) instead of history (strings) ---
        if status == "SUCCESS":
            final_x = xs[-1]  # <--- FIXED
            final_y = ys[-1]  # <--- FIXED
            
            # Draw weapon release vector
            ax.annotate("", xy=(target.pos[0], target.pos[1]), xytext=(final_x, final_y),
                        arrowprops=dict(arrowstyle="->", linestyle="--", color='green', lw=2))
            
            # Label
            mid_x = (final_x + target.pos[0]) / 2.0 + 1000
            mid_y = ((final_y + target.pos[1]) / 2.0) + 2000
            ax.text(mid_x, mid_y, "Weapon Release\n(8km Stand-off)", color='green', 
                    fontsize=9, ha='center', fontweight='bold')

    # 4. Final Formatting
    ax.set_title(f"UCAV Survivability Analysis - Result: {status}\n(Green=Safe, Orange=Detected, Red=Locked)")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    
    plt.show()

if __name__ == "__main__":
    run_simulation()