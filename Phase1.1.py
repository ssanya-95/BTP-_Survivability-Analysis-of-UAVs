import numpy as np
import matplotlib.pyplot as plt

# --- CONSTANTS ---
DT = 0.5            # Time step (seconds)
RADAR_FREQ = 10e9   # 10 GHz (X-Band Radar)
C = 3e8             # Speed of light
K_BOLTZMANN = 1.38e-23

def get_aspect_dependent_rcs(drone_pos, drone_heading, radar_pos):
    """
    Returns RCS (m^2) based on the Aspect Angle.
    Logic: Stealthy from front, vulnerable from side.
    """
    # 1. Vector from Drone to Radar
    dx = radar_pos[0] - drone_pos[0]
    dy = radar_pos[1] - drone_pos[1]
    
    # 2. Angle of Radar relative to Global North (0 degrees)
    radar_angle_global = np.degrees(np.arctan2(dx, dy)) # Note: using (dx, dy) for North=0 convention if y is North
    
    # Simple atan2(y, x) usually gives angle from East. Let's stick to standard math:
    # We'll treat East as 0 degrees for simplicity in this script.
    radar_angle_standard = np.degrees(np.arctan2(dy, dx))
    
    # 3. Aspect Angle = Difference between heading and radar direction
    # We take absolute difference and wrap to [0, 180] (symmetry)
    raw_angle = abs(radar_angle_standard - drone_heading) % 360
    aspect_angle = 360 - raw_angle if raw_angle > 180 else raw_angle
    
    # 4. Lookup Table [cite: 3225]
    if aspect_angle < 30:
        return 0.01   # Frontal: Very Stealthy
    elif aspect_angle < 100:
        return 5.0    # Side: "Barn Door" (Very Loud)
    else:
        return 0.1    # Rear: Moderate

class Radar:
    def __init__(self, x, y, peak_power=50e3):
        self.pos = np.array([x, y])
        self.power = peak_power
        self.threshold = 15.0 # SNR Threshold for detection (dB)

    def get_transition_matrix(self, drone_pos, rcs):
        """
        Calculates the 5x5 Transition Intensity Matrix based on physics.
        States: 0:Search, 1:Detect, 2:Track, 3:Engage, 4:Hit
        """
        # Calculate Distance
        dist = np.linalg.norm(self.pos - drone_pos)
        if dist == 0: dist = 0.1 # Avoid divide by zero
        
        # --- PHYSICS: Radar Range Equation [cite: 3211] ---
        # SNR = (Pt * G^2 * lambda^2 * RCS) / ((4pi)^3 * R^4 * Noise)
        # Simplified proportional model for simulation speed:
        # We'll just say Signal Strength scales with Power * RCS / Dist^4
        signal_strength = (self.power * rcs) / (dist**4)
        detection_factor = signal_strength * 1e12 # Scaling constant for simulation
        
        # --- LOGIC: Define Intensities (Lambdas) ---
        # Default: No danger
        lambda_12 = 0.0 # Search -> Detect
        lambda_23 = 0.0 # Detect -> Track
        lambda_34 = 0.0 # Track -> Engage
        lambda_45 = 0.0 # Engage -> Hit
        
        if detection_factor > 2.0: # Weak Signal
            lambda_12 = 0.5
        
        if detection_factor > 10.0: # Strong Signal
            lambda_12 = 2.0 # Fast detection
            lambda_23 = 0.8 # Tracking starts
            
        if detection_factor > 50.0: # "Burn Through" Range
            lambda_23 = 2.0
            lambda_34 = 0.5 # Missile launch probable
            
        if dist < 3000: # Inside Kill Zone (3km)
            lambda_34 = 3.0
            lambda_45 = 0.2 # Missile Kill Probability
            
        # --- MATRIX CONSTRUCTION [cite: 2610] ---
        # Rows = From State, Cols = To State
        # Note: In the differential equation dot(p, Lambda), Lambda is often defined 
        # such that columns sum to 0. Let's use the explicit "From -> To" rate.
        
        # 5x5 Matrix of Zeros
        L = np.zeros((5, 5))
        
        # Fill transitions
        L[0, 1] = lambda_12
        L[1, 2] = lambda_23
        L[2, 3] = lambda_34
        L[3, 4] = lambda_45
        
        # Add Recovery (Escaping back to previous state if signal drops)
        if detection_factor < 2.0:
            L[1, 0] = 1.0 # Detected -> Search (Lost contact)
            L[2, 1] = 1.0 # Track -> Detected (Broken lock)
            
        # Diagonal elements (Flux out) must balance rows for probability conservation
        # p_dot = p * Q (where Q is rate matrix). 
        # We will handle the update step manually in the UCAV class to be clear.
        
        return L

class UCAV:
    def __init__(self, x, y, speed, heading):
        self.pos = np.array([float(x), float(y)])
        self.speed = speed # m/s
        self.heading = heading # Degrees (0 = East)
        
        # 5-State Probability Vector: [Search, Detect, Track, Engage, Hit]
        # Start: 100% in "Searching" (Undetected), 0% everywhere else
        self.probs = np.array([1.0, 0.0, 0.0, 0.0, 0.0]) 
        
        self.history_pos = []
        self.history_probs = []

    def update_physics(self, dt):
        """ Moves the drone based on speed and heading """
        rad = np.radians(self.heading)
        self.pos[0] += self.speed * np.cos(rad) * dt
        self.pos[1] += self.speed * np.sin(rad) * dt
        self.history_pos.append(self.pos.copy())

    def update_survival(self, radar, dt):
        """ Updates the Markov Probabilities  """
        # 1. Get RCS for current angle
        rcs = get_aspect_dependent_rcs(self.pos, self.heading, radar.pos)
        
        # 2. Get Transition Rates from the Enemy
        L = radar.get_transition_matrix(self.pos, rcs)
        
        # 3. Compute Delta Probabilities (Flux)
        # New_State[j] += Old_State[i] * Rate_i_to_j * dt
        # Old_State[i] -= Old_State[i] * Rate_i_to_j * dt
        
        new_probs = self.probs.copy()
        
        for i in range(5): # From State
            for j in range(5): # To State
                if i == j: continue
                rate = L[i, j]
                if rate > 0:
                    flow = self.probs[i] * rate * dt
                    # Clamp flow so we don't drain more than 100% (simulation stability)
                    if flow > self.probs[i]: flow = self.probs[i]
                    
                    new_probs[i] -= flow
                    new_probs[j] += flow
        
        self.probs = new_probs
        self.history_probs.append(self.probs[4]) # Log 'Hit' probability

# --- SIMULATION SETUP ---
# Create Radar at (5000m, 5000m)
enemy = Radar(x=5000, y=5000)

# Create Drone starting at (0, 4000) flying East past the radar
# This means it will fly PAST the radar, showing "Side Aspect" (High RCS)
my_drone = UCAV(x=0, y=4000, speed=200, heading=0)

time_steps = int(60 / DT)
times = np.linspace(0, 60, time_steps)

print("Starting Simulation...")

# --- MAIN LOOP ---
for t in times:
    # 1. Move
    my_drone.update_physics(DT)
    
    # 2. Update Health (Markov Math)
    my_drone.update_survival(enemy, DT)
    
    # Optional: Logic to turn away if "Tracked" probability gets too high
    if my_drone.probs[2] > 0.5: 
        print(f"[{t:.1f}s] THREAT LOCK! Evasive Maneuvers!")
        # my_drone.heading += 90 # Turn North (Show tail aspect)

# --- VISUALIZATION ---
# 1. Plot Trajectory
pos_data = np.array(my_drone.history_pos)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("UCAV Path vs Radar")
plt.plot(pos_data[:,0], pos_data[:,1], 'b-', label='UCAV Path')
plt.plot(enemy.pos[0], enemy.pos[1], 'ro', markersize=10, label='Radar')
plt.xlabel("X (m)"); plt.ylabel("Y (m)")
plt.legend(); plt.grid(True)
plt.axis('equal')

# 2. Plot Hit Probability (Risk)
plt.subplot(1, 2, 2)
plt.title("Probability of Being Hit (State 5)")
plt.plot(my_drone.history_probs, 'r-')
plt.xlabel("Time Steps"); plt.ylabel("Probability (0-1)")
plt.ylim(0, 1.1)
plt.grid(True)

plt.tight_layout()
plt.show()
