import numpy as np
import config
import physics

class Target:
    def __init__(self, x, y):
        self.pos = np.array([float(x), float(y)])
        self.is_destroyed = False
        
    def check_destroyed(self, ucav_pos):
        dist = np.linalg.norm(self.pos - ucav_pos)
        
        # LOGIC CHANGE:
        # Instead of 500m, we check if we are inside the Weapon Release Basket.
        # This allows the UCAV to "Shoot and Scoot" from a safe distance.
        if dist < config.WEAPON_MAX_RANGE:
            self.is_destroyed = True
            return True
        return False

class UCAV:
    def __init__(self, x, y, heading_deg=0):
        # Kinematics
        self.pos = np.array([float(x), float(y)])
        self.heading = np.deg2rad(heading_deg) # Radians
        self.velocity = config.UCAV_CRUISE_SPEED
        
        # State
        self.state = "SAFE" # SAFE, DETECTED, TRACKED, LOCKED, ENGAGED, HIT
        self.jammer_on = False
        self.history = [] # For plotting later (x, y, state)

    def move(self, dt, control_input):
        """
        Updates the physics of the aircraft based on control inputs.
        """
        # 1. Parse Controls
        # Default to current heading (converted to deg for consistency) if no input
        current_heading_deg = np.rad2deg(self.heading)
        command_heading = np.deg2rad(control_input.get('heading', current_heading_deg))
        
        jam_mode = control_input.get('jam', 'OFF')
        self.jam_mode = jam_mode
        self.jammer_on = (jam_mode != 'OFF')

        self.chaff_active = control_input.get('chaff', False)

        # 2. Limit Turn Rate (Physics Constraint)
        heading_diff = command_heading - self.heading
        
        # Normalize diff to -pi to +pi (The "Shortest Turn" logic)
        heading_diff = (heading_diff + np.pi) % (2 * np.pi) - np.pi
        
        max_turn = config.MAX_TURN_RATE * dt
        
        # Store old heading for integration
        heading_start = self.heading
        
        if abs(heading_diff) < max_turn:
            self.heading = command_heading
        else:
            self.heading += max_turn * np.sign(heading_diff)
            
        # --- IMPROVEMENT 1: State Normalization ---
        # Keep heading strictly between -pi and +pi
        self.heading = (self.heading + np.pi) % (2 * np.pi) - np.pi

        # --- IMPROVEMENT 2: Average Heading Integration ---
        # Flies along the "average" angle of the turn, not the end angle.
        # Reduces position error during hard turns.
        avg_heading = (heading_start + self.heading) / 2.0
        
        # Handle the edge case where avg wraps around pi/-pi (e.g. 179 to -179)
        # If the turn crossed the "cut", strictly averaging gives ~0 (wrong direction).
        # Simple fix: If diff was huge, just use new heading.
        if abs(heading_start - self.heading) > np.pi:
            avg_heading = self.heading

        # 3. Update Position
        dx = self.velocity * np.cos(avg_heading) * dt
        dy = self.velocity * np.sin(avg_heading) * dt
        self.pos += np.array([dx, dy])
        
        # 4. Save History
        self.history.append((self.pos[0], self.pos[1], self.state))

class ThreatRadar:
    def __init__(self, x, y):
        self.pos = np.array([float(x), float(y)])
        self.id = id(self)
        
    def get_perceived_state(self, ucav):
        """
        The 'Referee' function.
        Calculates physics to decide what state the UCAV should be in.
        """
        # 1. Calculate Core Metrics
        # We need to calculate Aspect Angle here to pass it to Burn-Through logic
        # (This math repeats slightly from inside calculate_snr, but it's necessary for fidelity)
        vector_to_radar = self.pos - ucav.pos
        angle_radar_direction = np.arctan2(vector_to_radar[1], vector_to_radar[0])
        aspect_angle = angle_radar_direction - ucav.heading
        
        # Get the Dynamic RCS for this specific moment
        current_rcs = physics.get_rcs(aspect_angle)
        
        # Calculate SNR (The signal strength)
        snr = physics.calculate_snr(self.pos, ucav.pos, ucav.heading)
        dist = np.linalg.norm(vector_to_radar)
        
        # 2. Calculate Modifiers
        notch_factor = physics.get_doppler_notch_factor(ucav.velocity, ucav.heading, self.pos, ucav.pos)
        
        # Jamming Impact (High Fidelity J/S Logic)
        jamming_penalty = 0.0
        if ucav.jammer_on:
            bt_range = physics.calculate_burn_through_range(config.UCAV_MAX_JAMMER_POWER, current_rcs)
            
            if dist > bt_range:
                # Base Noise Penalty
                jamming_penalty = 15.0 
                
                # CHECK FOR RGPO BONUS
                # We need to access the mode from the UCAV object
                # (You might need to store self.jam_mode in the UCAV class move function)
                if hasattr(ucav, 'jam_mode') and ucav.jam_mode == 'RGPO':
                    # RGPO is specifically designed to break LOCKS.
                    # It adds extra confusion to the tracking loop.
                    jamming_penalty += 10.0 # Total 25.0 dB reduction against LOCK
        
        effective_snr = snr - jamming_penalty

        # 3. Determine State Logic Layer
        
        # HIT Condition (Terminal Defense)
        if effective_snr > config.SNR_THRESHOLD_LOCK and dist < 1000:
            # --- FIX 4: Check UCAV Memory for Chaff ---
            # If UCAV dropped chaff, 60% chance to spoof the missile
            if ucav.chaff_active:
                if np.random.rand() > 0.4: # 60% survival chance
                    return "ENGAGED" # Survived the hit!
            
            # If no chaff, or bad luck:
            return "HIT"

        # LOCK Condition (STT - Single Target Track)
        if effective_snr > config.SNR_THRESHOLD_LOCK:
            return "LOCKED"
            
        # TRACK Condition (TWS - Track While Scan)
        if effective_snr > config.SNR_THRESHOLD_TRACK:
            # DOPPLER GATE: If we are notching (factor < 0.2), the radar filters us out.
            # The simulator correctly drops the track.
            if notch_factor > 0.2: 
                return "TRACKED"
            else:
                return "SAFE" 
                
        # DETECT Condition (Search / RWR Tickle)
        if effective_snr > config.SNR_THRESHOLD_DETECT:
            return "DETECTED"
            
        return "SAFE"