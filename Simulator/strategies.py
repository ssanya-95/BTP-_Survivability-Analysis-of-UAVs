import numpy as np
import config
import physics

class GradientOptimizer:
    def __init__(self):
        # We assume the "Mission Weight" and "Risk Weight" from config
        pass

    def get_optimal_control(self, ucav, radars, target):
        """
        Main decision function.
        1. Identifies the primary threat (closest/loudest radar).
        2. Routes to the specific optimization logic based on current State.
        """
        # 1. Identify Primary Threat
        # (For this vertical slice, we just find the closest radar)
        primary_threat = None
        min_dist = float('inf')
        
        for radar in radars:
            d = np.linalg.norm(radar.pos - ucav.pos)
            if d < min_dist:
                min_dist = d
                primary_threat = radar
        
        # If no threats exist (unlikely in this sim), fly to target
        if primary_threat is None:
            return self._fly_to_target(ucav, target)

        # 2. State-Based Switching
        state = ucav.state
        
        if state == "SAFE":
            return self._optimize_safe(ucav, primary_threat, target)
        elif state == "DETECTED":
            return self._optimize_detected(ucav, primary_threat, target)
        elif state == "TRACKED":
            return self._optimize_tracked(ucav, primary_threat, target)
        elif state == "LOCKED":
            return self._optimize_locked(ucav, primary_threat, target)
        elif state == "ENGAGED":
            return self._optimize_engaged(ucav, primary_threat, target)
        elif state == "TERMINAL":
            return self._optimize_terminal(ucav, primary_threat)
        else:
            # Fallback or HIT state
            return {"heading": np.rad2deg(ucav.heading), "jam": "OFF"}

    # =========================================================
    # STATE 1: SAFE (Normalized Potential Field)
    # Goal: Feel the threat from far away and curve early.
    # =========================================================
    def _optimize_safe(self, ucav, radar, target):
        best_u = None
        min_total_cost = float('inf')
        
        # 1. Search Space: Wider angles to enable early curvature
        candidate_turns = [-45, -30, -15, 0, 15, 30, 45]
        
        current_heading_deg = np.rad2deg(ucav.heading)
        look_ahead = config.SAFE_LOOKAHEAD_TIME
        keep_out = config.RADAR_KEEP_OUT_RADIUS
        
        # --- NEW LOGIC: DYNAMIC AGGRESSION --- for the cases when the target is much nearer to the radar
        # Calculate distance from UCAV to Target to scale fear.
        current_dist_to_target = np.linalg.norm(target.pos - ucav.pos)
        
        # Define a "Commitment Radius" (e.g., 15 km). 
        # Inside this radius, the mission priority overrides the safety barrier.
        commitment_radius = 15000.0 
        
        # Scale from 0.0 (At Target) to 1.0 (Far away)
        aggression_factor = np.clip(current_dist_to_target / commitment_radius, 0.0, 1.0)
        
        # Scale the penalties dynamically
        # If we are close to target, the wall becomes a soft curtain.
        keep_out_penalty = config.RADAR_KEEP_OUT_PENALTY * aggression_factor
        approach_penalty = config.RADAR_APPROACH_PENALTY * aggression_factor
        
        # Weights (Constant)
        w_mission = config.WEIGHT_MISSION
        w_risk = config.WEIGHT_RISK
        
        for d_deg in candidate_turns:
            test_heading = current_heading_deg + d_deg
            
            # A. PREDICT FUTURE
            rad_heading = np.deg2rad(test_heading)
            
            dx = config.UCAV_CRUISE_SPEED * np.cos(rad_heading) * look_ahead
            dy = config.UCAV_CRUISE_SPEED * np.sin(rad_heading) * look_ahead
            pred_pos = ucav.pos + np.array([dx, dy])
            
            # B. CALCULATE NORMALIZED MISSION COST
            dist_to_target = np.linalg.norm(target.pos - pred_pos)
            mission_cost = dist_to_target / 1000.0 
            
            # C. GEOMETRIC KEEP-OUT RISK (Scaled by Aggression)
            seg = pred_pos - ucav.pos
            seg_len_sq = np.dot(seg, seg)
            
            # Point-to-Segment Distance Logic
            if seg_len_sq < 1e-6:
                min_path_dist = np.linalg.norm(radar.pos - ucav.pos)
            else:
                t = np.dot(radar.pos - ucav.pos, seg) / seg_len_sq
                t = np.clip(t, 0.0, 1.0)
                closest = ucav.pos + t * seg
                min_path_dist = np.linalg.norm(radar.pos - closest)
            
            # 1. The Wall Penalty (Softened by aggression_factor)
            if min_path_dist < keep_out:
                # Quadratic barrier inside the radius
                raw_barrier = ((keep_out - min_path_dist) / keep_out) ** 2
                keep_out_cost = keep_out_penalty * raw_barrier
            else:
                # Soft inverse square repulsion outside (Pre-sensing)
                soft_risk = (keep_out / max(min_path_dist, keep_out)) ** 2
                keep_out_cost = 0.0 # We separate them usually, or add soft_risk here
                # In your original code you summed them, let's keep consistency:
                keep_out_cost += soft_risk * aggression_factor # Scale this too!

            # 2. The Approach Penalty (Softened by aggression_factor)
            to_radar = radar.pos - ucav.pos
            to_radar_norm = np.linalg.norm(to_radar)
            if to_radar_norm > 1e-6:
                heading_vec = np.array([np.cos(rad_heading), np.sin(rad_heading)])
                approach = np.dot(heading_vec, to_radar / to_radar_norm)
                # Only penalize if approaching (>0)
                approach_cost = approach_penalty * max(0.0, approach) ** 2
            else:
                approach_cost = 0.0
            
            risk_cost = keep_out_cost + approach_cost
            
            # D. TOTAL COST
            J = (w_mission * mission_cost) + (w_risk * risk_cost)
            
            if J < min_total_cost:
                min_total_cost = J
                best_u = {"heading": test_heading, "jam": "OFF"}
                
        return best_u
    
    # =========================================================
    # STATE 2: DETECTED (Balanced Notch)
    # Goal: Minimize Total Signal (RCS * Doppler Visibility)
    #       Avoids the "Broadside Trap" of a perfect 90-degree turn.
    # =========================================================
    def _optimize_detected(self, ucav, radar, target):
        # 1. Vector to Radar
        vec_to_radar = radar.pos - ucav.pos
        angle_radar = np.arctan2(vec_to_radar[1], vec_to_radar[0])
        dist_to_radar = np.linalg.norm(vec_to_radar)
        
        # 2. Search Space: Inspect angles AROUND the notch (Perpendicular)
        # Instead of just +/- 90, we look at 70, 80, 85, 90, 95, 100, 110
        # relative to the radar beam.
        notch_offsets = [-110, -100, -90, -80, -70, 70, 80, 90, 100, 110]
        
        best_heading = ucav.heading
        min_detectability_score = float('inf')
        
        current_rad_heading = ucav.heading
        
        for offset in notch_offsets:
            # Candidate Heading
            test_heading_deg = np.rad2deg(angle_radar) + offset
            test_heading_rad = np.deg2rad(test_heading_deg)
            
            # A. Calculate Resulting Aspect Angle (For RCS)
            # If I fly this heading, what part of me does the radar see?
            # Aspect = (Radar Angle) - (My Heading)
            test_aspect = angle_radar - test_heading_rad
            pred_rcs = physics.get_rcs(test_aspect)
            
            # B. Calculate Resulting Doppler Visibility
            # If I fly this heading, how fast do I move relative to radar?
            pred_notch = physics.get_doppler_notch_factor(
                config.UCAV_CRUISE_SPEED, 
                test_heading_rad, 
                radar.pos, 
                ucav.pos
            )
            
            # C. The "Balanced" Score
            # We want to minimize the signal that actually passes the filter.
            # Signal ~ RCS * VisibilityFactor
            # (If Notch is 0.0 (Invisible), Score is 0 regardless of RCS)
            # (If Notch is 1.0 (Visible), Score depends purely on RCS)
            score = pred_rcs * pred_notch
            
            # Tie-Breaker: If scores are similar, pick the one closer to Target
            # (Small penalty for flying away from mission)
            if score < min_detectability_score:
                min_detectability_score = score
                best_heading = test_heading_deg

        # Return Best Balanced Heading
        return {"heading": best_heading, "jam": "OFF"}

    # =========================================================
    # STATE 3: TRACKED (Stand-Off Jamming)
    # Goal: Jam logic + Maintain Range
    # =========================================================
    def _optimize_tracked(self, ucav, radar, target):
        dist = np.linalg.norm(radar.pos - ucav.pos)
        
        # --- FIX: Calculate Dynamic Burn-Through Range ---
        # 1. Determine current aspect angle
        vec_to_radar = radar.pos - ucav.pos
        angle_radar = np.arctan2(vec_to_radar[1], vec_to_radar[0])
        aspect_angle = angle_radar - ucav.heading
        
        # 2. Get current RCS
        current_rcs = physics.get_rcs(aspect_angle)
        
        # 3. Calculate BT Range with specific RCS
        bt_range = physics.calculate_burn_through_range(config.UCAV_MAX_JAMMER_POWER, current_rcs)
        
        # LOGIC BRANCH
        if dist < bt_range:
            # PANIC: We are too close. Jamming won't work. 
            # Strategy: Turn 180 away from radar (Cold Aspect) and Run.
            # Note: This points our TAIL at the radar. 
            # In physics.py, Tail RCS is ~0.5m^2 (Medium). This is better than Broadside (5.0m^2).
            escape_heading = np.rad2deg(angle_radar) + 180
            return {"heading": escape_heading, "jam": "NOISE"} 
        else:
            # SAFE-ISH: We can jam effectively.
            # Strategy: Fly to target, but keep Jammer ON.
            return self._fly_to_target(ucav, target, jam_mode="NOISE")

    # =========================================================
    # STATE 4: LOCKED (Deception & Drag)
    # Goal: RGPO + "Crank" (Slow down closure rate)
    # =========================================================
    def _optimize_locked(self, ucav, radar, target):
        # 1. Calculate Geometry
        vec_to_radar = radar.pos - ucav.pos
        angle_radar = np.arctan2(vec_to_radar[1], vec_to_radar[0])
        deg_radar = np.rad2deg(angle_radar)
        
        # 2. The "Crank" Maneuver
        # Instead of flying straight (Closure ~250m/s), we fly at 50 degrees offset.
        # Closure becomes 250 * cos(50) = ~160 m/s. 
        # We buy 35% more time for the Jammer to work.
        
        # Decide direction: Which way gets us closer to target?
        vec_to_target = target.pos - ucav.pos
        deg_target = np.rad2deg(np.arctan2(vec_to_target[1], vec_to_target[0]))
        
        # Check angle difference to target
        diff = (deg_target - deg_radar + 180) % 360 - 180
        
        # If target is to the left of radar, Crank Left (Radar + 50). 
        # Otherwise Crank Right (Radar - 50).
        offset = 50.0 if diff > 0 else -50.0
        
        crank_heading = deg_radar + offset
        
        # 3. Execution
        # We use "RGPO" mode. (Ensure entities.py assigns a higher penalty to this!)
        return {"heading": crank_heading, "jam": "RGPO"}

    # =========================================================
    # STATE 5: ENGAGED (The F-Pole / Crank)
    # Goal: Maintain Radar Contact at Gimbal Limit (60 deg) 
    #       while slowing closure rate to bleed missile energy.
    # =========================================================
    def _optimize_engaged(self, ucav, radar, target):
        # 1. Vector to Threat
        vec_to_threat = radar.pos - ucav.pos
        angle_threat = np.arctan2(vec_to_threat[1], vec_to_threat[0])
        deg_threat = np.rad2deg(angle_threat)
        
        # 2. Calculate Crank Candidates (Standard 60 degree offset)
        # This keeps the threat at the edge of our sensor field (if we had one)
        # and reduces radial velocity significantly.
        crank_left = deg_threat + 60
        crank_right = deg_threat - 60
        
        # 3. INTELLIGENT SELECTION (The Fix)
        # Which crank gets us closer to the Target?
        
        # Vector to Target
        vec_to_target = target.pos - ucav.pos
        deg_target = np.rad2deg(np.arctan2(vec_to_target[1], vec_to_target[0]))
        
        # Normalize angles to compare difference
        diff_left = abs((crank_left - deg_target + 180) % 360 - 180)
        diff_right = abs((crank_right - deg_target + 180) % 360 - 180)
        
        best_heading = crank_left if diff_left < diff_right else crank_right
        
        # 4. Return Command
        # 'NOISE' jamming is standard here to deny range info to the missile seeker.
        return {"heading": best_heading, "jam": "NOISE"}
    
    # =========================================================
    # STATE 6: TERMINAL (The "Break")
    # Goal: Max G turn 90 deg to threat (Orthogonal) + Chaff
    #       Forces missile to overshoot (High G Intercept)
    # =========================================================
    def _optimize_terminal(self, ucav, radar):
        # 1. Vector to Threat
        vec_to_threat = radar.pos - ucav.pos
        angle_threat = np.arctan2(vec_to_threat[1], vec_to_threat[0])
        deg_threat = np.rad2deg(angle_threat)
        
        # 2. Calculate Break Candidates (90 degrees off threat)
        # This puts us on the "Beam" - maximum angular error for the missile
        break_left = deg_threat + 90
        break_right = deg_threat - 90
        
        # 3. Selection Logic: Minimize Turn Angle
        # We are in a panic. We can't turn 270 degrees. We take the shortest turn 
        # to get to a beam aspect.
        
        current_heading = np.rad2deg(ucav.heading)
        
        # Normalize diffs
        diff_left = abs((break_left - current_heading + 180) % 360 - 180)
        diff_right = abs((break_right - current_heading + 180) % 360 - 180)
        
        best_heading = break_left if diff_left < diff_right else break_right
        
        # 4. Return Command with Chaff
        # NOTE: Ensure entities.py logic reduces hit probability when 'chaff' is True!
        return {"heading": best_heading, "jam": "NOISE", "chaff": True}

    # =========================================================
    # HELPER FUNCTIONS
    # =========================================================
    def _fly_to_target(self, ucav, target, jam_mode="OFF"):
        """
        Simple guidance logic to fly straight to the target.
        """
        vec_to_target = target.pos - ucav.pos
        angle_target = np.arctan2(vec_to_target[1], vec_to_target[0])
        return {"heading": np.rad2deg(angle_target), "jam": jam_mode}
