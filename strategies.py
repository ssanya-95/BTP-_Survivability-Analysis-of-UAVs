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
        """
        primary_threat = None
        highest_threat_val = -1
        min_dist = float('inf')
        
        # Priority Map
        threat_levels = {"HIT": 5, "LOCKED": 4, "TRACKED": 3, "DETECTED": 2, "SAFE": 1}

        # 1. Identify the Primary Threat (The one shouting the loudest)
        for radar in radars:
            state = radar.get_perceived_state(ucav)
            val = threat_levels.get(state, 0)
            dist = np.linalg.norm(radar.pos - ucav.pos)
            
            if val > highest_threat_val:
                highest_threat_val = val
                primary_threat = radar
                min_dist = dist
            elif val == highest_threat_val:
                # Tie-breaker: Closer threat is more dangerous
                if dist < min_dist:
                    min_dist = dist
                    primary_threat = radar
        
        if primary_threat is None:
            return self._fly_to_target(ucav, target)

        # 2. Select Strategy
        # We use the state OF THE PRIMARY THREAT
        state = primary_threat.get_perceived_state(ucav)
        
        if state == "SAFE":
            # STRATEGY: Global Awareness (Avoid ALL radars)
            return self._optimize_safe(ucav, radars, target)
            
        elif state == "DETECTED":
            # STRATEGY: Global Stealth (Find a heading that minimizes TOTAL radar return)
            # UPGRADE: Now passing 'radars' list instead of just 'primary_threat'
            return self._optimize_detected(ucav, radars, target)
            
        elif state == "TRACKED":
            # STRATEGY: Focused Defense (Jam/Burn-through check on Primary)
            return self._optimize_tracked(ucav, primary_threat, target)
            
        elif state == "LOCKED":
            # STRATEGY: Immediate Survival (Crank/Drag against the Locker)
            return self._optimize_locked(ucav, primary_threat, target)
            
        elif state == "ENGAGED":
            return self._optimize_engaged(ucav, primary_threat, target)
            
        elif state == "TERMINAL":
            return self._optimize_terminal(ucav, primary_threat)
            
        else:
            return {"heading": np.rad2deg(ucav.heading), "jam": "OFF"}

    def _optimize_safe(self, ucav, radars, target):
        """
        Calculates repulsive potential from ALL radars to create a global safety map.
        TUNED for 'The Iron Gate' scenario.

        Fixes:
        1) Prevents TIMEOUT when UCAV starts with a "bad" heading (e.g. 170 deg away).
        -> Adds candidates around BOTH current heading and target bearing.
        2) Prevents the UCAV from drifting away forever.
        -> Adds penalty if predicted action increases distance to target.
        """
        best_u = None
        min_total_cost = float('inf')

        # ------------------------------------------------------------------
        # 1) Search Space
        # ------------------------------------------------------------------
        # Candidates around current heading (small corrections)
        candidate_turns = [-60, -45, -30, -15, 0, 15, 30, 45, 60]

        # Candidates around target bearing (helps recovery from wrong start heading)
        vec_to_target_now = target.pos - ucav.pos
        target_heading_deg = np.rad2deg(np.arctan2(vec_to_target_now[1], vec_to_target_now[0]))
        target_turns = [-45, -30, -15, 0, 15, 30, 45]

        current_heading_deg = np.rad2deg(ucav.heading)

        # Combine both sets (avoid duplicates)
        candidate_headings = []
        for d in candidate_turns:
            candidate_headings.append(current_heading_deg + d)
        for d in target_turns:
            candidate_headings.append(target_heading_deg + d)

        # Deduplicate while keeping order
        seen = set()
        candidate_headings = [
            h for h in candidate_headings
            if not (round(h, 3) in seen or seen.add(round(h, 3)))
        ]

        look_ahead = config.SAFE_LOOKAHEAD_TIME
        keep_out = config.RADAR_KEEP_OUT_RADIUS

        # Weights from config
        w_mission = config.WEIGHT_MISSION
        w_risk = config.WEIGHT_RISK

        # Current dist to target (used to discourage flying away)
        dist_to_target_now = np.linalg.norm(target.pos - ucav.pos)

        # ------------------------------------------------------------------
        # 2) Evaluate each candidate
        # ------------------------------------------------------------------
        for test_heading in candidate_headings:
            rad_heading = np.deg2rad(test_heading)

            # A) Predict future position
            dx = config.UCAV_CRUISE_SPEED * np.cos(rad_heading) * look_ahead
            dy = config.UCAV_CRUISE_SPEED * np.sin(rad_heading) * look_ahead
            pred_pos = ucav.pos + np.array([dx, dy])

            # B) Mission Cost (remaining distance to target, in km)
            dist_to_target = np.linalg.norm(target.pos - pred_pos)
            mission_cost = dist_to_target / 1000.0

            # Guardrail: penalize "moving away" in SAFE mode
            if dist_to_target > dist_to_target_now:
                mission_cost += 25.0

            # C) Total geometric risk (sum over all radars)
            total_risk_cost = 0.0

            for radar in radars:
                dist_to_radar = np.linalg.norm(radar.pos - pred_pos)

                # 1) Hard barrier: Keep-out zone
                if dist_to_radar < keep_out:
                    penetration = (keep_out - dist_to_radar) / 1000.0  # km inside
                    barrier_cost = config.RADAR_KEEP_OUT_PENALTY * (1.0 + penetration**2)
                    total_risk_cost += barrier_cost
                else:
                    # 2) Soft repulsion outside keep-out
                    buffer_dist = (dist_to_radar - keep_out) / 1000.0  # km outside
                    soft_cost = 100.0 / (buffer_dist + 0.1)
                    total_risk_cost += soft_cost

            # D) Total cost
            J = (w_mission * mission_cost) + (w_risk * total_risk_cost)

            if J < min_total_cost:
                min_total_cost = J
                best_u = {"heading": test_heading, "jam": "OFF"}

        return best_u
    
    # =========================================================
    # STATE 2: DETECTED (Global Notch Optimization)
    # Goal: Find a heading that balances stealth against ALL radars.
    #       (e.g., Don't notch Radar A if it shows your broadside to Radar B)
    # =========================================================
    def _optimize_detected(self, ucav, radars, target):
        
        # Search Space: Angles around the UCAV
        # We scan a full 360 to find the best "Global Stealth" angle
        # (Coarse search first for speed)
        search_angles = np.arange(0, 360, 10) 
        
        best_heading = ucav.heading
        min_total_score = float('inf')
        
        for deg in search_angles:
            rad_heading = np.deg2rad(deg)
            
            # Calculate Total Detectability Score for this heading
            # Score = Sum of (RCS * DopplerFactor * ProximityWeight) for all radars
            total_score = 0.0
            
            for radar in radars:
                # 1. Physics Inputs
                vec_to_radar = radar.pos - ucav.pos
                dist = np.linalg.norm(vec_to_radar)
                angle_radar = np.arctan2(vec_to_radar[1], vec_to_radar[0])
                
                # 2. Predicted RCS (Aspect Angle)
                aspect = angle_radar - rad_heading
                pred_rcs = physics.get_rcs(aspect)
                
                # 3. Predicted Notch Factor (Doppler)
                pred_notch = physics.get_doppler_notch_factor(
                    config.UCAV_CRUISE_SPEED, 
                    rad_heading, 
                    radar.pos, 
                    ucav.pos
                )
                
                # 4. Proximity Weight
                # Threats closer are MORE important to hide from.
                # Weight decays as 1/distance^2
                # We normalize by 10km to keep numbers readable
                weight = (10000.0 / (dist + 1.0)) ** 2
                
                # 5. Component Score
                # If Notch is 0.0 (Invisible), score is 0.
                # If Notch is 1.0 (Visible), score is proportional to RCS * Distance.
                total_score += (pred_rcs * pred_notch * weight)
            
            # Tie-Breaker: Mission alignment
            # Add a tiny penalty for deviation from target to prevent "Spinning in circles"
            # if all angles are equally safe.
            vec_to_target = target.pos - ucav.pos
            angle_target = np.arctan2(vec_to_target[1], vec_to_target[0])
            angle_diff = abs(np.arctan2(np.sin(rad_heading - angle_target), np.cos(rad_heading - angle_target)))
            mission_penalty = angle_diff * 0.1 # Small weight
            
            final_score = total_score + mission_penalty
            
            if final_score < min_total_score:
                min_total_score = final_score
                best_heading = deg

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
    # Goal: RGPO + "Crank" (if safe) OR "Drag" (if critical)
    # =========================================================
    def _optimize_locked(self, ucav, radar, target):
        # 1. Analyze Geometry
        vec_to_radar = radar.pos - ucav.pos
        dist_to_radar = np.linalg.norm(vec_to_radar)
        angle_radar = np.arctan2(vec_to_radar[1], vec_to_radar[0])
        deg_radar = np.rad2deg(angle_radar)

        v_hat = np.array([np.cos(ucav.heading), np.sin(ucav.heading)])
        if np.dot(vec_to_radar, v_hat) < 0:
            return self._fly_to_target(ucav, target, jam_mode="RGPO")

        
        # 2. PANIC CHECK: Are we inside the Burn-Through / lethal zone?
        # If we are closer than 8km, a "Crank" (50 deg turn) keeps us in the danger zone too long.
        # We must "Drag" (Turn 180 away) to break the lock distance immediately.
        if dist_to_radar < 9000.0: # 9km Safety Buffer
            # DRAG MANEUVER (Turn Tail)
            # This minimizes closure rate (makes it negative)
            best_heading = deg_radar + 180
            return {"heading": best_heading, "jam": "RGPO"}
            
        # 3. STANDARD LOGIC: The "Crank" Maneuver
        # If we have space, we turn 50 degrees to slow closure but keep jamming active.
        
        # Decide direction: Which way gets us closer to target?
        vec_to_target = target.pos - ucav.pos
        deg_target = np.rad2deg(np.arctan2(vec_to_target[1], vec_to_target[0]))
        
        # Check angle difference to target
        diff = (deg_target - deg_radar + 180) % 360 - 180
        
        # If target is to the left of radar, Crank Left (Radar + 50). 
        # Otherwise Crank Right (Radar - 50).
        offset = 50.0 if diff > 0 else -50.0
        
        crank_heading = (deg_radar + offset) % 360.0
        
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
