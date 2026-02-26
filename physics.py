import numpy as np
import config

def get_rcs(aspect_angle_rad):
    """
    Calculates RCS based on aspect angle using a smooth 'Butterfly' model.
    0 rad = Nose-on (Min RCS)
    pi/2 rad = Broadside (Max RCS)
    pi rad = Tail-on (Medium RCS due to exhaust)
    """
    # 1. Normalize angle to 0 -> pi range (Symmetry)
    angle = np.abs(aspect_angle_rad) 
    
    # 2. Broadside Flash (The Wings) - modeled as sin^4 for a sharp spike at 90 deg
    broadside_factor = np.sin(angle)**4
    rcs_broadside = config.UCAV_MIN_RCS + (config.UCAV_MAX_RCS - config.UCAV_MIN_RCS) * broadside_factor
    
    # 3. Tail Penalty (The Engine Exhaust) - Modeled using Cosine
    # formula: (1 - cos(theta)) / 2  --> Maps 0(nose) to 0, and 180(tail) to 1
    tail_base_factor = (1.0 - np.cos(angle)) / 2.0
    
    # RAISE TO POWER 6: This keeps the penalty near 0 until we get to the rear (~120 deg+)
    tail_shape = tail_base_factor ** 6
    
    # Apply the penalty magnitude (e.g., 0.5 m^2 extra for tail)
    tail_penalty_magnitude = 0.5 
    rcs_tail = tail_penalty_magnitude * tail_shape
    
    # 4. Total RCS
    total_rcs = rcs_broadside + rcs_tail
    
    return total_rcs

def calculate_snr(radar_pos, ucav_pos, ucav_heading):
    """
    Calculates Signal-to-Noise Ratio (dB) using the Radar Range Equation.
    Matches LaTeX Equation: SNR = (Pt * G^2 * lambda^2 * sigma) / ((4pi)^3 * R^4 * k * T * B * L)
    """
    # --- 1. GEOMETRY FIX ---
    # We need the vector pointing TO the radar to see if our nose is aligned with it.
    # Old: R_vec = ucav_pos - radar_pos (Points at UCAV) -> Wrong angles
    # New: Vector FROM UCAV TO Radar
    vector_to_radar = radar_pos - ucav_pos 
    R = np.linalg.norm(vector_to_radar)
    
    if R < 1.0: return 999.0 # Avoid division by zero
    
    # Calculate the angle of this vector (Where is the radar relative to me?)
    angle_radar_direction = np.arctan2(vector_to_radar[1], vector_to_radar[0])
    
    # Aspect Angle = (Radar Direction) - (My Heading)
    # If I am flying West (180) and Radar is West (180), result is 0 (Nose-on).
    aspect_angle = angle_radar_direction - ucav_heading
    
    # Get RCS using your shape factor function
    sigma = get_rcs(aspect_angle)

    # --- 2. RADAR RANGE EQUATION ---
    # Numerator: Pt * G^2 * Lambda^2 * Sigma
    # This part was ALREADY CORRECT in your code.
    num = (config.RADAR_PEAK_POWER * (config.RADAR_GAIN_LIN**2) * (config.RADAR_WAVELENGTH**2) * sigma)
    
    # Denominator: (4pi)^3 * R^4 * Noise * Losses
    # This part was ALREADY CORRECT.
    # config.RADAR_LOSSES_LIN is > 1.0, so putting it in denominator correctly reduces SNR.
    den = ((4 * np.pi)**3 * (R**4) * config.BOLTZMANN * config.TEMP_0 * config.RADAR_BANDWIDTH * config.RADAR_LOSSES_LIN)
    
    snr_linear = num / den
    
    # Safety floor for log10
    if snr_linear <= 1e-9: return -99.0 
    
    return 10.0 * np.log10(snr_linear)

def calculate_burn_through_range(ucav_jammer_power, current_rcs=None):
    """
    Calculates the distance (meters) where Radar Power > Jamming Power.
    Args:
        ucav_jammer_power: Current power output of the jammer (Watts).
        current_rcs: (Optional) The current RCS based on aspect angle. 
                     Defaults to MIN_RCS if not provided.
    """
    # 1. RADAR CONSTANT (k_radar)
    # Derived from: Pt * G_radar (Transmit Gain)
    # Physics Note: Receiver Gain (Gr) cancels out on both sides (Signal vs Jammer).
    k_radar = config.RADAR_PEAK_POWER * config.RADAR_GAIN_LIN

    # 2. JAMMER CONSTANT (k_jam)
    # Use the function argument 'ucav_jammer_power' instead of the global config constant
    # so we can simulate variable jamming power levels.
    JAMMER_BANDWIDTH = 200e6 
    bw_efficiency = config.RADAR_BANDWIDTH / JAMMER_BANDWIDTH
    jammer_gain = 1.0 

    # --- FIX: Used the argument variable here ---
    k_jam = (ucav_jammer_power * bw_efficiency) * jammer_gain
    
    # 3. SELECT RCS
    # Use the dynamic RCS if provided, otherwise use the conservative config value
    sigma = current_rcs if current_rcs is not None else config.UCAV_MIN_RCS

    # 4. CALCULATE
    # Formula: R^2 = (Pt * G * sigma) / (4pi * Pj_eff * Gj)
    # We simplified (4pi)^3 / (4pi)^2 to just (4pi) in the denominator.
    numerator = k_radar * sigma
    denominator = k_jam * 4 * np.pi
    
    term = numerator / denominator
    
    return np.sqrt(term)

def get_doppler_notch_factor(ucav_vel, ucav_heading, radar_pos, ucav_pos):
    """
    Returns factor 0.0 (Invisible/Notched) to 1.0 (Fully Visible).
    """
    r_vec = radar_pos - ucav_pos
    dist = np.linalg.norm(r_vec)
    
    # SAFETY: Avoid division by zero
    if dist < 1.0: return 1.0 # If you hit the radar, you are definitely visible!
    
    r_unit = r_vec / dist
    
    v_vec = np.array([
        ucav_vel * np.cos(ucav_heading),
        ucav_vel * np.sin(ucav_heading)
    ])
    
    radial_vel = np.dot(v_vec, r_unit)
    
    # HIGH FIDELITY TUNING:
    # A modern FCR (Fire Control Radar) has a narrow notch.
    # sigma_v = 4.0 m/s implies you need to be very precise to hide.
    sigma_v = 4.0 
    
    notch_effectiveness = np.exp(- (radial_vel**2) / (2 * sigma_v**2))
    
    return 1.0 - notch_effectiveness