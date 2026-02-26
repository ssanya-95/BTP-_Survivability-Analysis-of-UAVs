import numpy as np

# ==========================================
# GLOBAL PHYSICS CONSTANTS
# ==========================================
C = 3e8                  # Speed of light (m/s)
BOLTZMANN = 1.38e-23     # Boltzmann constant (J/K)
TEMP_0 = 290.0           # Standard noise temperature (Kelvin)

# ==========================================
# THREAT RADAR PARAMETERS (Adversary)
# ==========================================
# Modeled after a generic modern SAM acquisition radar (X-Band)
RADAR_FREQ = 10e9        # 10 GHz
RADAR_PEAK_POWER = 200e3  # 200 kW
RADAR_GAIN_DB = 35.0     # Antenna Gain in dB
RADAR_LOSSES_DB = 5.0    # System losses (cabling, atmosphere)
RADAR_BANDWIDTH = 5e6    # 5 MHz receiver bandwidth

# Derived Linear Values (Calculated once here to save CPU time)
RADAR_WAVELENGTH = C / RADAR_FREQ
RADAR_GAIN_LIN = 10**(RADAR_GAIN_DB / 10.0)
RADAR_LOSSES_LIN = 10**(RADAR_LOSSES_DB / 10.0)

# Detection Logic Thresholds
SNR_THRESHOLD_DETECT = 13.0  # dB (Minimum to be seen)
SNR_THRESHOLD_TRACK = 18.0   # dB (Minimum to establish track)
SNR_THRESHOLD_LOCK = 25.0    # dB (Minimum to guide missile)

# ==========================================
# UCAV PARAMETERS (The Hero)
# ==========================================
UCAV_CRUISE_SPEED = 250.0    # m/s (~Mach 0.75)
UCAV_MAX_G_LOAD = 9.0        # Max turn rate limit (9G)
UCAV_MAX_JAMMER_POWER = 1500.0 # Watts
UCAV_MIN_RCS = 0.01          # m^2 (Best case nose-on)
UCAV_MAX_RCS = 5.0           # m^2 (Worst case broadside)
MAX_TURN_RATE = np.deg2rad(20.0) # Limit turn to 20 degrees per second (calculated from the formula)

# ==========================================
# PAYLOAD PARAMETERS
# ==========================================
# For a generic Guided Bomb Unit (GBU) or SDB
WEAPON_MAX_RANGE = 8000.0  # 8 km standoff range
WEAPON_PK_HIT = 0.9        # Probability of Kill if released in range

# ==========================================
# OPTIMIZATION WEIGHTS
# ==========================================
# "How paranoid should the UCAV be?"
# Higher Risk Weight = More evasion, longer path.
# Higher Mission Weight = straighter line, more danger.
WEIGHT_RISK = 25.0
WEIGHT_MISSION = 15.0

# ==========================================
# SAFE-STATE PATH SHAPING
# ==========================================
SAFE_LOOKAHEAD_TIME = 25.0          # seconds
RADAR_KEEP_OUT_RADIUS = 12000.0     # meters
RADAR_KEEP_OUT_PENALTY = 100.0      # unitless barrier weight
RADAR_APPROACH_PENALTY = 1.0        # unitless approach penalty

# ==========================================
# SIMULATION SETTINGS
# ==========================================
DT = 0.5          # Time step (seconds)
MAX_TIME = 300.0   # Max mission duration (seconds)