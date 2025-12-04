"""
Centralized configuration for OVRO-LWA Solar pipeline.
"""


# Reference antenna for calibration
# 2025-12 changed from 202 to 283
REFANT = '283'

# Examples: '>10lambda', '>1lambda', '<1000lambda'
DEFAULT_UV_RANGE = '>10lambda'

# Default number of phase calibration rounds
DEFAULT_NUM_PHASE_CAL = 1

# Default number of amplitude-phase calibration rounds
DEFAULT_NUM_APCAL = 1

# Default polarization
DEFAULT_POL = 'I'  # Options: 'I', 'XX,YY', etc.