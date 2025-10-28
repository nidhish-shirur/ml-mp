"""
Configuration settings for the NEO ML application
"""

# Cache settings
CACHE_TTL = 3600  # 1 hour

# Model settings
DEFAULT_N_ESTIMATORS = 100
DEFAULT_MAX_DEPTH = 10
USE_PARALLEL = True  # Use n_jobs=-1 for parallel processing

# Data sampling for visualization
MAX_PLOT_SAMPLES = 905  # All NEO samples (small dataset)

# Streamlit settings
WIDE_MODE = True
INITIAL_SIDEBAR_STATE = "expanded"

# Session state keys
SESSION_DATA_KEY = "neo_data"
SESSION_MODELS_KEY = "trained_models"

# NEO Dataset info
DATA_FILE = "neo_data.csv"

# Feature columns
FEATURE_COLS = [
    'Absolute Magnitude (H)',
    'Min Diameter (km)',
    'Max Diameter (km)',
    'Relative Velocity (km/s)',
    'Miss Distance (astronomical)',
    'Epoch Date Close Approach'
]

# Target columns
TARGET_HAZARDOUS = 'Is Potentially Hazardous'
TARGET_DISTANCE = 'Miss Distance (km)'
TARGET_VELOCITY = 'Relative Velocity (km/s)'

# Page settings
PAGE_TITLE = "NEO Detection ML Project"
PAGE_ICON = "🔭"

# Model training settings
DEFAULT_TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model file naming convention
MODEL_SUFFIX = "_neo"
