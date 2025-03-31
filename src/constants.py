import os

# Paths
RAW_DIR = os.path.join("PATH_TO_MIMIC_III_CSV_GZ_FILES")
INTERMEDIATE_DIR = os.path.join("./data/")
CHARTS_DIR = os.path.join("./data/charts/")

# Data processing
CHUNKSIZE = 500000
WINDOW_HOURS = 6
LABEL_WINDOW_HOURS = 24
NO_SEPSIS = 0
HAS_SEPSIS = 1
DEFAULT_GENDER = 0
OUTPUT_DIM = 1
SECONDS_PER_HOUR = 3600

# Age calculation
DAYS_PER_YEAR = 365.25
AGE_CAP = 90
YEAR_SHIFT = 100

# ITEMIDs
VITAL_SIGNS = {
	"heart_rate": [211],
	"sbp": [51],
	"dbp": [8368],
	"resp_rate": [618],
	"temperature": [223761],
	"spo2": [220277]
}
LAB_TESTS = {
	"hemoglobin": [50811],
	"hematocrit": [51221],
	"platelets": [51265],
	"wbc": [51301],
	"lactate": [50813],
	"creatinine": [50912]
}

# SIRS criteria conditions
SIRS_CONDITIONS = {
	223761: lambda x: (x > 38) | (x < 36),  # Temperature
	211: lambda x: x > 90,                  # Heart rate
	618: lambda x: x > 20,                  # Respiratory rate
	51301: lambda x: (x > 12) | (x < 4)     # WBC
}

# Model and training parameters
MODEL_HIDDEN = 64
MODEL_DROPOUT = 0.2
TRAIN_BATCH_SIZE = 64
TRAIN_NUM_EPOCHS = 15
TRAIN_LEARNING_RATE = 1e-4
TRAIN_WEIGHT_DECAY = 1e-4
SPLIT_TEST_SIZE = 0.3
SPLIT_VAL_RATIO = 0.5
RANDOM_STATE = 42
PREDICTION_THRESHOLD = 0.7
