"""
Global configuration for CSEE-DML research pipeline.
All experiment parameters are centralized here for reproducibility.
"""
import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

# ── Panel dimensions ───────────────────────────────────────────────────────
N_CITIES = 280          # prefecture-level cities
START_YEAR = 2005
END_YEAR = 2023
POLICY_YEAR = 2017      # first batch of climate-resilient city pilots
N_PILOT_CITIES = 28     # pilot cities (住建部 first batch)

# ── NDVI simulation ─────────────────────────────────────────────────────────
NDVI_INTERVAL_DAYS = 16     # MODIS MOD13Q1 temporal resolution
NDVI_SPATIAL_RES_M = 250    # meters
NDVI_PERIODS_PER_YEAR = 23  # 365 / 16 ≈ 23

# ── Extreme weather thresholds ─────────────────────────────────────────────
HEAT_PERCENTILE = 90         # daily max temp > 90th percentile
HEAT_MIN_DURATION = 3        # consecutive days
RAIN_PERCENTILE = 95          # daily precipitation > 95th percentile
DROUGHT_NO_RAIN_DAYS = 30    # consecutive days without effective precipitation
DROUGHT_SPEI_THRESHOLD = -1.5

# ── CSEE computation ───────────────────────────────────────────────────────
NORMAL_LOOKBACK_YEARS = 5    # years used to compute NDVI_normal baseline
RECOVERY_WINDOW_MONTHS = 3   # months after event to measure recovery
EVENT_WINDOW_DAYS = 30       # NDVI window around extreme weather event

# ── DML parameters ──────────────────────────────────────────────────────────
N_FOLDS = 5                  # cross-fitting folds
N_BOOTSTRAP = 500            # bootstrap iterations for placebo/se
RANDOM_SEED = 42

# ML learners to compare
ML_LEARNERS = ["random_forest", "xgboost", "neural_network", "lasso"]

# Random Forest hyperparameters
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,
    "min_samples_leaf": 20,
    "max_features": "sqrt",
    "random_state": RANDOM_SEED,
}

# XGBoost hyperparameters
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# Neural Network hyperparameters
NN_PARAMS = {
    "hidden_layer_sizes": (128, 64, 32),
    "activation": "relu",
    "alpha": 0.01,
    "max_iter": 500,
    "random_state": RANDOM_SEED,
}

# LASSO hyperparameters
LASSO_PARAMS = {
    "alpha": 0.01,
    "max_iter": 5000,
    "random_state": RANDOM_SEED,
}

# ── True treatment effect (for simulation validation) ─────────────────────
TRUE_BUFFER_EFFECT = 0.08    # policy increases CSEE by 0.08 under shock
TRUE_RESISTANCE_EFFECT = 0.06
TRUE_RECOVERY_EFFECT = 0.10

# ── Control variables ───────────────────────────────────────────────────────
CONTROL_VARS = [
    "ln_gdppc",       # log GDP per capita
    "gdp_growth",     # GDP growth rate
    "ind_share",      # secondary industry share
    "ter_share",      # tertiary industry share
    "pop_density",    # population density
    "urban_rate",     # urbanization rate
    "annual_temp",    # annual mean temperature
    "annual_precip",  # annual precipitation
    "elevation",      # elevation
    "built_area",     # built-up area
    "road_density",   # road density
    "green_rate",     # green coverage rate
    "env_exp_share",  # environmental expenditure share
    "edu_level",      # education level
    "tech_exp",        # science & technology expenditure
]

# ── Mediation variables (for mechanism analysis) ───────────────────────────
MEDIATION_VARS = [
    "green_rate",       # green coverage rate (resistance path)
    "sponge_inv",       # sponge city investment (resistance path)
    "blue_green_ratio", # blue-green space ratio (blue-green infra path)
    "coupling_coord",   # coupling coordination degree (coupling path)
]

# ── Heterogeneity grouping variables ──────────────────────────────────────
HETERO_VARS = {
    "city_size": "large vs small",         # by population
    "region": "east/central/west",          # by geography
    "ns": "north/south",                    # by climate
    "coastal": "coastal/inland",            # by location
    "ecology_baseline": "fragile/good",     # by ecological base
    "shock_type": "heat/rain/drought",      # by extreme weather type
    "policy_intensity": "high/low",         # by sponge city investment
}

# ── Visualization ───────────────────────────────────────────────────────────
FIG_DPI = 300
FIG_FORMAT = "png"
COLOR_PALETTE = "Set2"


def ensure_dirs():
    for d in [RESULTS_DIR, FIGURES_DIR, TABLES_DIR]:
        os.makedirs(d, exist_ok=True)
