"""Schema and encoding constants shared across the package."""

TARGET_STRESS = "Stress Level"
TARGET_DISORDER = "Sleep Disorder"
TARGETS = [TARGET_STRESS, TARGET_DISORDER]

# Columns present in some exports of the dataset that carry no signal.
ID_COLUMNS = ["Unnamed: 0", "Person ID"]

RAW_REQUIRED = [
    "Gender", "Age", "Occupation", "Sleep Duration", "Quality of Sleep",
    "Physical Activity Level", "Stress Level", "BMI Category", "Blood Pressure",
    "Heart Rate", "Daily Steps", "Sleep Disorder",
]

# Feature groups used by the preprocessing ColumnTransformer.
BINARY_FEATURES = ["Gender"]            # Female=0, Male=1
NOMINAL_FEATURES = ["Occupation"]       # one-hot encoded (no natural order)
ORDINAL_FEATURES = ["BMI Category"]     # Underweight < Normal < Overweight < Obese
NUMERIC_FEATURES = [
    "Age", "Sleep Duration", "Quality of Sleep", "Physical Activity Level",
    "Heart Rate", "Daily Steps", "SystolicBP", "DiastolicBP",
]
FEATURES = BINARY_FEATURES + NOMINAL_FEATURES + ORDINAL_FEATURES + NUMERIC_FEATURES

GENDER_ORDER = ["Female", "Male"]
BMI_ORDER = ["Underweight", "Normal", "Overweight", "Obese"]
BMI_ALIASES = {"Normal Weight": "Normal"}  # the source data uses both spellings

NO_DISORDER = "No Disorder"
RANDOM_STATE = 42
