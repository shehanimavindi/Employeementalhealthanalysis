import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load Dataset
df = pd.read_csv(r"C:\Users\diluk\OneDrive\Desktop\Projects\Employee mental health analysis\mental_health_workplace.csv")

print(df.columns)
# Show first 5 rows
print(df.head())

# Show dataset info
print("\nDataset Info:")
print(df.info())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

df.columns = df.columns.str.strip().str.lower()

# Fill missing values
categorical_cols = df.select_dtypes(include=['object']).columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())

# Encode categorical columns
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Target column
target = 'burnout_risk_score'

# Features and labels
X = df.drop(columns=[target, 'record_id'])
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Create model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Absolute Error:")
print(mae)

print("\nR2 Score:")
print(r2)

# Feature Importance
importance = model.feature_importances_
feature_names = X.columns

feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nTop Important Features:")
print(feature_importance.head(10))

# Burnout Distribution Plot
# Count burnout values
burnout_counts = df['burnout_risk_score'].value_counts()

# Create figure
plt.figure(figsize=(8,5))

# Bar chart
bars = plt.bar(
    burnout_counts.index.astype(str),
    burnout_counts.values
)

# Add count labels on bars
for bar in bars:
    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        str(height),
        ha='center',
        va='bottom',
        fontsize=10
    )

# Labels and title
plt.xlabel("Burnout Risk Level", fontsize=12)
plt.ylabel("Number of Employees", fontsize=12)
plt.title("Burnout Risk Distribution", fontsize=15)

# Rotate labels if crowded
plt.xticks(rotation=0)

# Grid for readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Tight layout
plt.tight_layout()

# Show graph
plt.show()

# Stress vs Weekly Hours Scatter Plot
plt.figure(figsize=(8,5))

plt.scatter(
    df['weekly_hours'],
    df['stress_level']
)

plt.xlabel("Weekly Work Hours")
plt.ylabel("Stress Level")
plt.title("Stress Level vs Weekly Work Hours")
plt.show()