import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

# 1. Load the Dataset
df = pd.read_csv("B1_toyato_feature.csv")

# 2. Handle missing values
df.fillna(df.median(), inplace=True)

# 3. Separate features (X) and target (y)
X = df.drop(columns=["Price"])
y = df["Price"]

# 4. Split dataset into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Initialize and Train the Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Make Predictions
y_pred = model.predict(X_test)

# 7. Evaluate Model Performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.2f}")

# 8. Compute Permutation Feature Importances
perm_importance = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42)

# 9. Extract Feature Importances and Feature Names
feature_importances = perm_importance.importances_mean
feature_names = X.columns

# 10. Print Feature Importances
print("\nFeature Importances:")
for feature_name, importance in zip(feature_names, feature_importances):
    print(f"{feature_name}: {importance:.4f}")

# 11. Visualize Feature Importances
plt.figure(figsize=(10, 5))
plt.barh(feature_names, feature_importances, color='skyblue')
plt.xlabel("Permutation Importance")
plt.title("Permutation Feature Importance (Toyota Price Prediction)")
plt.show()
# 10. Print Feature Importances in Tabular Format
print("\nFeature\t\tImportance")
sorted_idx = feature_importances.argsort()[::-1]  # Sort from high to low

for idx in sorted_idx:
    feature = feature_names[idx]
    importance = feature_importances[idx]
    print(f"{feature:<16}{importance:.3f}")