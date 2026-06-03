# Iris Flower Classification Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load Dataset
print("Script started successfully!")
df = pd.read_csv("Iris.csv")
print("Dataset loaded successfully!")
print(df.head())


# Step 2: Explore Data
print("\nDataset Info:\n")
print(df.info())
print("\nSummary Statistics:\n")
print(df.describe())
print("\nSpecies Count:\n", df['species'].value_counts())

# Visualization
sns.pairplot(df, hue="species")
plt.show()

sns.heatmap(df.drop("species", axis=1).corr(), annot=True, cmap="coolwarm")
plt.show()


# Step 3: Prepare Features & Labels
X = df.drop(["species"], axis=1)
y = df["species"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Scale Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Train Model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Step 6: Evaluate Model
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 7: Predict New Sample
sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)
print("\nPredicted species:", prediction[0])

plt.show()
