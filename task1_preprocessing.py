# ==============================
# STEP 1: Import Libraries
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ==============================
# STEP 2: Load Dataset
# ==============================
df = pd.read_csv("titanic.csv")   # Make sure file is in same folder
print("First 5 rows:")
print(df.head())


# ==============================
# STEP 3: Explore Dataset
# ==============================
print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())


# ==============================
# STEP 4: Handle Missing Values
# ==============================
# Fill Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill Embarked with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin (too many nulls)
if 'Cabin' in df.columns:
    df.drop('Cabin', axis=1, inplace=True)

print("\nAfter Handling Missing Values:")
print(df.isnull().sum())


# ==============================
# STEP 5: Encode Categorical Data
# ==============================
# Convert Sex column
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# One-hot encoding for Embarked
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

print("\nAfter Encoding:")
print(df.head())


# ==============================
# STEP 6: Feature Scaling (Normalization)
# ==============================
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

print("\nAfter Scaling:")
print(df[['Age', 'Fare']].head())


# ==============================
# STEP 7: Detect Outliers
# ==============================
plt.figure()
sns.boxplot(x=df['Fare'])
plt.title("Boxplot for Fare")
plt.show()


# ==============================
# STEP 8: Remove Outliers (IQR Method)
# ==============================
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1

df = df[(df['Fare'] >= Q1 - 1.5 * IQR) & (df['Fare'] <= Q3 + 1.5 * IQR)]

print("\nAfter Removing Outliers:")
print(df.shape)


# ==============================
# STEP 9: Final Cleaned Data
# ==============================
print("\nFinal Dataset:")
print(df.head())

# Save cleaned dataset
df.to_csv("cleaned_titanic.csv", index=False)

print("\n✅ Data Cleaning Completed Successfully!")