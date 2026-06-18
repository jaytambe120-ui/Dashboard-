import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("data.csv")

print(df.head())
print(df.info())

# Check Missing Values
print(df.isnull().sum())

# Fill Missing Values
df.fillna(df.mean(numeric_only=True), inplace=True)

for col in df.select_dtypes(include='object'):
    df[col].fillna(df[col].mode()[0], inplace=True)

# Remove Duplicates
print("Duplicates:", df.duplicated().sum())
df.drop_duplicates(inplace=True)

# Detect and Remove Outliers (IQR Method)
numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df = df[(df[col] >= lower) & (df[col] <= upper)]

# Basic Statistics
print(df.describe())

# Histogram
df.hist(figsize=(10, 8))
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.show()

# Box Plot
plt.figure(figsize=(10, 5))
sns.boxplot(data=df)
plt.xticks(rotation=45)
plt.show()

# Scatter Plot (change column names as needed)
# sns.scatterplot(x=df["Age"], y=df["Salary"])
# plt.show()

# Save Cleaned Dataset
df.to_csv("cleaned_data.csv", index=False)

print("Cleaned dataset saved successfully!")
