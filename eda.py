import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define column names based on dataset-info.txt
columns = [
    "status", "duration", "credit_history", "purpose", "credit_amount",
    "savings", "employment_duration", "installment_rate", "personal_status_sex",
    "other_debtors", "residence_since", "property", "age", "other_installment_plans",
    "housing", "existing_credits", "job", "people_liable", "telephone", "foreign_worker",
    "credit_risk"
]

def load_data():
    """Loads the dataset from dataset.data"""
    filepath = "dataset.data"
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return None
    
    # The dataset is space-separated
    df = pd.read_csv(filepath, sep=" ", names=columns, header=None)
    return df

def perform_eda(df):
    """Performs basic EDA and saves plots"""
    if df is None:
        return

    print("Dataset Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nDescriptive Statistics:")
    print(df.describe())
    
    # Create 'plots' directory if it doesn't exist
    if not os.path.exists("plots"):
        os.makedirs("plots")

    # Target variable distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='credit_risk', data=df)
    plt.title('Distribution of Credit Risk (1=Good, 2=Bad)')
    plt.savefig('plots/credit_risk_distribution.png')
    plt.close()

    # Numerical distributions
    numerical_cols = ["duration", "credit_amount", "installment_rate", "residence_since", "age", "existing_credits", "people_liable"]
    for col in numerical_cols:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.savefig(f'plots/{col}_distribution.png')
        plt.close()

    # Correlation matrix (only for numerical columns)
    plt.figure(figsize=(10, 8))
    corr = df[numerical_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Numerical Features')
    plt.savefig('plots/correlation_matrix.png')
    plt.close()
    
    print("\nEDA complete. Plots saved to 'plots/' directory.")

if __name__ == "__main__":
    df = load_data()
    perform_eda(df)
