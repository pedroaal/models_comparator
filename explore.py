import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def basic_info(df):
  print("\n=== DATASET OVERVIEW ===")
  print(f"Dataset shape: {df.shape}")
  print(f"Number of rows: {df.shape[0]}")
  print(f"Number of columns: {df.shape[1]}")
  
  print("\n=== DATA TYPES ===")
  print(df.dtypes)
  
  print("\n=== MISSING VALUES ===")
  missing = df.isnull().sum()
  missing_percent = (missing / len(df)) * 100
  missing_info = pd.DataFrame({
      'Missing Values': missing,
      'Percentage': missing_percent
  })
  print(missing_info[missing_info['Missing Values'] > 0])
  
  print("\n=== DUPLICATED ROWS ===")
  print(f"Number of duplicated rows: {df.duplicated().sum()}")
  
  return missing_info

def numerical_analysis(df):
  numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
  
  if len(numerical_cols) == 0:
    print("\nNo numerical columns found in the dataset.")
    return
  
  print("\n=== NUMERICAL COLUMNS STATISTICS ===")
  stats_df = df[numerical_cols].describe().T
  stats_df['skew'] = df[numerical_cols].skew()
  stats_df['kurtosis'] = df[numerical_cols].kurtosis()
  print(stats_df)
  
  plt.figure(figsize=(15, len(numerical_cols) * 4))
  for i, col in enumerate(numerical_cols):
    plt.subplot(len(numerical_cols), 2, 2*i+1)
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f'Distribution of {col}')
    
    plt.subplot(len(numerical_cols), 2, 2*i+2)
    sns.boxplot(x=df[col].dropna())
    plt.title(f'Boxplot of {col}')
  
  plt.tight_layout()
  plt.savefig('numerical_analysis.png')
  plt.close()
  print("\nNumerical analysis plots saved as 'numerical_analysis.png'")

def categorical_analysis(df):
  categorical_cols = df.select_dtypes(include=['object', 'category']).columns
  
  if len(categorical_cols) == 0:
    print("\nNo categorical columns found in the dataset.")
    return
  
  print("\n=== CATEGORICAL COLUMNS ANALYSIS ===")
  for col in categorical_cols:
    print(f"\nColumn: {col}")
    value_counts = df[col].value_counts()
    print(f"Number of unique values: {len(value_counts)}")

    if len(value_counts) <= 20:  # Only show if not too many categories
      print(value_counts)
      
      # Bar plot for categorical columns
      plt.figure(figsize=(12, 6))
      sns.countplot(y=col, data=df, order=df[col].value_counts().index[:20])
      plt.title(f'Count of {col}')
      plt.tight_layout()
      plt.savefig(f'categorical_{col}.png')
      plt.close()
      print(f"Plot for {col} saved as 'categorical_{col}.png'")
    else:
      print(f"Too many unique values ({len(value_counts)}). Showing top 20:")
      print(value_counts.head(20))

def correlation_analysis(df):
  numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
  
  if len(numerical_cols) < 2:
    print("\nNot enough numerical columns for correlation analysis.")
    return
  
  print("\n=== CORRELATION ANALYSIS ===")
  corr_matrix = df[numerical_cols].corr()
  
  # Plot correlation heatmap
  plt.figure(figsize=(12, 10))
  sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
  plt.title('Correlation Matrix')
  plt.tight_layout()
  plt.savefig('correlation_matrix.png')
  plt.close()
  print("\nCorrelation matrix saved as 'correlation_matrix.png'")
  
  # Find highly correlated features
  print("\nHighly correlated features (|r| > 0.7):")
  high_corr = np.where(np.abs(corr_matrix) > 0.7)
  high_corr_list = [(corr_matrix.columns[x], corr_matrix.columns[y], corr_matrix.iloc[x, y]) 
                    for x, y in zip(*high_corr) if x != y and x < y]
  
  if high_corr_list:
    for col1, col2, corr_val in high_corr_list:
      print(f"{col1} and {col2}: {corr_val:.3f}")
  else:
    print("No highly correlated features found.")

def outlier_detection(df):
  numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
  
  if len(numerical_cols) == 0:
    print("\nNo numerical columns for outlier detection.")
    return
  
  print("\n=== OUTLIER DETECTION ===")
  for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]

    if len(outliers) > 0:
      print(f"\nColumn: {col}")
      print(f"Number of outliers: {len(outliers)}")
      print(f"Percentage of outliers: {(len(outliers) / len(df)) * 100:.2f}%")
      print(f"Range of outliers: [{outliers.min()}, {outliers.max()}]")

def time_series_check(df):
  date_cols = df.select_dtypes(include=['datetime64']).columns
  
  # Also check for columns that might be dates but stored as objects
  object_cols = df.select_dtypes(include=['object']).columns
  potential_date_cols = []
  
  for col in object_cols:
    # Try to convert to datetime
    try:
      pd.to_datetime(df[col], errors='raise')
      potential_date_cols.append(col)
    except:
      pass
  
  if len(date_cols) == 0 and len(potential_date_cols) == 0:
    print("\nNo datetime columns found in the dataset.")
    return
  
  print("\n=== TIME SERIES ANALYSIS ===")
  
  # Process actual datetime columns
  for col in date_cols:
    print(f"\nDatetime column: {col}")
    print(f"Min date: {df[col].min()}")
    print(f"Max date: {df[col].max()}")
    print(f"Range: {df[col].max() - df[col].min()}")
  
  # Process potential datetime columns
  for col in potential_date_cols:
    print(f"\nPotential datetime column: {col}")
    print("Sample values:")
    print(df[col].sample(5).values)
    print("Consider converting this column to datetime using pd.to_datetime()")

def main():
  df = pd.read_csv('data.csv')
  df.fillna(df.mean(), inplace=True)
  
  print("\n=== FIRST 5 ROWS ===")
  print(df.head())
  
  # Run all analysis functions
  basic_info(df)
  numerical_analysis(df)
  categorical_analysis(df)
  correlation_analysis(df)
  outlier_detection(df)
  time_series_check(df)
  
  print("\nExploratory data analysis completed!")

if __name__ == "__main__":
  main()