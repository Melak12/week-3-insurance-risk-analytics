import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


## This script provides Exploratory Data Analysis (EDA) for insurance data which is located in the ../data/MachineLearningRating_v3.txt file. The following are key functionalities:
# 1. Data Summarization
    # - Descriptive Statistics: Calculate the variability for numerical features such as TotalPremium, TotalClaim, etc.
    # - Data Structure: Review the dtype of each column to confirm if categorical variables, dates, etc. are properly formatted.
# 2. Data Quality Assessment: 
    # - Missing Values: Identify and handle missing values in the dataset.
    # - Duplicates: Check for and remove duplicate entries.
    # - Data Types: Ensure that data types are appropriate for analysis (e.g., numerical, categorical).
# 3. Univariate Analysis
    #    - Distribution Analysis: Analyze the distribution of numerical features like TotalPremium, TotalClaim, etc.
    # - Plot histograms for numerical columns and bar charts for categorical columns to understand distributions.
# 4. Bivariate or Multivariate Analysis:
    # - Correlations and Associations: Explore relationships between the monthly changes TotalPremium and TotalClaims as a function of ZipCode, using scatter plots and correlation matrices.
    # - Correlation Analysis: Calculate correlation coefficients between numerical features to identify relationships.
    # - Grouped Analysis: Analyze how different categories (e.g., age
# 5. Data Comparison
    # - Trends Over Geography: Compare the change in insurance cover type, premium, auto make, etc. 
# 6. Outlier Detection
   # - Use box plots to detect outliers in numerical data
# 7. Visualization
   # - Use seaborn and matplotlib to create visualizations for better understanding of the data.
   # - Visualize the distribution of premiums and claims, and how they vary by different categories like age
   # - Visualize the relationship between different features using scatter plots, bar charts, and heatmaps.


class InsuranceAnalysis:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.load_data()

    # Load the data from the specified file path and preprocess it
    def load_data(self):
        print(f"Loading data from {self.file_path}...")
        if not self.file_path.endswith('.txt'):
            raise ValueError("File must be a .txt file with | separated values.")
        try:
            data = pd.read_csv(self.file_path, sep='|')
            print("Data loaded successfully.")
            return data 
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def convert_to_datetime(self, column_name):
        if column_name in self.data.columns:
            try:
                self.data[column_name] = pd.to_datetime(self.data[column_name], errors='coerce')
                print(f"Converted {column_name} to datetime format.")
            except Exception as e:
                print(f"Error converting {column_name} to datetime: {e}")
        else:
            print(f"{column_name} does not exist in the data.")

    def summarize_data(self):

        #print shape of the data
        print(f"Data Shape: {self.data.shape}")

        print("Data Summary:")
        print(self.data.describe())

        # Review the dtype of each column to confirm if categorical variables, dates, etc. are properly formatted.
        self.check_data_types()
        print("\nData Types:")
        print(self.data.dtypes)
    
    def check_data_types(self):
        print("\nData Types:")
        for column in self.data.columns:
            print(f"{column}: {self.data[column].dtype}")

    def check_missing_values(self):
        # Show only columns with missing values
        missing_values = self.data.isnull().sum()
        #print total columns with missing values
        print("\nTotal Columns with Missing Values:")
        print(missing_values[missing_values > 0].count())
        print("\nMissing Values:")
        print(missing_values[missing_values > 0])

    def check_duplicates(self):
        # Check for duplicate rows
        print("\nDuplicate Rows:")
        print(self.data.duplicated().sum())

        # View duplicate rows
        self.view_duplicates()
    
    def clean_data(self):
        # Convert the date columns to datetime format
        self.convert_to_datetime('VehicleIntroDate')
        self.convert_to_datetime('TransactionMonth')

        

        # Drop columns with more than 30% missing values
        threshold = 0.3 * len(self.data)
        cols_to_drop = [col for col in self.data.columns if self.data[col].isnull().sum() > threshold]
        if cols_to_drop:
            print(f"Dropping columns with >30% missing values: {cols_to_drop}")
            self.data.drop(columns=cols_to_drop, inplace=True)

        # Impute missing values for categorical columns (mode) and numerical columns (median)
        for col in self.data.columns:
            if self.data[col].isnull().any():
                if self.data[col].dtype == 'object':
                    mode_val = self.data[col].mode(dropna=True)
                    if not mode_val.empty:
                        self.data[col] = self.data[col].fillna(mode_val[0])
                        print(f"Filled missing values in {col} with mode: {mode_val[0]}")
                    else:
                        print(f"No mode found for {col}, leaving missing values as is.")
                else:
                    median_val = self.data[col].median()
                    self.data[col] = self.data[col].fillna(median_val)
                    print(f"Filled missing values in {col} with median: {median_val}")

        # Remove duplicate entries
        if self.data.duplicated().any():
            print("Removing duplicate rows...")
            self.data.drop_duplicates(inplace=True)
            print("Duplicate rows removed.")

        # Return the cleaned DataFrame
        return self.data
    
    def export_cleaned_data(self, output_path):
        # Export the cleaned data to a new file
        try:
            self.data.to_csv(output_path, index=False)
            print(f"Cleaned data exported to {output_path}.")
        except Exception as e:
            print(f"Error exporting cleaned data: {e}")

    def view_duplicates(self):
        duplicates = self.data[self.data.duplicated()]
        if not duplicates.empty:
            print("Duplicate Rows:")
            print(duplicates)
        else:
            print("No duplicate rows found.")
    
    def univariate_analysis(self, save_plots=False, output_dir=None, num_cols=None, cat_cols=None):
        """
        Plots histograms for up to 5 key numerical columns and bar charts for up to 5 key categorical columns to understand distributions.
        If save_plots is True, saves plots to output_dir; otherwise, displays them.
        Optionally, you can specify which columns to plot using num_cols and cat_cols.
        """
        # Select up to 5 numerical columns
        if num_cols is None:
            num_cols = self.data.select_dtypes(include=['int64', 'float64']).columns[:5]
        else:
            num_cols = [col for col in num_cols if col in self.data.columns][:5]
        # Select up to 5 categorical columns
        if cat_cols is None:
            cat_cols = self.data.select_dtypes(include=['object', 'category']).columns[:5]
        else:
            cat_cols = [col for col in cat_cols if col in self.data.columns][:5]

        # Plot histograms for numerical columns
        for col in num_cols:
            plt.figure(figsize=(7, 4))
            sns.histplot(self.data[col].dropna(), kde=True, bins=30, color='skyblue')
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.tight_layout()
            if save_plots and output_dir:
                os.makedirs(output_dir, exist_ok=True)
                plt.savefig(os.path.join(output_dir, f"hist_{col}.png"))
                plt.close()
            else:
                plt.show()

        # Plot bar charts for categorical columns
        for col in cat_cols:
            plt.figure(figsize=(8, 4))
            self.data[col].value_counts(dropna=False).plot(kind='bar', color='orange')
            plt.title(f'Bar Chart of {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.tight_layout()
            if save_plots and output_dir:
                os.makedirs(output_dir, exist_ok=True)
                plt.savefig(os.path.join(output_dir, f"bar_{col}.png"))
                plt.close()
            else:
                plt.show()

