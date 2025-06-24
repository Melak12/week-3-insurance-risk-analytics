import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

'''
This script is used to build and evaluate predictive models that form the core of a dynamic, risk-based pricing system.

Dataset location: ../data/MachineLearningRating_v3.txt file
Information about the dataset: ../docs/business_docs.md under "Dataset Information" section.

Modeling Goals:

1) Claim Severity Prediction (Risk Model): For policies that have a claim, build a model to predict the TotalClaims amount. This model estimates the financial liability associated with a policy.
Target Variable: TotalClaims (on the subset of data where claims > 0).

Evaluation Metric: Root Mean Squared Error (RMSE) to penalize large prediction errors, and R-squared.

2) Premium Optimization (Pricing Framework): Develop a machine learning model to predict an appropriate premium. A naive approach is to predict CalculatedPremiumPerTerm, but a more sophisticated, business-driven approach is required.

3)Advanced Task: Build a model to predict the probability of a claim occurring (a binary classification problem). The Risk-Based Premium can then be conceptually framed as: Premium = (Predicted Probability of Claim * Predicted Claim Severity) + Expense Loading + Profit Margin.
'''

class InsurancePredictionModel:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = self.load_data()

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

    def handle_missing_data(self, drop_threshold=0.3):
        """
        Impute or remove missing values based on their nature and the quantity missing.
        Columns with >drop_threshold missing are dropped. Categorical: mode, Numeric: median.
        """
        df = self.df.copy()
        # Drop columns with too many missing values
        threshold = drop_threshold * len(df)
        cols_to_drop = [col for col in df.columns if df[col].isnull().sum() > threshold]
        df.drop(columns=cols_to_drop, inplace=True)
        # Impute
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype == 'object':
                    mode_val = df[col].mode(dropna=True)
                    if not mode_val.empty:
                        df[col] = df[col].fillna(mode_val[0])
                else:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
        self.df = df
        return df
    
    def check_missing_values(self):
        # Show only columns with missing values
        missing_values = self.df.isnull().sum()
        #print total columns with missing values
        print("\nTotal Columns with Missing Values:")
        print(missing_values[missing_values > 0].count())
        print("\nMissing Values:")
        print(missing_values[missing_values > 0])

    def feature_engineering(self):
        """
        Create new features relevant to TotalPremium and TotalClaims.
        Example: ClaimFrequency, Margin, PremiumPerVehicle, etc.
        """
        df = self.df.copy()
        if 'TotalClaims' in df.columns:
            df['ClaimFrequency'] = (df['TotalClaims'] > 0).astype(int)
        if 'TotalPremium' in df.columns and 'TotalClaims' in df.columns:
            df['Margin'] = df['TotalPremium'] - df['TotalClaims']
        if 'TotalPremium' in df.columns and 'NumberOfVehiclesInFleet' in df.columns:
            df['PremiumPerVehicle'] = df['TotalPremium'] / (df['NumberOfVehiclesInFleet'].replace(0, np.nan))
        self.df = df
        return df

    def encode_categorical(self, encoding='onehot', drop_first=True, onehot_cols=None):
        """
        Convert categorical data into numeric format using one-hot or label encoding.
        Only one-hot encode columns specified in onehot_cols (list of column names).
        Label encode all other categorical columns.
        """
        df = self.df.copy()
        # Convert boolean columns to int
        bool_cols = df.select_dtypes(include=['bool']).columns
        for col in bool_cols:
            df[col] = df[col].astype(int)
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if onehot_cols is None:
            onehot_cols = []
        # Only keep columns that are actually categorical
        onehot_cols = [col for col in onehot_cols if col in cat_cols]
        label_cols = [col for col in cat_cols if col not in onehot_cols]
        if encoding == 'onehot':
            if onehot_cols:
                df = pd.get_dummies(df, columns=onehot_cols, drop_first=drop_first)
            for col in label_cols:
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        elif encoding == 'label':
            for col in cat_cols:
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        self.df = df
        return df

    def train_test_split(self, target, test_size=0.2, random_state=42):
        """
        Split the data into train and test sets.
        """
        df = self.df.copy()
        X = df.drop(columns=[target])
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test