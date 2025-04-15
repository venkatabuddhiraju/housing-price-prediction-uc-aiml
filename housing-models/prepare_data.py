import numpy as np
#cleanup tasks - venkata
import pandas as pd
from scipy.stats._mstats_basic import winsorize
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
from matplotlib import pyplot as plt

scaler = StandardScaler()

test_df = pd.read_csv('../docs/test.csv')
# test_df.head()
train_df = pd.read_csv('../docs/train.csv')
train_df.head()

# missing_values_in_test = test_df.isnull().sum()
# print(missing_values_in_test)

def remove_highly_correlated_features(train_df, test_df, threshold=0.85):
    # Compute the correlation matrix
    corr_matrix = train_df.corr(numeric_only=True)

    to_drop = [column for column in corr_matrix.columns if any(corr_matrix[column] > threshold)]

    # Drop the highly correlated columns
    train_df = train_df.drop(columns=to_drop)
    test_df = test_df.drop(columns=to_drop)
    return train_df, test_df


def process_correlation(train_df, test_df):
    # Compute the correlation matrix
    correlation_matrix = train_df.corr(numeric_only=True)

    # Visualize the correlation matrix
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    # plt.title('Correlation Matrix')
    # plt.show()
    remove_highly_correlated_features(train_df, test_df)

def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Remove rows with outliers
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

def winsorize_data(df, columns, limits=(0.01, 0.01)):
   for col in columns:
       df[col] = winsorize(df[col], limits=limits)
   return df

numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns

def preprocess_data(train_df, test_df):
    # Identify numerical and categorical columns
    numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = train_df.select_dtypes(include=['object', 'category']).columns

    # Scale numerical columns
    scaler = StandardScaler()
    train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
    test_df[numerical_cols] = scaler.transform(test_df[numerical_cols])
    # Encode categorical columns
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_categorical = encoder.fit_transform(train_df[categorical_cols])
    encoded_categorical_test = encoder.transform(test_df[categorical_cols])
    # Convert encoded values to a DataFrame
    encoded_categorical_df = pd.DataFrame(
        encoded_categorical,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=train_df.index
    )
    encoded_categorical_df_test = pd.DataFrame(encoded_categorical_test,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=test_df.index
    )
    # Drop original categorical columns and concatenate the encoded DataFrame
    train_df = pd.concat([train_df.drop(categorical_cols, axis=1), encoded_categorical_df], axis=1)
    test_df = pd.concat([test_df.drop(categorical_cols, axis=1), encoded_categorical_df_test], axis=1)

    # Align the train and test DataFrames

    # process correlation features
    process_correlation(train_df, test_df)
    numerical_cols_winsor = [col for col in numerical_cols if col != 'Id']  # If it's a list
    train_df = winsorize_data(train_df, numerical_cols_winsor)

    train_df = train_df.drop(columns=['Id'])  # Drop ID column
    # test_ids = test_df['Id']
    test_df = test_df.drop(columns=['Id'])

    return train_df, test_df

def preprocess_test_data(test_df, scaler, encoder, categorical_cols, numerical_cols):
    # Scale numerical columns
    test_df[numerical_cols] = scaler.transform(test_df[numerical_cols])

    # Encode categorical columns
    encoded_categorical_test = encoder.transform(test_df[categorical_cols])
    encoded_categorical_df_test = pd.DataFrame(
        encoded_categorical_test,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=test_df.index
    )

    # Drop original categorical columns and concatenate the encoded DataFrame
    test_df = pd.concat([test_df.drop(categorical_cols, axis=1), encoded_categorical_df_test], axis=1)

    # Drop unnecessary columns (e.g., 'Id')
    test_df = test_df.drop(columns=['Id'], errors='ignore')

    return test_df

def get_training_data():
    global y_train, y_test, X_train_ready, X_test_ready, test_df
    # Load the dataset
    X = train_df.drop('SalePrice', axis=1)  # Replace 'target_column' with the actual target column name
    y = train_df['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_copy = X_train.copy()
    # preprocess
    X_train_ready, X_test_ready = preprocess_data(X_train, X_test)
    X_train_ready_copy, output_test_ready = preprocess_data(X_train_copy, test_df)
    return X_train_ready, X_test_ready, y_train, y_test, output_test_ready


