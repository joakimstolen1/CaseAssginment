# pandas for data manipulation,
# sqlite3 for database operations,
# and scikit-learn for ML preprocessing
import requests
import pandas as pd
import sqlite3
import json
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Configuration
API_URL = "https://fakestoreapi.com/products"
DB_FILE = "product_data.db"


def fetch_data_from_api(url):
    """Fetch data from the FakeStore API."""
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def prepare_raw_data(data):
    """Prepare raw data for storage by converting dict to JSON string."""
    df = pd.DataFrame(data)

    #Convert the 'rating' object to a JSON string for storage in DB
    df['rating'] = df['rating'].apply(json.dumps)
    return df


def transform_data(df):
    """Transform the data for analysis."""

    # Create a copy to avoid modifying the original data
    df = df.copy()

    df['price'] = pd.to_numeric(df['price'], errors='coerce')

    # Parse the 'rating' JSON string back to dict
    df['rating'] = df['rating'].apply(json.loads)

    # Extracts rate and count from the rating column
    df['rating_rate'] = df['rating'].apply(lambda x: x['rate'])
    df['rating_count'] = df['rating'].apply(lambda x: x['count'])
    df = df.drop('rating', axis=1)

    # MAke category readable for ML. Turn category string values into columns with numeric(bools)
    category_dummies = pd.get_dummies(df['category'], prefix='category')
    df = pd.concat([df, category_dummies], axis=1)

    return df


def prepare_for_ml(df):
    """Prepare the data for machine learning."""
    numeric_features = ['price', 'rating_rate', 'rating_count']
    category_features = [col for col in df.columns if col.startswith('category_')]
    features = numeric_features + category_features

    X = df[features]

    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns)

    for col in X_scaled.columns:
        df[f'scaled_{col}'] = X_scaled[col]

    return df


def store_data(df, table_name, db_file):
    """Store the data in a SQLite database using pandas."""
    conn = sqlite3.connect(db_file)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
    print(f"Data successfully stored in table '{table_name}'")


def main():
    # Fetch data
    raw_data = fetch_data_from_api(API_URL)

    # Prepare and store raw data
    raw_df = prepare_raw_data(raw_data)
    store_data(raw_df, 'raw_products', DB_FILE)

    # Transform data
    transformed_df = transform_data(raw_df)
    store_data(transformed_df, 'transformed_products', DB_FILE)

    # Prepare for ML
    ml_ready_data = prepare_for_ml(transformed_df)

    # Store ML-ready data
    store_data(ml_ready_data, 'ml_ready_products', DB_FILE)

    print("Data import and preparation completed successfully.")
    print(f"Data stored in {DB_FILE}")


if __name__ == "__main__":
    main()