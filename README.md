# CaseAssignment

This script fetches product data from the FakeStore API, processes it, and prepares it for machine learning tasks.

## Features

- Fetches product data from Fakestore API
- Stores the raw data in a SQLite database
- Transforms and prepares the data for machine learning
- Creates three database tables: raw, transformed, and ML-ready data

## Requirements

- Python 3.7+
- Required Python packages:
  - requests
  - pandas
  - scikit-learn

## Installation

1. Clone this repository or download the script.
2. Install the required packages:
   ```
   pip install requests pandas scikit-learn
   ```

## Usage

Run the script using Python:

```
python data_import_and_preparation.py
```

The script will:
1. Fetch data from the FakeStore API
2. Display a sample of the raw data
3. Process and transform the data
4. Store the data in a SQLite database named `product_data.db`

## Output

The script creates a SQLite database file named `product_data.db` with three tables:
- `raw_products`: Original data from the API
- `transformed_products`: Data after initial transformations
- `ml_ready_products`: Fully prepared data for machine learning
