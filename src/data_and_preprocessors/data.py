import pandas as pd


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the dataframe by removing outliers based on age and loan-to-income ratio.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        
    Returns:
        pd.DataFrame: Filtered dataframe.
    """
    df = df[df['person_age'] <= 60]
    df = df[df['loan_percent_income'] <= 0.45]
    return df


def load_data(TRAIN_PATH, TEST_PATH):
    """
    Loads train and test datasets, applies filtering, and separates target and IDs.
    
    Args:
        TRAIN_PATH (str): Path to the training CSV file.
        TEST_PATH (str): Path to the test CSV file.
        
    Returns:
        tuple: (test_data, train_data, target_data, idx)
            - test_data (pd.DataFrame): Test features.
            - train_data (pd.DataFrame): Train features.
            - target_data (pd.Series): Target variable (loan_status).
            - idx (pd.Series): IDs from the test dataset.
    """
    test_data = pd.read_csv(TEST_PATH)
    train_data = pd.read_csv(TRAIN_PATH)

    train_data = filter_data(train_data)

    target_data = train_data['loan_status']
    train_data = train_data.drop(columns=['loan_status'])

    idx = test_data['id']
    test_data = test_data.drop(columns=['id'])
    train_data = train_data.drop(columns=['id'])

    return test_data, train_data, target_data, idx

