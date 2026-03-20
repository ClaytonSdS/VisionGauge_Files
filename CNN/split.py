import pandas as pd
from sklearn.model_selection import train_test_split

def split_train_valid(dataset:pd.DataFrame, 
                      feature_column:str, 
                      target_column:str, 
                      train_size:float=0.7):

    valid_size = 1 - train_size

    X_train, X_valid, y_train, y_valid = train_test_split(
        dataset[feature_column], dataset[target_column], test_size=valid_size, random_state=42
    )

    # Converter para DataFrame (X, y)
    train_df = pd.concat([X_train, y_train], axis=1)
    valid_df = pd.concat([X_val, y_val], axis=1)

    return train_df, valid_df

def split_train_valid_test(
    dataset: pd.DataFrame,
    feature_column: str,
    target_column: str,
    train_size: float = 0.7,
    valid_size: float = 0.15
):

    test_size = 1 - train_size - valid_size

    # SPLIT 1
    X_train, X_temp, y_train, y_temp = train_test_split(
        dataset[feature_column],
        dataset[target_column],
        test_size=(valid_size + test_size),
        random_state=42
    )

    # proporção do test dentro do temp
    test_ratio = test_size / (valid_size + test_size)

    # SPLIT 2
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=test_ratio,
        random_state=42
    )

    # Converter para DataFrame (X, y)
    train_df = pd.concat([X_train, y_train], axis=1)
    valid_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    return train_df, valid_df, test_df
   