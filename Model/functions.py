from sklearn.model_selection import train_test_split


def split_train_test(
    df,
    target_col,
    test_size=0.2,
    random_state=42
):
    """
    Split a DataFrame into training and testing sets based on a target column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing features and the target.
    target_col : str
        Name of the target column in df.
    test_size : float, optional
        Proportion of the dataset to include in the test split (default=0.2).
    random_state : int, optional
        Seed used by the random number generator (default=42) for reproducibility.

    Returns
    -------
    X_train : pandas.DataFrame
        Training feature matrix.
    X_test : pandas.DataFrame
        Testing feature matrix.
    y_train : pandas.Series
        Training target values.
    y_test : pandas.Series
        Testing target values.
    """
    # Separate features and target
    X = df.loc[:, df.columns != target_col]
    y = df[target_col]

    # Perform train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


import xgboost as xgb
from sklearn.metrics import classification_report

def train_and_report_xgb(
    X_train,
    y_train,
    X_test,
    y_test,
    n_estimators=200,
    learning_rate=0.1,
    eval_metric='mlogloss',
    **kwargs
):
    """
    Train an XGBoost classifier and print a classification report.

    Parameters
    ----------
    X_train, y_train : training data
    X_test, y_test   : test data
    n_estimators     : number of trees
    learning_rate    : learning rate
    eval_metric      : evaluation metric for training
    **kwargs         : extra XGBClassifier args

    Returns
    -------
    model  : trained XGBClassifier
    y_pred : predicted labels
    report : classification report text
    """
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        eval_metric=eval_metric,
        **kwargs
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)
    return model, y_pred, report