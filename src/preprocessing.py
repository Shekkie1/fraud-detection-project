from sklearn.preprocessing import StandardScaler


def preprocess_data(df):
    """
    Split dataset into features and target.
    Fit scaler and return it.
    """

    X = df.drop(
        "Class",
        axis=1
    )

    y = df["Class"]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return (
        X_scaled,
        y,
        scaler
    )