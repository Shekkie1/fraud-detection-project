import matplotlib.pyplot as plt
import seaborn as sns

def fraud_distribution(df):

    plt.figure(figsize=(8,5))

    sns.countplot(
        x="Class",
        data=df
    )

    plt.title("Fraud Distribution")

    plt.show()