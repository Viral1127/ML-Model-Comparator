import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_csv(file):
    """Load CSV file into a Pandas DataFrame."""
    return pd.read_csv(file)

def plot_correlation_matrix(df):
    """Generate and return a correlation matrix heatmap figure."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Feature Correlation Matrix")
    return fig
