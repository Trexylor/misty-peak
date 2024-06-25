import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from contextlib import contextmanager
import time

# Set up logging for efficiency monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@contextmanager
def timer(name):
    """Context manager to measure time of execution."""
    start_time = time.time()
    yield
    end_time = time.time()
    logger.info(f"[{name}] done in {end_time - start_time:.2f} s")

def load_data(file_path):
    """Load data from a CSV file."""
    with timer('Loading data'):
        data = pd.read_csv(file_path)
    return data

def process_data(data):
    """Process the data: fill missing values, encode categorical variables."""
    with timer('Processing data'):
        data = data.fillna(data.median())
        data = pd.get_dummies(data, drop_first=True)
    return data

def visualize_data(data):
    """Visualize the data using seaborn pairplot."""
    with timer('Visualizing data'):
        sns.pairplot(data.sample(500))  # Sample for quicker visualization
        plt.show()

def main():
    file_path = 'path/to/your/dataset.csv'
    
    data = load_data(file_path)
    processed_data = process_data(data)
    visualize_data(processed_data)

    logger.info("Data processing and visualization complete.")

if __name__ == "__main__":
    main()
