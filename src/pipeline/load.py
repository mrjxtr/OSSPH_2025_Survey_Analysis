import os
import pandas as pd


class Printer:
    def __init__(self, df):
        self.df = df

    def print_data(self):
        """Print data summary and save processed data."""
        # Print basic information about the dataframe
        self._print_summary()

        # Save the processed dataframe
        self._save_processed_data()

    def _print_summary(self):
        """Print summary information about the dataframe."""
        print("Preview of the dataset:")
        print(self.df.head())

        print(f"\nDataset shape: {self.df.shape}")

        print("\nDataset info:")
        self.df.info()

        # Print summary statistics for numerical columns
        numerical_columns = self.df.select_dtypes(
            include=["int", "float"]
        ).columns
        numerical_columns = [
            col for col in numerical_columns if not col.endswith("_binary")
        ]
        numerical_columns = [
            col for col in numerical_columns if not col.endswith("_subjectivity")
        ]
        numerical_columns = [
            col for col in numerical_columns if not col.endswith("_polarity")
        ]
        numerical_columns = [
            col for col in numerical_columns if not col.endswith("_subjectivity")
        ]

        if len(numerical_columns) > 0:
            print("\nNumerical data summary:")
            print(self.df[numerical_columns].describe())

        # Print unique values for categorical columns
        categorical_columns = self.df.select_dtypes(
            include=["category"]
        ).columns

        for col in categorical_columns:
            print(f"\nUnique values in {col}:")
            print(self.df[col].unique())

    def _save_processed_data(self):
        """Save the processed dataframe to CSV."""
        # Get the base directory (project root)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(os.path.dirname(current_dir))

        # Create directory if it doesn't exist
        output_dir = os.path.join(base_dir, "src", "data", "processed")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save to CSV
        output_path = os.path.join(output_dir, "survey_with_sentiment.csv")
        self.df.to_csv(output_path, index=False)

        print(f"\nProcessed data saved to '{output_path}'")
