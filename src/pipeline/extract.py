import os
import pandas as pd


class Extractor:
    def __init__(self):
        self.df = None
        self.file_path = None

    def extract(self):
        """Extract data from the Excel file."""
        # Set the file path
        self._set_file_path()

        # Read the Excel file
        self._read_excel_file()

        return self.df

    def _set_file_path(self):
        """Set the file path to the Excel file."""
        # Get the base directory (project root)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(os.path.dirname(current_dir))

        self.file_path = os.path.join(
            base_dir,
            "src",
            "data",
            "OSSPH_Discord_Server_Survey_Responses_2025.xlsx",
        )

    def _read_excel_file(self):
        """Read the Excel file into a pandas DataFrame."""
        try:
            self.df = pd.read_excel(self.file_path)
            print(f"Successfully loaded data from {self.file_path}")
            print(f"Number of rows: {len(self.df)}")
        except FileNotFoundError:
            print(f"Error: File not found at {self.file_path}")
            self.df = pd.DataFrame()  # Create empty DataFrame
        except Exception as e:
            print(f"Error loading file: {str(e)}")
            self.df = pd.DataFrame()  # Create empty DataFrame
