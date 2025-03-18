import os

import matplotlib
import nltk

from pipeline.clean import Cleaner
from pipeline.extract import Extractor
from pipeline.load import Printer
from plots.plots import *

# Set the backend to 'Agg' to prevent windows from popping up
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Configure matplotlib to avoid warnings about too many open figures
matplotlib.rcParams["figure.max_open_warning"] = 50


def download_nltk_resources() -> None:
    """Download required NLTK resources."""
    nltk.download("stopwords")
    nltk.download("punkt")


def generate_plots(df) -> None:
    """Generate all plots for the analysis."""
    # Define categorical columns
    categorical_columns = [
        "participation_frequency",
        "primary_usage",
        "feeling_included",
        "event_participation",
        "volunteer_interest",
    ]

    # Define text columns for sentiment analysis
    text_columns = [
        "positive_feedback",
        "improvement_feedback",
        "additional_feedback",
        "desired_content",
        "channel_suggestions",
    ]

    # Define score columns and labels
    score_columns = ["satisfaction_score", "moderation_score"]
    score_labels = {
        "satisfaction_score": "Overall Satisfaction with Discord Server",
        "moderation_score": "Rating of Moderation and Server Atmosphere",
    }

    # Plot categorical distributions
    print("Plotting categorical distributions...")
    plot_categorical_distribution(df, categorical_columns)

    # Plot score histograms
    print("Plotting score histograms...")
    plot_score_histograms(df, score_columns, score_labels)

    # For each text column, generate sentiment plots
    print("Plotting sentiment distributions...")
    for col in text_columns:
        if col in df.columns:
            plot_sentiment_distribution(df, col)
            plot_polarity_distribution(df, col)
            generate_wordcloud(df, col)

    # Plot channel preferences
    print("Plotting channel preferences...")
    plot_channel_preferences(df)

    # Plot sentiment by category
    print("Plotting sentiment by category...")
    plot_sentiment_by_category(df, text_columns, categorical_columns)

    # Plot satisfaction vs sentiment
    print("Plotting satisfaction vs sentiment...")
    plot_satisfaction_vs_sentiment(df, text_columns)

    # Plot participation vs satisfaction
    print("Plotting participation vs satisfaction...")
    plot_participation_vs_satisfaction(df)


def main() -> None:
    print("Starting OSSPH 2025 Survey Analysis...")

    # Download NLTK resources
    download_nltk_resources()

    # Extract data
    data_extractor = Extractor()
    data_extractor.extract()

    # Clean and transform data
    data_cleaner = Cleaner(data_extractor.df)
    data_cleaner.clean()

    # Print summary and save processed data
    data_printer = Printer(data_cleaner.df)
    data_printer.print_data()

    # Generate plots
    generate_plots(data_cleaner.df)

    # Make sure all figures are closed
    plt.close("all")

    # Get the base directory (project root)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    output_dir = os.path.join(base_dir, "reports", "figures")

    print(
        f"\nEDA and Sentiment Analysis completed. Results saved to '{output_dir}' directory."
    )


if __name__ == "__main__":
    main()
