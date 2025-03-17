# OSSPH 2025 Discord Server Survey Analysis

This project analyzes survey responses from the OSSPH Discord Server for 2025. It processes the data, performs sentiment analysis, and generates various visualizations to understand user satisfaction, participation patterns, and feedback.

> This document was automatically generated with AI and may not be 100% accurate. Please check the source code for details.

## Project Structure

```plaintext
OSSPH_2025_Survey/
├── reports/
│   └── figures/           # Generated visualizations and CSV output
├── src/
│   ├── data/              # Input data files
│   ├── EDA/               # Exploratory Data Analysis scripts (meant for testing)
│   ├── pipeline/          # Data processing pipeline
│   │   ├── extract.py     # Data extraction module
│   │   ├── clean.py       # Data cleaning and transformation module
│   │   └── load.py        # Data output and printing module
│   ├── plots/             # Visualization modules
│   │   └── plots.py       # Plot generation functions
│   └── main.py            # Main execution script
```

## Features

- **Data Processing Pipeline**: Extracts, cleans, and transforms survey data
- **Sentiment Analysis**: Analyzes sentiment in text responses
- **Visualization**: Generates various plots to visualize survey results
  - Distribution of categorical variables
  - Histograms of satisfaction and moderation scores
  - Sentiment analysis visualizations
  - Word clouds for text responses
  - Channel preference analysis
  - Relationship between sentiment and categorical variables
  - Correlation between satisfaction and sentiment
  - Impact of participation frequency on satisfaction

## Requirements

- Python 3.13+
- pandas
- numpy
- matplotlib
- seaborn
- nltk
- textblob
- wordcloud

## Usage

To run the analysis:

```bash
cd src
python main.py
```

The script will:

1. Download required NLTK resources
2. Extract data from the Excel file
3. Clean and transform the data
4. Generate visualizations
5. Save processed data to CSV

## Visualization Outputs

The analysis generates the following types of visualizations:

- **Distribution Plots**: Shows the distribution of categorical variables
- **Histograms**: Displays the distribution of satisfaction and moderation scores
- **Sentiment Analysis**: Visualizes sentiment in text responses
- **Word Clouds**: Generates word clouds for text responses
- **Channel Analysis**: Shows the most popular Discord channels
- **Sentiment by Category**: Analyzes sentiment across different categorical variables
- **Satisfaction vs. Sentiment**: Explores the relationship between overall satisfaction and sentiment
- **Participation vs. Satisfaction**: Examines how participation frequency affects satisfaction

All visualizations are saved in the `reports/figures/` directory.
