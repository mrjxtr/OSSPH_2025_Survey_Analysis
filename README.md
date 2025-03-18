# OSSPH 2025 Discord Server Survey Analysis

This project analyzes survey responses from the OSSPH Discord Server for 2025.
It processes the data, performs sentiment analysis, and generates various visualizations to understand user satisfaction, participation patterns, and feedback.

> This document was automatically generated with AI and may not be 100% accurate.
> Please check the source code for details.

## Project Structure

```plaintext
OSSPH_2025_Survey/
├── reports                                 # Generated reports and figures
│   └── figures                                 # Visualizations and processed survey data
│       ├── channel_analysis/                       # Analysis of activity per channel
│       ├── distribution/                           # Distribution of survey responses
│       ├── histogram/                              # Histograms of survey data
│       ├── participation_vs_satisfaction/          # Correlation between participation and satisfaction
│       ├── polarity/                               # Polarity analysis results
│       ├── satisfaction_vs_sentiment/              # Satisfaction vs sentiment analysis
│       ├── sentiment/                              # Overall sentiment analysis
│       ├── sentiment_by_category/                  # Sentiment grouped by category
│       ├── wordcloud/                              # Word cloud visualizations
│       └── survey_with_sentiment.csv               # Survey data with sentiment analysis results
├── src                                     # Source code
│   ├── __init__.py                             # Package init file
│   ├── main.py                                 # Main script to run analysis
│   ├── data                                    # Data handling modules
│   │   └── processed                               # Processed survey data
│   ├── EDA                                     # Exploratory Data Analysis scripts
│   │   ├── eda.py                                  # EDA logic
│   │   └── __init__.py                             # Package init file
│   ├── pipeline                                # Data processing pipeline
│   │   ├── clean.py                                # Data cleaning logic
│   │   ├── extract.py                              # Data extraction logic
│   │   ├── __init__.py                             # Package init file
│   │   └── load.py                                 # Data loading logic
│   └── plots                                   # Plot generation scripts
│       ├── __init__.py                             # Package init file
│       └── plots.py                                # Plot creation logic
├── LICENSE                                 # Project license
├── notebooks                               # Jupyter notebooks for analysis
├── pyproject.toml                          # Project dependencies and configuration
├── README.md                               # Project overview and setup instructions
├── requirements.txt                        # Python package dependencies
└── SECURITY.md                             # Security guidelines
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
