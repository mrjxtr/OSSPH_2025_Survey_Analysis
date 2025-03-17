"""
! RUN THE main.py SCRIPT

This script is only meant for initial EDA and testing
Although this will work, it might not produce the same results
as running the main.py script
"""

import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from textblob import TextBlob
from wordcloud import WordCloud

# Download NLTK resources
nltk.download("stopwords")
nltk.download("punkt")

# Read the Excel file
script_dir = os.getcwd()
file_path = os.path.join(
    script_dir, "../data/OSSPH_Discord_Server_Survey_Responses_2025.xlsx"
)
df = pd.read_excel(file_path)

# Check the first few rows to understand the data structure
print("Preview of the dataset:")
df.head()
print("Original column names:")
for col in df.columns:
    print(f"- {col}")

# Check the shape of the dataframe (rows, columns)
print(f"Dataset shape: {df.shape}")

# Rename columns by position since the special characters are causing issues
new_column_names = [
    "timestamp",
    "satisfaction_score",
    "participation_frequency",
    "primary_usage",
    "feeling_included",
    "moderation_score",
    "desired_content",
    "useful_channels",
    "channel_suggestions",
    "event_participation",
    "positive_feedback",
    "improvement_feedback",
    "additional_feedback",
    "volunteer_interest",
]
df.columns = new_column_names

# Check basic information (data types, non-null values)
print("\nDataset info:")
df.info()

# Convert appropriate columns to categorical
categorical_columns = [
    "participation_frequency",
    "primary_usage",
    "feeling_included",
    "event_participation",
    "volunteer_interest",
]

for col in categorical_columns:
    if col in df.columns:
        df[col] = df[col].astype("category")

print("\nAfter converting to categorical types:")
df.info()

# Create binary columns from categorical responses where appropriate
# For feeling_included (based on your data: "Yes", "Sometimes", "No")
# After converting to categorical types, check the unique values in feeling_included
if "feeling_included" in df.columns:
    print("\nUnique values in feeling_included:")
    print(df["feeling_included"].unique())

    # Create binary columns from categorical responses where appropriate
    # For feeling_included - check exact values and case sensitivity
    df["feeling_included_binary"] = (
        df["feeling_included"]
        .apply(lambda x: 1 if str(x).strip() == "Yes" else 0)
        .astype(int)
    )  # Explicitly set the data type to int

    # Verify the transformation worked
    print("\nCross-tabulation of feeling_included and binary version:")
    print(pd.crosstab(df["feeling_included"], df["feeling_included_binary"]))
    print(
        f"Data type of feeling_included_binary: {df['feeling_included_binary'].dtype}"
    )

# For event_participation (based on your data: "Yes", "Sometimes", "No")
if "event_participation" in df.columns:
    print("\nUnique values in event_participation:")
    print(df["event_participation"].unique())

    df["event_participation_binary"] = (
        df["event_participation"]
        .apply(
            lambda x: (
                1
                if str(x).strip() == "Yes"
                else (0.5 if str(x).strip() == "Sometimes" else 0)
            )
        )
        .astype(float)
    )  # Explicitly set the data type to float

    # Verify the transformation worked
    print("\nCross-tabulation of event_participation and binary version:")
    print(pd.crosstab(df["event_participation"], df["event_participation_binary"]))
    print(
        f"Data type of event_participation_binary: {df['event_participation_binary'].dtype}"
    )

# For volunteer_interest (based on your data's various responses)
if "volunteer_interest" in df.columns:
    print("\nUnique values in volunteer_interest:")
    print(df["volunteer_interest"].unique())

    df["volunteer_interest_binary"] = (
        df["volunteer_interest"]
        .apply(lambda x: 1 if str(x).strip() == "Yes" else 0)
        .astype(int)
    )  # Explicitly set the data type to int

    # Verify the transformation worked
    print("\nCross-tabulation of volunteer_interest and binary version:")
    print(pd.crosstab(df["volunteer_interest"], df["volunteer_interest_binary"]))
    print(
        f"Data type of volunteer_interest_binary: {df['volunteer_interest_binary'].dtype}"
    )

# Check basic statistics for numerical columns
numerical_columns = df.select_dtypes(include=[int, float]).columns
numerical_columns = [col for col in numerical_columns if not col.endswith("_binary")]
print("\nNumerical data summary:")
df[numerical_columns].describe()

# Distribution of categorical variables
print("\nDistribution of categorical variables:")
categorical_cols = df.select_dtypes(include=["category"]).columns
categorical_cols = [col for col in categorical_cols if not col.endswith("_binary")]

for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    value_counts = df[col].value_counts().sort_values(ascending=False)

    # If there are too many categories, show only top 10
    if len(value_counts) > 10:
        value_counts = value_counts.head(10)
        plt.title(f"Top 10 values in {col}")
    else:
        plt.title(f"Distribution of {col}")

    sns.barplot(x=value_counts.index, y=value_counts.values)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"../../reports/figures/distribution/distribution_{col}.png")
    plt.show()
    plt.close()

# Analyze satisfaction and moderation scores
print("\nAnalyzing satisfaction and moderation scores:")
score_columns = ["satisfaction_score", "moderation_score"]
score_labels = {
    "satisfaction_score": "Overall Satisfaction with Discord Server",
    "moderation_score": "Rating of Moderation and Server Atmosphere",
}

for col in score_columns:
    if col in df.columns:
        plt.figure(figsize=(10, 6))

        # Create histogram
        sns.histplot(df[col], kde=True, bins=10)

        plt.title(
            f"Histogram of {score_labels[col]} Ratings",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel(f"{score_labels[col]} (1-10 Scale)", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.tight_layout()

        plt.savefig(f"../../reports/figures/histogram/histogram_of_{col}.png")
        plt.show()
        plt.close()

# Define text columns for sentiment analysis based on the actual data
text_columns = [
    "positive_feedback",
    "improvement_feedback",
    "additional_feedback",
    "desired_content",
    "channel_suggestions",
]


# Sentiment Analysis
def analyze_sentiment(text):
    if pd.isna(text) or not isinstance(text, str):
        return np.nan, np.nan, np.nan

    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    subjectivity = analysis.sentiment.subjectivity

    # Determine sentiment category
    if polarity > 0.1:
        sentiment = "Positive"
    elif polarity < -0.1:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return polarity, subjectivity, sentiment


# Perform sentiment analysis on text columns
for col in text_columns:
    print(f"\nPerforming sentiment analysis on column: {col}")

    # Create new columns for sentiment metrics
    df[f"{col}_polarity"] = np.nan
    df[f"{col}_subjectivity"] = np.nan
    df[f"{col}_sentiment"] = np.nan

    # Apply sentiment analysis
    sentiments = df[col].apply(analyze_sentiment)
    df[f"{col}_polarity"], df[f"{col}_subjectivity"], df[f"{col}_sentiment"] = zip(
        *sentiments
    )

    # Visualize sentiment distribution
    sentiment_counts = df[f"{col}_sentiment"].value_counts()
    if not sentiment_counts.empty:  # Only plot if we have sentiment data
        plt.figure(figsize=(10, 6))

        # Create bar plot
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)

        feedback_type = col.replace("_", " ").title()
        plt.title(
            f"Sentiment Analysis of {feedback_type} Responses",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Sentiment Category", fontsize=12)
        plt.ylabel("Number of Responses", fontsize=12)
        plt.tight_layout()

        plt.savefig(f"../../reports/figures/sentiment/sentiment_in_{col}.png")
        plt.show()
        plt.close()

    # Visualize polarity distribution
    plt.figure(figsize=(10, 6))
    if df[f"{col}_polarity"].notna().any():  # Only plot if we have polarity data
        sns.histplot(df[f"{col}_polarity"].dropna(), kde=True)
        plt.title(f"Polarity Distribution for {col}")
        plt.tight_layout()
        plt.savefig(f"../../reports/figures/polarity/polarity_{col}.png")
    plt.show()
    plt.close()

    # Generate word cloud for this text column
    if df[col].count() > 0:
        all_text = " ".join([str(text) for text in df[col].dropna()])
        stop_words = set(stopwords.words("english"))

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
            stopwords=stop_words,
            max_words=100,
        ).generate(all_text)

        plt.figure(figsize=(12, 8))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud for {col}")
        plt.tight_layout()
        plt.savefig(f"../../reports/figures/wordcloud/wordcloud_{col}.png")
        plt.show()
        plt.close()

        # Extract common words and phrases
        words = re.findall(r"\b\w+\b", all_text.lower())
        words = [word for word in words if word not in stop_words and len(word) > 2]
        word_freq = Counter(words).most_common(20)

        print(f"\nMost common words in {col}:")
        for word, count in word_freq:
            print(f"{word}: {count}")

# Analyze channel preferences
if "useful_channels" in df.columns:
    print("\nAnalyzing channel preferences:")

    # Extract all mentioned channels
    all_channels = []
    for channels in df["useful_channels"].dropna():
        # Split by commas and other separators
        channel_list = re.split(r"[,\s]+", channels)
        # Extract channel names (those starting with #)
        for item in channel_list:
            if item.startswith("#"):
                all_channels.append(item.strip().lower())

    # Count channel mentions
    channel_counts = Counter(all_channels)

    plt.figure(figsize=(12, 8))

    top_channels = dict(channel_counts.most_common(10))
    sns.barplot(x=list(top_channels.keys()), y=list(top_channels.values()))

    plt.title("Top 10 Most Popular Discord Channels", fontsize=14, fontweight="bold")
    plt.xlabel("Channel Name", fontsize=12)
    plt.ylabel("Number of Mentions", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig("../../reports/figures/channel_analysis/most_popular_channels.png")
    plt.show()
    plt.close()

# Bivariate analysis: Sentiment vs categorical variables
for text_col in text_columns:
    sentiment_col = f"{text_col}_sentiment"

    if sentiment_col in df.columns and df[sentiment_col].notna().any():
        for cat_col in categorical_columns:
            if (
                cat_col in df.columns and df[cat_col].nunique() < 10
            ):  # Only for categorical columns with few categories
                # Create a crosstab
                cross_data = df[[cat_col, sentiment_col]].dropna()

                if not cross_data.empty and cross_data[sentiment_col].nunique() > 1:
                    # Create figure with normal layout
                    plt.figure(figsize=(12, 8))

                    # Create a crosstab of sentiment vs category
                    cross_tab = pd.crosstab(
                        cross_data[cat_col],
                        cross_data[sentiment_col],
                        normalize="index",
                    )
                    cross_tab.plot(kind="bar", stacked=True, colormap="viridis")

                    # Create more descriptive title and labels
                    feedback_type = text_col.replace("_", " ").title()
                    category_name = cat_col.replace("_", " ").title()

                    plt.title(
                        f"Sentiment in {feedback_type} by {category_name}",
                        fontsize=14,
                        fontweight="bold",
                    )
                    plt.xlabel(f"{category_name}", fontsize=12)
                    plt.ylabel("Proportion of Responses", fontsize=12)
                    plt.legend(title="Sentiment Type", fontsize=10)
                    plt.xticks(rotation=45, ha="right")
                    plt.tight_layout()

                    plt.savefig(
                        f"../../reports/figures/sentiment_by_category/sentiment_in_{text_col}_by_{cat_col}.png"
                    )
                    plt.show()
                    plt.close()

# Analyze relationship between satisfaction score and sentiment
for text_col in text_columns:
    polarity_col = f"{text_col}_polarity"

    if polarity_col in df.columns and "satisfaction_score" in df.columns:
        # Filter out rows with missing values
        scatter_data = df[["satisfaction_score", polarity_col]].dropna()

        if not scatter_data.empty:
            plt.figure(figsize=(10, 7))

            # Create scatter plot
            sns.scatterplot(x="satisfaction_score", y=polarity_col, data=scatter_data)

            # Add regression line
            sns.regplot(
                x="satisfaction_score",
                y=polarity_col,
                data=scatter_data,
                scatter=False,
                line_kws={"color": "red"},
            )

            # Create more descriptive title and labels
            feedback_type = text_col.replace("_", " ").title()

            plt.title(
                f"Relationship: Overall Satisfaction vs. {feedback_type} Sentiment",
                fontsize=14,
                fontweight="bold",
            )
            plt.xlabel("Overall Satisfaction Score (1-10)", fontsize=12)
            plt.ylabel(f"Sentiment Polarity in {feedback_type}", fontsize=12)
            plt.tight_layout()

            plt.savefig(
                f"../../reports/figures/satisfaction_vs_sentiment/satisfaction_vs_{text_col}_sentiment.png"
            )
            plt.show()
            plt.close()

# Analyze participation frequency vs satisfaction
if "participation_frequency" in df.columns and "satisfaction_score" in df.columns:
    plt.figure(figsize=(12, 8))

    # Create boxplot
    sns.boxplot(x="participation_frequency", y="satisfaction_score", data=df)

    # Create more descriptive title and labels
    plt.title(
        "How Participation Frequency Affects Overall Satisfaction",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Frequency of Participation", fontsize=12)
    plt.ylabel("Satisfaction Score (1-10)", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(
        "../../reports/figures/participation_vs_satisfaction/participation_frequency_vs_satisfaction.png"
    )
    plt.show()
    plt.close()

# Save the processed dataframe with sentiment analysis
df.to_csv("../../reports/figures/survey_with_sentiment.csv", index=False)

print(
    "\nEDA and Sentiment Analysis completed. Results saved to '../../reports/figures/' directory."
)
