import numpy as np
import pandas as pd
from textblob import TextBlob


class Cleaner:
    def __init__(self, df):
        self.df = df

    def clean(self):
        """Clean and transform the dataframe."""
        # Rename columns if needed
        self._rename_columns()

        # Convert appropriate columns to categorical
        self._convert_to_categorical()

        # Create binary columns from categorical responses
        self._create_binary_columns()

        # Perform sentiment analysis on text columns
        self._perform_sentiment_analysis()

        return self.df

    def _rename_columns(self):
        """Rename columns if they have special characters or need standardization."""
        # Check if columns need renaming (if they have special characters)
        if any(["?" in col or ":" in col for col in self.df.columns]):
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
            self.df.columns = new_column_names

    def _convert_to_categorical(self):
        """Convert appropriate columns to categorical type."""
        categorical_columns = [
            "participation_frequency",
            "primary_usage",
            "feeling_included",
            "event_participation",
            "volunteer_interest",
        ]

        for col in categorical_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype("category")

    def _create_binary_columns(self):
        """Create binary columns from categorical responses."""
        # For feeling_included
        if "feeling_included" in self.df.columns:
            self.df["feeling_included_binary"] = (
                self.df["feeling_included"]
                .apply(lambda x: 1 if str(x).strip() == "Yes" else 0)
                .astype(int)
            )

        # For event_participation
        if "event_participation" in self.df.columns:
            self.df["event_participation_binary"] = (
                self.df["event_participation"]
                .apply(
                    lambda x: 1
                    if str(x).strip() == "Yes"
                    else (0.5 if str(x).strip() == "Sometimes" else 0)
                )
                .astype(float)
            )

        # For volunteer_interest
        if "volunteer_interest" in self.df.columns:
            self.df["volunteer_interest_binary"] = (
                self.df["volunteer_interest"]
                .apply(lambda x: 1 if str(x).strip() == "Yes" else 0)
                .astype(int)
            )

    def _perform_sentiment_analysis(self):
        """Perform sentiment analysis on text columns."""
        text_columns = [
            "positive_feedback",
            "improvement_feedback",
            "additional_feedback",
            "desired_content",
            "channel_suggestions",
        ]

        for col in text_columns:
            if col in self.df.columns:
                # Create new columns for sentiment metrics
                self.df[f"{col}_polarity"] = np.nan
                self.df[f"{col}_subjectivity"] = np.nan
                self.df[f"{col}_sentiment"] = np.nan

                # Apply sentiment analysis
                sentiments = self.df[col].apply(self._analyze_sentiment)
                (
                    self.df[f"{col}_polarity"],
                    self.df[f"{col}_subjectivity"],
                    self.df[f"{col}_sentiment"],
                ) = zip(*sentiments)

    def _analyze_sentiment(self, text):
        """Analyze sentiment of text using TextBlob."""
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
