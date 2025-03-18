import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

# Set default figure size and DPI for better quality
plt.rcParams["figure.figsize"] = (16, 10)  # Larger default figure size
plt.rcParams["figure.dpi"] = 150  # Higher DPI for better resolution
plt.rcParams["savefig.dpi"] = 300  # Even higher DPI for saved figures


def get_base_dir():
    """Get the base directory for the project."""
    # Get the current script directory
    script_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    return script_dir


def create_directory_if_not_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def plot_categorical_distribution(df, categorical_cols):
    """Plot distribution of categorical variables."""
    base_dir = get_base_dir()
    output_dir = os.path.join(base_dir, "reports", "figures", "distribution")
    create_directory_if_not_exists(output_dir)

    for col in categorical_cols:
        try:
            plt.figure(figsize=(16, 10))
            value_counts = df[col].value_counts().sort_values(ascending=False)

            # If there are too many categories, show only top 10
            if len(value_counts) > 10:
                value_counts = value_counts.head(10)
                plt.title(f"Top 10 values in {col}", fontsize=20)
            else:
                plt.title(f"Distribution of {col}", fontsize=20)

            sns.barplot(x=value_counts.index, y=value_counts.values)
            plt.xticks(rotation=45, ha="right", fontsize=14)
            plt.yticks(fontsize=14)
            plt.xlabel(col, fontsize=16)
            plt.ylabel("Count", fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"distribution_{col}.png"))
        except Exception as e:
            print(f"Error plotting distribution for {col}: {str(e)}")
        finally:
            plt.close("all")


def plot_score_histograms(df, score_columns, score_labels):
    """Plot histograms for satisfaction and moderation scores."""
    base_dir = get_base_dir()
    output_dir = os.path.join(base_dir, "reports", "figures", "histogram")
    create_directory_if_not_exists(output_dir)

    for col in score_columns:
        if col in df.columns:
            try:
                plt.figure(figsize=(16, 10))

                # Create histogram
                sns.histplot(df[col], kde=True, bins=10)

                plt.title(
                    f"Histogram of {score_labels[col]} Ratings",
                    fontsize=20,
                    fontweight="bold",
                )
                plt.xlabel(f"{score_labels[col]} (1-10 Scale)", fontsize=16)
                plt.ylabel("Frequency", fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.tight_layout()

                plt.savefig(os.path.join(output_dir, f"histogram_of_{col}.png"))
            except Exception as e:
                print(f"Error plotting histogram for {col}: {str(e)}")
            finally:
                plt.close("all")


def plot_sentiment_distribution(df, col):
    """Plot sentiment distribution for a text column."""
    base_dir = get_base_dir()
    output_dir = os.path.join(base_dir, "reports", "figures", "sentiment")
    create_directory_if_not_exists(output_dir)

    sentiment_col = f"{col}_sentiment"
    sentiment_counts = df[sentiment_col].value_counts()

    if not sentiment_counts.empty:
        try:
            plt.figure(figsize=(16, 10))

            # Create bar plot
            sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)

            feedback_type = col.replace("_", " ").title()
            plt.title(
                f"Sentiment Analysis of {feedback_type} Responses",
                fontsize=20,
                fontweight="bold",
            )
            plt.xlabel("Sentiment Category", fontsize=16)
            plt.ylabel("Number of Responses", fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.tight_layout()

            plt.savefig(os.path.join(output_dir, f"sentiment_in_{col}.png"))
        except Exception as e:
            print(f"Error plotting sentiment distribution for {col}: {str(e)}")
        finally:
            plt.close("all")


def plot_polarity_distribution(df, col):
    """Plot polarity distribution for a text column."""
    base_dir = get_base_dir()
    output_dir = os.path.join(base_dir, "reports", "figures", "polarity")
    create_directory_if_not_exists(output_dir)

    polarity_col = f"{col}_polarity"

    try:
        plt.figure(figsize=(16, 10))

        if df[polarity_col].notna().any():
            sns.histplot(df[polarity_col].dropna(), kde=True)
            plt.title(f"Polarity Distribution for {col}", fontsize=20)
            plt.xlabel("Polarity", fontsize=16)
            plt.ylabel("Frequency", fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"polarity_{col}.png"))
    except Exception as e:
        print(f"Error plotting polarity distribution for {col}: {str(e)}")
    finally:
        plt.close("all")


def generate_wordcloud(df, col):
    """Generate word cloud for a text column."""
    base_dir = get_base_dir()
    output_dir = os.path.join(base_dir, "reports", "figures", "wordcloud")
    create_directory_if_not_exists(output_dir)

    if df[col].count() > 0:
        try:
            all_text = " ".join([str(text) for text in df[col].dropna()])

            wordcloud = WordCloud(
                width=1600,
                height=1000,
                background_color="white",
                max_words=150,
                min_font_size=10,
                max_font_size=200,
            ).generate(all_text)

            plt.figure(figsize=(20, 12))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.title(f"Word Cloud for {col}", fontsize=24)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"wordcloud_{col}.png"))
        except Exception as e:
            print(f"Error generating word cloud for {col}: {str(e)}")
        finally:
            plt.close("all")


def plot_channel_preferences(df):
    """Analyze and plot channel preferences."""
    base_dir = get_base_dir()
    output_dir = os.path.join(base_dir, "reports", "figures", "channel_analysis")
    create_directory_if_not_exists(output_dir)

    if "useful_channels" in df.columns:
        try:
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

            plt.figure(figsize=(18, 12))

            top_channels = dict(channel_counts.most_common(10))
            sns.barplot(x=list(top_channels.keys()), y=list(top_channels.values()))

            plt.title(
                "Top 10 Most Popular Discord Channels",
                fontsize=24,
                fontweight="bold",
            )
            plt.xlabel("Channel Name", fontsize=18)
            plt.ylabel("Number of Mentions", fontsize=18)
            plt.xticks(rotation=45, ha="right", fontsize=16)
            plt.yticks(fontsize=16)  # Larger tick labels
            plt.tight_layout()

            plt.savefig(os.path.join(output_dir, "most_popular_channels.png"))
        except Exception as e:
            print(f"Error plotting channel preferences: {str(e)}")
        finally:
            plt.close("all")


def plot_sentiment_by_category(df, text_columns, categorical_columns):
    """Plot sentiment by categorical variables."""
    base_dir = get_base_dir()
    output_dir = os.path.join(base_dir, "reports", "figures", "sentiment_by_category")
    create_directory_if_not_exists(output_dir)

    for text_col in text_columns:
        sentiment_col = f"{text_col}_sentiment"

        if sentiment_col in df.columns and df[sentiment_col].notna().any():
            for cat_col in categorical_columns:
                if cat_col in df.columns and df[cat_col].nunique() < 10:
                    # Create a crosstab
                    cross_data = df[[cat_col, sentiment_col]].dropna()

                    if not cross_data.empty and cross_data[sentiment_col].nunique() > 1:
                        try:
                            # Create figure with normal layout
                            plt.figure(figsize=(18, 12))

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
                                fontsize=24,
                                fontweight="bold",
                            )
                            plt.xlabel(f"{category_name}", fontsize=18)
                            plt.ylabel("Proportion of Responses", fontsize=18)
                            plt.legend(
                                title="Sentiment Type",
                                fontsize=16,
                                title_fontsize=18,
                            )
                            plt.xticks(rotation=45, ha="right", fontsize=16)
                            plt.yticks(fontsize=16)
                            plt.tight_layout()

                            plt.savefig(
                                os.path.join(
                                    output_dir,
                                    f"sentiment_in_{text_col}_by_{cat_col}.png",
                                )
                            )
                        except Exception as e:
                            print(
                                f"Error plotting sentiment by category for {text_col} and {cat_col}: {str(e)}"
                            )
                        finally:
                            plt.close("all")


def plot_satisfaction_vs_sentiment(df, text_columns):
    """Plot relationship between satisfaction score and sentiment."""
    base_dir = get_base_dir()
    output_dir = os.path.join(
        base_dir, "reports", "figures", "satisfaction_vs_sentiment"
    )
    create_directory_if_not_exists(output_dir)

    for text_col in text_columns:
        polarity_col = f"{text_col}_polarity"

        if polarity_col in df.columns and "satisfaction_score" in df.columns:
            # Filter out rows with missing values
            scatter_data = df[["satisfaction_score", polarity_col]].dropna()

            if not scatter_data.empty:
                try:
                    plt.figure(figsize=(16, 12))

                    # Create scatter plot with larger markers
                    sns.scatterplot(
                        x="satisfaction_score",
                        y=polarity_col,
                        data=scatter_data,
                        s=150,
                    )

                    # Add regression line
                    sns.regplot(
                        x="satisfaction_score",
                        y=polarity_col,
                        data=scatter_data,
                        scatter=False,
                        line_kws={"color": "red", "lw": 3},
                    )

                    # Create more descriptive title and labels
                    feedback_type = text_col.replace("_", " ").title()

                    plt.title(
                        f"Relationship: Overall Satisfaction vs. {feedback_type} Sentiment",
                        fontsize=24,
                        fontweight="bold",
                    )
                    plt.xlabel("Overall Satisfaction Score (1-10)", fontsize=18)
                    plt.ylabel(
                        f"Sentiment Polarity in {feedback_type}",
                        fontsize=18,
                    )
                    plt.xticks(fontsize=16)
                    plt.yticks(fontsize=16)
                    plt.tight_layout()

                    plt.savefig(
                        os.path.join(
                            output_dir,
                            f"satisfaction_vs_{text_col}_sentiment.png",
                        )
                    )
                except Exception as e:
                    print(
                        f"Error plotting satisfaction vs sentiment for {text_col}: {str(e)}"
                    )
                finally:
                    plt.close("all")


def plot_participation_vs_satisfaction(df):
    """Plot participation frequency vs satisfaction."""
    base_dir = get_base_dir()
    output_dir = os.path.join(
        base_dir, "reports", "figures", "participation_vs_satisfaction"
    )
    create_directory_if_not_exists(output_dir)

    if "participation_frequency" in df.columns and "satisfaction_score" in df.columns:
        try:
            plt.figure(figsize=(18, 12))

            # Create boxplot with larger line width
            sns.boxplot(
                x="participation_frequency",
                y="satisfaction_score",
                data=df,
                linewidth=2.5,
            )

            # Create more descriptive title and labels
            plt.title(
                "How Participation Frequency Affects Overall Satisfaction",
                fontsize=24,
                fontweight="bold",
            )
            plt.xlabel("Frequency of Participation", fontsize=18)
            plt.ylabel("Satisfaction Score (1-10)", fontsize=18)
            plt.xticks(rotation=45, ha="right", fontsize=16)
            plt.yticks(fontsize=16)
            plt.tight_layout()

            plt.savefig(
                os.path.join(output_dir, "participation_frequency_vs_satisfaction.png")
            )
        except Exception as e:
            print(f"Error plotting participation vs satisfaction: {str(e)}")
        finally:
            plt.close("all")
