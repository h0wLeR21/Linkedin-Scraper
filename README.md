# LinkedIn Sentiment Analysis 📊

A comprehensive sentiment analysis toolkit for LinkedIn posts and comments using NLTK's VADER sentiment analyzer.

## 🚀 Features

- **Sentiment Analysis** for both posts and comments
- **Multiple output formats**: Long CSV and Nested JSON
- **Comprehensive visualizations** with 10+ professional charts
- **Batch processing** for multiple CSV files
- **URN-based linking** between posts and comments


## 🛠️ Installation

```bash
# Install required packages
pip install pandas nltk matplotlib seaborn

# Download NLTK data
python -m nltk.downloader vader_lexicon punkt stopwords
```

## 📊 Visualizations

1. Sentiment Distribution (Posts & Comments)
2. Sentiment Score Distribution
3. Post vs Comment Sentiment Heatmap
4. Sentiment Match Analysis
5. Engagement Metrics by Sentiment
6. Reaction Types Distribution
7. Posts Over Time
8. Top Authors Analysis
9. Text Length Analysis
10. Correlation Matrix

## 🔗 URN Linking

Comments are automatically linked to posts using LinkedIn URN IDs:
- Post URN: `urn:li:activity:7357944725993086976`
- Comment file: `linkedin-7357944725993086976.csv`

## 📝 Input Data Format

### Posts CSV
Required columns:
- `text` - Post content
- `full_urn` - LinkedIn URN for linking

### Comments CSV
Required columns:
- `Comment` - Comment text
- Filename must contain URN: `linkedin-[URN_ID].csv`

## 🎯 Key Features

- **Text Preprocessing**: Lowercasing, URL removal, mention cleaning, stop words removal
- **VADER Sentiment**: Specialized for social media text
- **Batch Processing**: Handles multiple files automatically
- **Error Handling**: Graceful failures with detailed logging
- **Flexible Output**: CSV for analysis, JSON for programming

## 📊 Example Results

```python
# Long Format CSV
post_urn_id | post_text | post_sentiment_label | comment_text | comment_sentiment_label
7357944...  | "Hiring..." | Positive | "Great!" | Positive
7357944...  | "Hiring..." | Positive | "Thanks" | Positive

# Nested JSON
{
  "post_urn_id": "7357944...",
  "post_sentiment": {"label": "Positive", "compound": 0.67},
  "comment_stats": {"total_comments": 15, "positive_comments": 10},
  "comments": [...]
}
```
