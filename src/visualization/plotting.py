import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_CSV = BASE_DIR / "data" / "results" / "combined_posts_comments_long.csv"
OUTPUT_DIR = BASE_DIR / "src" / "visualization" / 'plots'

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_data():
    """Load the combined CSV data."""
    print("=" * 60)
    print("Loading Data")
    print("=" * 60)
    
    if not INPUT_CSV.exists():
        print(f"Error: File not found: {INPUT_CSV}")
        print("Please update INPUT_CSV path in the script.")
        return None
    
    df = pd.read_csv(INPUT_CSV, encoding='utf-8')
    print(f"Loaded {len(df)} rows")
    print(f"Unique posts: {df['post_urn_id'].nunique()}")
    print(f"Posts with comments: {df['has_comment'].sum()}")
    
    return df

def plot_1_sentiment_distribution(df):
    """Plot sentiment distribution for posts and comments."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Post sentiment
    post_sentiment = df.drop_duplicates('post_urn_id')['post_sentiment_label'].value_counts()
    colors = {'Positive': '#2ecc71', 'Neutral': '#95a5a6', 'Negative': '#e74c3c'}
    post_colors = [colors.get(x, '#3498db') for x in post_sentiment.index]
    
    axes[0].bar(post_sentiment.index, post_sentiment.values, color=post_colors, alpha=0.8)
    axes[0].set_title('Post Sentiment Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Sentiment', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    
    # Add percentage labels
    total_posts = post_sentiment.sum()
    for i, v in enumerate(post_sentiment.values):
        axes[0].text(i, v + 5, f'{v}\n({v/total_posts*100:.1f}%)', 
                    ha='center', va='bottom', fontweight='bold')
    
    # Comment sentiment
    comment_data = df[df['has_comment']]
    if len(comment_data) > 0:
        comment_sentiment = comment_data['comment_sentiment_label'].value_counts()
        comment_colors = [colors.get(x, '#3498db') for x in comment_sentiment.index]
        
        axes[1].bar(comment_sentiment.index, comment_sentiment.values, color=comment_colors, alpha=0.8)
        axes[1].set_title('Comment Sentiment Distribution', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Sentiment', fontsize=12)
        axes[1].set_ylabel('Count', fontsize=12)
        
        # Add percentage labels
        total_comments = comment_sentiment.sum()
        for i, v in enumerate(comment_sentiment.values):
            axes[1].text(i, v + 5, f'{v}\n({v/total_comments*100:.1f}%)', 
                        ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_sentiment_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 01_sentiment_distribution.png")
    plt.close()

def plot_2_sentiment_scores_distribution(df):
    """Plot distribution of compound sentiment scores."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Post sentiment scores
    post_data = df.drop_duplicates('post_urn_id')
    axes[0].hist(post_data['post_sentiment_compound'], bins=30, 
                color='#3498db', alpha=0.7, edgecolor='black')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Neutral (0)')
    axes[0].set_title('Post Sentiment Score Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Compound Sentiment Score', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].legend()
    
    # Comment sentiment scores
    comment_data = df[df['has_comment']]
    if len(comment_data) > 0:
        axes[1].hist(comment_data['comment_sentiment_compound'], bins=30, 
                    color='#e67e22', alpha=0.7, edgecolor='black')
        axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='Neutral (0)')
        axes[1].set_title('Comment Sentiment Score Distribution', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Compound Sentiment Score', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_sentiment_scores_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 02_sentiment_scores_distribution.png")
    plt.close()

def plot_3_post_vs_comment_sentiment(df):
    """Cross-tabulation of post vs comment sentiment."""
    comment_data = df[df['has_comment']]
    
    if len(comment_data) == 0:
        print("⚠ Skipping post vs comment plot (no comments)")
        return
    
    # Create crosstab
    crosstab = pd.crosstab(comment_data['post_sentiment_label'], 
                           comment_data['comment_sentiment_label'])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(crosstab, annot=True, fmt='d', cmap='YlOrRd', 
                cbar_kws={'label': 'Count'}, linewidths=1, linecolor='white')
    plt.title('Post Sentiment vs Comment Sentiment\n(Heatmap)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Comment Sentiment', fontsize=12)
    plt.ylabel('Post Sentiment', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_post_vs_comment_sentiment.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 03_post_vs_comment_sentiment.png")
    plt.close()

def plot_4_sentiment_match_analysis(df):
    """Analyze sentiment match between posts and comments."""
    comment_data = df[df['has_comment']]
    
    if len(comment_data) == 0 or 'sentiment_match' not in comment_data.columns:
        print("⚠ Skipping sentiment match plot (no data)")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Overall match rate
    match_counts = comment_data['sentiment_match'].value_counts()
    colors = ['#2ecc71', '#e74c3c']
    
    axes[0].pie(match_counts.values, labels=['Match', 'No Match'], autopct='%1.1f%%',
               colors=colors, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    axes[0].set_title('Overall Sentiment Match Rate', fontsize=14, fontweight='bold')
    
    # Match rate by post sentiment
    match_by_sentiment = comment_data.groupby('post_sentiment_label')['sentiment_match'].mean() * 100
    
    axes[1].bar(match_by_sentiment.index, match_by_sentiment.values, 
               color=['#2ecc71', '#95a5a6', '#e74c3c'], alpha=0.8)
    axes[1].set_title('Sentiment Match Rate by Post Sentiment', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Post Sentiment', fontsize=12)
    axes[1].set_ylabel('Match Rate (%)', fontsize=12)
    axes[1].set_ylim(0, 100)
    
    # Add percentage labels
    for i, v in enumerate(match_by_sentiment.values):
        axes[1].text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_sentiment_match_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 04_sentiment_match_analysis.png")
    plt.close()

def plot_5_engagement_metrics(df):
    """Plot engagement metrics."""
    post_data = df.drop_duplicates('post_urn_id')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Total reactions by sentiment
    reactions_by_sentiment = post_data.groupby('post_sentiment_label')['post_stats/total_reactions'].mean()
    axes[0, 0].bar(reactions_by_sentiment.index, reactions_by_sentiment.values, 
                   color=['#2ecc71', '#95a5a6', '#e74c3c'], alpha=0.8)
    axes[0, 0].set_title('Avg Total Reactions by Post Sentiment', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Average Reactions')
    
    # Comments by sentiment
    comments_by_sentiment = post_data.groupby('post_sentiment_label')['post_stats/comments'].mean()
    axes[0, 1].bar(comments_by_sentiment.index, comments_by_sentiment.values, 
                   color=['#2ecc71', '#95a5a6', '#e74c3c'], alpha=0.8)
    axes[0, 1].set_title('Avg Comments by Post Sentiment', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Average Comments')
    
    # Reposts by sentiment
    reposts_by_sentiment = post_data.groupby('post_sentiment_label')['post_stats/reposts'].mean()
    axes[1, 0].bar(reposts_by_sentiment.index, reposts_by_sentiment.values, 
                   color=['#2ecc71', '#95a5a6', '#e74c3c'], alpha=0.8)
    axes[1, 0].set_title('Avg Reposts by Post Sentiment', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Average Reposts')
    
    # Correlation between sentiment score and engagement
    axes[1, 1].scatter(post_data['post_sentiment_compound'], 
                      post_data['post_stats/total_reactions'],
                      alpha=0.6, s=50, color='#3498db')
    axes[1, 1].set_title('Sentiment Score vs Total Reactions', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Sentiment Compound Score')
    axes[1, 1].set_ylabel('Total Reactions')
    
    # Add trend line
    z = np.polyfit(post_data['post_sentiment_compound'].dropna(), 
                   post_data['post_stats/total_reactions'].dropna(), 1)
    p = np.poly1d(z)
    x_line = np.linspace(post_data['post_sentiment_compound'].min(), 
                         post_data['post_sentiment_compound'].max(), 100)
    axes[1, 1].plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '05_engagement_metrics.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 05_engagement_metrics.png")
    plt.close()

def plot_6_reaction_types(df):
    """Plot different reaction types distribution."""
    post_data = df.drop_duplicates('post_urn_id')
    
    reaction_types = ['post_stats/like', 'post_stats/celebrate', 'post_stats/support', 
                     'post_stats/love', 'post_stats/insight', 'post_stats/funny']
    
    reaction_totals = {col.replace('post_stats/', ''): post_data[col].sum() 
                       for col in reaction_types if col in post_data.columns}
    
    if not reaction_totals:
        print("⚠ Skipping reaction types plot (no data)")
        return
    
    plt.figure(figsize=(12, 6))
    colors = ['#3498db', '#f39c12', '#9b59b6', '#e74c3c', '#1abc9c', '#34495e']
    
    bars = plt.bar(reaction_totals.keys(), reaction_totals.values(), 
                   color=colors[:len(reaction_totals)], alpha=0.8)
    plt.title('Distribution of Reaction Types', fontsize=14, fontweight='bold')
    plt.xlabel('Reaction Type', fontsize=12)
    plt.ylabel('Total Count', fontsize=12)
    plt.xticks(rotation=45)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '06_reaction_types.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 06_reaction_types.png")
    plt.close()

def plot_7_posts_over_time(df):
    """Plot posts and sentiment over time."""
    post_data = df.drop_duplicates('post_urn_id')
    
    if 'post_posted_at/date' not in post_data.columns:
        print("⚠ Skipping time series plot (no date column)")
        return
    
    # Convert to datetime
    post_data['date'] = pd.to_datetime(post_data['post_posted_at/date'], errors='coerce')
    post_data = post_data.dropna(subset=['date'])
    
    if len(post_data) == 0:
        print("⚠ Skipping time series plot (no valid dates)")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Posts per day
    posts_per_day = post_data.groupby(post_data['date'].dt.date).size()
    axes[0].plot(posts_per_day.index, posts_per_day.values, marker='o', linewidth=2, color='#3498db')
    axes[0].fill_between(posts_per_day.index, posts_per_day.values, alpha=0.3, color='#3498db')
    axes[0].set_title('Posts Over Time', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Date', fontsize=12)
    axes[0].set_ylabel('Number of Posts', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Sentiment over time
    sentiment_by_date = post_data.groupby([post_data['date'].dt.date, 'post_sentiment_label']).size().unstack(fill_value=0)
    
    if not sentiment_by_date.empty:
        sentiment_by_date.plot(kind='area', stacked=True, ax=axes[1], 
                              color=['#2ecc71', '#95a5a6', '#e74c3c'], alpha=0.7)
        axes[1].set_title('Sentiment Distribution Over Time', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Date', fontsize=12)
        axes[1].set_ylabel('Number of Posts', fontsize=12)
        axes[1].legend(title='Sentiment', loc='upper left')
        axes[1].grid(True, alpha=0.3)
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '07_posts_over_time.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 07_posts_over_time.png")
    plt.close()

def plot_8_top_authors(df):
    """Plot top authors by engagement and sentiment."""
    post_data = df.drop_duplicates('post_urn_id')
    
    if 'post_author/full_name' not in post_data.columns:
        print("⚠ Skipping top authors plot (no author column)")
        return
    
    # Top 10 authors by total reactions
    top_authors = post_data.groupby('post_author/full_name')['post_stats/total_reactions'].sum().nlargest(10)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Total reactions
    axes[0].barh(range(len(top_authors)), top_authors.values, color='#3498db', alpha=0.8)
    axes[0].set_yticks(range(len(top_authors)))
    axes[0].set_yticklabels(top_authors.index, fontsize=10)
    axes[0].set_xlabel('Total Reactions', fontsize=12)
    axes[0].set_title('Top 10 Authors by Total Reactions', fontsize=14, fontweight='bold')
    axes[0].invert_yaxis()
    
    # Add value labels
    for i, v in enumerate(top_authors.values):
        axes[0].text(v + 5, i, f'{int(v)}', va='center', fontweight='bold')
    
    # Sentiment distribution for top authors
    top_author_names = top_authors.index.tolist()
    top_author_data = post_data[post_data['post_author/full_name'].isin(top_author_names)]
    
    sentiment_by_author = pd.crosstab(top_author_data['post_author/full_name'], 
                                       top_author_data['post_sentiment_label'], 
                                       normalize='index') * 100
    
    sentiment_by_author.plot(kind='barh', stacked=True, ax=axes[1], 
                            color=['#2ecc71', '#95a5a6', '#e74c3c'], alpha=0.8)
    axes[1].set_xlabel('Percentage', fontsize=12)
    axes[1].set_title('Sentiment Distribution for Top Authors', fontsize=14, fontweight='bold')
    axes[1].legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].set_xlim(0, 100)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '08_top_authors.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 08_top_authors.png")
    plt.close()

def plot_9_word_length_analysis(df):
    """Analyze text length vs sentiment."""
    post_data = df.drop_duplicates('post_urn_id')
    
    # Calculate text length
    post_data['text_length'] = post_data['post_text'].astype(str).str.len()
    
    plt.figure(figsize=(12, 6))
    
    # Box plot of text length by sentiment
    sentiment_order = ['Negative', 'Neutral', 'Positive']
    colors = ['#e74c3c', '#95a5a6', '#2ecc71']
    
    data_to_plot = [post_data[post_data['post_sentiment_label'] == sent]['text_length'].dropna() 
                    for sent in sentiment_order]
    
    box = plt.boxplot(data_to_plot, labels=sentiment_order, patch_artist=True)
    
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.title('Post Text Length by Sentiment', fontsize=14, fontweight='bold')
    plt.xlabel('Sentiment', fontsize=12)
    plt.ylabel('Text Length (characters)', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '09_text_length_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 09_text_length_analysis.png")
    plt.close()

def plot_10_correlation_matrix(df):
    """Plot correlation matrix of numerical features."""
    post_data = df.drop_duplicates('post_urn_id')
    
    # Select numerical columns
    numerical_cols = [
        'post_sentiment_compound', 'post_sentiment_positive', 
        'post_sentiment_negative', 'post_sentiment_neutral',
        'post_stats/total_reactions', 'post_stats/comments', 'post_stats/reposts'
    ]
    
    # Keep only existing columns
    numerical_cols = [col for col in numerical_cols if col in post_data.columns]
    
    if len(numerical_cols) < 2:
        print("⚠ Skipping correlation matrix (not enough numerical columns)")
        return
    
    corr_data = post_data[numerical_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix of Post Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '10_correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 10_correlation_matrix.png")
    plt.close()

def main():
    print("=" * 60)
    print("Creating Sentiment Analysis Visualizations")
    print("=" * 60)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    print("\n" + "=" * 60)
    print("Generating Plots")
    print("=" * 60 + "\n")
    
    # Generate all plots
    plot_1_sentiment_distribution(df)
    plot_2_sentiment_scores_distribution(df)
    plot_3_post_vs_comment_sentiment(df)
    plot_4_sentiment_match_analysis(df)
    plot_5_engagement_metrics(df)
    plot_6_reaction_types(df)
    plot_7_posts_over_time(df)
    plot_8_top_authors(df)
    plot_9_word_length_analysis(df)
    plot_10_correlation_matrix(df)
    
    print("\n" + "=" * 60)
    print("All Visualizations Complete!")
    print("=" * 60)
    print(f"\nAll plots saved to: {OUTPUT_DIR}")
    print("\nGenerated plots:")
    print("  01. Sentiment Distribution (Posts & Comments)")
    print("  02. Sentiment Score Distribution")
    print("  03. Post vs Comment Sentiment Heatmap")
    print("  04. Sentiment Match Analysis")
    print("  05. Engagement Metrics by Sentiment")
    print("  06. Reaction Types Distribution")
    print("  07. Posts Over Time")
    print("  08. Top Authors Analysis")
    print("  09. Text Length Analysis")
    print("  10. Correlation Matrix")

if __name__ == "__main__":
    main()