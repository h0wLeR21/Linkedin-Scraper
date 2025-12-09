import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
input_folder = BASE_DIR / 'data' / 'raw'
output_folder = BASE_DIR / 'data' / 'processed'

# Create output folder if it doesn't exist
output_folder.mkdir(parents=True, exist_ok=True)

# Columns to remove (including first_name and last_name which we'll combine first)
cols_to_drop = [
    'article/subtitle', 'article/thumbnail', 'article/title', 'article/url',
    'document/page_count', 'document/thumbnail', 'document/title', 'document/url',
    'job_data/company', 'job_data/company_logo', 'job_data/id', 'job_data/location',
    'job_data/title', 'job_data/url',
    'media/images/0/height', 'media/images/0/url', 'media/images/0/width',
    'media/images/1/height', 'media/images/1/url', 'media/images/1/width',
    'media/images/2/height', 'media/images/2/url', 'media/images/2/width',
    'media/images/3/height', 'media/images/3/url', 'media/images/3/width',
    'media/images/4/height', 'media/images/4/url', 'media/images/4/width',
    'media/images/5/height', 'media/images/5/url', 'media/images/5/width',
    'media/thumbnail', 'media/type', 'media/url',
    'reshared_post/article/subtitle', 'reshared_post/article/thumbnail',
    'reshared_post/article/title', 'reshared_post/article/url',
    'reshared_post/author/first_name', 'reshared_post/author/headline',
    'reshared_post/author/last_name', 'reshared_post/author/profile_picture',
    'reshared_post/author/profile_url', 'reshared_post/author/username',
    'reshared_post/document/page_count', 'reshared_post/document/thumbnail',
    'reshared_post/document/title', 'reshared_post/document/url',
    'reshared_post/media/images/0/height', 'reshared_post/media/images/0/url',
    'reshared_post/media/images/0/width', 'reshared_post/media/thumbnail',
    'reshared_post/media/type', 'reshared_post/media/url',
    'reshared_post/post_type', 'reshared_post/posted_at/date',
    'reshared_post/posted_at/relative', 'reshared_post/posted_at/timestamp',
    'reshared_post/stats/celebrate', 'reshared_post/stats/comments',
    'reshared_post/stats/funny', 'reshared_post/stats/insight',
    'reshared_post/stats/like', 'reshared_post/stats/love',
    'reshared_post/stats/reposts', 'reshared_post/stats/support',
    'reshared_post/stats/total_reactions', 'reshared_post/text',
    'reshared_post/url', 'reshared_post/urn/activity_urn',
    'reshared_post/urn/share_urn', 'reshared_post/urn/ugcPost_urn',
    'urn/share_urn', 'urn/ugcPost_urn', 'pagination_token',
    'posted_at/relative', 'author/profile_picture',
    'author/first_name', 'author/last_name', 'urn/activity_urn'  # Added these here
]

# Process all CSV files in data/raw/
for csv_file in input_folder.glob("*.csv"):
    print(f"Processing: {csv_file.name}")
    
    try:
        df = pd.read_csv(csv_file)
        
        # Combine first_name and last_name into full_name
        if 'author/first_name' in df.columns and 'author/last_name' in df.columns:
            df['author/full_name'] = (
                df['author/first_name'].fillna('') + ' ' + df['author/last_name'].fillna('')
            ).str.strip()
            print(f"  Created 'author/full_name' column")
        
        # Drop columns if they exist
        cols_present = [c for c in cols_to_drop if c in df.columns]
        if cols_present:
            df.drop(columns=cols_present, inplace=True)
            print(f"  Removed {len(cols_present)} columns")
        else:
            print(f"  No matching columns to remove")
        
        # Save to results folder with same filename
        output_path = output_folder / csv_file.name
        df.to_csv(output_path, index=False)
        print(f"  Saved → {output_path}\n")
        
    except Exception as e:
        print(f"  ❌ Error processing {csv_file.name}: {e}\n")
        continue

print("Processing completed!")