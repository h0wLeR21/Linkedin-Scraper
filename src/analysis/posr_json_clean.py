import pandas as pd
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
input_folder = BASE_DIR / 'data' / 'processed'
output_folder = BASE_DIR / 'data' / 'processed'

def clean_value(value):
    """Return None for NaN/empty values, otherwise return the value"""
    if pd.isna(value) or value == '' or str(value).lower() == 'nan':
        return None
    return value

def restructure_to_nested(row):
    """Convert flat CSV row to nested JSON structure"""
    
    # Build nested structure
    post = {}
    
    full_urn = clean_value(row.get('full_urn'))
    if full_urn:
        post['full_urn'] = full_urn
    
    # Posted_at object
    date = clean_value(row.get('posted_at/date'))
    timestamp = clean_value(row.get('posted_at/timestamp'))
    if date or timestamp:
        post['posted_at'] = {}
        if date:
            post['posted_at']['date'] = date
        if timestamp:
            post['posted_at']['timestamp'] = int(timestamp)
    
    text = clean_value(row.get('text'))
    if text:
        post['text'] = text
    
    url = clean_value(row.get('url'))
    if url:
        post['url'] = url
    
    post_type = clean_value(row.get('post_type'))
    if post_type:
        post['post_type'] = post_type
    
    # Author object - only add if at least one field exists
    author = {}
    full_name = clean_value(row.get('author/full_name'))
    if full_name:
        author['full_name'] = full_name
    
    headline = clean_value(row.get('author/headline'))
    if headline:
        author['headline'] = headline
    
    username = clean_value(row.get('author/username'))
    if username:
        author['username'] = username
    
    profile_url = clean_value(row.get('author/profile_url'))
    if profile_url:
        author['profile_url'] = profile_url
    
    if author:  # Only add author if it has at least one field
        post['author'] = author
    
    # Stats object
    stats = {}
    stat_fields = ['total_reactions', 'like', 'support', 'love', 'insight', 
                   'celebrate', 'funny', 'comments', 'reposts']
    
    for field in stat_fields:
        value = clean_value(row.get(f'stats/{field}'))
        if value is not None:
            stats[field] = int(value)
    
    if stats:  # Only add stats if it has at least one field
        post['stats'] = stats
    
    return post

# Process all CSV files in data/results/
for csv_file in input_folder.glob("*.csv"):
    print(f"Processing: {csv_file.name}")
    
    try:
        # Read CSV file
        df = pd.read_csv(csv_file)
        
        # Convert each row to nested structure
        posts = []
        for _, row in df.iterrows():
            post = restructure_to_nested(row)
            posts.append(post)
        
        # Save as JSON
        json_output = csv_file.with_suffix('.json')
        
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(posts, f, indent=2, ensure_ascii=False)
        
        print(f"  ✅ Converted to nested JSON → {json_output}")
        print(f"  Total posts: {len(posts)}\n")
        
    except Exception as e:
        print(f"  ❌ Error processing {csv_file.name}: {e}\n")
        continue

print("Conversion completed!")