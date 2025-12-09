import pandas as pd
from pathlib import Path
import re

BASE_DIR = Path(__file__).resolve().parent.parent.parent
folder = BASE_DIR / 'data' / 'raw' / 'cmts'

# columns to remove
cols_to_drop = ['First Name', 'Last Name', 'Type', 'Is Edited', 
                'Replies Count', 'Is Post Author', 'Has Reply From Post Author']

# Loop through every CSV in folder
for csv_file in folder.glob("*.csv"):
    print(f"Processing: {csv_file.name}")
    
    try:
        df = pd.read_csv(csv_file)

        # Drop columns if present
        df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True, errors='ignore')

        # Extract number from "Comment URN"
        # Handles both formats:
        # urn:li:comment:(activity:7401857889449279488,7402383095112441857)
        # urn:li:comment:(ugcPost:7401218035778301952,7401867563011334144)
        urn = df['Comment URN'].iloc[0]
        
        # Try to match either 'activity:' or 'ugcPost:'
        match = re.search(r'(?:activity|ugcPost):(\d+)', urn)
        
        if match:
            number = match.group(1)
        else:
            # Fallback: use the original filename
            print(f"  ⚠️ Could not extract number from URN: {urn}")
            print(f"  Using original filename as fallback\n")
            number = csv_file.stem
        
        new_name = f"linkedin-{number}.csv"
        output_path = folder / 'formated' / new_name

        df.to_csv(output_path, index=False)
        print(f"Saved → {output_path}\n")
        
    except Exception as e:
        print(f"  ❌ Error processing {csv_file.name}: {e}\n")
        continue

print("Automation completed.")