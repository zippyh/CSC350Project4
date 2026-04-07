#
# data_cleaning.py
# Reads through the big CSV and cleans up the data, removing any excess characters or spaces
# Code by Max Cheezic, Nicholas Demetrio, Hayden Ward
#
import pandas as pd
import numpy as np
import os

def deep_clean_csv(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Error: Could not find {input_file}")
        return

    # Load the data
    df = pd.read_csv(input_file, low_memory=False)

    # Clean the column headers first
    # This removes any ticks and spaces from the top row of the CSV
    df.columns = [str(c).strip().strip("'").strip('"') for c in df.columns]

    # Cell cleaning logic
    def cleaner(val):
        if pd.isna(val):
            return np.nan
        
        # Convert to string and handle non-breaking spaces (\xa0) and normal spaces
        s = str(val).replace('\xa0', ' ').strip()
        
        # Remove leading and trailing apostrophes (ticks) or quotes
        s = s.strip("'").strip('"')
        
        # Catch various forms of "Nothing" and turn them into true NaNs
        if s.lower() in ['', 'nan', 'none', 'null', 'n/a', '-', '.']:
            return np.nan
            
        # Try to turn it back into a number if it looks like one
        try:
            # This handles cases like '4.0' or ' 3 '
            return pd.to_numeric(s)
        except:
            # If it's a word like 'Participant_1', return it as is
            return s

    # Apply the cleaner to every single cell (applymap handles every cell)
    cleaned_df = df.map(cleaner)

    # Save the result
    cleaned_df.to_csv(output_file, index=False)
    print(f"SUCCESS: {input_file} has been cleaned.")
    print(f"Result saved to: {output_file}")

if __name__ == "__main__":
    deep_clean_csv('data_filled.csv', 'data_cleaned.csv')