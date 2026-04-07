#
# data_gathering.py
# Reads through each participant's survey responses and data and collects the important features into one large CSV.
# Code by Max Cheezic, Nicholas Demetrio, Hayden Ward
#
import os
import pandas as pd
import numpy as np
from datetime import datetime

# This detects whatever folder the terminal is currently "in"
BASE_PATH = os.getcwd()
all_survey_files = os.listdir('.')

if not os.path.exists(BASE_PATH):
    print(f"ERROR: Could not find the dataset at {BASE_PATH}")
    print("Make sure the script is in the correct folder!")
    exit()

# Mappings based on the questions given in the PDF from the original data source
# We assign all these answers to a series of numbers so we can sum them up to get a score
# A higher score does not necessarily mean a good outcome. For instance, the UCLA test
# has it where higher score = lonelier. 
sf36_map = {
    'Excellent': 5, 'Very Good': 4, 'Good': 3, 'Fair': 2, 'Poor': 1,
    'Much better now than one year ago': 5, 'Somewhat better now than one year ago': 4, 
    'About the same': 3, 'Somewhat worse now than one year ago': 2, 'Much worse than one year ago': 1,
    'No, Not Limited at all': 3, 'Yes, Limited a Little': 2, 'Yes, Limited a lot': 1,
    'Yes': 1, 'No': 2, 'Not at all': 1, 'Slightly Moderately': 2, 'Severe': 3, 'Very Severe': 4,
    'All of the time': 6, 'Most of the time': 5, 'A good Bit of the Time': 4, 
    'Some of the time': 3, 'A little bit of the time': 2, 'None of the Time': 1,
    'Definitely true': 5, 'Mostly true': 4, "Don't know": 3, 'Mostly false': 2, 'Definitely false': 1
}

pss_map = {
    'Never': 0, 'Almost Never': 1, 'Sometimes': 2, 'Fairly Often': 3, 'Very Often': 4
}

ucla_map = {
    'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Always': 3
}

generic_likert_5 = { 
    'Strongly Disagree': 1, 'Disagree': 2, 'Neutral': 3, 'Agree': 4, 'Strongly Agree': 5,
    'Not at all': 1, 'A little bit': 2, 'Somewhat': 3, 'Quite a bit': 4, 'Very much': 5
}

def get_survey_score(file_path, survey_type):
    if not os.path.exists(file_path): return np.nan
    try:
        df = pd.read_csv(file_path)
        # Select columns that start with 'q'
        q_cols = [c for c in df.columns if c.lower().startswith('q')]
        if not q_cols: return np.nan

        if survey_type == 'BDI':
            def extract_bdi(val):
                if pd.isna(val): return 0
                if isinstance(val, (int, float)): return int(val)
                return int(val[0]) if isinstance(val, str) and val[0].isdigit() else 0
            return df[q_cols].map(extract_bdi).sum(axis=1).iloc[0]

        elif survey_type == 'SF36':
            return df[q_cols].map(lambda x: sf36_map.get(x, 0) if isinstance(x, str) else x).sum(axis=1).iloc[0]

        elif survey_type in ['PSS', 'TWEETS', 'Social']:
            mapping = {
                'PSS': pss_map,  
                'TWEETS': generic_likert_5, 
                'Social': generic_likert_5
            }.get(survey_type)
            
            if mapping:
                # Map the text labels to numbers
                mapped_df = df[q_cols].map(lambda x: mapping.get(x, x) if isinstance(x, str) else x)
                # Force everything to be a number (converts any remaining text to NaN)
                numeric_df = mapped_df.apply(pd.to_numeric, errors='coerce')
                # Sum, treating NaNs as 0
                return numeric_df.sum(axis=1).iloc[0]
            else:
                return pd.to_numeric(df[q_cols].stack(), errors='coerce').unstack().sum(axis=1).iloc[0]

        elif survey_type == 'UCLA':
            # Apply the standard mapping first
            mapped_df = df[q_cols].map(lambda x: ucla_map.get(x, x) if isinstance(x, str) else x)
            numeric_df = mapped_df.apply(pd.to_numeric, errors='coerce')
            
            # Identify reverse-scored columns 
            # Certain questions (ones specifically about not feeling lonely) are reverse-scored (so "always" is 1 and "never" is 4 instead of the other way around)
            # So we identify which questions we feel are about not feeling lonely and reverse their answer values to get a more accurate score
            reverse_indices = [0, 4, 5, 8, 9, 14, 15, 18, 19]
            
            for idx in reverse_indices:
                if idx < len(q_cols):
                    col_name = q_cols[idx]
                    # The "Flip" formula: (Max + Min) - Current Value
                    # (3 + 1) - 3 = 1 | (3 + 1) - 1 = 3
                    numeric_df[col_name] = 3 - numeric_df[col_name]

            return numeric_df.sum(axis=1).iloc[0]
        return df[q_cols].sum(axis=1).iloc[0] 
    except Exception as e:
        print(f"Error processing {survey_type} at {file_path}: {e}")
        return np.nan

surveys_map = {
    'PSS': ['perceived stress', 'pss'],
    'TWEETS': ['twente engagement', 'tweets'],
    'Social': ['social connectedness', 'social_support', 'social support'],
    'SOC': ['sense of coherence', 'soc'],
    'SF36': ['short form health', 'sf36', 'sf-36'],
    'BDI': ['beck depression', 'bdi']
}

def process_participant(p_id):
    p_path = os.path.join(BASE_PATH, p_id)
    data = {'Participant': p_id}

    try:
        current_contents = os.listdir(p_path)
        existing_subfolders = {f.lower(): f for f in current_contents if os.path.isdir(os.path.join(p_path, f))}
    except Exception as e:
        print(f"Could not read directory for {p_id}: {e}")
        return data

    # Aware data
    aware_f = existing_subfolders.get('aware')
    if aware_f:
        aware_dir = os.path.join(p_path, aware_f)
        # Check for files inside Aware
        aware_files = os.listdir(aware_dir)
        if 'screen.csv' in aware_files:
            df_s = pd.read_csv(os.path.join(aware_dir, 'screen.csv')).sort_values('timestamp')
            df_s['duration'] = df_s['timestamp'].shift(-1) - df_s['timestamp']
            total_on_ms = df_s[df_s['screen_status'].isin([1, 3])]['duration'].sum()
            data['Average screen time'] = (total_on_ms / 1000 / 60) / 28 # Divide by 28 since the study was over 28 days
        
        if 'calls.csv' in aware_files:
            df_c = pd.read_csv(os.path.join(aware_dir, 'calls.csv'))
            data['Average call duration'] = df_c['dur'].mean() 
            data['Average daily calls'] = len(df_c) / 28 

        if 'messages.csv' in aware_files:
            data['Total messages'] = len(pd.read_csv(os.path.join(aware_dir, 'messages.csv'))) 

    # Oura data
    oura_f = existing_subfolders.get('oura')
    if oura_f:
        oura_dir = os.path.join(p_path, oura_f)
        oura_files = [f for f in os.listdir(oura_dir) if f.endswith('.csv')]
        if oura_files:
            df_o = pd.read_csv(os.path.join(oura_dir, oura_files[0]))
            
            # Helper function to safely get the mean of a column if it exists
            def safe_mean(df, col):
                return df[col].mean() if col in df.columns else np.nan

            # Use safe_mean to ensure we don't get errors from some columns not existing (looking at you, participants 26, 28, and 45...)
            data['Average MET'] = safe_mean(df_o, 'OURA_activity_average_met')
            data['Average Activity Score'] = safe_mean(df_o, 'OURA_activity_score')
            data['Average Total Active Time'] = safe_mean(df_o, 'OURA_activity_total')
            data['Average Sleep Score'] = safe_mean(df_o, 'OURA_sleep_score')
            data['Average Inactivity Alerts'] = safe_mean(df_o, 'OURA_activity_inactivity_alerts')
            data['Average Sleep Efficiency'] = safe_mean(df_o, 'OURA_sleep_efficiency')
            data['Sleep Alignment Score'] = safe_mean(df_o, 'OURA_sleep_score_alignment')
            data['Average sleep duration'] = safe_mean(df_o, 'OURA_sleep_duration') / 60 # Convert to hours
            data['Average daily steps'] = safe_mean(df_o, 'OURA_activity_steps')
            data['Average HRV (RMSSD)'] = safe_mean(df_o, 'OURA_sleep_rmssd')

    # Watch data
    watch_f = existing_subfolders.get('watch')
    if watch_f:
        watch_dir = os.path.join(p_path, watch_f)
        hr_values = []
        for f in os.listdir(watch_dir):
            if f.endswith('.csv'):
                try:
                    df_w = pd.read_csv(os.path.join(watch_dir, f))
                    if 'hrm' in df_w.columns:
                        clean_hr = pd.to_numeric(df_w['hrm'], errors='coerce').dropna()
                        if not clean_hr.empty:
                            hr_values.append(clean_hr.mean())
                except: continue
        if hr_values:
            data['Average Heartrate'] = np.mean([v for v in hr_values if v > 30]) # Filter out heartrates below 30 since anything less than that is probably a tech issue and not accurate. Lowest ever recorded resting heartrate was allegedly 26 

    # Survey data
    survey_f = existing_subfolders.get('survey') or existing_subfolders.get('surveys')
    if survey_f:
        survey_dir = os.path.join(p_path, survey_f)
        all_survey_files = os.listdir(survey_dir)
        
        surveys_map = {
            'PSS': 'perceived stress',
            'TWEETS': 'twente engagement',
            'Social': 'social connectedness',
            'SOC': 'sense of coherence',
            'SF36': 'short form health',
            'BDI': 'beck depression'
        }

        for key, search_term in surveys_map.items():
            beg = [f for f in all_survey_files if search_term in f.lower() and 'beginning' in f.lower()]
            end = [f for f in all_survey_files if search_term in f.lower() and 'end' in f.lower()]
            if beg: data[f'{key} (Beginning)'] = get_survey_score(os.path.join(survey_dir, beg[0]), key)
            if end: data[f'{key} (End)'] = get_survey_score(os.path.join(survey_dir, end[0]), key)

        ema = [f for f in all_survey_files if 'ema' in f.lower()]
        if ema: data['Average EMA Loneliness'] = pd.read_csv(os.path.join(survey_dir, ema[0]))['lonely'].mean() 

        # Label
        ucla = [f for f in all_survey_files if 'ucla' in f.lower() and 'end' in f.lower()]
        if ucla: data['UCLA Loneliness Total (Label)'] = get_survey_score(os.path.join(survey_dir, ucla[0]), 'UCLA')   

    return data

# Main execution loop
all_results = []

# Get the base path dynamically
script_dir = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(script_dir, "Loneliness_Dataset_Nov10")

# Check if the directory exists to prevent a crash
if not os.path.exists(BASE_PATH):
    print(f"Error: Directory not found at {BASE_PATH}")
else:
    # Only look at folders that actually exist
    # This filters out missing participants (a bunch of participants e.g. 4, 14, 30, 35, etc do not exist)
    existing_folders = [f for f in os.listdir(BASE_PATH) 
                        if os.path.isdir(os.path.join(BASE_PATH, f)) and f.startswith("Participant_")]

    # Sort numerically so Participant_2 comes before Participant_10 and so on
    existing_folders.sort(key=lambda x: int(x.split('_')[1]))

    print(f"Found {len(existing_folders)} participants. Processing now...")

    for p_id in existing_folders:
        try:
            print(f"Processing {p_id}...")
            result = process_participant(p_id)
            all_results.append(result)
        except Exception as e:
            # This ensures that if one participant has a file error, the script doesn't stop for everyone else
            print(f"Skipping {p_id} due to an error: {e}")

    # Save to the final CSV
    final_df = pd.DataFrame(all_results)
    final_df.to_csv("data_filled.csv", index=False)
    print(f"Done")