#%%[md]
# # 1. Data Preprocessing
# In this section, we will load the data, clean it, and prepare it for analysis.

#%%
import pandas as pd

# Global state mapping dictionaries:
STATE_MAPPING = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia",
    "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming"
}
# Reverse mapping: full state names to their abbreviations
REVERSE_STATE_MAPPING = {v: k for k, v in STATE_MAPPING.items()}


def load_excel_sheets(file_path, sheet_names):
    """
    Load specified sheets from an Excel file.
    
    Parameters:
        file_path (str): Path to the Excel file.
        sheet_names (list): List of sheet names to load.
    
    Returns:
        dict: A dictionary where keys are sheet names and values are DataFrames.
    """
    # Read all specified sheets using the 'xlrd' engine
    dataframes = pd.read_excel(file_path, sheet_name=sheet_names, engine='xlrd')
    return dataframes


def clean_state_column(df, state_column='State'):
    """
    Clean and map the state column in the DataFrame.
    
    Steps:
      - Convert values to string.
      - Strip extra whitespace.
      - Convert to title case.
      - Map full state names to abbreviations.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the state column.
        state_column (str): The name of the state column.
    
    Returns:
        pd.DataFrame: DataFrame with the cleaned and mapped state column.
    """
    # Display unique state values before mapping for debugging
    print("Unique state values before mapping:", df[state_column].unique())
    
    # Clean the state values
    df[state_column] = df[state_column].astype(str).str.strip().str.title()
    df[state_column] = df[state_column].map(REVERSE_STATE_MAPPING)
    
    # Debug output: unique values after mapping
    print("Unique state values after mapping:", df[state_column].unique())
    print("First few mapped state values:", df[state_column].head())
    
    return df


def merge_excel_sheets(base_df, dataframes, join_keys=['State', 'County'], sheets_to_join=None):
    """
    Merge additional Excel sheets into the base DataFrame using join keys.
    
    Parameters:
        base_df (pd.DataFrame): The base DataFrame (e.g., 'Supplemental Data - County').
        dataframes (dict): Dictionary of DataFrames loaded from Excel.
        join_keys (list): List of columns to join on.
        sheets_to_join (list): List of sheet names to merge (first sheet should be the base).
    
    Returns:
        pd.DataFrame: The final merged DataFrame.
    """
    # If no specific order is provided, use all keys (assuming first is base)
    if sheets_to_join is None:
        sheets_to_join = list(dataframes.keys())
    
    final_df = base_df.copy()
    
    # Loop through each additional sheet to merge
    for sheet in sheets_to_join[1:]:
        df_to_merge = dataframes[sheet].copy()
        
        # Drop duplicate rows based on join keys to avoid many-to-many merges
        df_to_merge = df_to_merge.drop_duplicates(subset=join_keys)
        
        # Rename columns that are duplicates (except the join keys)
        rename_dict = {}
        for col in df_to_merge.columns:
            if col not in join_keys and col in final_df.columns:
                rename_dict[col] = f"{col}_{sheet}"
        if rename_dict:
            df_to_merge = df_to_merge.rename(columns=rename_dict)
        
        # Perform a left merge on the join keys
        final_df = final_df.merge(df_to_merge, on=join_keys, how='left')
    
    # Debug output: show the first few rows of the merged DataFrame
    print("Merged Excel DataFrame preview:")
    print(final_df.head())
    return final_df
