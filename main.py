
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
def load_csv_and_group(file_path, state_mapping):
    """
    Load a CSV file and group data by State and County.
    
    Steps:
      - Read the CSV file.
      - Remove the text ' County' from the County column.
      - Map full state names to abbreviations.
      - Group by State and County, computing the mean for numeric columns.
    
    Parameters:
        file_path (str): Path to the CSV file.
        state_mapping (dict): State mapping dictionary.
    
    Returns:
        pd.DataFrame: Grouped DataFrame with the mean values.
    """
    df = pd.read_csv(file_path)
    
    # Remove ' County' from the County column
    df['County'] = df['County'].str.replace(' County', '')
    
    # Create a reverse mapping (full state name -> abbreviation)
    reverse_map = {v: k for k, v in state_mapping.items()}
    df['State'] = df['State'].map(reverse_map)
    
    # Group by 'State' and 'County' and calculate the mean for numeric columns
    grouped_df = df.groupby(['State', 'County'], as_index=False).mean()
    return grouped_df


def merge_with_csv(excel_df, csv_grouped_df, columns_from_csv):
    """
    Merge the Excel DataFrame with the grouped CSV DataFrame.
    
    Parameters:
        excel_df (pd.DataFrame): DataFrame from the processed Excel file.
        csv_grouped_df (pd.DataFrame): Grouped DataFrame from the CSV.
        columns_from_csv (list): List of columns from the CSV to merge.
    
    Returns:
        pd.DataFrame: The final merged DataFrame.
    """
    # Remove any extraneous column if it exists
    if 'Unnamed: 0' in excel_df.columns:
        excel_df.drop(['Unnamed: 0'], axis=1, inplace=True)
    
    # Merge on 'State' and 'County' using a left join
    merged_df = pd.merge(excel_df, csv_grouped_df[columns_from_csv], on=['State', 'County'], how='left')
    return merged_df


def processing():
    # File paths for the Excel and CSV files
    excel_file_path = '/data/FoodEnvironmentAtlas.xls'
    csv_file_path = '/data/FoodAccessResearchAtlasData2019.csv'
    
    # Define the sheet names to load from the Excel file
    sheet_names = [
        'Supplemental Data - County', 'Supplemental Data - State', 'ACCESS', 'STORES',
        'RESTAURANTS', 'ASSISTANCE', 'INSECURITY', 'TAXES', 'LOCAL', 'HEALTH', 'SOCIOECONOMIC'
    ]
    
    # Load all sheets from the Excel file
    dataframes = load_excel_sheets(excel_file_path, sheet_names)
    
    # Clean and map the state column in the base DataFrame
    base_df = clean_state_column(dataframes['Supplemental Data - County'], state_column='State')
    
    # Define the order of sheets to join (base sheet first)
    sheets_to_join = [
        'Supplemental Data - County', 'ACCESS', 'STORES',
        'RESTAURANTS', 'ASSISTANCE', 'INSECURITY', 'TAXES', 'LOCAL', 'HEALTH', 'SOCIOECONOMIC'
    ]
    
    # Merge additional sheets into the base DataFrame
    merged_excel_df = merge_excel_sheets(base_df, dataframes, join_keys=['State', 'County'], sheets_to_join=sheets_to_join)
    
    # Save the merged Excel DataFrame to a new Excel file (optional)
    merged_excel_df.to_excel('kk.xlsx', index=False)
    
    # Load the merged Excel file into a DataFrame
    excel_df = pd.read_excel('kk.xlsx', sheet_name='Sheet1')
    
    # Load and process the CSV file, grouping by State and County
    grouped_csv_df = load_csv_and_group(csv_file_path, STATE_MAPPING)
    
    # Specify the columns from the CSV to merge with the Excel DataFrame
    columns_from_csv = [
        'State', 'County', 'Urban', 'Pop2010', 'OHU2010', 'GroupQuartersFlag', 'NUMGQTRS', 'PCTGQTRS',
        'LILATracts_1And10', 'LILATracts_halfAnd10', 'LILATracts_1And20', 'LILATracts_Vehicle', 'HUNVFlag',
        'LowIncomeTracts', 'PovertyRate', 'MedianFamilyIncome', 'TractLOWI', 'TractKids', 'TractSeniors',
        'TractWhite', 'TractBlack', 'TractAsian', 'TractNHOPI', 'TractAIAN', 'TractOMultir', 'TractHispanic',
        'TractHUNV', 'TractSNAP'
    ]
    
    # Merge the processed Excel DataFrame with the CSV data
    final_df = merge_with_csv(excel_df, grouped_csv_df, columns_from_csv)
    
    # Optionally, save the final merged DataFrame to an Excel file
    final_df.to_excel('/data/final_merged.xlsx', index=False)
    
    # Return the final DataFrame for further processing if needed
    return final_df

final_dataframe = processing()
print("Final DataFrame shape:", final_dataframe.shape)
