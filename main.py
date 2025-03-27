
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

#%%[md]
# # Data Visualization

#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the main data
data = pd.read_csv('../data/FoodAccessResearchAtlasData2019.csv')

# ======================================================================
# Section 1: Basic Food Access & Demographic Visualizations
# ======================================================================
selected_columns = [
    'Pop2010', 'lapophalf', 'lakidshalf', 'laseniorshalf', 'lalowihalf',
    'lablackhalf', 'lawhitehalf', 'lahunvhalf', 'lasnaphalf', 'TractSNAP',
    'LowIncomeTracts', 'LATracts_half', 'LATractsVehicle_20'
]
viz_data = data[selected_columns]

# Distribution of Population in 2010
plt.figure(figsize=(10, 6))
sns.histplot(viz_data['Pop2010'], bins=50, kde=True)
plt.title('Distribution of Population in Census Tracts (2010)')
plt.xlabel('Population in 2010')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Correlation Heatmap of Selected Features
plt.figure(figsize=(12, 10))
sns.heatmap(viz_data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Selected Features')
plt.show()

# Bar Chart for Low Income Tracts
plt.figure(figsize=(8, 6))
viz_data['LowIncomeTracts'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Count of Low Income vs Non-Low Income Tracts')
plt.xlabel('Low Income Tracts')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['Non-Low Income', 'Low Income'], rotation=0)
plt.grid(axis='y')
plt.show()

# Bar Chart: Tracts with Limited Food Access (Half-mile)
plt.figure(figsize=(8, 6))
viz_data['LATracts_half'].value_counts().plot(kind='bar', color='orange')
plt.title('Tracts with Limited Food Access (Half-mile)')
plt.xlabel('Limited Access Tracts (Half-mile)')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['No', 'Yes'], rotation=0)
plt.grid(axis='y')
plt.show()

# Scatter Plot: Population vs SNAP Recipients
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Pop2010', y='TractSNAP', data=viz_data, alpha=0.5)
plt.title('Population vs SNAP Recipients')
plt.xlabel('Population (2010)')
plt.ylabel('SNAP Recipients')
plt.grid(True)
plt.show()


# ======================================================================
# Section 2: Demographic Analysis
# ======================================================================
# Create additional demographic percentage columns
data['White_pct'] = data['TractWhite'] / data['Pop2010'] * 100
data['Black_pct'] = data['TractBlack'] / data['Pop2010'] * 100
data['Asian_pct'] = data['TractAsian'] / data['Pop2010'] * 100
data['Hispanic_pct'] = data['TractHispanic'] / data['Pop2010'] * 100
data['Seniors_pct'] = data['TractSeniors'] / data['Pop2010'] * 100

# Urban vs Rural Analysis: Count and Population Distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='Urban', data=data)
plt.title('Count of Urban vs Rural Census Tracts')
plt.xlabel('Urban Indicator (0 = Rural, 1 = Urban)')
plt.ylabel('Number of Tracts')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Urban', y='Pop2010', data=data)
plt.title('Population Distribution by Urban vs Rural Tracts')
plt.xlabel('Urban Indicator (0 = Rural, 1 = Urban)')
plt.ylabel('Population (2010)')
plt.show()

# Racial Composition Histograms
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
sns.histplot(data['White_pct'], bins=50, kde=True, ax=axes[0,0])
axes[0,0].set_title('White Population (%)')
sns.histplot(data['Black_pct'], bins=50, kde=True, ax=axes[0,1])
axes[0,1].set_title('Black Population (%)')
sns.histplot(data['Asian_pct'], bins=50, kde=True, ax=axes[0,2])
axes[0,2].set_title('Asian Population (%)')
sns.histplot(data['Hispanic_pct'], bins=50, kde=True, ax=axes[1,0])
axes[1,0].set_title('Hispanic Population (%)')
sns.histplot(data['Seniors_pct'], bins=50, kde=True, ax=axes[1,1])
axes[1,1].set_title('Seniors (%)')
axes[1,2].axis('off')
plt.tight_layout()
plt.show()

# Correlation Heatmap: Demographics & Food Access Measures
cols_for_corr = ['lapophalfshare', 'TractSNAP', 'LowIncomeTracts', 
                 'White_pct', 'Black_pct', 'Asian_pct', 'Hispanic_pct', 'Seniors_pct']
plt.figure(figsize=(10, 8))
sns.heatmap(data[cols_for_corr].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap: Demographics & Food Access Measures')
plt.show()


# ======================================================================
# Section 3: Housing Units & Population Density Analysis
# ======================================================================
# Compute population per housing unit
data['pop_per_house'] = data['Pop2010'] / data['OHU2010']

# Distribution of Occupied Housing Units (OHU2010)
plt.figure(figsize=(10, 6))
sns.histplot(data['OHU2010'], bins=50, kde=True)
plt.title("Distribution of Occupied Housing Units (OHU2010)")
plt.xlabel("Occupied Housing Units (2010)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Distribution of Population per Housing Unit
plt.figure(figsize=(10, 6))
sns.histplot(data['pop_per_house'], bins=50, kde=True)
plt.title("Population per Housing Unit (Pop2010/OHU2010)")
plt.xlabel("Population per Housing Unit")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Scatter Plot: Population vs Occupied Housing Units
plt.figure(figsize=(10, 6))
sns.scatterplot(x='OHU2010', y='Pop2010', data=data, alpha=0.5)
plt.title("Population vs Occupied Housing Units")
plt.xlabel("Occupied Housing Units (OHU2010)")
plt.ylabel("Population (Pop2010)")
plt.grid(True)
plt.show()

# Box Plot: Population per Housing Unit by Urban/Rural
plt.figure(figsize=(10, 6))
sns.boxplot(x='Urban', y='pop_per_house', data=data)
plt.title("Population per Housing Unit by Urban vs Rural")
plt.xlabel("Urban (0 = Rural, 1 = Urban)")
plt.ylabel("Population per Housing Unit")
plt.show()


# ======================================================================
# Section 4: Group Quarters Analysis
# ======================================================================
# Count of Group Quarters Flag
plt.figure(figsize=(10, 6))
sns.countplot(x='GroupQuartersFlag', data=data)
plt.title("Group Quarters Flag Count")
plt.xlabel("Group Quarters Flag (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.grid(axis='y')
plt.show()

# Distribution of Number and Percentage of Group Quarters
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.histplot(data['NUMGQTRS'], bins=50, kde=True, ax=axes[0])
axes[0].set_title("Number of Group Quarters (NUMGQTRS)")
axes[0].set_xlabel("Number of Group Quarters")
axes[0].set_ylabel("Frequency")
sns.histplot(data['PCTGQTRS'], bins=50, kde=True, ax=axes[1])
axes[1].set_title("Percentage Group Quarters (PCTGQTRS)")
axes[1].set_xlabel("Percentage Group Quarters")
axes[1].set_ylabel("Frequency")
plt.tight_layout()
plt.show()

# Scatter Plot: NUMGQTRS vs PCTGQTRS by Urban Status
plt.figure(figsize=(10, 6))
sns.scatterplot(x='NUMGQTRS', y='PCTGQTRS', hue='Urban', data=data, alpha=0.5)
plt.title("NUMGQTRS vs PCTGQTRS by Urban/Rural")
plt.xlabel("Number of Group Quarters")
plt.ylabel("Percentage Group Quarters")
plt.grid(True)
plt.show()


# ======================================================================
# Section 5: Vehicle Access Analysis
# ======================================================================
# Count of Households with No Vehicle (LATractsVehicle_20)
plt.figure(figsize=(10, 6))
sns.countplot(x='LATractsVehicle_20', data=data)
plt.title("Households with No Vehicle (LATractsVehicle_20)")
plt.xlabel("LATractsVehicle_20 (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.grid(axis='y')
plt.show()

# Urban vs Rural Comparison for No Vehicle Indicator
plt.figure(figsize=(10, 6))
sns.countplot(x='Urban', hue='LATractsVehicle_20', data=data)
plt.title("Urban vs Rural: Households with No Vehicle")
plt.xlabel("Urban (0 = Rural, 1 = Urban)")
plt.ylabel("Count")
plt.legend(title="LATractsVehicle_20", labels=["No", "Yes"])
plt.grid(axis='y')
plt.show()


# ======================================================================
# Section 6: Extended Analysis: Binary Indicators & Demographics
# ======================================================================
# Frequency Distributions for Selected Binary Food Access Indicators (LILA Family)
binary_lila_cols = ['LILATracts_1And10', 'LILATracts_halfAnd10', 'LILATracts_1And20', 'LILATracts_Vehicle']
for col in binary_lila_cols:
    plt.figure(figsize=(8,6))
    sns.countplot(x=col, data=data)
    plt.title(f"Frequency Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.grid(axis='y')
    plt.show()

# Urban vs Rural Comparison for LA1and10 (Example)
plt.figure(figsize=(10,6))
sns.countplot(x='Urban', hue='LA1and10', data=data)
plt.title("Urban vs Rural Comparison for LA1and10")
plt.xlabel("Urban (0 = Rural, 1 = Urban)")
plt.ylabel("Count")
plt.legend(title="LA1and10", labels=["No", "Yes"])
plt.grid(axis='y')
plt.show()

# Racial Limited Access Shares in Urban Areas
racial_limited_measures = ['lawhitehalfshare', 'lablackhalfshare', 'laasianhalfshare', 'lahisphalfshare']
urban_data = data[data['Urban'] == 1]
urban_means = urban_data[racial_limited_measures].mean()
plt.figure(figsize=(10,6))
urban_means.plot(kind='bar', color='skyblue')
plt.title("Average Limited Access Share by Race in Urban Tracts")
plt.xlabel("Racial Group")
plt.ylabel("Average Limited Access Share (%)")
plt.grid(axis='y')
plt.show()

# State-Level Proportion of Low Income Tracts
state_low_income = data.groupby('State')['LowIncomeTracts'].mean().sort_values()
plt.figure(figsize=(12,8))
state_low_income.plot(kind='barh', color='coral')
plt.title("Average Proportion of Low Income Tracts by State")
plt.xlabel("Proportion of Low Income Tracts")
plt.ylabel("State")
plt.grid(axis='x')
plt.show()

# Violin Plot: Group Quarters Percentage by Urban Status
plt.figure(figsize=(10,6))
sns.violinplot(x='Urban', y='PCTGQTRS', data=data)
plt.title("Group Quarters Percentage by Urban Status")
plt.xlabel("Urban (0 = Rural, 1 = Urban)")
plt.ylabel("Percentage Group Quarters")
plt.grid(axis='y')
plt.show()

# Scatter Plot: Limited Access Share for Kids vs Seniors
plt.figure(figsize=(10,6))
sns.scatterplot(x='lakidshalfshare', y='laseniorshalfshare', data=data, alpha=0.5)
plt.title("Limited Access Share: Kids vs Seniors (Urban)")
plt.xlabel("Kids Limited Access Share (%)")
plt.ylabel("Seniors Limited Access Share (%)")
plt.grid(True)
plt.show()


# ======================================================================
# Section 7: Extended Correlation Analysis
# ======================================================================
extended_cols = ['Pop2010', 'OHU2010', 'lapophalf', 'LAPOP05_10', 'LAPOP1_10', 'LAPOP1_20',
                 'LALOWI1_10', 'LALOWI05_10', 'LALOWI1_20',
                 'lawhitehalfshare', 'lablackhalfshare', 'laasianhalfshare', 'lahisphalfshare']
plt.figure(figsize=(14,12))
sns.heatmap(data[extended_cols].corr(), annot=True, cmap='viridis')
plt.title("Extended Correlation Heatmap: Housing, Food Access & Racial Limited Access Measures")
plt.show()
