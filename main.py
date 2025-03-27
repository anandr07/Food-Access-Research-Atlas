
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
    excel_file_path = 'FoodEnvironmentAtlas.xls'
    csv_file_path = 'FoodAccessResearchAtlasData2019.csv'
    
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
    final_df.to_excel('final_merged.xlsx', index=False)
    
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
data = pd.read_csv('FoodAccessResearchAtlasData2019.csv')

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

#%%[md]
# # Causal Inference Analysis

#%%
import pandas as pd
df = pd.read_excel('final_merged.xlsx',sheet_name ='Sheet1')

#%%
# Install required packages if not already installed
# !pip install causalinference
# !pip install causalinference==0.0.8
# !pip install numpy==1.23.5
# !pip install econml

#%%
from graphviz import Digraph

# Identify all confounder columns automatically
confounders = [
    col for col in data.columns
    if col not in ['PCH_SNAP_12_17', 'FOODINSEC_15_17', 'High_SNAP']
]

dot = Digraph()

# Treatment and outcome
dot.node("T", "High_SNAP")
dot.node("Y", "FOODINSEC_15_17")

# Add each confounder as its own node, with arrows into T and Y
for c in confounders:
    dot.node(c, c)
    dot.edge(c, "T")
    dot.edge(c, "Y")

# Direct causal effect of treatment on outcome
dot.edge("T", "Y")

# Render (creates causal_dag.pdf and opens it)
dot.render("causal_dag", view=True)

#%%
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
# Assuming df is your original dataframe loaded already
columns = [
    'PCH_SNAP_12_17', 'FOODINSEC_15_17', 'PCT_LACCESS_POP10','PCT_LACCESS_LOWI10',
    'PCT_LACCESS_HHNV10','PCT_LACCESS_CHILD10','PCT_LACCESS_SENIORS10','GROCPTH11','SUPERCPTH11','CONVSPTH11','SPECSPTH11','SNAPSPTH12',
    'WICSPTH11','FSRPTH11','PC_FFRSALES07','PCT_NSLP12','PCT_FREE_LUNCH10','PCT_REDUCED_LUNCH10','PCT_SBP12','PCT_SFSP12','PCT_WIC12','PCT_WICINFANTCHILD14',
    'PCT_WICWOMEN14','PCT_CACFP12','FDPIR12','FMRKTPTH13','VEG_ACRESPTH07','FRESHVEG_ACRESPTH07','ORCHARD_ACRESPTH12','BERRY_ACRESPTH07','SLHOUSE07','GHVEG_SQFTPTH07',
    'AGRITRSM_OPS07','PCT_DIABETES_ADULTS08','PCT_OBESE_ADULTS12','RECFACPTH11','PCT_NHWHITE10','PCT_NHBLACK10','PCT_HISP10','PCT_NHASIAN10','PCT_NHNA10','PCT_NHPI10',
    'PCT_65OLDER10','PCT_18YOUNGER10','PERPOV10','METRO13','POPLOSS10','Urban'
]

# Drop NA values to maintain data quality
data = df[columns].dropna()

# Define treatment properly using median
median_snap = data['PCH_SNAP_12_17'].median()
data['High_SNAP'] = (data['PCH_SNAP_12_17'] > median_snap).astype(int)
# 1️⃣ Descriptive statistics by group
summary = data.groupby('High_SNAP')['FOODINSEC_15_17'].agg(
    count='count',
    mean='mean',
    median='median',
    std='std',
    min='min',
    max='max'
)
print("Descriptive stats:\n", summary)

# 2️⃣ Independent t‑test (with variance check)
levene_p = stats.levene(
    data.loc[data.High_SNAP==1, 'FOODINSEC_15_17'],
    data.loc[data.High_SNAP==0, 'FOODINSEC_15_17']
).pvalue
equal_var = True if levene_p > 0.05 else False

t_stat, t_p = stats.ttest_ind(
    data.loc[data.High_SNAP==1, 'FOODINSEC_15_17'],
    data.loc[data.High_SNAP==0, 'FOODINSEC_15_17'],
    equal_var=equal_var
)
print(f"\nLevene’s test p-value = {levene_p:.4f} → equal_var={equal_var}")
print(f"T‑test: t = {t_stat:.3f}, p = {t_p:.4f}")

# 3️⃣ Cohen’s d
mean_diff = summary.loc[1,'mean'] - summary.loc[0,'mean']
pooled_sd = np.sqrt(((summary.loc[1,'count']-1)*summary.loc[1,'std']**2 +
                     (summary.loc[0,'count']-1)*summary.loc[0,'std']**2) /
                    (summary.loc[1,'count'] + summary.loc[0,'count'] - 2))
cohens_d = mean_diff / pooled_sd
print(f"Cohen’s d = {cohens_d:.3f}")

# 4️⃣ Mann–Whitney U test (non-parametric)
u_stat, u_p = stats.mannwhitneyu(
    data.loc[data.High_SNAP==1, 'FOODINSEC_15_17'],
    data.loc[data.High_SNAP==0, 'FOODINSEC_15_17'],
    alternative='two-sided'
)
print(f"Mann–Whitney U: U = {u_stat:.3f}, p = {u_p:.4f}")

# 5️⃣ Visualizations
plt.figure()
data.boxplot(column='FOODINSEC_15_17', by='High_SNAP')
plt.title('Food Insecurity by SNAP Level')
plt.suptitle('')
plt.xlabel('High SNAP (1 = above median)')
plt.ylabel('Food Insecurity (%)')
plt.show()

plt.figure()
data[data.High_SNAP==1]['FOODINSEC_15_17'].hist(alpha=0.7, label='High SNAP')
data[data.High_SNAP==0]['FOODINSEC_15_17'].hist(alpha=0.7, label='Low SNAP')
plt.legend()
plt.title('Distribution of Food Insecurity by SNAP Level')
plt.xlabel('Food Insecurity (%)')
plt.ylabel('Frequency')
plt.show()

#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from causalinference import CausalModel
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

# Assuming df is your original dataframe loaded already
# columns = [
#     'PCH_SNAP_12_17', 'FOODINSEC_15_17', 'PCT_LACCESS_POP10','PCT_LACCESS_LOWI10',
#     'PCT_LACCESS_HHNV10','PCT_LACCESS_CHILD10','PCT_LACCESS_SENIORS10','GROCPTH11','SUPERCPTH11','CONVSPTH11','SPECSPTH11','SNAPSPTH12',
#     'WICSPTH11','FSRPTH11','PC_FFRSALES07','PCT_NSLP12','PCT_FREE_LUNCH10','PCT_REDUCED_LUNCH10','PCT_SBP12','PCT_SFSP12','PCT_WIC12','PCT_WICINFANTCHILD14',
#     'PCT_WICWOMEN14','PCT_CACFP12','FDPIR12','FMRKTPTH13','VEG_ACRESPTH07','FRESHVEG_ACRESPTH07','ORCHARD_ACRESPTH12','BERRY_ACRESPTH07','SLHOUSE07','GHVEG_SQFTPTH07',
#     'AGRITRSM_OPS07','PCT_DIABETES_ADULTS08','PCT_OBESE_ADULTS12','RECFACPTH11','PCT_NHWHITE10','PCT_NHBLACK10','PCT_HISP10','PCT_NHASIAN10','PCT_NHNA10','PCT_NHPI10',
#     'PCT_65OLDER10','PCT_18YOUNGER10','PERPOV10','METRO13','POPLOSS10','Urban'
# ]

# # Drop NA values to maintain data quality
# data = df[columns].dropna()

# # Define treatment properly using median
# median_snap = data['PCH_SNAP_12_17'].median()
# data['High_SNAP'] = (data['PCH_SNAP_12_17'] > median_snap).astype(int)

# Outcome and Treatment
Y = data['FOODINSEC_15_17'].values
T = data['High_SNAP'].values

# Confounders (covariates)
X = data.drop(columns=['PCH_SNAP_12_17', 'FOODINSEC_15_17', 'High_SNAP']).values

# Standardize confounders
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

### 🌟 **1. Propensity Score Matching (PSM)**
causal = CausalModel(Y, T, X_std)
causal.est_propensity()

print("\nBefore Matching Balance:")
print(causal.summary_stats)

causal.est_via_matching()

print("\nMatching Estimates:")
print(causal.estimates)

print("\nAfter Matching Balance:")
print(causal.summary_stats)

### 🌟 **2. Inverse Probability of Treatment Weighting (IPTW)**
# Logistic Regression for Propensity Scores
ps_model = LogisticRegression().fit(X_std, T)
propensity_scores = ps_model.predict_proba(X_std)[:, 0]

# Ensure stable weights (clipping extreme propensity scores)
# eps = 1e-6
# propensity_scores = np.clip(propensity_scores, eps, 1 - eps)
# weights = T / propensity_scores + (1 - T) / (1 - propensity_scores)

# Stabilized IPTW weights (standard practice):
treated_weights = T.mean() / propensity_scores
control_weights = (1 - T).mean() / (1 - propensity_scores)

weights = T * treated_weights + (1 - T) * control_weights


# Weighted regression to estimate causal effects
X_treatment = sm.add_constant(T)
iptw_model = sm.WLS(Y, X_treatment, weights=weights)
iptw_results = iptw_model.fit()
print("\nIPTW Regression Results (Corrected):")
print(iptw_results.summary())




### 🌟 **3. Doubly Robust Estimation**
seed = 12345
from sklearn.ensemble import RandomForestRegressor
from econml.dr import LinearDRLearner

regressor = RandomForestRegressor(random_state=seed)
propensity = LogisticRegression(random_state=seed)




dr_model = LinearDRLearner(model_regression=regressor,
                           model_propensity=propensity)
dr_model.fit(Y, T, X=X_std)
treatment_effect = dr_model.ate(X_std)
print(f"\nDoubly Robust ATE: {treatment_effect:.4f}")

### 🌟 **Additional Techniques for ATE Estimation**

# 4. Regression Adjustment (Outcome Modeling)
# Here, we run an OLS regression that includes the treatment indicator and the standardized covariates.
X_reg = sm.add_constant(np.column_stack([T, X_std]))
reg_model = sm.OLS(Y, X_reg).fit()
ate_reg_adj = reg_model.params[1]
print(f"\nRegression Adjustment ATE: {ate_reg_adj:.4f}")

#%%
w_treated = weights[T == 1]
w_control = weights[T == 0]
y_treated = Y[T == 1]
y_control = Y[T == 0]

# Weighted means
mean_treated = np.sum(w_treated * y_treated) / np.sum(w_treated)
mean_control = np.sum(w_control * y_control) / np.sum(w_control)

iptw_diff = mean_treated - mean_control
print("IPTW difference (manual) =", iptw_diff)

#%%
import matplotlib.pyplot as plt
plt.hist(propensity_scores[T==1], bins=30, alpha=0.5, label='Treated')
plt.hist(propensity_scores[T==0], bins=30, alpha=0.5, label='Control')
plt.legend()
plt.show()

#%%
dr_model = LinearDRLearner(model_regression=regressor,
                           model_propensity=propensity)
dr_model.fit(Y, T, X=X_std)
ate_dr = dr_model.ate(X_std)
print("DR ATE:", ate_dr)

#%%
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
# Assuming df is your original dataframe loaded already
columns = [
    'PCH_SNAP_12_17', 'PCT_OBESE_ADULTS17', 'PCT_LACCESS_POP10','PCT_LACCESS_LOWI10',
    'PCT_LACCESS_HHNV10','PCT_LACCESS_CHILD10','PCT_LACCESS_SENIORS10','GROCPTH11','SUPERCPTH11','CONVSPTH11','SPECSPTH11','SNAPSPTH12',
    'WICSPTH11','FSRPTH11','PC_FFRSALES07','PCT_NSLP12','PCT_FREE_LUNCH10','PCT_REDUCED_LUNCH10','PCT_SBP12','PCT_SFSP12','PCT_WIC12','PCT_WICINFANTCHILD14',
    'PCT_WICWOMEN14','PCT_CACFP12','FDPIR12','FMRKTPTH13','VEG_ACRESPTH07','FRESHVEG_ACRESPTH07','ORCHARD_ACRESPTH12','BERRY_ACRESPTH07','SLHOUSE07','GHVEG_SQFTPTH07',
    'AGRITRSM_OPS07','PCT_DIABETES_ADULTS08','PCT_OBESE_ADULTS12','RECFACPTH11','PCT_NHWHITE10','PCT_NHBLACK10','PCT_HISP10','PCT_NHASIAN10','PCT_NHNA10','PCT_NHPI10',
    'PCT_65OLDER10','PCT_18YOUNGER10','PERPOV10','METRO13','POPLOSS10','Urban'
]

# Drop NA values to maintain data quality
data = df[columns].dropna()

# Define treatment properly using median
median_snap = data['PCH_SNAP_12_17'].median()
data['High_SNAP'] = (data['PCH_SNAP_12_17'] > median_snap).astype(int)
# 1️⃣ Descriptive statistics by group
summary = data.groupby('High_SNAP')['PCT_OBESE_ADULTS17'].agg(
    count='count',
    mean='mean',
    median='median',
    std='std',
    min='min',
    max='max'
)
print("Descriptive stats:\n", summary)

# 2️⃣ Independent t‑test (with variance check)
levene_p = stats.levene(
    data.loc[data.High_SNAP==1, 'PCT_OBESE_ADULTS17'],
    data.loc[data.High_SNAP==0, 'PCT_OBESE_ADULTS17']
).pvalue
equal_var = True if levene_p > 0.05 else False

t_stat, t_p = stats.ttest_ind(
    data.loc[data.High_SNAP==1, 'PCT_OBESE_ADULTS17'],
    data.loc[data.High_SNAP==0, 'PCT_OBESE_ADULTS17'],
    equal_var=equal_var
)
print(f"\nLevene’s test p-value = {levene_p:.4f} → equal_var={equal_var}")
print(f"T‑test: t = {t_stat:.3f}, p = {t_p:.4f}")

# 3️⃣ Cohen’s d
mean_diff = summary.loc[1,'mean'] - summary.loc[0,'mean']
pooled_sd = np.sqrt(((summary.loc[1,'count']-1)*summary.loc[1,'std']**2 +
                     (summary.loc[0,'count']-1)*summary.loc[0,'std']**2) /
                    (summary.loc[1,'count'] + summary.loc[0,'count'] - 2))
cohens_d = mean_diff / pooled_sd
print(f"Cohen’s d = {cohens_d:.3f}")

# 4️⃣ Mann–Whitney U test (non-parametric)
u_stat, u_p = stats.mannwhitneyu(
    data.loc[data.High_SNAP==1, 'PCT_OBESE_ADULTS17'],
    data.loc[data.High_SNAP==0, 'PCT_OBESE_ADULTS17'],
    alternative='two-sided'
)
print(f"Mann–Whitney U: U = {u_stat:.3f}, p = {u_p:.4f}")

# 5️⃣ Visualizations
plt.figure()
data.boxplot(column='PCT_OBESE_ADULTS17', by='High_SNAP')
plt.title('Obesity by SNAP Level')
plt.suptitle('')
plt.xlabel('High SNAP (1 = above median)')
plt.ylabel('Obesity (%)')
plt.show()

plt.figure()
data[data.High_SNAP==1]['PCT_OBESE_ADULTS17'].hist(alpha=0.7, label='High SNAP')
data[data.High_SNAP==0]['PCT_OBESE_ADULTS17'].hist(alpha=0.7, label='Low SNAP')
plt.legend()
plt.title('Distribution of Onesity by SNAP Level')
plt.xlabel('Obesity (%)')
plt.ylabel('Frequency')
plt.show()

#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from causalinference import CausalModel
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

# Assuming df is your original dataframe loaded already
# columns = [
#     'PCH_SNAP_12_17', 'PCT_OBESE_ADULTS17', 'PCT_LACCESS_POP10','PCT_LACCESS_LOWI10',
#     'PCT_LACCESS_HHNV10','PCT_LACCESS_CHILD10','PCT_LACCESS_SENIORS10','GROCPTH11','SUPERCPTH11','CONVSPTH11','SPECSPTH11','SNAPSPTH12',
#     'WICSPTH11','FSRPTH11','PC_FFRSALES07','PCT_NSLP12','PCT_FREE_LUNCH10','PCT_REDUCED_LUNCH10','PCT_SBP12','PCT_SFSP12','PCT_WIC12','PCT_WICINFANTCHILD14',
#     'PCT_WICWOMEN14','PCT_CACFP12','FDPIR12','FMRKTPTH13','VEG_ACRESPTH07','FRESHVEG_ACRESPTH07','ORCHARD_ACRESPTH12','BERRY_ACRESPTH07','SLHOUSE07','GHVEG_SQFTPTH07',
#     'AGRITRSM_OPS07','PCT_DIABETES_ADULTS08','PCT_OBESE_ADULTS12','RECFACPTH11','PCT_NHWHITE10','PCT_NHBLACK10','PCT_HISP10','PCT_NHASIAN10','PCT_NHNA10','PCT_NHPI10',
#     'PCT_65OLDER10','PCT_18YOUNGER10','PERPOV10','METRO13','POPLOSS10','Urban'
# ]

# # Drop NA values to maintain data quality
# data = df[columns].dropna()

# # Define treatment properly using median
# median_snap = data['PCH_SNAP_12_17'].median()
# data['High_SNAP'] = (data['PCH_SNAP_12_17'] > median_snap).astype(int)

# Outcome and Treatment
Y = data['PCT_OBESE_ADULTS17'].values
T = data['High_SNAP'].values

# Confounders (covariates)
X = data.drop(columns=['PCH_SNAP_12_17', 'PCT_OBESE_ADULTS17', 'High_SNAP']).values

# Standardize confounders
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

### 🌟 **1. Propensity Score Matching (PSM)**

causal = CausalModel(Y, T, X_std)
causal.est_propensity()

print("\nBefore Matching Balance:")
print(causal.summary_stats)

causal.est_via_matching()

print("\nMatching Estimates:")
print(causal.estimates)

print("\nAfter Matching Balance:")
print(causal.summary_stats)

### 🌟 **2. Inverse Probability of Treatment Weighting (IPTW)**

# Logistic Regression for Propensity Scores
ps_model = LogisticRegression(max_iter=1000).fit(X_std, T)
propensity_scores = ps_model.predict_proba(X_std)[:, 0]

# Ensure stable weights (clipping extreme propensity scores)
# eps = 1e-6
# propensity_scores = np.clip(propensity_scores, eps, 1 - eps)
treated_weights = T.mean() / propensity_scores
control_weights = (1 - T).mean() / (1 - propensity_scores)

weights = T * treated_weights + (1 - T) * control_weights

# Weighted regression to estimate causal effects
X_treatment = sm.add_constant(T)
iptw_model = sm.WLS(Y, X_treatment, weights=weights)
iptw_results = iptw_model.fit()

print("\nIPTW Regression Results:")
print(iptw_results.summary())



## 📌 **Additional Robust Causal Inference Methods**

# Include these methods to strengthen your analysis:

### 🌟 **3. Doubly Robust Estimation (Recommended)**
seed = 12345

from sklearn.ensemble import RandomForestRegressor
from econml.dr import LinearDRLearner

regressor = RandomForestRegressor(random_state=seed, n_jobs=-1)
propensity = LogisticRegression(max_iter=1000, random_state=seed)

dr_model = LinearDRLearner(model_regression=regressor,
                           model_propensity=propensity)
dr_model.fit(Y, T, X=X_std)
treatment_effect = dr_model.ate(X_std)
print(f"Doubly Robust ATE: {treatment_effect:.4f}")

X_reg = sm.add_constant(np.column_stack([T, X_std]))
reg_model = sm.OLS(Y, X_reg).fit()
ate_reg_adj = reg_model.params[1]
print(f"\nRegression Adjustment ATE: {ate_reg_adj:.4f}")

#%%
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
# Assuming df is your original dataframe loaded already
columns = [
    'PCH_SNAP_12_17', 'PCT_DIABETES_ADULTS13', 'PCT_LACCESS_POP10','PCT_LACCESS_LOWI10',
    'PCT_LACCESS_HHNV10','PCT_LACCESS_CHILD10','PCT_LACCESS_SENIORS10','GROCPTH11','SUPERCPTH11','CONVSPTH11','SPECSPTH11','SNAPSPTH12',
    'WICSPTH11','FSRPTH11','PC_FFRSALES07','PCT_NSLP12','PCT_FREE_LUNCH10','PCT_REDUCED_LUNCH10','PCT_SBP12','PCT_SFSP12','PCT_WIC12','PCT_WICINFANTCHILD14',
    'PCT_WICWOMEN14','PCT_CACFP12','FDPIR12','FMRKTPTH13','VEG_ACRESPTH07','FRESHVEG_ACRESPTH07','ORCHARD_ACRESPTH12','BERRY_ACRESPTH07','SLHOUSE07','GHVEG_SQFTPTH07',
    'AGRITRSM_OPS07','PCT_DIABETES_ADULTS08','PCT_OBESE_ADULTS12','RECFACPTH11','PCT_NHWHITE10','PCT_NHBLACK10','PCT_HISP10','PCT_NHASIAN10','PCT_NHNA10','PCT_NHPI10',
    'PCT_65OLDER10','PCT_18YOUNGER10','PERPOV10','METRO13','POPLOSS10','Urban'
]

# Drop NA values to maintain data quality
data = df[columns].dropna()

# Define treatment properly using median
median_snap = data['PCH_SNAP_12_17'].median()
data['High_SNAP'] = (data['PCH_SNAP_12_17'] > median_snap).astype(int)
# 1️⃣ Descriptive statistics by group
summary = data.groupby('High_SNAP')['PCT_DIABETES_ADULTS13'].agg(
    count='count',
    mean='mean',
    median='median',
    std='std',
    min='min',
    max='max'
)
print("Descriptive stats:\n", summary)

# 2️⃣ Independent t‑test (with variance check)
levene_p = stats.levene(
    data.loc[data.High_SNAP==1, 'PCT_DIABETES_ADULTS13'],
    data.loc[data.High_SNAP==0, 'PCT_DIABETES_ADULTS13']
).pvalue
equal_var = True if levene_p > 0.05 else False

t_stat, t_p = stats.ttest_ind(
    data.loc[data.High_SNAP==1, 'PCT_DIABETES_ADULTS13'],
    data.loc[data.High_SNAP==0, 'PCT_DIABETES_ADULTS13'],
    equal_var=equal_var
)
print(f"\nLevene’s test p-value = {levene_p:.4f} → equal_var={equal_var}")
print(f"T‑test: t = {t_stat:.3f}, p = {t_p:.4f}")

# 3️⃣ Cohen’s d
mean_diff = summary.loc[1,'mean'] - summary.loc[0,'mean']
pooled_sd = np.sqrt(((summary.loc[1,'count']-1)*summary.loc[1,'std']**2 +
                     (summary.loc[0,'count']-1)*summary.loc[0,'std']**2) /
                    (summary.loc[1,'count'] + summary.loc[0,'count'] - 2))
cohens_d = mean_diff / pooled_sd
print(f"Cohen’s d = {cohens_d:.3f}")

# 4️⃣ Mann–Whitney U test (non-parametric)
u_stat, u_p = stats.mannwhitneyu(
    data.loc[data.High_SNAP==1, 'PCT_DIABETES_ADULTS13'],
    data.loc[data.High_SNAP==0, 'PCT_DIABETES_ADULTS13'],
    alternative='two-sided'
)
print(f"Mann–Whitney U: U = {u_stat:.3f}, p = {u_p:.4f}")

# 5️⃣ Visualizations
plt.figure()
data.boxplot(column='PCT_DIABETES_ADULTS13', by='High_SNAP')
plt.title('Diabetes by SNAP Level')
plt.suptitle('')
plt.xlabel('High SNAP (1 = above median)')
plt.ylabel('Diabetes (%)')
plt.show()

plt.figure()
data[data.High_SNAP==1]['PCT_DIABETES_ADULTS13'].hist(alpha=0.7, label='High SNAP')
data[data.High_SNAP==0]['PCT_DIABETES_ADULTS13'].hist(alpha=0.7, label='Low SNAP')
plt.legend()
plt.title('Distribution of Diabetes by SNAP Level')
plt.xlabel('Diabetes (%)')
plt.ylabel('Frequency')
plt.show()


#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from causalinference import CausalModel
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

# Assuming df is your original dataframe loaded already
# columns = [
#     'PCH_SNAP_12_17', 'PCT_DIABETES_ADULTS13', 'PCT_LACCESS_POP10','PCT_LACCESS_LOWI10',
#     'PCT_LACCESS_HHNV10','PCT_LACCESS_CHILD10','PCT_LACCESS_SENIORS10','GROCPTH11','SUPERCPTH11','CONVSPTH11','SPECSPTH11','SNAPSPTH12',
#     'WICSPTH11','FSRPTH11','PC_FFRSALES07','PCT_NSLP12','PCT_FREE_LUNCH10','PCT_REDUCED_LUNCH10','PCT_SBP12','PCT_SFSP12','PCT_WIC12','PCT_WICINFANTCHILD14',
#     'PCT_WICWOMEN14','PCT_CACFP12','FDPIR12','FMRKTPTH13','VEG_ACRESPTH07','FRESHVEG_ACRESPTH07','ORCHARD_ACRESPTH12','BERRY_ACRESPTH07','SLHOUSE07','GHVEG_SQFTPTH07',
#     'AGRITRSM_OPS07','PCT_DIABETES_ADULTS08','PCT_OBESE_ADULTS12','RECFACPTH11','PCT_NHWHITE10','PCT_NHBLACK10','PCT_HISP10','PCT_NHASIAN10','PCT_NHNA10','PCT_NHPI10',
#     'PCT_65OLDER10','PCT_18YOUNGER10','PERPOV10','METRO13','POPLOSS10','Urban'
# ]

# # Drop NA values to maintain data quality
# data = df[columns].dropna()

# # Define treatment properly using median
# median_snap = data['PCH_SNAP_12_17'].median()
# data['High_SNAP'] = (data['PCH_SNAP_12_17'] > median_snap).astype(int)

# Outcome and Treatment
Y = data['PCT_DIABETES_ADULTS13'].values
T = data['High_SNAP'].values

# Confounders (covariates)
X = data.drop(columns=['PCH_SNAP_12_17', 'PCT_DIABETES_ADULTS13', 'High_SNAP']).values

# Standardize confounders
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

### 🌟 **1. Propensity Score Matching (PSM)**

causal = CausalModel(Y, T, X_std)
causal.est_propensity()

print("\nBefore Matching Balance:")
print(causal.summary_stats)

causal.est_via_matching()

print("\nMatching Estimates:")
print(causal.estimates)

print("\nAfter Matching Balance:")
print(causal.summary_stats)

### 🌟 **2. Inverse Probability of Treatment Weighting (IPTW)**

# Logistic Regression for Propensity Scores
ps_model = LogisticRegression(max_iter=1000).fit(X_std, T)
propensity_scores = ps_model.predict_proba(X_std)[:, 0]

# Ensure stable weights (clipping extreme propensity scores)
eps = 1e-6
propensity_scores = np.clip(propensity_scores, eps, 1 - eps)
treated_weights = T.mean() / propensity_scores
control_weights = (1 - T).mean() / (1 - propensity_scores)

weights = T * treated_weights + (1 - T) * control_weights

# Weighted regression to estimate causal effects
X_treatment = sm.add_constant(T)
iptw_model = sm.WLS(Y, X_treatment, weights=weights)
iptw_results = iptw_model.fit()

print("\nIPTW Regression Results:")
print(iptw_results.summary())



## 📌 **Additional Robust Causal Inference Methods**

# Include these methods to strengthen your analysis:

### 🌟 **3. Doubly Robust Estimation (Recommended)**
seed = 12345

from sklearn.ensemble import RandomForestRegressor
from econml.dr import LinearDRLearner

regressor = RandomForestRegressor(random_state=seed, n_jobs=-1)
propensity = LogisticRegression(max_iter=1000, random_state=seed)

dr_model = LinearDRLearner(model_regression=regressor,
                           model_propensity=propensity)
dr_model.fit(Y, T, X=X_std)
treatment_effect = dr_model.ate(X_std)
print(f"Doubly Robust ATE: {treatment_effect:.4f}")

X_reg = sm.add_constant(np.column_stack([T, X_std]))
reg_model = sm.OLS(Y, X_reg).fit()
ate_reg_adj = reg_model.params[1]
print(f"\nRegression Adjustment ATE: {ate_reg_adj:.4f}")


#%%
from econml.cate_interpreter import SingleTreeCateInterpreter
import matplotlib.pyplot as plt

# Reduce complexity by using a shallower tree (max_depth=2)
cate_interpreter = SingleTreeCateInterpreter(max_depth=2, min_samples_leaf=30).interpret(dr_model, X_std)

plt.figure(figsize=(12,8))  # Increase plot size for readability
cate_interpreter.plot(
    feature_names=data.drop(columns=['PCH_SNAP_12_17', 'PCT_DIABETES_ADULTS13', 'High_SNAP']).columns,
    fontsize=12,
    precision=2,  # limit decimal points
    filled=True,  # colored boxes
    rounded=True
)

plt.title("Simplified Heterogeneous Treatment Effect (HTE) Analysis", fontsize=14, weight='bold')
plt.tight_layout()
plt.show()


#%%
col_name_map = {
    'PCH_SNAP_12_17':           'SNAP participants (change % pop), 2012‑17',
    'FOODINSEC_15_17':         'Household food insecurity (%), 2015‑17',
    'PCT_LACCESS_POP10':       'Population low access to store (%), 2010',
    'PCT_LACCESS_LOWI10':      'Low income & low access to store (%), 2010',
    'PCT_LACCESS_HHNV10':      'Households no car & low access to store (%), 2010',
    'PCT_LACCESS_CHILD10':     'Children low access to store (%), 2010',
    'PCT_LACCESS_SENIORS10':   'Seniors low access to store (%), 2010',
    'GROCPTH11':               'Grocery stores/1,000 pop, 2011',
    'SUPERCPTH11':             'Supercenters & club stores/1,000 pop, 2011',
    'CONVSPTH11':              'Convenience stores/1,000 pop, 2011',
    'SPECSPTH11':              'Specialized food stores/1,000 pop, 2011',
    'SNAPSPTH12':              'SNAP‑authorized stores/1,000 pop, 2012',
    'WICSPTH11':               'WIC‑authorized stores/1,000 pop, 2011',
    'FSRPTH11':                'Full‑service restaurants/1,000 pop, 2011',
    'PC_FFRSALES07':           'Fast food expenditures per capita, 2007',
    'PCT_NSLP12':              'National School Lunch Program (% children), 2012',
    'PCT_FREE_LUNCH10':        'Students eligible for free lunch (%), 2010',
    'PCT_REDUCED_LUNCH10':     'Students eligible for reduced‑price lunch (%), 2010',
    'PCT_SBP12':               'School Breakfast Program (% children), 2012',
    'PCT_SFSP12':              'Summer Food Service Program (% children), 2012',
    'PCT_WIC12':               'WIC participants (% pop), 2012',
    'PCT_WICINFANTCHILD14':    'WIC infant & children participants (%), 2014',
    'PCT_WICWOMEN14':          'WIC women participants (%), 2014',
    'PCT_CACFP12':             'Child & Adult Care Program (% pop), 2012',
    'FDPIR12':                 'FDPIR sites, 2012',
    'FMRKTPTH13':              'Farmers’ markets/1,000 pop, 2013',
    'VEG_ACRESPTH07':          'Vegetable acres/1,000 pop, 2007',
    'FRESHVEG_ACRESPTH07':     'Fresh market vegetable acres/1,000 pop, 2007',
    'ORCHARD_ACRESPTH12':      'Orchard acres/1,000 pop, 2012',
    'BERRY_ACRESPTH07':        'Berry acres/1,000 pop, 2007',
    'SLHOUSE07':               'Small slaughterhouse facilities, 2007',
    'GHVEG_SQFTPTH07':         'Greenhouse veg sq ft/1,000 pop, 2007',
    'AGRITRSM_OPS07':          'Agritourism operations, 2007',
    'PCT_DIABETES_ADULTS08':   'Adult diabetes rate (%), 2008',
    'PCT_OBESE_ADULTS12':      'Adult obesity rate (%), 2012',
    'RECFACPTH11':             'Recreation & fitness facilities/1,000 pop, 2011',
    'PCT_NHWHITE10':           '% White, 2010',
    'PCT_NHBLACK10':           '% Black, 2010',
    'PCT_HISP10':              '% Hispanic, 2010',
    'PCT_NHASIAN10':           '% Asian, 2010',
    'PCT_NHNA10':              '% American Indian/Alaska Native, 2010',
    'PCT_NHPI10':              '% Hawaiian/Pacific Islander, 2010',
    'PCT_65OLDER10':           '% Population ≥65, 2010',
    'PCT_18YOUNGER10':         '% Population <18, 2010',
    'PERPOV10':                'Persistent‑poverty counties, 2010',
    'METRO13':                 'Metro/nonmetro status, 2010',
    'POPLOSS10':               'Population‑loss counties, 2010'
}

#%%
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
# Assuming df is your original dataframe loaded already
columns = [
    'PCH_CACFP_12_17', 'FOODINSEC_15_17', 'PCT_LACCESS_POP10','PCT_LACCESS_LOWI10',
    'PCT_LACCESS_HHNV10','PCT_LACCESS_CHILD10','PCT_LACCESS_SENIORS10','GROCPTH11','SUPERCPTH11','CONVSPTH11','SPECSPTH11','SNAPSPTH12',
    'WICSPTH11','FSRPTH11','PC_FFRSALES07','PCT_NSLP12','PCT_FREE_LUNCH10','PCT_REDUCED_LUNCH10','PCT_SBP12','PCT_SFSP12','PCT_WIC12','PCT_WICINFANTCHILD14',
    'PCT_WICWOMEN14','PCT_CACFP12','FDPIR12','FMRKTPTH13','VEG_ACRESPTH07','FRESHVEG_ACRESPTH07','ORCHARD_ACRESPTH12','BERRY_ACRESPTH07','SLHOUSE07','GHVEG_SQFTPTH07',
    'AGRITRSM_OPS07','PCT_DIABETES_ADULTS08','PCT_OBESE_ADULTS12','RECFACPTH11','PCT_NHWHITE10','PCT_NHBLACK10','PCT_HISP10','PCT_NHASIAN10','PCT_NHNA10','PCT_NHPI10',
    'PCT_65OLDER10','PCT_18YOUNGER10','PERPOV10','METRO13','POPLOSS10','Urban'
]

# Drop NA values to maintain data quality
data = df[columns].dropna()

# Define treatment properly using median
median_snap = data['PCH_CACFP_12_17'].median()
data['High_CACFP'] = (data['PCH_CACFP_12_17'] > median_snap).astype(int)
# 1️⃣ Descriptive statistics by group
summary = data.groupby('High_CACFP')['FOODINSEC_15_17'].agg(
    count='count',
    mean='mean',
    median='median',
    std='std',
    min='min',
    max='max'
)
print("Descriptive stats:\n", summary)

# 2️⃣ Independent t‑test (with variance check)
levene_p = stats.levene(
    data.loc[data.High_CACFP==1, 'FOODINSEC_15_17'],
    data.loc[data.High_CACFP==0, 'FOODINSEC_15_17']
).pvalue
equal_var = True if levene_p > 0.05 else False

t_stat, t_p = stats.ttest_ind(
    data.loc[data.High_CACFP==1, 'FOODINSEC_15_17'],
    data.loc[data.High_CACFP==0, 'FOODINSEC_15_17'],
    equal_var=equal_var
)
print(f"\nLevene’s test p-value = {levene_p:.4f} → equal_var={equal_var}")
print(f"T‑test: t = {t_stat:.3f}, p = {t_p:.4f}")

# 3️⃣ Cohen’s d
mean_diff = summary.loc[1,'mean'] - summary.loc[0,'mean']
pooled_sd = np.sqrt(((summary.loc[1,'count']-1)*summary.loc[1,'std']**2 +
                     (summary.loc[0,'count']-1)*summary.loc[0,'std']**2) /
                    (summary.loc[1,'count'] + summary.loc[0,'count'] - 2))
cohens_d = mean_diff / pooled_sd
print(f"Cohen’s d = {cohens_d:.3f}")

# 4️⃣ Mann–Whitney U test (non-parametric)
u_stat, u_p = stats.mannwhitneyu(
    data.loc[data.High_CACFP==1, 'FOODINSEC_15_17'],
    data.loc[data.High_CACFP==0, 'FOODINSEC_15_17'],
    alternative='two-sided'
)
print(f"Mann–Whitney U: U = {u_stat:.3f}, p = {u_p:.4f}")

# 5️⃣ Visualizations
plt.figure()
data.boxplot(column='FOODINSEC_15_17', by='High_CACFP')
plt.title('Food Insecurity by CACFP Level')
plt.suptitle('')
plt.xlabel('High CACFP (1 = above median)')
plt.ylabel('Food Insecurity (%)')
plt.show()

plt.figure()
data[data.High_CACFP==1]['FOODINSEC_15_17'].hist(alpha=0.7, label='High SNAP')
data[data.High_CACFP==0]['FOODINSEC_15_17'].hist(alpha=0.7, label='Low SNAP')
plt.legend()
plt.title('Distribution of Food Insecurity by CACFP Level')
plt.xlabel('Food Insecurity (%)')
plt.ylabel('Frequency')
plt.show()


#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from causalinference import CausalModel
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm



# Assuming df is your original dataframe loaded already
# columns = [
#     'PCH_CACFP_12_17', 'FOODINSEC_15_17', 'PCT_LACCESS_POP10','PCT_LACCESS_LOWI10',
#     'PCT_LACCESS_HHNV10','PCT_LACCESS_CHILD10','PCT_LACCESS_SENIORS10','GROCPTH11','SUPERCPTH11','CONVSPTH11','SPECSPTH11','SNAPSPTH12',
#     'WICSPTH11','FSRPTH11','PC_FFRSALES07','PCT_NSLP12','PCT_FREE_LUNCH10','PCT_REDUCED_LUNCH10','PCT_SBP12','PCT_SFSP12','PCT_WIC12','PCT_WICINFANTCHILD14',
#     'PCT_WICWOMEN14','PCT_CACFP12','FDPIR12','FMRKTPTH13','VEG_ACRESPTH07','FRESHVEG_ACRESPTH07','ORCHARD_ACRESPTH12','BERRY_ACRESPTH07','SLHOUSE07','GHVEG_SQFTPTH07',
#     'AGRITRSM_OPS07','PCT_DIABETES_ADULTS08','PCT_OBESE_ADULTS12','RECFACPTH11','PCT_NHWHITE10','PCT_NHBLACK10','PCT_HISP10','PCT_NHASIAN10','PCT_NHNA10','PCT_NHPI10',
#     'PCT_65OLDER10','PCT_18YOUNGER10','PERPOV10','METRO13','POPLOSS10','Urban'
# ]

# # Drop NA values to maintain data quality
# data = df[columns].dropna()

# # Define treatment properly using median
# median_snap = data['PCH_CACFP_12_17'].median()
# data['High_CACFP'] = (data['PCH_CACFP_12_17'] > median_snap).astype(int)

# Outcome and Treatment
Y = data['FOODINSEC_15_17'].values
T = data['High_CACFP'].values

# Confounders (covariates)
X = data.drop(columns=['PCH_CACFP_12_17', 'FOODINSEC_15_17', 'High_CACFP']).values

# Standardize confounders
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

### 🌟 **1. Propensity Score Matching (PSM)**

causal = CausalModel(Y, T, X_std)
causal.est_propensity()

print("\nBefore Matching Balance:")
print(causal.summary_stats)

causal.est_via_matching()

print("\nMatching Estimates:")
print(causal.estimates)

print("\nAfter Matching Balance:")
print(causal.summary_stats)

### 🌟 **2. Inverse Probability of Treatment Weighting (IPTW)**

# Logistic Regression for Propensity Scores
ps_model = LogisticRegression(max_iter=1000).fit(X_std, T)
propensity_scores = ps_model.predict_proba(X_std)[:, 1]

# Ensure stable weights (clipping extreme propensity scores)
eps = 1e-6
propensity_scores = np.clip(propensity_scores, eps, 1 - eps)
weights = T / propensity_scores + (1 - T) / (1 - propensity_scores)

# Weighted regression to estimate causal effects
X_treatment = sm.add_constant(T)
iptw_model = sm.WLS(Y, X_treatment, weights=weights)
iptw_results = iptw_model.fit()

print("\nIPTW Regression Results:")
print(iptw_results.summary())



## 📌 **Additional Robust Causal Inference Methods**

# Include these methods to strengthen your analysis:

### 🌟 **3. Doubly Robust Estimation (Recommended)**
seed = 12345

from sklearn.ensemble import RandomForestRegressor
from econml.dr import LinearDRLearner

regressor = RandomForestRegressor(random_state=seed, n_jobs=-1)
propensity = LogisticRegression(max_iter=1000, random_state=seed)

dr_model = LinearDRLearner(model_regression=regressor,
                           model_propensity=propensity)
dr_model.fit(Y, T, X=X_std)
treatment_effect = dr_model.ate(X_std)
print(f"Doubly Robust ATE: {treatment_effect:.4f}")


#%%
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
# Assuming df is your original dataframe loaded already
columns = [
    'PCH_CACFP_12_17', 'PCT_OBESE_ADULTS17', 'PCT_LACCESS_POP10','PCT_LACCESS_LOWI10',
    'PCT_LACCESS_HHNV10','PCT_LACCESS_CHILD10','PCT_LACCESS_SENIORS10','GROCPTH11','SUPERCPTH11','CONVSPTH11','SPECSPTH11','SNAPSPTH12',
    'WICSPTH11','FSRPTH11','PC_FFRSALES07','PCT_NSLP12','PCT_FREE_LUNCH10','PCT_REDUCED_LUNCH10','PCT_SBP12','PCT_SFSP12','PCT_WIC12','PCT_WICINFANTCHILD14',
    'PCT_WICWOMEN14','PCT_CACFP12','FDPIR12','FMRKTPTH13','VEG_ACRESPTH07','FRESHVEG_ACRESPTH07','ORCHARD_ACRESPTH12','BERRY_ACRESPTH07','SLHOUSE07','GHVEG_SQFTPTH07',
    'AGRITRSM_OPS07','PCT_DIABETES_ADULTS08','PCT_OBESE_ADULTS12','RECFACPTH11','PCT_NHWHITE10','PCT_NHBLACK10','PCT_HISP10','PCT_NHASIAN10','PCT_NHNA10','PCT_NHPI10',
    'PCT_65OLDER10','PCT_18YOUNGER10','PERPOV10','METRO13','POPLOSS10','Urban'
]

# Drop NA values to maintain data quality
data = df[columns].dropna()

# Define treatment properly using median
median_snap = data['PCH_CACFP_12_17'].median()
data['High_CACFP'] = (data['PCH_CACFP_12_17'] > median_snap).astype(int)
# 1️⃣ Descriptive statistics by group
summary = data.groupby('High_CACFP')['PCT_OBESE_ADULTS17'].agg(
    count='count',
    mean='mean',
    median='median',
    std='std',
    min='min',
    max='max'
)
print("Descriptive stats:\n", summary)

# 2️⃣ Independent t‑test (with variance check)
levene_p = stats.levene(
    data.loc[data.High_CACFP==1, 'PCT_OBESE_ADULTS17'],
    data.loc[data.High_CACFP==0, 'PCT_OBESE_ADULTS17']
).pvalue
equal_var = True if levene_p > 0.05 else False

t_stat, t_p = stats.ttest_ind(
    data.loc[data.High_CACFP==1, 'PCT_OBESE_ADULTS17'],
    data.loc[data.High_CACFP==0, 'PCT_OBESE_ADULTS17'],
    equal_var=equal_var
)
print(f"\nLevene’s test p-value = {levene_p:.4f} → equal_var={equal_var}")
print(f"T‑test: t = {t_stat:.3f}, p = {t_p:.4f}")

# 3️⃣ Cohen’s d
mean_diff = summary.loc[1,'mean'] - summary.loc[0,'mean']
pooled_sd = np.sqrt(((summary.loc[1,'count']-1)*summary.loc[1,'std']**2 +
                     (summary.loc[0,'count']-1)*summary.loc[0,'std']**2) /
                    (summary.loc[1,'count'] + summary.loc[0,'count'] - 2))
cohens_d = mean_diff / pooled_sd
print(f"Cohen’s d = {cohens_d:.3f}")

# 4️⃣ Mann–Whitney U test (non-parametric)
u_stat, u_p = stats.mannwhitneyu(
    data.loc[data.High_CACFP==1, 'PCT_OBESE_ADULTS17'],
    data.loc[data.High_CACFP==0, 'PCT_OBESE_ADULTS17'],
    alternative='two-sided'
)
print(f"Mann–Whitney U: U = {u_stat:.3f}, p = {u_p:.4f}")

# 5️⃣ Visualizations
plt.figure()
data.boxplot(column='PCT_OBESE_ADULTS17', by='High_CACFP')
plt.title('Obesity by CACFP Level')
plt.suptitle('')
plt.xlabel('High CACFP (1 = above median)')
plt.ylabel('Obesity (%)')
plt.show()

plt.figure()
data[data.High_CACFP==1]['PCT_OBESE_ADULTS17'].hist(alpha=0.7, label='High SNAP')
data[data.High_CACFP==0]['PCT_OBESE_ADULTS17'].hist(alpha=0.7, label='Low SNAP')
plt.legend()
plt.title('Distribution of Obesity by CACFP Level')
plt.xlabel('Obesity (%)')
plt.ylabel('Frequency')
plt.show()


#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from causalinference import CausalModel
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm



# Assuming df is your original dataframe loaded already
columns = [
    'PCH_CACFP_12_17', 'PCT_OBESE_ADULTS17', 'PCT_LACCESS_POP10','PCT_LACCESS_LOWI10',
    'PCT_LACCESS_HHNV10','PCT_LACCESS_CHILD10','PCT_LACCESS_SENIORS10','GROCPTH11','SUPERCPTH11','CONVSPTH11','SPECSPTH11','SNAPSPTH12',
    'WICSPTH11','FSRPTH11','PC_FFRSALES07','PCT_NSLP12','PCT_FREE_LUNCH10','PCT_REDUCED_LUNCH10','PCT_SBP12','PCT_SFSP12','PCT_WIC12','PCT_WICINFANTCHILD14',
    'PCT_WICWOMEN14','PCT_CACFP12','FDPIR12','FMRKTPTH13','VEG_ACRESPTH07','FRESHVEG_ACRESPTH07','ORCHARD_ACRESPTH12','BERRY_ACRESPTH07','SLHOUSE07','GHVEG_SQFTPTH07',
    'AGRITRSM_OPS07','PCT_DIABETES_ADULTS08','PCT_OBESE_ADULTS12','RECFACPTH11','PCT_NHWHITE10','PCT_NHBLACK10','PCT_HISP10','PCT_NHASIAN10','PCT_NHNA10','PCT_NHPI10',
    'PCT_65OLDER10','PCT_18YOUNGER10','PERPOV10','METRO13','POPLOSS10','Urban'
]

# Drop NA values to maintain data quality
data = df[columns].dropna()

# Define treatment properly using median
median_snap = data['PCH_CACFP_12_17'].median()
data['High_CACFP'] = (data['PCH_CACFP_12_17'] > median_snap).astype(int)

# Outcome and Treatment
Y = data['PCT_OBESE_ADULTS17'].values
T = data['High_CACFP'].values

# Confounders (covariates)
X = data.drop(columns=['PCH_CACFP_12_17', 'PCT_OBESE_ADULTS17', 'High_CACFP']).values

# Standardize confounders
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

### 🌟 **1. Propensity Score Matching (PSM)**

causal = CausalModel(Y, T, X_std)
causal.est_propensity()

print("\nBefore Matching Balance:")
print(causal.summary_stats)

causal.est_via_matching()

print("\nMatching Estimates:")
print(causal.estimates)

print("\nAfter Matching Balance:")
print(causal.summary_stats)

### 🌟 **2. Inverse Probability of Treatment Weighting (IPTW)**

# Logistic Regression for Propensity Scores
ps_model = LogisticRegression(max_iter=1000).fit(X_std, T)
propensity_scores = ps_model.predict_proba(X_std)[:, 1]

# Ensure stable weights (clipping extreme propensity scores)
eps = 1e-6
propensity_scores = np.clip(propensity_scores, eps, 1 - eps)
weights = T / propensity_scores + (1 - T) / (1 - propensity_scores)

# Weighted regression to estimate causal effects
X_treatment = sm.add_constant(T)
iptw_model = sm.WLS(Y, X_treatment, weights=weights)
iptw_results = iptw_model.fit()

print("\nIPTW Regression Results:")
print(iptw_results.summary())



## 📌 **Additional Robust Causal Inference Methods**

# Include these methods to strengthen your analysis:

### 🌟 **3. Doubly Robust Estimation (Recommended)**
seed = 12345

from sklearn.ensemble import RandomForestRegressor
from econml.dr import LinearDRLearner

regressor = RandomForestRegressor(random_state=seed, n_jobs=-1)
propensity = LogisticRegression(max_iter=1000, random_state=seed)

dr_model = LinearDRLearner(model_regression=regressor,
                           model_propensity=propensity)
dr_model.fit(Y, T, X=X_std)
treatment_effect = dr_model.ate(X_std)
print(f"Doubly Robust ATE: {treatment_effect:.4f}")


#%%
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
# Assuming df is your original dataframe loaded already
columns = [
    'PCH_CACFP_12_17', 'PCT_DIABETES_ADULTS13', 'PCT_LACCESS_POP10','PCT_LACCESS_LOWI10',
    'PCT_LACCESS_HHNV10','PCT_LACCESS_CHILD10','PCT_LACCESS_SENIORS10','GROCPTH11','SUPERCPTH11','CONVSPTH11','SPECSPTH11','SNAPSPTH12',
    'WICSPTH11','FSRPTH11','PC_FFRSALES07','PCT_NSLP12','PCT_FREE_LUNCH10','PCT_REDUCED_LUNCH10','PCT_SBP12','PCT_SFSP12','PCT_WIC12','PCT_WICINFANTCHILD14',
    'PCT_WICWOMEN14','PCT_CACFP12','FDPIR12','FMRKTPTH13','VEG_ACRESPTH07','FRESHVEG_ACRESPTH07','ORCHARD_ACRESPTH12','BERRY_ACRESPTH07','SLHOUSE07','GHVEG_SQFTPTH07',
    'AGRITRSM_OPS07','PCT_DIABETES_ADULTS08','PCT_OBESE_ADULTS12','RECFACPTH11','PCT_NHWHITE10','PCT_NHBLACK10','PCT_HISP10','PCT_NHASIAN10','PCT_NHNA10','PCT_NHPI10',
    'PCT_65OLDER10','PCT_18YOUNGER10','PERPOV10','METRO13','POPLOSS10','Urban'
]

# Drop NA values to maintain data quality
data = df[columns].dropna()

# Define treatment properly using median
median_snap = data['PCH_CACFP_12_17'].median()
data['High_CACFP'] = (data['PCH_CACFP_12_17'] > median_snap).astype(int)
# 1️⃣ Descriptive statistics by group
summary = data.groupby('High_CACFP')['PCT_DIABETES_ADULTS13'].agg(
    count='count',
    mean='mean',
    median='median',
    std='std',
    min='min',
    max='max'
)
print("Descriptive stats:\n", summary)

# 2️⃣ Independent t‑test (with variance check)
levene_p = stats.levene(
    data.loc[data.High_CACFP==1, 'PCT_DIABETES_ADULTS13'],
    data.loc[data.High_CACFP==0, 'PCT_DIABETES_ADULTS13']
).pvalue
equal_var = True if levene_p > 0.05 else False

t_stat, t_p = stats.ttest_ind(
    data.loc[data.High_CACFP==1, 'PCT_DIABETES_ADULTS13'],
    data.loc[data.High_CACFP==0, 'PCT_DIABETES_ADULTS13'],
    equal_var=equal_var
)
print(f"\nLevene’s test p-value = {levene_p:.4f} → equal_var={equal_var}")
print(f"T‑test: t = {t_stat:.3f}, p = {t_p:.4f}")

# 3️⃣ Cohen’s d
mean_diff = summary.loc[1,'mean'] - summary.loc[0,'mean']
pooled_sd = np.sqrt(((summary.loc[1,'count']-1)*summary.loc[1,'std']**2 +
                     (summary.loc[0,'count']-1)*summary.loc[0,'std']**2) /
                    (summary.loc[1,'count'] + summary.loc[0,'count'] - 2))
cohens_d = mean_diff / pooled_sd
print(f"Cohen’s d = {cohens_d:.3f}")

# 4️⃣ Mann–Whitney U test (non-parametric)
u_stat, u_p = stats.mannwhitneyu(
    data.loc[data.High_CACFP==1, 'PCT_DIABETES_ADULTS13'],
    data.loc[data.High_CACFP==0, 'PCT_DIABETES_ADULTS13'],
    alternative='two-sided'
)
print(f"Mann–Whitney U: U = {u_stat:.3f}, p = {u_p:.4f}")

# 5️⃣ Visualizations
plt.figure()
data.boxplot(column='PCT_DIABETES_ADULTS13', by='High_CACFP')
plt.title('Diabetes by CACFP Level')
plt.suptitle('')
plt.xlabel('High CACFP (1 = above median)')
plt.ylabel('Diabetes (%)')
plt.show()

plt.figure()
data[data.High_CACFP==1]['PCT_DIABETES_ADULTS13'].hist(alpha=0.7, label='High SNAP')
data[data.High_CACFP==0]['PCT_DIABETES_ADULTS13'].hist(alpha=0.7, label='Low SNAP')
plt.legend()
plt.title('Distribution of Diabetes by CACFP Level')
plt.xlabel('Diabetes (%)')
plt.ylabel('Frequency')
plt.show()

#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from causalinference import CausalModel
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm



# Assuming df is your original dataframe loaded already
columns = [
    'PCH_CACFP_12_17', 'PCT_DIABETES_ADULTS13', 'PCT_LACCESS_POP10','PCT_LACCESS_LOWI10',
    'PCT_LACCESS_HHNV10','PCT_LACCESS_CHILD10','PCT_LACCESS_SENIORS10','GROCPTH11','SUPERCPTH11','CONVSPTH11','SPECSPTH11','SNAPSPTH12',
    'WICSPTH11','FSRPTH11','PC_FFRSALES07','PCT_NSLP12','PCT_FREE_LUNCH10','PCT_REDUCED_LUNCH10','PCT_SBP12','PCT_SFSP12','PCT_WIC12','PCT_WICINFANTCHILD14',
    'PCT_WICWOMEN14','PCT_CACFP12','FDPIR12','FMRKTPTH13','VEG_ACRESPTH07','FRESHVEG_ACRESPTH07','ORCHARD_ACRESPTH12','BERRY_ACRESPTH07','SLHOUSE07','GHVEG_SQFTPTH07',
    'AGRITRSM_OPS07','PCT_DIABETES_ADULTS08','PCT_OBESE_ADULTS12','RECFACPTH11','PCT_NHWHITE10','PCT_NHBLACK10','PCT_HISP10','PCT_NHASIAN10','PCT_NHNA10','PCT_NHPI10',
    'PCT_65OLDER10','PCT_18YOUNGER10','PERPOV10','METRO13','POPLOSS10','Urban'
]

# Drop NA values to maintain data quality
data = df[columns].dropna()

# Define treatment properly using median
median_snap = data['PCH_CACFP_12_17'].median()
data['High_CACFP'] = (data['PCH_CACFP_12_17'] > median_snap).astype(int)

# Outcome and Treatment
Y = data['PCT_DIABETES_ADULTS13'].values
T = data['High_CACFP'].values

# Confounders (covariates)
X = data.drop(columns=['PCH_CACFP_12_17', 'PCT_DIABETES_ADULTS13', 'High_CACFP']).values

# Standardize confounders
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

### 🌟 **1. Propensity Score Matching (PSM)**

causal = CausalModel(Y, T, X_std)
causal.est_propensity()

print("\nBefore Matching Balance:")
print(causal.summary_stats)

causal.est_via_matching()

print("\nMatching Estimates:")
print(causal.estimates)

print("\nAfter Matching Balance:")
print(causal.summary_stats)

### 🌟 **2. Inverse Probability of Treatment Weighting (IPTW)**

# Logistic Regression for Propensity Scores
ps_model = LogisticRegression(max_iter=1000).fit(X_std, T)
propensity_scores = ps_model.predict_proba(X_std)[:, 1]

# Ensure stable weights (clipping extreme propensity scores)
eps = 1e-6
propensity_scores = np.clip(propensity_scores, eps, 1 - eps)
weights = T / propensity_scores + (1 - T) / (1 - propensity_scores)

# Weighted regression to estimate causal effects
X_treatment = sm.add_constant(T)
iptw_model = sm.WLS(Y, X_treatment, weights=weights)
iptw_results = iptw_model.fit()

print("\nIPTW Regression Results:")
print(iptw_results.summary())



## 📌 **Additional Robust Causal Inference Methods**

# Include these methods to strengthen your analysis:

### 🌟 **3. Doubly Robust Estimation (Recommended)**
seed = 12345

from sklearn.ensemble import RandomForestRegressor
from econml.dr import LinearDRLearner

regressor = RandomForestRegressor(random_state=seed, n_jobs=-1)
propensity = LogisticRegression(max_iter=1000, random_state=seed)

dr_model = LinearDRLearner(model_regression=regressor,
                           model_propensity=propensity)
dr_model.fit(Y, T, X=X_std)
treatment_effect = dr_model.ate(X_std)
print(f"Doubly Robust ATE: {treatment_effect:.4f}")

#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from causalinference import CausalModel
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm



# Assuming df is your original dataframe loaded already
columns = [
    'PCH_REDEMP_WICS_11_16', 'FOODINSEC_15_17', 'PCT_LACCESS_POP10','PCT_LACCESS_LOWI10',
    'PCT_LACCESS_HHNV10','PCT_LACCESS_CHILD10','PCT_LACCESS_SENIORS10','GROCPTH11','SUPERCPTH11','CONVSPTH11','SPECSPTH11','SNAPSPTH12',
    'WICSPTH11','FSRPTH11','PC_FFRSALES07','PCT_NSLP12','PCT_FREE_LUNCH10','PCT_REDUCED_LUNCH10','PCT_SBP12','PCT_SFSP12','PCT_WIC12','PCT_WICINFANTCHILD14',
    'PCT_WICWOMEN14','PCT_CACFP12','FDPIR12','FMRKTPTH13','VEG_ACRESPTH07','FRESHVEG_ACRESPTH07','ORCHARD_ACRESPTH12','BERRY_ACRESPTH07','SLHOUSE07','GHVEG_SQFTPTH07',
    'AGRITRSM_OPS07','PCT_DIABETES_ADULTS08','PCT_OBESE_ADULTS12','RECFACPTH11','PCT_NHWHITE10','PCT_NHBLACK10','PCT_HISP10','PCT_NHASIAN10','PCT_NHNA10','PCT_NHPI10',
    'PCT_65OLDER10','PCT_18YOUNGER10','PERPOV10','METRO13','POPLOSS10','Urban'
]

# Drop NA values to maintain data quality
data = df[columns].dropna()

# Define treatment properly using median
median_snap = data['PCH_REDEMP_WICS_11_16'].median()
data['High_WIC'] = (data['PCH_REDEMP_WICS_11_16'] > median_snap).astype(int)

# Outcome and Treatment
Y = data['FOODINSEC_15_17'].values
T = data['High_WIC'].values

# Confounders (covariates)
X = data.drop(columns=['PCH_REDEMP_WICS_11_16', 'FOODINSEC_15_17', 'High_WIC']).values

# Standardize confounders
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

### 🌟 **1. Propensity Score Matching (PSM)**

causal = CausalModel(Y, T, X_std)
causal.est_propensity()

print("\nBefore Matching Balance:")
print(causal.summary_stats)

causal.est_via_matching()

print("\nMatching Estimates:")
print(causal.estimates)

print("\nAfter Matching Balance:")
print(causal.summary_stats)

### 🌟 **2. Inverse Probability of Treatment Weighting (IPTW)**

# Logistic Regression for Propensity Scores
ps_model = LogisticRegression(max_iter=1000).fit(X_std, T)
propensity_scores = ps_model.predict_proba(X_std)[:, 1]

# Ensure stable weights (clipping extreme propensity scores)
eps = 1e-6
propensity_scores = np.clip(propensity_scores, eps, 1 - eps)
weights = T / propensity_scores + (1 - T) / (1 - propensity_scores)

# Weighted regression to estimate causal effects
X_treatment = sm.add_constant(T)
iptw_model = sm.WLS(Y, X_treatment, weights=weights)
iptw_results = iptw_model.fit()

print("\nIPTW Regression Results:")
print(iptw_results.summary())



## 📌 **Additional Robust Causal Inference Methods**

# Include these methods to strengthen your analysis:

### 🌟 **3. Doubly Robust Estimation (Recommended)**
seed = 12345

from sklearn.ensemble import RandomForestRegressor
from econml.dr import LinearDRLearner

regressor = RandomForestRegressor(random_state=seed, n_jobs=-1)
propensity = LogisticRegression(max_iter=1000, random_state=seed)

dr_model = LinearDRLearner(model_regression=regressor,
                           model_propensity=propensity)
dr_model.fit(Y, T, X=X_std)
treatment_effect = dr_model.ate(X_std)
print(f"Doubly Robust ATE: {treatment_effect:.4f}")

#%%
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
# Assuming df is your original dataframe loaded already
columns = [
    'PCH_PC_WIC_REDEMP_11_16', 'FOODINSEC_15_17', 'PCT_LACCESS_POP10','PCT_LACCESS_LOWI10',
    'PCT_LACCESS_HHNV10','PCT_LACCESS_CHILD10','PCT_LACCESS_SENIORS10','GROCPTH11','SUPERCPTH11','CONVSPTH11','SPECSPTH11','SNAPSPTH12',
    'WICSPTH11','FSRPTH11','PC_FFRSALES07','PCT_NSLP12','PCT_FREE_LUNCH10','PCT_REDUCED_LUNCH10','PCT_SBP12','PCT_SFSP12','PCT_WIC12','PCT_WICINFANTCHILD14',
    'PCT_WICWOMEN14','PCT_CACFP12','FDPIR12','FMRKTPTH13','VEG_ACRESPTH07','FRESHVEG_ACRESPTH07','ORCHARD_ACRESPTH12','BERRY_ACRESPTH07','SLHOUSE07','GHVEG_SQFTPTH07',
    'AGRITRSM_OPS07','PCT_DIABETES_ADULTS08','PCT_OBESE_ADULTS12','RECFACPTH11','PCT_NHWHITE10','PCT_NHBLACK10','PCT_HISP10','PCT_NHASIAN10','PCT_NHNA10','PCT_NHPI10',
    'PCT_65OLDER10','PCT_18YOUNGER10','PERPOV10','METRO13','POPLOSS10'
]

# Drop NA values to maintain data quality
data = df[columns].dropna()

# Define treatment properly using median
median_snap = data['PCH_PC_WIC_REDEMP_11_16'].median()
data['High_WIC_perC'] = (data['PCH_PC_WIC_REDEMP_11_16'] > median_snap).astype(int)
# 1️⃣ Descriptive statistics by group
summary = data.groupby('High_WIC_perC')['FOODINSEC_15_17'].agg(
    count='count',
    mean='mean',
    median='median',
    std='std',
    min='min',
    max='max'
)
print("Descriptive stats:\n", summary)

# 2️⃣ Independent t‑test (with variance check)
levene_p = stats.levene(
    data.loc[data.High_WIC_perC==1, 'FOODINSEC_15_17'],
    data.loc[data.High_WIC_perC==0, 'FOODINSEC_15_17']
).pvalue
equal_var = True if levene_p > 0.05 else False

t_stat, t_p = stats.ttest_ind(
    data.loc[data.High_WIC_perC==1, 'FOODINSEC_15_17'],
    data.loc[data.High_WIC_perC==0, 'FOODINSEC_15_17'],
    equal_var=equal_var
)
print(f"\nLevene’s test p-value = {levene_p:.4f} → equal_var={equal_var}")
print(f"T‑test: t = {t_stat:.3f}, p = {t_p:.4f}")

# 3️⃣ Cohen’s d
mean_diff = summary.loc[1,'mean'] - summary.loc[0,'mean']
pooled_sd = np.sqrt(((summary.loc[1,'count']-1)*summary.loc[1,'std']**2 +
                     (summary.loc[0,'count']-1)*summary.loc[0,'std']**2) /
                    (summary.loc[1,'count'] + summary.loc[0,'count'] - 2))
cohens_d = mean_diff / pooled_sd
print(f"Cohen’s d = {cohens_d:.3f}")

# 4️⃣ Mann–Whitney U test (non-parametric)
u_stat, u_p = stats.mannwhitneyu(
    data.loc[data.High_WIC_perC==1, 'FOODINSEC_15_17'],
    data.loc[data.High_WIC_perC==0, 'FOODINSEC_15_17'],
    alternative='two-sided'
)
print(f"Mann–Whitney U: U = {u_stat:.3f}, p = {u_p:.4f}")

# 5️⃣ Visualizations
plt.figure()
data.boxplot(column='FOODINSEC_15_17', by='High_WIC_perC')
plt.title('Food Insecurity by WIC Level')
plt.suptitle('')
plt.xlabel('High WIC (1 = above median)')
plt.ylabel('Food Insecurity (%)')
plt.show()

plt.figure()
data[data.High_WIC_perC==1]['FOODINSEC_15_17'].hist(alpha=0.7, label='High WIC perC')
data[data.High_WIC_perC==0]['FOODINSEC_15_17'].hist(alpha=0.7, label='Low WIC perC')
plt.legend()
plt.title('Distribution of Food Insecurity by SNAP Level')
plt.xlabel('Food Insecurity (%)')
plt.ylabel('Frequency')
plt.show()

#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from causalinference import CausalModel
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm



# # Assuming df is your original dataframe loaded already
# columns = [
#     'PCH_PC_WIC_REDEMP_11_16', 'FOODINSEC_15_17', 'PCT_LACCESS_POP10','PCT_LACCESS_LOWI10',
#     'PCT_LACCESS_HHNV10','PCT_LACCESS_CHILD10','PCT_LACCESS_SENIORS10','GROCPTH11','SUPERCPTH11','CONVSPTH11','SPECSPTH11','SNAPSPTH12',
#     'WICSPTH11','FSRPTH11','PC_FFRSALES07','PCT_NSLP12','PCT_FREE_LUNCH10','PCT_REDUCED_LUNCH10','PCT_SBP12','PCT_SFSP12','PCT_WIC12','PCT_WICINFANTCHILD14',
#     'PCT_WICWOMEN14','PCT_CACFP12','FDPIR12','FMRKTPTH13','VEG_ACRESPTH07','FRESHVEG_ACRESPTH07','ORCHARD_ACRESPTH12','BERRY_ACRESPTH07','SLHOUSE07','GHVEG_SQFTPTH07',
#     'AGRITRSM_OPS07','PCT_DIABETES_ADULTS08','PCT_OBESE_ADULTS12','RECFACPTH11','PCT_NHWHITE10','PCT_NHBLACK10','PCT_HISP10','PCT_NHASIAN10','PCT_NHNA10','PCT_NHPI10',
#     'PCT_65OLDER10','PCT_18YOUNGER10','PERPOV10','METRO13','POPLOSS10'
# ]

# # Drop NA values to maintain data quality
# data = df[columns].dropna()

# # Define treatment properly using median
# median_snap = data['PCH_PC_WIC_REDEMP_11_16'].median()
# data['High_WIC_perC'] = (data['PCH_PC_WIC_REDEMP_11_16'] > median_snap).astype(int)

# Outcome and Treatment
Y = data['FOODINSEC_15_17'].values
T = data['High_WIC_perC'].values

# Confounders (covariates)
X = data.drop(columns=['PCH_PC_WIC_REDEMP_11_16', 'FOODINSEC_15_17', 'High_WIC_perC']).values

# Standardize confounders
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

### 🌟 **1. Propensity Score Matching (PSM)**

causal = CausalModel(Y, T, X_std)
causal.est_propensity()

print("\nBefore Matching Balance:")
print(causal.summary_stats)

causal.est_via_matching()

print("\nMatching Estimates:")
print(causal.estimates)

print("\nAfter Matching Balance:")
print(causal.summary_stats)

### 🌟 **2. Inverse Probability of Treatment Weighting (IPTW)**

# Logistic Regression for Propensity Scores
ps_model = LogisticRegression(max_iter=1000).fit(X_std, T)
propensity_scores = ps_model.predict_proba(X_std)[:, 1]

# Ensure stable weights (clipping extreme propensity scores)
eps = 1e-6
propensity_scores = np.clip(propensity_scores, eps, 1 - eps)
weights = T / propensity_scores + (1 - T) / (1 - propensity_scores)

# Weighted regression to estimate causal effects
X_treatment = sm.add_constant(T)
iptw_model = sm.WLS(Y, X_treatment, weights=weights)
iptw_results = iptw_model.fit()

print("\nIPTW Regression Results:")
print(iptw_results.summary())



## 📌 **Additional Robust Causal Inference Methods**

# Include these methods to strengthen your analysis:

### 🌟 **3. Doubly Robust Estimation (Recommended)**
seed = 12345

from sklearn.ensemble import RandomForestRegressor
from econml.dr import LinearDRLearner

regressor = RandomForestRegressor(random_state=seed, n_jobs=-1)
propensity = LogisticRegression(max_iter=1000, random_state=seed)

dr_model = LinearDRLearner(model_regression=regressor,
                           model_propensity=propensity)
dr_model.fit(Y, T, X=X_std)
treatment_effect = dr_model.ate(X_std)
print(f"Doubly Robust ATE: {treatment_effect:.4f}")


#%%
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
# Assuming df is your original dataframe loaded already
columns = [
    'PCH_PC_WIC_REDEMP_11_16', 'PCT_OBESE_ADULTS17', 'PCT_LACCESS_POP10','PCT_LACCESS_LOWI10',
    'PCT_LACCESS_HHNV10','PCT_LACCESS_CHILD10','PCT_LACCESS_SENIORS10','GROCPTH11','SUPERCPTH11','CONVSPTH11','SPECSPTH11','SNAPSPTH12',
    'WICSPTH11','FSRPTH11','PC_FFRSALES07','PCT_NSLP12','PCT_FREE_LUNCH10','PCT_REDUCED_LUNCH10','PCT_SBP12','PCT_SFSP12','PCT_WIC12','PCT_WICINFANTCHILD14',
    'PCT_WICWOMEN14','PCT_CACFP12','FDPIR12','FMRKTPTH13','VEG_ACRESPTH07','FRESHVEG_ACRESPTH07','ORCHARD_ACRESPTH12','BERRY_ACRESPTH07','SLHOUSE07','GHVEG_SQFTPTH07',
    'AGRITRSM_OPS07','PCT_DIABETES_ADULTS08','PCT_OBESE_ADULTS12','RECFACPTH11','PCT_NHWHITE10','PCT_NHBLACK10','PCT_HISP10','PCT_NHASIAN10','PCT_NHNA10','PCT_NHPI10',
    'PCT_65OLDER10','PCT_18YOUNGER10','PERPOV10','METRO13','POPLOSS10'
]

# Drop NA values to maintain data quality
data = df[columns].dropna()

# Define treatment properly using median
median_snap = data['PCH_PC_WIC_REDEMP_11_16'].median()
data['High_WIC_perC'] = (data['PCH_PC_WIC_REDEMP_11_16'] > median_snap).astype(int)
# 1️⃣ Descriptive statistics by group
summary = data.groupby('High_WIC_perC')['PCT_OBESE_ADULTS17'].agg(
    count='count',
    mean='mean',
    median='median',
    std='std',
    min='min',
    max='max'
)
print("Descriptive stats:\n", summary)

# 2️⃣ Independent t‑test (with variance check)
levene_p = stats.levene(
    data.loc[data.High_WIC_perC==1, 'PCT_OBESE_ADULTS17'],
    data.loc[data.High_WIC_perC==0, 'PCT_OBESE_ADULTS17']
).pvalue
equal_var = True if levene_p > 0.05 else False

t_stat, t_p = stats.ttest_ind(
    data.loc[data.High_WIC_perC==1, 'PCT_OBESE_ADULTS17'],
    data.loc[data.High_WIC_perC==0, 'PCT_OBESE_ADULTS17'],
    equal_var=equal_var
)
print(f"\nLevene’s test p-value = {levene_p:.4f} → equal_var={equal_var}")
print(f"T‑test: t = {t_stat:.3f}, p = {t_p:.4f}")

# 3️⃣ Cohen’s d
mean_diff = summary.loc[1,'mean'] - summary.loc[0,'mean']
pooled_sd = np.sqrt(((summary.loc[1,'count']-1)*summary.loc[1,'std']**2 +
                     (summary.loc[0,'count']-1)*summary.loc[0,'std']**2) /
                    (summary.loc[1,'count'] + summary.loc[0,'count'] - 2))
cohens_d = mean_diff / pooled_sd
print(f"Cohen’s d = {cohens_d:.3f}")

# 4️⃣ Mann–Whitney U test (non-parametric)
u_stat, u_p = stats.mannwhitneyu(
    data.loc[data.High_WIC_perC==1, 'PCT_OBESE_ADULTS17'],
    data.loc[data.High_WIC_perC==0, 'PCT_OBESE_ADULTS17'],
    alternative='two-sided'
)
print(f"Mann–Whitney U: U = {u_stat:.3f}, p = {u_p:.4f}")

# 5️⃣ Visualizations
plt.figure()
data.boxplot(column='PCT_OBESE_ADULTS17', by='High_WIC_perC')
plt.title('Obesity by WIC Level')
plt.suptitle('')
plt.xlabel('High WIC (1 = above median)')
plt.ylabel('Obesity (%)')
plt.show()

plt.figure()
data[data.High_WIC_perC==1]['PCT_OBESE_ADULTS17'].hist(alpha=0.7, label='High WIC perC')
data[data.High_WIC_perC==0]['PCT_OBESE_ADULTS17'].hist(alpha=0.7, label='Low WIC perC')
plt.legend()
plt.title('Distribution of Obesity by SNAP Level')
plt.xlabel('Obesity (%)')
plt.ylabel('Frequency')
plt.show()


#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from causalinference import CausalModel
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm



# # Assuming df is your original dataframe loaded already
# columns = [
#     'PCH_PC_WIC_REDEMP_11_16', 'FOODINSEC_15_17', 'PCT_LACCESS_POP10','PCT_LACCESS_LOWI10',
#     'PCT_LACCESS_HHNV10','PCT_LACCESS_CHILD10','PCT_LACCESS_SENIORS10','GROCPTH11','SUPERCPTH11','CONVSPTH11','SPECSPTH11','SNAPSPTH12',
#     'WICSPTH11','FSRPTH11','PC_FFRSALES07','PCT_NSLP12','PCT_FREE_LUNCH10','PCT_REDUCED_LUNCH10','PCT_SBP12','PCT_SFSP12','PCT_WIC12','PCT_WICINFANTCHILD14',
#     'PCT_WICWOMEN14','PCT_CACFP12','FDPIR12','FMRKTPTH13','VEG_ACRESPTH07','FRESHVEG_ACRESPTH07','ORCHARD_ACRESPTH12','BERRY_ACRESPTH07','SLHOUSE07','GHVEG_SQFTPTH07',
#     'AGRITRSM_OPS07','PCT_DIABETES_ADULTS08','PCT_OBESE_ADULTS12','RECFACPTH11','PCT_NHWHITE10','PCT_NHBLACK10','PCT_HISP10','PCT_NHASIAN10','PCT_NHNA10','PCT_NHPI10',
#     'PCT_65OLDER10','PCT_18YOUNGER10','PERPOV10','METRO13','POPLOSS10'
# ]

# # Drop NA values to maintain data quality
# data = df[columns].dropna()

# # Define treatment properly using median
# median_snap = data['PCH_PC_WIC_REDEMP_11_16'].median()
# data['High_WIC_perC'] = (data['PCH_PC_WIC_REDEMP_11_16'] > median_snap).astype(int)

# Outcome and Treatment
Y = data['PCT_OBESE_ADULTS17'].values
T = data['High_WIC_perC'].values

# Confounders (covariates)
X = data.drop(columns=['PCH_PC_WIC_REDEMP_11_16', 'PCT_OBESE_ADULTS17', 'High_WIC_perC']).values

# Standardize confounders
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

### 🌟 **1. Propensity Score Matching (PSM)**

causal = CausalModel(Y, T, X_std)
causal.est_propensity()

print("\nBefore Matching Balance:")
print(causal.summary_stats)

causal.est_via_matching()

print("\nMatching Estimates:")
print(causal.estimates)

print("\nAfter Matching Balance:")
print(causal.summary_stats)

### 🌟 **2. Inverse Probability of Treatment Weighting (IPTW)**

# Logistic Regression for Propensity Scores
ps_model = LogisticRegression(max_iter=1000).fit(X_std, T)
propensity_scores = ps_model.predict_proba(X_std)[:, 1]

# Ensure stable weights (clipping extreme propensity scores)
eps = 1e-6
propensity_scores = np.clip(propensity_scores, eps, 1 - eps)
weights = T / propensity_scores + (1 - T) / (1 - propensity_scores)

# Weighted regression to estimate causal effects
X_treatment = sm.add_constant(T)
iptw_model = sm.WLS(Y, X_treatment, weights=weights)
iptw_results = iptw_model.fit()

print("\nIPTW Regression Results:")
print(iptw_results.summary())



## 📌 **Additional Robust Causal Inference Methods**

# Include these methods to strengthen your analysis:

### 🌟 **3. Doubly Robust Estimation (Recommended)**
seed = 12345

from sklearn.ensemble import RandomForestRegressor
from econml.dr import LinearDRLearner

regressor = RandomForestRegressor(random_state=seed, n_jobs=-1)
propensity = LogisticRegression(max_iter=1000, random_state=seed)

dr_model = LinearDRLearner(model_regression=regressor,
                           model_propensity=propensity)
dr_model.fit(Y, T, X=X_std)
treatment_effect = dr_model.ate(X_std)
print(f"Doubly Robust ATE: {treatment_effect:.4f}")

#%%
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
# Assuming df is your original dataframe loaded already
columns = [
    'PCH_PC_WIC_REDEMP_11_16', 'PCT_DIABETES_ADULTS13', 'PCT_LACCESS_POP10','PCT_LACCESS_LOWI10',
    'PCT_LACCESS_HHNV10','PCT_LACCESS_CHILD10','PCT_LACCESS_SENIORS10','GROCPTH11','SUPERCPTH11','CONVSPTH11','SPECSPTH11','SNAPSPTH12',
    'WICSPTH11','FSRPTH11','PC_FFRSALES07','PCT_NSLP12','PCT_FREE_LUNCH10','PCT_REDUCED_LUNCH10','PCT_SBP12','PCT_SFSP12','PCT_WIC12','PCT_WICINFANTCHILD14',
    'PCT_WICWOMEN14','PCT_CACFP12','FDPIR12','FMRKTPTH13','VEG_ACRESPTH07','FRESHVEG_ACRESPTH07','ORCHARD_ACRESPTH12','BERRY_ACRESPTH07','SLHOUSE07','GHVEG_SQFTPTH07',
    'AGRITRSM_OPS07','PCT_DIABETES_ADULTS08','PCT_OBESE_ADULTS12','RECFACPTH11','PCT_NHWHITE10','PCT_NHBLACK10','PCT_HISP10','PCT_NHASIAN10','PCT_NHNA10','PCT_NHPI10',
    'PCT_65OLDER10','PCT_18YOUNGER10','PERPOV10','METRO13','POPLOSS10'
]

# Drop NA values to maintain data quality
data = df[columns].dropna()

# Define treatment properly using median
median_snap = data['PCH_PC_WIC_REDEMP_11_16'].median()
data['High_WIC_perC'] = (data['PCH_PC_WIC_REDEMP_11_16'] > median_snap).astype(int)
# 1️⃣ Descriptive statistics by group
summary = data.groupby('High_WIC_perC')['PCT_DIABETES_ADULTS13'].agg(
    count='count',
    mean='mean',
    median='median',
    std='std',
    min='min',
    max='max'
)
print("Descriptive stats:\n", summary)

# 2️⃣ Independent t‑test (with variance check)
levene_p = stats.levene(
    data.loc[data.High_WIC_perC==1, 'PCT_DIABETES_ADULTS13'],
    data.loc[data.High_WIC_perC==0, 'PCT_DIABETES_ADULTS13']
).pvalue
equal_var = True if levene_p > 0.05 else False

t_stat, t_p = stats.ttest_ind(
    data.loc[data.High_WIC_perC==1, 'PCT_DIABETES_ADULTS13'],
    data.loc[data.High_WIC_perC==0, 'PCT_DIABETES_ADULTS13'],
    equal_var=equal_var
)
print(f"\nLevene’s test p-value = {levene_p:.4f} → equal_var={equal_var}")
print(f"T‑test: t = {t_stat:.3f}, p = {t_p:.4f}")

# 3️⃣ Cohen’s d
mean_diff = summary.loc[1,'mean'] - summary.loc[0,'mean']
pooled_sd = np.sqrt(((summary.loc[1,'count']-1)*summary.loc[1,'std']**2 +
                     (summary.loc[0,'count']-1)*summary.loc[0,'std']**2) /
                    (summary.loc[1,'count'] + summary.loc[0,'count'] - 2))
cohens_d = mean_diff / pooled_sd
print(f"Cohen’s d = {cohens_d:.3f}")

# 4️⃣ Mann–Whitney U test (non-parametric)
u_stat, u_p = stats.mannwhitneyu(
    data.loc[data.High_WIC_perC==1, 'PCT_DIABETES_ADULTS13'],
    data.loc[data.High_WIC_perC==0, 'PCT_DIABETES_ADULTS13'],
    alternative='two-sided'
)
print(f"Mann–Whitney U: U = {u_stat:.3f}, p = {u_p:.4f}")

# 5️⃣ Visualizations
plt.figure()
data.boxplot(column='PCT_DIABETES_ADULTS13', by='High_WIC_perC')
plt.title('Diabetes by WIC Level')
plt.suptitle('')
plt.xlabel('High WIC (1 = above median)')
plt.ylabel('Diabetes (%)')
plt.show()

plt.figure()
data[data.High_WIC_perC==1]['PCT_DIABETES_ADULTS13'].hist(alpha=0.7, label='High WIC perC')
data[data.High_WIC_perC==0]['PCT_DIABETES_ADULTS13'].hist(alpha=0.7, label='Low WIC perC')
plt.legend()
plt.title('Distribution of Diabetes by SNAP Level')
plt.xlabel('Diabetes (%)')
plt.ylabel('Frequency')
plt.show()

#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from causalinference import CausalModel
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm



# # Assuming df is your original dataframe loaded already
# columns = [
#     'PCH_PC_WIC_REDEMP_11_16', 'FOODINSEC_15_17', 'PCT_LACCESS_POP10','PCT_LACCESS_LOWI10',
#     'PCT_LACCESS_HHNV10','PCT_LACCESS_CHILD10','PCT_LACCESS_SENIORS10','GROCPTH11','SUPERCPTH11','CONVSPTH11','SPECSPTH11','SNAPSPTH12',
#     'WICSPTH11','FSRPTH11','PC_FFRSALES07','PCT_NSLP12','PCT_FREE_LUNCH10','PCT_REDUCED_LUNCH10','PCT_SBP12','PCT_SFSP12','PCT_WIC12','PCT_WICINFANTCHILD14',
#     'PCT_WICWOMEN14','PCT_CACFP12','FDPIR12','FMRKTPTH13','VEG_ACRESPTH07','FRESHVEG_ACRESPTH07','ORCHARD_ACRESPTH12','BERRY_ACRESPTH07','SLHOUSE07','GHVEG_SQFTPTH07',
#     'AGRITRSM_OPS07','PCT_DIABETES_ADULTS08','PCT_OBESE_ADULTS12','RECFACPTH11','PCT_NHWHITE10','PCT_NHBLACK10','PCT_HISP10','PCT_NHASIAN10','PCT_NHNA10','PCT_NHPI10',
#     'PCT_65OLDER10','PCT_18YOUNGER10','PERPOV10','METRO13','POPLOSS10'
# ]

# # Drop NA values to maintain data quality
# data = df[columns].dropna()

# # Define treatment properly using median
# median_snap = data['PCH_PC_WIC_REDEMP_11_16'].median()
# data['High_WIC_perC'] = (data['PCH_PC_WIC_REDEMP_11_16'] > median_snap).astype(int)

# Outcome and Treatment
Y = data['PCT_DIABETES_ADULTS13'].values
T = data['High_WIC_perC'].values

# Confounders (covariates)
X = data.drop(columns=['PCH_PC_WIC_REDEMP_11_16', 'PCT_DIABETES_ADULTS13', 'High_WIC_perC']).values

# Standardize confounders
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

### 🌟 **1. Propensity Score Matching (PSM)**

causal = CausalModel(Y, T, X_std)
causal.est_propensity()

print("\nBefore Matching Balance:")
print(causal.summary_stats)

causal.est_via_matching()

print("\nMatching Estimates:")
print(causal.estimates)

print("\nAfter Matching Balance:")
print(causal.summary_stats)

### 🌟 **2. Inverse Probability of Treatment Weighting (IPTW)**

# Logistic Regression for Propensity Scores
ps_model = LogisticRegression(max_iter=1000).fit(X_std, T)
propensity_scores = ps_model.predict_proba(X_std)[:, 1]

# Ensure stable weights (clipping extreme propensity scores)
eps = 1e-6
propensity_scores = np.clip(propensity_scores, eps, 1 - eps)
weights = T / propensity_scores + (1 - T) / (1 - propensity_scores)

# Weighted regression to estimate causal effects
X_treatment = sm.add_constant(T)
iptw_model = sm.WLS(Y, X_treatment, weights=weights)
iptw_results = iptw_model.fit()

print("\nIPTW Regression Results:")
print(iptw_results.summary())



## 📌 **Additional Robust Causal Inference Methods**

# Include these methods to strengthen your analysis:

### 🌟 **3. Doubly Robust Estimation (Recommended)**
seed = 12345

from sklearn.ensemble import RandomForestRegressor
from econml.dr import LinearDRLearner

regressor = RandomForestRegressor(random_state=seed, n_jobs=-1)
propensity = LogisticRegression(max_iter=1000, random_state=seed)

dr_model = LinearDRLearner(model_regression=regressor,
                           model_propensity=propensity)
dr_model.fit(Y, T, X=X_std)
treatment_effect = dr_model.ate(X_std)
print(f"Doubly Robust ATE: {treatment_effect:.4f}")

#%%
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
# Assuming df is your original dataframe loaded already
columns = [
    'PCH_SFSP_12_17', 'FOODINSEC_15_17', 'PCT_LACCESS_POP10','PCT_LACCESS_LOWI10',
    'PCT_LACCESS_HHNV10','PCT_LACCESS_CHILD10','PCT_LACCESS_SENIORS10','GROCPTH11','SUPERCPTH11','CONVSPTH11','SPECSPTH11','SNAPSPTH12',
    'WICSPTH11','FSRPTH11','PC_FFRSALES07','PCT_NSLP12','PCT_FREE_LUNCH10','PCT_REDUCED_LUNCH10','PCT_SBP12','PCT_SFSP12','PCT_WIC12','PCT_WICINFANTCHILD14',
    'PCT_WICWOMEN14','PCT_CACFP12','FDPIR12','FMRKTPTH13','VEG_ACRESPTH07','FRESHVEG_ACRESPTH07','ORCHARD_ACRESPTH12','BERRY_ACRESPTH07','SLHOUSE07','GHVEG_SQFTPTH07',
    'AGRITRSM_OPS07','PCT_DIABETES_ADULTS08','PCT_OBESE_ADULTS12','RECFACPTH11','PCT_NHWHITE10','PCT_NHBLACK10','PCT_HISP10','PCT_NHASIAN10','PCT_NHNA10','PCT_NHPI10',
    'PCT_65OLDER10','PCT_18YOUNGER10','PERPOV10','METRO13','POPLOSS10'
]

# Drop NA values to maintain data quality
data = df[columns].dropna()

# Define treatment properly using median
median_snap = data['PCH_SFSP_12_17'].median()
data['High_Foodservice_summer'] = (data['PCH_SFSP_12_17'] > median_snap).astype(int)
# 1️⃣ Descriptive statistics by group
summary = data.groupby('High_Foodservice_summer')['FOODINSEC_15_17'].agg(
    count='count',
    mean='mean',
    median='median',
    std='std',
    min='min',
    max='max'
)
print("Descriptive stats:\n", summary)

# 2️⃣ Independent t‑test (with variance check)
levene_p = stats.levene(
    data.loc[data.High_Foodservice_summer==1, 'FOODINSEC_15_17'],
    data.loc[data.High_Foodservice_summer==0, 'FOODINSEC_15_17']
).pvalue
equal_var = True if levene_p > 0.05 else False

t_stat, t_p = stats.ttest_ind(
    data.loc[data.High_Foodservice_summer==1, 'FOODINSEC_15_17'],
    data.loc[data.High_Foodservice_summer==0, 'FOODINSEC_15_17'],
    equal_var=equal_var
)
print(f"\nLevene’s test p-value = {levene_p:.4f} → equal_var={equal_var}")
print(f"T‑test: t = {t_stat:.3f}, p = {t_p:.4f}")

# 3️⃣ Cohen’s d
mean_diff = summary.loc[1,'mean'] - summary.loc[0,'mean']
pooled_sd = np.sqrt(((summary.loc[1,'count']-1)*summary.loc[1,'std']**2 +
                     (summary.loc[0,'count']-1)*summary.loc[0,'std']**2) /
                    (summary.loc[1,'count'] + summary.loc[0,'count'] - 2))
cohens_d = mean_diff / pooled_sd
print(f"Cohen’s d = {cohens_d:.3f}")

# 4️⃣ Mann–Whitney U test (non-parametric)
u_stat, u_p = stats.mannwhitneyu(
    data.loc[data.High_Foodservice_summer==1, 'FOODINSEC_15_17'],
    data.loc[data.High_Foodservice_summer==0, 'FOODINSEC_15_17'],
    alternative='two-sided'
)
print(f"Mann–Whitney U: U = {u_stat:.3f}, p = {u_p:.4f}")

# 5️⃣ Visualizations
plt.figure()
data.boxplot(column='FOODINSEC_15_17', by='High_Foodservice_summer')
plt.title('Food Insecurity by Foodservice summer Level')
plt.suptitle('')
plt.xlabel('High Foodservice summer (1 = above median)')
plt.ylabel('Food Insecurity (%)')
plt.show()

plt.figure()
data[data.High_Foodservice_summer==1]['FOODINSEC_15_17'].hist(alpha=0.7, label='High WIC perC')
data[data.High_Foodservice_summer==0]['FOODINSEC_15_17'].hist(alpha=0.7, label='Low WIC perC')
plt.legend()
plt.title('Distribution of Food Insecurity by Foodservice summer Level')
plt.xlabel('Food Insecurity (%)')
plt.ylabel('Frequency')
plt.show()

#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from causalinference import CausalModel
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm



# Assuming df is your original dataframe loaded already
# columns = [
#     'PCH_SFSP_12_17', 'FOODINSEC_15_17', 'PCT_LACCESS_POP10','PCT_LACCESS_LOWI10',
#     'PCT_LACCESS_HHNV10','PCT_LACCESS_CHILD10','PCT_LACCESS_SENIORS10','GROCPTH11','SUPERCPTH11','CONVSPTH11','SPECSPTH11','SNAPSPTH12',
#     'WICSPTH11','FSRPTH11','PC_FFRSALES07','PCT_NSLP12','PCT_FREE_LUNCH10','PCT_REDUCED_LUNCH10','PCT_SBP12','PCT_SFSP12','PCT_WIC12','PCT_WICINFANTCHILD14',
#     'PCT_WICWOMEN14','PCT_CACFP12','FDPIR12','FMRKTPTH13','VEG_ACRESPTH07','FRESHVEG_ACRESPTH07','ORCHARD_ACRESPTH12','BERRY_ACRESPTH07','SLHOUSE07','GHVEG_SQFTPTH07',
#     'AGRITRSM_OPS07','PCT_DIABETES_ADULTS08','PCT_OBESE_ADULTS12','RECFACPTH11','PCT_NHWHITE10','PCT_NHBLACK10','PCT_HISP10','PCT_NHASIAN10','PCT_NHNA10','PCT_NHPI10',
#     'PCT_65OLDER10','PCT_18YOUNGER10','PERPOV10','METRO13','POPLOSS10'
# ]

# # Drop NA values to maintain data quality
# data = df[columns].dropna()

# # Define treatment properly using median
# median_snap = data['PCH_SFSP_12_17'].median()
# data['High_Foodservice_summer'] = (data['PCH_SFSP_12_17'] > median_snap).astype(int)

# Outcome and Treatment
Y = data['FOODINSEC_15_17'].values
T = data['High_Foodservice_summer'].values

# Confounders (covariates)
X = data.drop(columns=['PCH_SFSP_12_17', 'FOODINSEC_15_17', 'High_Foodservice_summer']).values

# Standardize confounders
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

### 🌟 **1. Propensity Score Matching (PSM)**

causal = CausalModel(Y, T, X_std)
causal.est_propensity()

print("\nBefore Matching Balance:")
print(causal.summary_stats)

causal.est_via_matching()

print("\nMatching Estimates:")
print(causal.estimates)

print("\nAfter Matching Balance:")
print(causal.summary_stats)

### 🌟 **2. Inverse Probability of Treatment Weighting (IPTW)**

# Logistic Regression for Propensity Scores
ps_model = LogisticRegression(max_iter=1000).fit(X_std, T)
propensity_scores = ps_model.predict_proba(X_std)[:, 1]

# Ensure stable weights (clipping extreme propensity scores)
eps = 1e-6
propensity_scores = np.clip(propensity_scores, eps, 1 - eps)
weights = T / propensity_scores + (1 - T) / (1 - propensity_scores)

# Weighted regression to estimate causal effects
X_treatment = sm.add_constant(T)
iptw_model = sm.WLS(Y, X_treatment, weights=weights)
iptw_results = iptw_model.fit()

print("\nIPTW Regression Results:")
print(iptw_results.summary())



## 📌 **Additional Robust Causal Inference Methods**

# Include these methods to strengthen your analysis:

### 🌟 **3. Doubly Robust Estimation (Recommended)**
seed = 12345

from sklearn.ensemble import RandomForestRegressor
from econml.dr import LinearDRLearner

regressor = RandomForestRegressor(random_state=seed, n_jobs=-1)
propensity = LogisticRegression(max_iter=1000, random_state=seed)

dr_model = LinearDRLearner(model_regression=regressor,
                        model_propensity=propensity)
dr_model.fit(Y, T, X=X_std)
treatment_effect = dr_model.ate(X_std)
print(f"Doubly Robust ATE: {treatment_effect:.4f}")

#%%
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
# Assuming df is your original dataframe loaded already
columns = [
    'PCH_SFSP_12_17', 'PCT_OBESE_ADULTS17', 'PCT_LACCESS_POP10','PCT_LACCESS_LOWI10',
    'PCT_LACCESS_HHNV10','PCT_LACCESS_CHILD10','PCT_LACCESS_SENIORS10','GROCPTH11','SUPERCPTH11','CONVSPTH11','SPECSPTH11','SNAPSPTH12',
    'WICSPTH11','FSRPTH11','PC_FFRSALES07','PCT_NSLP12','PCT_FREE_LUNCH10','PCT_REDUCED_LUNCH10','PCT_SBP12','PCT_SFSP12','PCT_WIC12','PCT_WICINFANTCHILD14',
    'PCT_WICWOMEN14','PCT_CACFP12','FDPIR12','FMRKTPTH13','VEG_ACRESPTH07','FRESHVEG_ACRESPTH07','ORCHARD_ACRESPTH12','BERRY_ACRESPTH07','SLHOUSE07','GHVEG_SQFTPTH07',
    'AGRITRSM_OPS07','PCT_DIABETES_ADULTS08','PCT_OBESE_ADULTS12','RECFACPTH11','PCT_NHWHITE10','PCT_NHBLACK10','PCT_HISP10','PCT_NHASIAN10','PCT_NHNA10','PCT_NHPI10',
    'PCT_65OLDER10','PCT_18YOUNGER10','PERPOV10','METRO13','POPLOSS10'
]

# Drop NA values to maintain data quality
data = df[columns].dropna()

# Define treatment properly using median
median_snap = data['PCH_SFSP_12_17'].median()
data['High_Foodservice_summer'] = (data['PCH_SFSP_12_17'] > median_snap).astype(int)
# 1️⃣ Descriptive statistics by group
summary = data.groupby('High_Foodservice_summer')['PCT_OBESE_ADULTS17'].agg(
    count='count',
    mean='mean',
    median='median',
    std='std',
    min='min',
    max='max'
)
print("Descriptive stats:\n", summary)

# 2️⃣ Independent t‑test (with variance check)
levene_p = stats.levene(
    data.loc[data.High_Foodservice_summer==1, 'PCT_OBESE_ADULTS17'],
    data.loc[data.High_Foodservice_summer==0, 'PCT_OBESE_ADULTS17']
).pvalue
equal_var = True if levene_p > 0.05 else False

t_stat, t_p = stats.ttest_ind(
    data.loc[data.High_Foodservice_summer==1, 'PCT_OBESE_ADULTS17'],
    data.loc[data.High_Foodservice_summer==0, 'PCT_OBESE_ADULTS17'],
    equal_var=equal_var
)
print(f"\nLevene’s test p-value = {levene_p:.4f} → equal_var={equal_var}")
print(f"T‑test: t = {t_stat:.3f}, p = {t_p:.4f}")

# 3️⃣ Cohen’s d
mean_diff = summary.loc[1,'mean'] - summary.loc[0,'mean']
pooled_sd = np.sqrt(((summary.loc[1,'count']-1)*summary.loc[1,'std']**2 +
                     (summary.loc[0,'count']-1)*summary.loc[0,'std']**2) /
                    (summary.loc[1,'count'] + summary.loc[0,'count'] - 2))
cohens_d = mean_diff / pooled_sd
print(f"Cohen’s d = {cohens_d:.3f}")

# 4️⃣ Mann–Whitney U test (non-parametric)
u_stat, u_p = stats.mannwhitneyu(
    data.loc[data.High_Foodservice_summer==1, 'PCT_OBESE_ADULTS17'],
    data.loc[data.High_Foodservice_summer==0, 'PCT_OBESE_ADULTS17'],
    alternative='two-sided'
)
print(f"Mann–Whitney U: U = {u_stat:.3f}, p = {u_p:.4f}")

# 5️⃣ Visualizations
plt.figure()
data.boxplot(column='PCT_OBESE_ADULTS17', by='High_Foodservice_summer')
plt.title('Obesity by Foodservice summer Level')
plt.suptitle('')
plt.xlabel('High Foodservice summer (1 = above median)')
plt.ylabel('Obesity (%)')
plt.show()

plt.figure()
data[data.High_Foodservice_summer==1]['PCT_OBESE_ADULTS17'].hist(alpha=0.7, label='High Foodservice summer')
data[data.High_Foodservice_summer==0]['PCT_OBESE_ADULTS17'].hist(alpha=0.7, label='Low Foodservice summer')
plt.legend()
plt.title('Distribution of Obesity by Foodservice summer Level')
plt.xlabel('Obesity (%)')
plt.ylabel('Frequency')
plt.show()

#%%
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
# Assuming df is your original dataframe loaded already
columns = [
    'PCH_SFSP_12_17', 'PCT_OBESE_ADULTS17', 'PCT_LACCESS_POP10','PCT_LACCESS_LOWI10',
    'PCT_LACCESS_HHNV10','PCT_LACCESS_CHILD10','PCT_LACCESS_SENIORS10','GROCPTH11','SUPERCPTH11','CONVSPTH11','SPECSPTH11','SNAPSPTH12',
    'WICSPTH11','FSRPTH11','PC_FFRSALES07','PCT_NSLP12','PCT_FREE_LUNCH10','PCT_REDUCED_LUNCH10','PCT_SBP12','PCT_SFSP12','PCT_WIC12','PCT_WICINFANTCHILD14',
    'PCT_WICWOMEN14','PCT_CACFP12','FDPIR12','FMRKTPTH13','VEG_ACRESPTH07','FRESHVEG_ACRESPTH07','ORCHARD_ACRESPTH12','BERRY_ACRESPTH07','SLHOUSE07','GHVEG_SQFTPTH07',
    'AGRITRSM_OPS07','PCT_DIABETES_ADULTS08','PCT_OBESE_ADULTS12','RECFACPTH11','PCT_NHWHITE10','PCT_NHBLACK10','PCT_HISP10','PCT_NHASIAN10','PCT_NHNA10','PCT_NHPI10',
    'PCT_65OLDER10','PCT_18YOUNGER10','PERPOV10','METRO13','POPLOSS10'
]

# Drop NA values to maintain data quality
data = df[columns].dropna()

# Define treatment properly using median
median_snap = data['PCH_SFSP_12_17'].median()
data['High_Foodservice_summer'] = (data['PCH_SFSP_12_17'] > median_snap).astype(int)
# 1️⃣ Descriptive statistics by group
summary = data.groupby('High_Foodservice_summer')['PCT_OBESE_ADULTS17'].agg(
    count='count',
    mean='mean',
    median='median',
    std='std',
    min='min',
    max='max'
)
print("Descriptive stats:\n", summary)

# 2️⃣ Independent t‑test (with variance check)
levene_p = stats.levene(
    data.loc[data.High_Foodservice_summer==1, 'PCT_OBESE_ADULTS17'],
    data.loc[data.High_Foodservice_summer==0, 'PCT_OBESE_ADULTS17']
).pvalue
equal_var = True if levene_p > 0.05 else False

t_stat, t_p = stats.ttest_ind(
    data.loc[data.High_Foodservice_summer==1, 'PCT_OBESE_ADULTS17'],
    data.loc[data.High_Foodservice_summer==0, 'PCT_OBESE_ADULTS17'],
    equal_var=equal_var
)
print(f"\nLevene’s test p-value = {levene_p:.4f} → equal_var={equal_var}")
print(f"T‑test: t = {t_stat:.3f}, p = {t_p:.4f}")

# 3️⃣ Cohen’s d
mean_diff = summary.loc[1,'mean'] - summary.loc[0,'mean']
pooled_sd = np.sqrt(((summary.loc[1,'count']-1)*summary.loc[1,'std']**2 +
                     (summary.loc[0,'count']-1)*summary.loc[0,'std']**2) /
                    (summary.loc[1,'count'] + summary.loc[0,'count'] - 2))
cohens_d = mean_diff / pooled_sd
print(f"Cohen’s d = {cohens_d:.3f}")

# 4️⃣ Mann–Whitney U test (non-parametric)
u_stat, u_p = stats.mannwhitneyu(
    data.loc[data.High_Foodservice_summer==1, 'PCT_OBESE_ADULTS17'],
    data.loc[data.High_Foodservice_summer==0, 'PCT_OBESE_ADULTS17'],
    alternative='two-sided'
)
print(f"Mann–Whitney U: U = {u_stat:.3f}, p = {u_p:.4f}")

# 5️⃣ Visualizations
plt.figure()
data.boxplot(column='PCT_OBESE_ADULTS17', by='High_Foodservice_summer')
plt.title('Obesity by Foodservice summer Level')
plt.suptitle('')
plt.xlabel('High Foodservice summer (1 = above median)')
plt.ylabel('Obesity (%)')
plt.show()

plt.figure()
data[data.High_Foodservice_summer==1]['PCT_OBESE_ADULTS17'].hist(alpha=0.7, label='High Foodservice summer')
data[data.High_Foodservice_summer==0]['PCT_OBESE_ADULTS17'].hist(alpha=0.7, label='Low Foodservice summer')
plt.legend()
plt.title('Distribution of Obesity by Foodservice summer Level')
plt.xlabel('Obesity (%)')
plt.ylabel('Frequency')
plt.show()

#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from causalinference import CausalModel
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm



# Assuming df is your original dataframe loaded already
# columns = [
#     'PCH_SFSP_12_17', 'FOODINSEC_15_17', 'PCT_LACCESS_POP10','PCT_LACCESS_LOWI10',
#     'PCT_LACCESS_HHNV10','PCT_LACCESS_CHILD10','PCT_LACCESS_SENIORS10','GROCPTH11','SUPERCPTH11','CONVSPTH11','SPECSPTH11','SNAPSPTH12',
#     'WICSPTH11','FSRPTH11','PC_FFRSALES07','PCT_NSLP12','PCT_FREE_LUNCH10','PCT_REDUCED_LUNCH10','PCT_SBP12','PCT_SFSP12','PCT_WIC12','PCT_WICINFANTCHILD14',
#     'PCT_WICWOMEN14','PCT_CACFP12','FDPIR12','FMRKTPTH13','VEG_ACRESPTH07','FRESHVEG_ACRESPTH07','ORCHARD_ACRESPTH12','BERRY_ACRESPTH07','SLHOUSE07','GHVEG_SQFTPTH07',
#     'AGRITRSM_OPS07','PCT_DIABETES_ADULTS08','PCT_OBESE_ADULTS12','RECFACPTH11','PCT_NHWHITE10','PCT_NHBLACK10','PCT_HISP10','PCT_NHASIAN10','PCT_NHNA10','PCT_NHPI10',
#     'PCT_65OLDER10','PCT_18YOUNGER10','PERPOV10','METRO13','POPLOSS10'
# ]

# # Drop NA values to maintain data quality
# data = df[columns].dropna()

# # Define treatment properly using median
# median_snap = data['PCH_SFSP_12_17'].median()
# data['High_Foodservice_summer'] = (data['PCH_SFSP_12_17'] > median_snap).astype(int)

# Outcome and Treatment
Y = data['PCT_OBESE_ADULTS17'].values
T = data['High_Foodservice_summer'].values

# Confounders (covariates)
X = data.drop(columns=['PCH_SFSP_12_17', 'PCT_OBESE_ADULTS17', 'High_Foodservice_summer']).values

# Standardize confounders
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

### 🌟 **1. Propensity Score Matching (PSM)**

causal = CausalModel(Y, T, X_std)
causal.est_propensity()

print("\nBefore Matching Balance:")
print(causal.summary_stats)

causal.est_via_matching()

print("\nMatching Estimates:")
print(causal.estimates)

print("\nAfter Matching Balance:")
print(causal.summary_stats)

### 🌟 **2. Inverse Probability of Treatment Weighting (IPTW)**

# Logistic Regression for Propensity Scores
ps_model = LogisticRegression(max_iter=1000).fit(X_std, T)
propensity_scores = ps_model.predict_proba(X_std)[:, 1]

# Ensure stable weights (clipping extreme propensity scores)
eps = 1e-6
propensity_scores = np.clip(propensity_scores, eps, 1 - eps)
weights = T / propensity_scores + (1 - T) / (1 - propensity_scores)

# Weighted regression to estimate causal effects
X_treatment = sm.add_constant(T)
iptw_model = sm.WLS(Y, X_treatment, weights=weights)
iptw_results = iptw_model.fit()

print("\nIPTW Regression Results:")
print(iptw_results.summary())



## 📌 **Additional Robust Causal Inference Methods**

# Include these methods to strengthen your analysis:

### 🌟 **3. Doubly Robust Estimation (Recommended)**
seed = 12345

from sklearn.ensemble import RandomForestRegressor
from econml.dr import LinearDRLearner

regressor = RandomForestRegressor(random_state=seed, n_jobs=-1)
propensity = LogisticRegression(max_iter=1000, random_state=seed)

dr_model = LinearDRLearner(model_regression=regressor,
                           model_propensity=propensity)
dr_model.fit(Y, T, X=X_std)
treatment_effect = dr_model.ate(X_std)
print(f"Doubly Robust ATE: {treatment_effect:.4f}")

#%%
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
# Assuming df is your original dataframe loaded already
columns = [
    'PCH_SFSP_12_17', 'PCT_DIABETES_ADULTS13', 'PCT_LACCESS_POP10','PCT_LACCESS_LOWI10',
    'PCT_LACCESS_HHNV10','PCT_LACCESS_CHILD10','PCT_LACCESS_SENIORS10','GROCPTH11','SUPERCPTH11','CONVSPTH11','SPECSPTH11','SNAPSPTH12',
    'WICSPTH11','FSRPTH11','PC_FFRSALES07','PCT_NSLP12','PCT_FREE_LUNCH10','PCT_REDUCED_LUNCH10','PCT_SBP12','PCT_SFSP12','PCT_WIC12','PCT_WICINFANTCHILD14',
    'PCT_WICWOMEN14','PCT_CACFP12','FDPIR12','FMRKTPTH13','VEG_ACRESPTH07','FRESHVEG_ACRESPTH07','ORCHARD_ACRESPTH12','BERRY_ACRESPTH07','SLHOUSE07','GHVEG_SQFTPTH07',
    'AGRITRSM_OPS07','PCT_DIABETES_ADULTS08','PCT_OBESE_ADULTS12','RECFACPTH11','PCT_NHWHITE10','PCT_NHBLACK10','PCT_HISP10','PCT_NHASIAN10','PCT_NHNA10','PCT_NHPI10',
    'PCT_65OLDER10','PCT_18YOUNGER10','PERPOV10','METRO13','POPLOSS10'
]

# Drop NA values to maintain data quality
data = df[columns].dropna()

# Define treatment properly using median
median_snap = data['PCH_SFSP_12_17'].median()
data['High_Foodservice_summer'] = (data['PCH_SFSP_12_17'] > median_snap).astype(int)
# 1️⃣ Descriptive statistics by group
summary = data.groupby('High_Foodservice_summer')['PCT_DIABETES_ADULTS13'].agg(
    count='count',
    mean='mean',
    median='median',
    std='std',
    min='min',
    max='max'
)
print("Descriptive stats:\n", summary)

# 2️⃣ Independent t‑test (with variance check)
levene_p = stats.levene(
    data.loc[data.High_Foodservice_summer==1, 'PCT_DIABETES_ADULTS13'],
    data.loc[data.High_Foodservice_summer==0, 'PCT_DIABETES_ADULTS13']
).pvalue
equal_var = True if levene_p > 0.05 else False

t_stat, t_p = stats.ttest_ind(
    data.loc[data.High_Foodservice_summer==1, 'PCT_DIABETES_ADULTS13'],
    data.loc[data.High_Foodservice_summer==0, 'PCT_DIABETES_ADULTS13'],
    equal_var=equal_var
)
print(f"\nLevene’s test p-value = {levene_p:.4f} → equal_var={equal_var}")
print(f"T‑test: t = {t_stat:.3f}, p = {t_p:.4f}")

# 3️⃣ Cohen’s d
mean_diff = summary.loc[1,'mean'] - summary.loc[0,'mean']
pooled_sd = np.sqrt(((summary.loc[1,'count']-1)*summary.loc[1,'std']**2 +
                     (summary.loc[0,'count']-1)*summary.loc[0,'std']**2) /
                    (summary.loc[1,'count'] + summary.loc[0,'count'] - 2))
cohens_d = mean_diff / pooled_sd
print(f"Cohen’s d = {cohens_d:.3f}")

# 4️⃣ Mann–Whitney U test (non-parametric)
u_stat, u_p = stats.mannwhitneyu(
    data.loc[data.High_Foodservice_summer==1, 'PCT_DIABETES_ADULTS13'],
    data.loc[data.High_Foodservice_summer==0, 'PCT_DIABETES_ADULTS13'],
    alternative='two-sided'
)
print(f"Mann–Whitney U: U = {u_stat:.3f}, p = {u_p:.4f}")

# 5️⃣ Visualizations
plt.figure()
data.boxplot(column='PCT_DIABETES_ADULTS13', by='High_Foodservice_summer')
plt.title('Diabetes by Foodservice summer Level')
plt.suptitle('')
plt.xlabel('High Foodservice summer (1 = above median)')
plt.ylabel('Diabetes (%)')
plt.show()

plt.figure()
data[data.High_Foodservice_summer==1]['PCT_DIABETES_ADULTS13'].hist(alpha=0.7, label='High Foodservice summer')
data[data.High_Foodservice_summer==0]['PCT_DIABETES_ADULTS13'].hist(alpha=0.7, label='Low Foodservice summer')
plt.legend()
plt.title('Distribution of Diabetes by Foodservice summer Level')
plt.xlabel('Diabetes (%)')
plt.ylabel('Frequency')
plt.show()

#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from causalinference import CausalModel
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm



# Assuming df is your original dataframe loaded already
# columns = [
#     'PCH_SFSP_12_17', 'FOODINSEC_15_17', 'PCT_LACCESS_POP10','PCT_LACCESS_LOWI10',
#     'PCT_LACCESS_HHNV10','PCT_LACCESS_CHILD10','PCT_LACCESS_SENIORS10','GROCPTH11','SUPERCPTH11','CONVSPTH11','SPECSPTH11','SNAPSPTH12',
#     'WICSPTH11','FSRPTH11','PC_FFRSALES07','PCT_NSLP12','PCT_FREE_LUNCH10','PCT_REDUCED_LUNCH10','PCT_SBP12','PCT_SFSP12','PCT_WIC12','PCT_WICINFANTCHILD14',
#     'PCT_WICWOMEN14','PCT_CACFP12','FDPIR12','FMRKTPTH13','VEG_ACRESPTH07','FRESHVEG_ACRESPTH07','ORCHARD_ACRESPTH12','BERRY_ACRESPTH07','SLHOUSE07','GHVEG_SQFTPTH07',
#     'AGRITRSM_OPS07','PCT_DIABETES_ADULTS08','PCT_OBESE_ADULTS12','RECFACPTH11','PCT_NHWHITE10','PCT_NHBLACK10','PCT_HISP10','PCT_NHASIAN10','PCT_NHNA10','PCT_NHPI10',
#     'PCT_65OLDER10','PCT_18YOUNGER10','PERPOV10','METRO13','POPLOSS10'
# ]

# # Drop NA values to maintain data quality
# data = df[columns].dropna()

# # Define treatment properly using median
# median_snap = data['PCH_SFSP_12_17'].median()
# data['High_Foodservice_summer'] = (data['PCH_SFSP_12_17'] > median_snap).astype(int)

# Outcome and Treatment
Y = data['PCT_DIABETES_ADULTS13'].values
T = data['High_Foodservice_summer'].values

# Confounders (covariates)
X = data.drop(columns=['PCH_SFSP_12_17', 'PCT_DIABETES_ADULTS13', 'High_Foodservice_summer']).values

# Standardize confounders
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

### 🌟 **1. Propensity Score Matching (PSM)**

causal = CausalModel(Y, T, X_std)
causal.est_propensity()

print("\nBefore Matching Balance:")
print(causal.summary_stats)

causal.est_via_matching()

print("\nMatching Estimates:")
print(causal.estimates)

print("\nAfter Matching Balance:")
print(causal.summary_stats)

### 🌟 **2. Inverse Probability of Treatment Weighting (IPTW)**

# Logistic Regression for Propensity Scores
ps_model = LogisticRegression(max_iter=1000).fit(X_std, T)
propensity_scores = ps_model.predict_proba(X_std)[:, 1]

# Ensure stable weights (clipping extreme propensity scores)
eps = 1e-6
propensity_scores = np.clip(propensity_scores, eps, 1 - eps)
weights = T / propensity_scores + (1 - T) / (1 - propensity_scores)

# Weighted regression to estimate causal effects
X_treatment = sm.add_constant(T)
iptw_model = sm.WLS(Y, X_treatment, weights=weights)
iptw_results = iptw_model.fit()

print("\nIPTW Regression Results:")
print(iptw_results.summary())



## 📌 **Additional Robust Causal Inference Methods**

# Include these methods to strengthen your analysis:

### 🌟 **3. Doubly Robust Estimation (Recommended)**
seed = 12345

from sklearn.ensemble import RandomForestRegressor
from econml.dr import LinearDRLearner

regressor = RandomForestRegressor(random_state=seed, n_jobs=-1)
propensity = LogisticRegression(max_iter=1000, random_state=seed)

dr_model = LinearDRLearner(model_regression=regressor,
                           model_propensity=propensity)
dr_model.fit(Y, T, X=X_std)
treatment_effect = dr_model.ate(X_std)
print(f"Doubly Robust ATE: {treatment_effect:.4f}")

#%%
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
# Assuming df is your original dataframe loaded already
columns = [
    'PCH_SBP_12_17', 'FOODINSEC_15_17', 'PCT_LACCESS_POP10','PCT_LACCESS_LOWI10',
    'PCT_LACCESS_HHNV10','PCT_LACCESS_CHILD10','PCT_LACCESS_SENIORS10','GROCPTH11','SUPERCPTH11','CONVSPTH11','SPECSPTH11','SNAPSPTH12',
    'WICSPTH11','FSRPTH11','PC_FFRSALES07','PCT_NSLP12','PCT_FREE_LUNCH10','PCT_REDUCED_LUNCH10','PCT_SBP12','PCT_SFSP12','PCT_WIC12','PCT_WICINFANTCHILD14',
    'PCT_WICWOMEN14','PCT_CACFP12','FDPIR12','FMRKTPTH13','VEG_ACRESPTH07','FRESHVEG_ACRESPTH07','ORCHARD_ACRESPTH12','BERRY_ACRESPTH07','SLHOUSE07','GHVEG_SQFTPTH07',
    'AGRITRSM_OPS07','PCT_DIABETES_ADULTS08','PCT_OBESE_ADULTS12','RECFACPTH11','PCT_NHWHITE10','PCT_NHBLACK10','PCT_HISP10','PCT_NHASIAN10','PCT_NHNA10','PCT_NHPI10',
    'PCT_65OLDER10','PCT_18YOUNGER10','PERPOV10','METRO13','POPLOSS10'
]

# Drop NA values to maintain data quality
data = df[columns].dropna()

# Define treatment properly using median
median_snap = data['PCH_SBP_12_17'].median()
data['High_breakfast'] = (data['PCH_SBP_12_17'] > median_snap).astype(int)

# 1️⃣ Descriptive statistics by group
summary = data.groupby('High_breakfast')['FOODINSEC_15_17'].agg(
    count='count',
    mean='mean',
    median='median',
    std='std',
    min='min',
    max='max'
)
print("Descriptive stats:\n", summary)

# 2️⃣ Independent t‑test (with variance check)
levene_p = stats.levene(
    data.loc[data.High_breakfast==1, 'FOODINSEC_15_17'],
    data.loc[data.High_breakfast==0, 'FOODINSEC_15_17']
).pvalue
equal_var = True if levene_p > 0.05 else False

t_stat, t_p = stats.ttest_ind(
    data.loc[data.High_breakfast==1, 'FOODINSEC_15_17'],
    data.loc[data.High_breakfast==0, 'FOODINSEC_15_17'],
    equal_var=equal_var
)
print(f"\nLevene’s test p-value = {levene_p:.4f} → equal_var={equal_var}")
print(f"T‑test: t = {t_stat:.3f}, p = {t_p:.4f}")

# 3️⃣ Cohen’s d
mean_diff = summary.loc[1,'mean'] - summary.loc[0,'mean']
pooled_sd = np.sqrt(((summary.loc[1,'count']-1)*summary.loc[1,'std']**2 +
                     (summary.loc[0,'count']-1)*summary.loc[0,'std']**2) /
                    (summary.loc[1,'count'] + summary.loc[0,'count'] - 2))
cohens_d = mean_diff / pooled_sd
print(f"Cohen’s d = {cohens_d:.3f}")

# 4️⃣ Mann–Whitney U test (non-parametric)
u_stat, u_p = stats.mannwhitneyu(
    data.loc[data.High_breakfast==1, 'FOODINSEC_15_17'],
    data.loc[data.High_breakfast==0, 'FOODINSEC_15_17'],
    alternative='two-sided'
)
print(f"Mann–Whitney U: U = {u_stat:.3f}, p = {u_p:.4f}")

# 5️⃣ Visualizations
plt.figure()
data.boxplot(column='FOODINSEC_15_17', by='High_breakfast')
plt.title('Food Insecurity by breakfast program Level')
plt.suptitle('')
plt.xlabel('High Breakfast program (1 = above median)')
plt.ylabel('Food Insecurity (%)')
plt.show()

plt.figure()
data[data.High_breakfast==1]['FOODINSEC_15_17'].hist(alpha=0.7, label='High Breakfast program ')
data[data.High_breakfast==0]['FOODINSEC_15_17'].hist(alpha=0.7, label='Low Breakfast program ')
plt.legend()
plt.title('Distribution of Food Insecurity by Breakfast program Level')
plt.xlabel('Food Insecurity (%)')
plt.ylabel('Frequency')
plt.show()

#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from causalinference import CausalModel
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm



# # Assuming df is your original dataframe loaded already
# columns = [
#     'PCH_SBP_12_17', 'FOODINSEC_15_17', 'PCT_LACCESS_POP10','PCT_LACCESS_LOWI10',
#     'PCT_LACCESS_HHNV10','PCT_LACCESS_CHILD10','PCT_LACCESS_SENIORS10','GROCPTH11','SUPERCPTH11','CONVSPTH11','SPECSPTH11','SNAPSPTH12',
#     'WICSPTH11','FSRPTH11','PC_FFRSALES07','PCT_NSLP12','PCT_FREE_LUNCH10','PCT_REDUCED_LUNCH10','PCT_SBP12','PCT_SFSP12','PCT_WIC12','PCT_WICINFANTCHILD14',
#     'PCT_WICWOMEN14','PCT_CACFP12','FDPIR12','FMRKTPTH13','VEG_ACRESPTH07','FRESHVEG_ACRESPTH07','ORCHARD_ACRESPTH12','BERRY_ACRESPTH07','SLHOUSE07','GHVEG_SQFTPTH07',
#     'AGRITRSM_OPS07','PCT_DIABETES_ADULTS08','PCT_OBESE_ADULTS12','RECFACPTH11','PCT_NHWHITE10','PCT_NHBLACK10','PCT_HISP10','PCT_NHASIAN10','PCT_NHNA10','PCT_NHPI10',
#     'PCT_65OLDER10','PCT_18YOUNGER10','PERPOV10','METRO13','POPLOSS10'
# ]

# # Drop NA values to maintain data quality
# data = df[columns].dropna()

# # Define treatment properly using median
# median_snap = data['PCH_SBP_12_17'].median()
# data['High_breakfast'] = (data['PCH_SBP_12_17'] > median_snap).astype(int)

# Outcome and Treatment
Y = data['FOODINSEC_15_17'].values
T = data['High_breakfast'].values

# Confounders (covariates)
X = data.drop(columns=['PCH_SBP_12_17', 'FOODINSEC_15_17', 'High_breakfast']).values

# Standardize confounders
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

### 🌟 **1. Propensity Score Matching (PSM)**

causal = CausalModel(Y, T, X_std)
causal.est_propensity()

print("\nBefore Matching Balance:")
print(causal.summary_stats)

causal.est_via_matching()

print("\nMatching Estimates:")
print(causal.estimates)

print("\nAfter Matching Balance:")
print(causal.summary_stats)

### 🌟 **2. Inverse Probability of Treatment Weighting (IPTW)**

# Logistic Regression for Propensity Scores
ps_model = LogisticRegression(max_iter=1000).fit(X_std, T)
propensity_scores = ps_model.predict_proba(X_std)[:, 1]

# Ensure stable weights (clipping extreme propensity scores)
eps = 1e-6
propensity_scores = np.clip(propensity_scores, eps, 1 - eps)
weights = T / propensity_scores + (1 - T) / (1 - propensity_scores)

# Weighted regression to estimate causal effects
X_treatment = sm.add_constant(T)
iptw_model = sm.WLS(Y, X_treatment, weights=weights)
iptw_results = iptw_model.fit()

print("\nIPTW Regression Results:")
print(iptw_results.summary())



## 📌 **Additional Robust Causal Inference Methods**

# Include these methods to strengthen your analysis:

### 🌟 **3. Doubly Robust Estimation (Recommended)**
seed = 12345

from sklearn.ensemble import RandomForestRegressor
from econml.dr import LinearDRLearner

regressor = RandomForestRegressor(random_state=seed, n_jobs=-1)
propensity = LogisticRegression(max_iter=1000, random_state=seed)

dr_model = LinearDRLearner(model_regression=regressor,
                           model_propensity=propensity)
dr_model.fit(Y, T, X=X_std)
treatment_effect = dr_model.ate(X_std)
print(f"Doubly Robust ATE: {treatment_effect:.4f}")

#%%
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
# Assuming df is your original dataframe loaded already
columns = [
    'PCH_SBP_12_17', 'PCT_OBESE_ADULTS17', 'PCT_LACCESS_POP10','PCT_LACCESS_LOWI10',
    'PCT_LACCESS_HHNV10','PCT_LACCESS_CHILD10','PCT_LACCESS_SENIORS10','GROCPTH11','SUPERCPTH11','CONVSPTH11','SPECSPTH11','SNAPSPTH12',
    'WICSPTH11','FSRPTH11','PC_FFRSALES07','PCT_NSLP12','PCT_FREE_LUNCH10','PCT_REDUCED_LUNCH10','PCT_SBP12','PCT_SFSP12','PCT_WIC12','PCT_WICINFANTCHILD14',
    'PCT_WICWOMEN14','PCT_CACFP12','FDPIR12','FMRKTPTH13','VEG_ACRESPTH07','FRESHVEG_ACRESPTH07','ORCHARD_ACRESPTH12','BERRY_ACRESPTH07','SLHOUSE07','GHVEG_SQFTPTH07',
    'AGRITRSM_OPS07','PCT_DIABETES_ADULTS08','PCT_OBESE_ADULTS12','RECFACPTH11','PCT_NHWHITE10','PCT_NHBLACK10','PCT_HISP10','PCT_NHASIAN10','PCT_NHNA10','PCT_NHPI10',
    'PCT_65OLDER10','PCT_18YOUNGER10','PERPOV10','METRO13','POPLOSS10'
]

# Drop NA values to maintain data quality
data = df[columns].dropna()

# Define treatment properly using median
median_snap = data['PCH_SBP_12_17'].median()
data['High_breakfast'] = (data['PCH_SBP_12_17'] > median_snap).astype(int)

# 1️⃣ Descriptive statistics by group
summary = data.groupby('High_breakfast')['PCT_OBESE_ADULTS17'].agg(
    count='count',
    mean='mean',
    median='median',
    std='std',
    min='min',
    max='max'
)
print("Descriptive stats:\n", summary)

# 2️⃣ Independent t‑test (with variance check)
levene_p = stats.levene(
    data.loc[data.High_breakfast==1, 'PCT_OBESE_ADULTS17'],
    data.loc[data.High_breakfast==0, 'PCT_OBESE_ADULTS17']
).pvalue
equal_var = True if levene_p > 0.05 else False

t_stat, t_p = stats.ttest_ind(
    data.loc[data.High_breakfast==1, 'PCT_OBESE_ADULTS17'],
    data.loc[data.High_breakfast==0, 'PCT_OBESE_ADULTS17'],
    equal_var=equal_var
)
print(f"\nLevene’s test p-value = {levene_p:.4f} → equal_var={equal_var}")
print(f"T‑test: t = {t_stat:.3f}, p = {t_p:.4f}")

# 3️⃣ Cohen’s d
mean_diff = summary.loc[1,'mean'] - summary.loc[0,'mean']
pooled_sd = np.sqrt(((summary.loc[1,'count']-1)*summary.loc[1,'std']**2 +
                     (summary.loc[0,'count']-1)*summary.loc[0,'std']**2) /
                    (summary.loc[1,'count'] + summary.loc[0,'count'] - 2))
cohens_d = mean_diff / pooled_sd
print(f"Cohen’s d = {cohens_d:.3f}")

# 4️⃣ Mann–Whitney U test (non-parametric)
u_stat, u_p = stats.mannwhitneyu(
    data.loc[data.High_breakfast==1, 'PCT_OBESE_ADULTS17'],
    data.loc[data.High_breakfast==0, 'PCT_OBESE_ADULTS17'],
    alternative='two-sided'
)
print(f"Mann–Whitney U: U = {u_stat:.3f}, p = {u_p:.4f}")

# 5️⃣ Visualizations
plt.figure()
data.boxplot(column='PCT_OBESE_ADULTS17', by='High_breakfast')
plt.title('Food Insecurity by breakfast program Level')
plt.suptitle('')
plt.xlabel('High Breakfast program (1 = above median)')
plt.ylabel('Obesity (%)')
plt.show()

plt.figure()
data[data.High_breakfast==1]['PCT_OBESE_ADULTS17'].hist(alpha=0.7, label='High Breakfast program ')
data[data.High_breakfast==0]['PCT_OBESE_ADULTS17'].hist(alpha=0.7, label='Low Breakfast program ')
plt.legend()
plt.title('Distribution of Food Insecurity by Breakfast program Level')
plt.xlabel('Food Insecurity (%)')
plt.ylabel('Frequency')
plt.show()

#%%
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
# Assuming df is your original dataframe loaded already
columns = [
    'PCH_SBP_12_17', 'PCT_DIABETES_ADULTS13', 'PCT_LACCESS_POP10','PCT_LACCESS_LOWI10',
    'PCT_LACCESS_HHNV10','PCT_LACCESS_CHILD10','PCT_LACCESS_SENIORS10','GROCPTH11','SUPERCPTH11','CONVSPTH11','SPECSPTH11','SNAPSPTH12',
    'WICSPTH11','FSRPTH11','PC_FFRSALES07','PCT_NSLP12','PCT_FREE_LUNCH10','PCT_REDUCED_LUNCH10','PCT_SBP12','PCT_SFSP12','PCT_WIC12','PCT_WICINFANTCHILD14',
    'PCT_WICWOMEN14','PCT_CACFP12','FDPIR12','FMRKTPTH13','VEG_ACRESPTH07','FRESHVEG_ACRESPTH07','ORCHARD_ACRESPTH12','BERRY_ACRESPTH07','SLHOUSE07','GHVEG_SQFTPTH07',
    'AGRITRSM_OPS07','PCT_DIABETES_ADULTS08','PCT_OBESE_ADULTS12','RECFACPTH11','PCT_NHWHITE10','PCT_NHBLACK10','PCT_HISP10','PCT_NHASIAN10','PCT_NHNA10','PCT_NHPI10',
    'PCT_65OLDER10','PCT_18YOUNGER10','PERPOV10','METRO13','POPLOSS10'
]

# Drop NA values to maintain data quality
data = df[columns].dropna()

# Define treatment properly using median
median_snap = data['PCH_SBP_12_17'].median()
data['High_breakfast'] = (data['PCH_SBP_12_17'] > median_snap).astype(int)

# 1️⃣ Descriptive statistics by group
summary = data.groupby('High_breakfast')['PCT_DIABETES_ADULTS13'].agg(
    count='count',
    mean='mean',
    median='median',
    std='std',
    min='min',
    max='max'
)
print("Descriptive stats:\n", summary)

# 2️⃣ Independent t‑test (with variance check)
levene_p = stats.levene(
    data.loc[data.High_breakfast==1, 'PCT_DIABETES_ADULTS13'],
    data.loc[data.High_breakfast==0, 'PCT_DIABETES_ADULTS13']
).pvalue
equal_var = True if levene_p > 0.05 else False

t_stat, t_p = stats.ttest_ind(
    data.loc[data.High_breakfast==1, 'PCT_DIABETES_ADULTS13'],
    data.loc[data.High_breakfast==0, 'PCT_DIABETES_ADULTS13'],
    equal_var=equal_var
)
print(f"\nLevene’s test p-value = {levene_p:.4f} → equal_var={equal_var}")
print(f"T‑test: t = {t_stat:.3f}, p = {t_p:.4f}")

# 3️⃣ Cohen’s d
mean_diff = summary.loc[1,'mean'] - summary.loc[0,'mean']
pooled_sd = np.sqrt(((summary.loc[1,'count']-1)*summary.loc[1,'std']**2 +
                     (summary.loc[0,'count']-1)*summary.loc[0,'std']**2) /
                    (summary.loc[1,'count'] + summary.loc[0,'count'] - 2))
cohens_d = mean_diff / pooled_sd
print(f"Cohen’s d = {cohens_d:.3f}")

# 4️⃣ Mann–Whitney U test (non-parametric)
u_stat, u_p = stats.mannwhitneyu(
    data.loc[data.High_breakfast==1, 'PCT_DIABETES_ADULTS13'],
    data.loc[data.High_breakfast==0, 'PCT_DIABETES_ADULTS13'],
    alternative='two-sided'
)
print(f"Mann–Whitney U: U = {u_stat:.3f}, p = {u_p:.4f}")

# 5️⃣ Visualizations
plt.figure()
data.boxplot(column='PCT_DIABETES_ADULTS13', by='High_breakfast')
plt.title('Food Insecurity by breakfast program Level')
plt.suptitle('')
plt.xlabel('High Breakfast program (1 = above median)')
plt.ylabel('Obesity (%)')
plt.show()

plt.figure()
data[data.High_breakfast==1]['PCT_DIABETES_ADULTS13'].hist(alpha=0.7, label='High Breakfast program ')
data[data.High_breakfast==0]['PCT_DIABETES_ADULTS13'].hist(alpha=0.7, label='Low Breakfast program ')
plt.legend()
plt.title('Distribution of Food Insecurity by Breakfast program Level')
plt.xlabel('Food Insecurity (%)')
plt.ylabel('Frequency')
plt.show()

#%%
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
# Assuming df is your original dataframe loaded already
columns = [
    'PCH_RECFACPTH_11_16', 'FOODINSEC_15_17', 'PCT_LACCESS_POP10','PCT_LACCESS_LOWI10',
    'PCT_LACCESS_HHNV10','PCT_LACCESS_CHILD10','PCT_LACCESS_SENIORS10','GROCPTH11','SUPERCPTH11','CONVSPTH11','SPECSPTH11','SNAPSPTH12',
    'WICSPTH11','FSRPTH11','PC_FFRSALES07','PCT_NSLP12','PCT_FREE_LUNCH10','PCT_REDUCED_LUNCH10','PCT_SBP12','PCT_SFSP12','PCT_WIC12','PCT_WICINFANTCHILD14',
    'PCT_WICWOMEN14','PCT_CACFP12','FDPIR12','FMRKTPTH13','VEG_ACRESPTH07','FRESHVEG_ACRESPTH07','ORCHARD_ACRESPTH12','BERRY_ACRESPTH07','SLHOUSE07','GHVEG_SQFTPTH07',
    'AGRITRSM_OPS07','PCT_DIABETES_ADULTS08','PCT_OBESE_ADULTS12','RECFACPTH11','PCT_NHWHITE10','PCT_NHBLACK10','PCT_HISP10','PCT_NHASIAN10','PCT_NHNA10','PCT_NHPI10',
    'PCT_65OLDER10','PCT_18YOUNGER10','PERPOV10','METRO13','POPLOSS10'
]

# Drop NA values to maintain data quality
data = df[columns].dropna()

# Define treatment properly using median
median_snap = data['PCH_RECFACPTH_11_16'].median()
data['High_fitness'] = (data['PCH_RECFACPTH_11_16'] > median_snap).astype(int)


# 1️⃣ Descriptive statistics by group
summary = data.groupby('High_fitness')['FOODINSEC_15_17'].agg(
    count='count',
    mean='mean',
    median='median',
    std='std',
    min='min',
    max='max'
)
print("Descriptive stats:\n", summary)

# 2️⃣ Independent t‑test (with variance check)
levene_p = stats.levene(
    data.loc[data.High_fitness==1, 'FOODINSEC_15_17'],
    data.loc[data.High_fitness==0, 'FOODINSEC_15_17']
).pvalue
equal_var = True if levene_p > 0.05 else False

t_stat, t_p = stats.ttest_ind(
    data.loc[data.High_fitness==1, 'FOODINSEC_15_17'],
    data.loc[data.High_fitness==0, 'FOODINSEC_15_17'],
    equal_var=equal_var
)
print(f"\nLevene’s test p-value = {levene_p:.4f} → equal_var={equal_var}")
print(f"T‑test: t = {t_stat:.3f}, p = {t_p:.4f}")

# 3️⃣ Cohen’s d
mean_diff = summary.loc[1,'mean'] - summary.loc[0,'mean']
pooled_sd = np.sqrt(((summary.loc[1,'count']-1)*summary.loc[1,'std']**2 +
                     (summary.loc[0,'count']-1)*summary.loc[0,'std']**2) /
                    (summary.loc[1,'count'] + summary.loc[0,'count'] - 2))
cohens_d = mean_diff / pooled_sd
print(f"Cohen’s d = {cohens_d:.3f}")

# 4️⃣ Mann–Whitney U test (non-parametric)
u_stat, u_p = stats.mannwhitneyu(
    data.loc[data.High_fitness==1, 'FOODINSEC_15_17'],
    data.loc[data.High_fitness==0, 'FOODINSEC_15_17'],
    alternative='two-sided'
)
print(f"Mann–Whitney U: U = {u_stat:.3f}, p = {u_p:.4f}")

# 5️⃣ Visualizations
plt.figure()
data.boxplot(column='FOODINSEC_15_17', by='High_fitness')
plt.title('Food Insecurity by fitness program Level')
plt.suptitle('')
plt.xlabel('High fitness program (1 = above median)')
plt.ylabel('Food Insecurity (%)')
plt.show()

plt.figure()
data[data.High_fitness==1]['FOODINSEC_15_17'].hist(alpha=0.7, label='High fitness program')
data[data.High_fitness==0]['FOODINSEC_15_17'].hist(alpha=0.7, label='Low fitness program')
plt.legend()
plt.title('Distribution of Food Insecurity by fitness program Level')
plt.xlabel('Food Insecurity (%)')
plt.ylabel('Frequency')
plt.show()

#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from causalinference import CausalModel
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm



# Assuming df is your original dataframe loaded already
columns = [
    'PCH_RECFACPTH_11_16', 'FOODINSEC_15_17', 'PCT_LACCESS_POP10','PCT_LACCESS_LOWI10',
    'PCT_LACCESS_HHNV10','PCT_LACCESS_CHILD10','PCT_LACCESS_SENIORS10','GROCPTH11','SUPERCPTH11','CONVSPTH11','SPECSPTH11','SNAPSPTH12',
    'WICSPTH11','FSRPTH11','PC_FFRSALES07','PCT_NSLP12','PCT_FREE_LUNCH10','PCT_REDUCED_LUNCH10','PCT_SBP12','PCT_SFSP12','PCT_WIC12','PCT_WICINFANTCHILD14',
    'PCT_WICWOMEN14','PCT_CACFP12','FDPIR12','FMRKTPTH13','VEG_ACRESPTH07','FRESHVEG_ACRESPTH07','ORCHARD_ACRESPTH12','BERRY_ACRESPTH07','SLHOUSE07','GHVEG_SQFTPTH07',
    'AGRITRSM_OPS07','PCT_DIABETES_ADULTS08','PCT_OBESE_ADULTS12','RECFACPTH11','PCT_NHWHITE10','PCT_NHBLACK10','PCT_HISP10','PCT_NHASIAN10','PCT_NHNA10','PCT_NHPI10',
    'PCT_65OLDER10','PCT_18YOUNGER10','PERPOV10','METRO13','POPLOSS10'
]

# Drop NA values to maintain data quality
data = df[columns].dropna()

# Define treatment properly using median
median_snap = data['PCH_RECFACPTH_11_16'].median()
data['High_fitness'] = (data['PCH_RECFACPTH_11_16'] > median_snap).astype(int)

# Outcome and Treatment
Y = data['FOODINSEC_15_17'].values
T = data['High_fitness'].values

# Confounders (covariates)
X = data.drop(columns=['PCH_RECFACPTH_11_16', 'FOODINSEC_15_17', 'High_fitness']).values

# Standardize confounders
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

### 🌟 **1. Propensity Score Matching (PSM)**

causal = CausalModel(Y, T, X_std)
causal.est_propensity()

print("\nBefore Matching Balance:")
print(causal.summary_stats)

causal.est_via_matching()

print("\nMatching Estimates:")
print(causal.estimates)

print("\nAfter Matching Balance:")
print(causal.summary_stats)

### 🌟 **2. Inverse Probability of Treatment Weighting (IPTW)**

# Logistic Regression for Propensity Scores
ps_model = LogisticRegression(max_iter=1000).fit(X_std, T)
propensity_scores = ps_model.predict_proba(X_std)[:, 1]

# Ensure stable weights (clipping extreme propensity scores)
eps = 1e-6
propensity_scores = np.clip(propensity_scores, eps, 1 - eps)
weights = T / propensity_scores + (1 - T) / (1 - propensity_scores)

# Weighted regression to estimate causal effects
X_treatment = sm.add_constant(T)
iptw_model = sm.WLS(Y, X_treatment, weights=weights)
iptw_results = iptw_model.fit()

print("\nIPTW Regression Results:")
print(iptw_results.summary())



## 📌 **Additional Robust Causal Inference Methods**

# Include these methods to strengthen your analysis:

### 🌟 **3. Doubly Robust Estimation (Recommended)**
seed = 12345

from sklearn.ensemble import RandomForestRegressor
from econml.dr import LinearDRLearner

regressor = RandomForestRegressor(random_state=seed, n_jobs=-1)
propensity = LogisticRegression(max_iter=1000, random_state=seed)

dr_model = LinearDRLearner(model_regression=regressor,
                           model_propensity=propensity)
dr_model.fit(Y, T, X=X_std)
treatment_effect = dr_model.ate(X_std)
print(f"Doubly Robust ATE: {treatment_effect:.4f}")

#%%
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
# Assuming df is your original dataframe loaded already
columns = [
    'PCH_RECFACPTH_11_16', 'PCT_OBESE_ADULTS17', 'PCT_LACCESS_POP10','PCT_LACCESS_LOWI10',
    'PCT_LACCESS_HHNV10','PCT_LACCESS_CHILD10','PCT_LACCESS_SENIORS10','GROCPTH11','SUPERCPTH11','CONVSPTH11','SPECSPTH11','SNAPSPTH12',
    'WICSPTH11','FSRPTH11','PC_FFRSALES07','PCT_NSLP12','PCT_FREE_LUNCH10','PCT_REDUCED_LUNCH10','PCT_SBP12','PCT_SFSP12','PCT_WIC12','PCT_WICINFANTCHILD14',
    'PCT_WICWOMEN14','PCT_CACFP12','FDPIR12','FMRKTPTH13','VEG_ACRESPTH07','FRESHVEG_ACRESPTH07','ORCHARD_ACRESPTH12','BERRY_ACRESPTH07','SLHOUSE07','GHVEG_SQFTPTH07',
    'AGRITRSM_OPS07','PCT_DIABETES_ADULTS08','PCT_OBESE_ADULTS12','RECFACPTH11','PCT_NHWHITE10','PCT_NHBLACK10','PCT_HISP10','PCT_NHASIAN10','PCT_NHNA10','PCT_NHPI10',
    'PCT_65OLDER10','PCT_18YOUNGER10','PERPOV10','METRO13','POPLOSS10'
]

# Drop NA values to maintain data quality
data = df[columns].dropna()

# Define treatment properly using median
median_snap = data['PCH_RECFACPTH_11_16'].median()
data['High_fitness'] = (data['PCH_RECFACPTH_11_16'] > median_snap).astype(int)


# 1️⃣ Descriptive statistics by group
summary = data.groupby('High_fitness')['PCT_OBESE_ADULTS17'].agg(
    count='count',
    mean='mean',
    median='median',
    std='std',
    min='min',
    max='max'
)
print("Descriptive stats:\n", summary)

# 2️⃣ Independent t‑test (with variance check)
levene_p = stats.levene(
    data.loc[data.High_fitness==1, 'PCT_OBESE_ADULTS17'],
    data.loc[data.High_fitness==0, 'PCT_OBESE_ADULTS17']
).pvalue
equal_var = True if levene_p > 0.05 else False

t_stat, t_p = stats.ttest_ind(
    data.loc[data.High_fitness==1, 'PCT_OBESE_ADULTS17'],
    data.loc[data.High_fitness==0, 'PCT_OBESE_ADULTS17'],
    equal_var=equal_var
)
print(f"\nLevene’s test p-value = {levene_p:.4f} → equal_var={equal_var}")
print(f"T‑test: t = {t_stat:.3f}, p = {t_p:.4f}")

# 3️⃣ Cohen’s d
mean_diff = summary.loc[1,'mean'] - summary.loc[0,'mean']
pooled_sd = np.sqrt(((summary.loc[1,'count']-1)*summary.loc[1,'std']**2 +
                     (summary.loc[0,'count']-1)*summary.loc[0,'std']**2) /
                    (summary.loc[1,'count'] + summary.loc[0,'count'] - 2))
cohens_d = mean_diff / pooled_sd
print(f"Cohen’s d = {cohens_d:.3f}")

# 4️⃣ Mann–Whitney U test (non-parametric)
u_stat, u_p = stats.mannwhitneyu(
    data.loc[data.High_fitness==1, 'PCT_OBESE_ADULTS17'],
    data.loc[data.High_fitness==0, 'PCT_OBESE_ADULTS17'],
    alternative='two-sided'
)
print(f"Mann–Whitney U: U = {u_stat:.3f}, p = {u_p:.4f}")

# 5️⃣ Visualizations
plt.figure()
data.boxplot(column='PCT_OBESE_ADULTS17', by='High_fitness')
plt.title('Food Insecurity by fitness program Level')
plt.suptitle('')
plt.xlabel('High fitness program (1 = above median)')
plt.ylabel('Obesity (%)')
plt.show()

plt.figure()
data[data.High_fitness==1]['PCT_OBESE_ADULTS17'].hist(alpha=0.7, label='High fitness program ')
data[data.High_fitness==0]['PCT_OBESE_ADULTS17'].hist(alpha=0.7, label='Low fitness program ')
plt.legend()
plt.title('Distribution of Food Insecurity by fitness program Level')
plt.xlabel('Food Insecurity (%)')
plt.ylabel('Frequency')
plt.show()

#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from causalinference import CausalModel
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm



# Assuming df is your original dataframe loaded already
columns = [
    'PCH_RECFACPTH_11_16', 'PCT_OBESE_ADULTS17', 'PCT_LACCESS_POP10','PCT_LACCESS_LOWI10',
    'PCT_LACCESS_HHNV10','PCT_LACCESS_CHILD10','PCT_LACCESS_SENIORS10','GROCPTH11','SUPERCPTH11','CONVSPTH11','SPECSPTH11','SNAPSPTH12',
    'WICSPTH11','FSRPTH11','PC_FFRSALES07','PCT_NSLP12','PCT_FREE_LUNCH10','PCT_REDUCED_LUNCH10','PCT_SBP12','PCT_SFSP12','PCT_WIC12','PCT_WICINFANTCHILD14',
    'PCT_WICWOMEN14','PCT_CACFP12','FDPIR12','FMRKTPTH13','VEG_ACRESPTH07','FRESHVEG_ACRESPTH07','ORCHARD_ACRESPTH12','BERRY_ACRESPTH07','SLHOUSE07','GHVEG_SQFTPTH07',
    'AGRITRSM_OPS07','PCT_DIABETES_ADULTS08','PCT_OBESE_ADULTS12','RECFACPTH11','PCT_NHWHITE10','PCT_NHBLACK10','PCT_HISP10','PCT_NHASIAN10','PCT_NHNA10','PCT_NHPI10',
    'PCT_65OLDER10','PCT_18YOUNGER10','PERPOV10','METRO13','POPLOSS10'
]

# Drop NA values to maintain data quality
data = df[columns].dropna()

# Define treatment properly using median
median_snap = data['PCH_RECFACPTH_11_16'].median()
data['High_fitness'] = (data['PCH_RECFACPTH_11_16'] > median_snap).astype(int)

# Outcome and Treatment
Y = data['PCT_OBESE_ADULTS17'].values
T = data['High_fitness'].values

# Confounders (covariates)
X = data.drop(columns=['PCH_RECFACPTH_11_16', 'PCT_OBESE_ADULTS17', 'High_fitness']).values

# Standardize confounders
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

### 🌟 **1. Propensity Score Matching (PSM)**

causal = CausalModel(Y, T, X_std)
causal.est_propensity()

print("\nBefore Matching Balance:")
print(causal.summary_stats)

causal.est_via_matching()

print("\nMatching Estimates:")
print(causal.estimates)

print("\nAfter Matching Balance:")
print(causal.summary_stats)

### 🌟 **2. Inverse Probability of Treatment Weighting (IPTW)**

# Logistic Regression for Propensity Scores
ps_model = LogisticRegression(max_iter=1000).fit(X_std, T)
propensity_scores = ps_model.predict_proba(X_std)[:, 1]

# Ensure stable weights (clipping extreme propensity scores)
eps = 1e-6
propensity_scores = np.clip(propensity_scores, eps, 1 - eps)
weights = T / propensity_scores + (1 - T) / (1 - propensity_scores)

# Weighted regression to estimate causal effects
X_treatment = sm.add_constant(T)
iptw_model = sm.WLS(Y, X_treatment, weights=weights)
iptw_results = iptw_model.fit()

print("\nIPTW Regression Results:")
print(iptw_results.summary())



## 📌 **Additional Robust Causal Inference Methods**

# Include these methods to strengthen your analysis:

### 🌟 **3. Doubly Robust Estimation (Recommended)**
seed = 12345

from sklearn.ensemble import RandomForestRegressor
from econml.dr import LinearDRLearner

regressor = RandomForestRegressor(random_state=seed, n_jobs=-1)
propensity = LogisticRegression(max_iter=1000, random_state=seed)

dr_model = LinearDRLearner(model_regression=regressor,
                           model_propensity=propensity)
dr_model.fit(Y, T, X=X_std)
treatment_effect = dr_model.ate(X_std)
print(f"Doubly Robust ATE: {treatment_effect:.4f}")


#%%
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
# Assuming df is your original dataframe loaded already
# Assuming df is your original dataframe loaded already
columns = [
    'PCH_RECFACPTH_11_16', 'PCT_DIABETES_ADULTS13', 'PCT_LACCESS_POP10','PCT_LACCESS_LOWI10',
    'PCT_LACCESS_HHNV10','PCT_LACCESS_CHILD10','PCT_LACCESS_SENIORS10','GROCPTH11','SUPERCPTH11','CONVSPTH11','SPECSPTH11','SNAPSPTH12',
    'WICSPTH11','FSRPTH11','PC_FFRSALES07','PCT_NSLP12','PCT_FREE_LUNCH10','PCT_REDUCED_LUNCH10','PCT_SBP12','PCT_SFSP12','PCT_WIC12','PCT_WICINFANTCHILD14',
    'PCT_WICWOMEN14','PCT_CACFP12','FDPIR12','FMRKTPTH13','VEG_ACRESPTH07','FRESHVEG_ACRESPTH07','ORCHARD_ACRESPTH12','BERRY_ACRESPTH07','SLHOUSE07','GHVEG_SQFTPTH07',
    'AGRITRSM_OPS07','PCT_DIABETES_ADULTS08','PCT_OBESE_ADULTS12','RECFACPTH11','PCT_NHWHITE10','PCT_NHBLACK10','PCT_HISP10','PCT_NHASIAN10','PCT_NHNA10','PCT_NHPI10',
    'PCT_65OLDER10','PCT_18YOUNGER10','PERPOV10','METRO13','POPLOSS10'
]

# Drop NA values to maintain data quality
data = df[columns].dropna()

# Define treatment properly using median
median_snap = data['PCH_RECFACPTH_11_16'].median()
data['High_fitness'] = (data['PCH_RECFACPTH_11_16'] > median_snap).astype(int)


# 1️⃣ Descriptive statistics by group
summary = data.groupby('High_fitness')['PCT_DIABETES_ADULTS13'].agg(
    count='count',
    mean='mean',
    median='median',
    std='std',
    min='min',
    max='max'
)
print("Descriptive stats:\n", summary)

# 2️⃣ Independent t‑test (with variance check)
levene_p = stats.levene(
    data.loc[data.High_fitness==1, 'PCT_DIABETES_ADULTS13'],
    data.loc[data.High_fitness==0, 'PCT_DIABETES_ADULTS13']
).pvalue
equal_var = True if levene_p > 0.05 else False

t_stat, t_p = stats.ttest_ind(
    data.loc[data.High_fitness==1, 'PCT_DIABETES_ADULTS13'],
    data.loc[data.High_fitness==0, 'PCT_DIABETES_ADULTS13'],
    equal_var=equal_var
)
print(f"\nLevene’s test p-value = {levene_p:.4f} → equal_var={equal_var}")
print(f"T‑test: t = {t_stat:.3f}, p = {t_p:.4f}")

# 3️⃣ Cohen’s d
mean_diff = summary.loc[1,'mean'] - summary.loc[0,'mean']
pooled_sd = np.sqrt(((summary.loc[1,'count']-1)*summary.loc[1,'std']**2 +
                     (summary.loc[0,'count']-1)*summary.loc[0,'std']**2) /
                    (summary.loc[1,'count'] + summary.loc[0,'count'] - 2))
cohens_d = mean_diff / pooled_sd
print(f"Cohen’s d = {cohens_d:.3f}")

# 4️⃣ Mann–Whitney U test (non-parametric)
u_stat, u_p = stats.mannwhitneyu(
    data.loc[data.High_fitness==1, 'PCT_DIABETES_ADULTS13'],
    data.loc[data.High_fitness==0, 'PCT_DIABETES_ADULTS13'],
    alternative='two-sided'
)
print(f"Mann–Whitney U: U = {u_stat:.3f}, p = {u_p:.4f}")

# 5️⃣ Visualizations
plt.figure()
data.boxplot(column='PCT_DIABETES_ADULTS13', by='High_fitness')
plt.title('Food Insecurity by fitness program Level')
plt.suptitle('')
plt.xlabel('High fitness program (1 = above median)')
plt.ylabel('Obesity (%)')
plt.show()

plt.figure()
data[data.High_fitness==1]['PCT_DIABETES_ADULTS13'].hist(alpha=0.7, label='High fitness program ')
data[data.High_fitness==0]['PCT_DIABETES_ADULTS13'].hist(alpha=0.7, label='Low fitness program ')
plt.legend()
plt.title('Distribution of Food Insecurity by fitness program Level')
plt.xlabel('Food Insecurity (%)')
plt.ylabel('Frequency')
plt.show()

#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from causalinference import CausalModel
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm



# Assuming df is your original dataframe loaded already
columns = [
    'PCH_RECFACPTH_11_16', 'PCT_DIABETES_ADULTS13', 'PCT_LACCESS_POP10','PCT_LACCESS_LOWI10',
    'PCT_LACCESS_HHNV10','PCT_LACCESS_CHILD10','PCT_LACCESS_SENIORS10','GROCPTH11','SUPERCPTH11','CONVSPTH11','SPECSPTH11','SNAPSPTH12',
    'WICSPTH11','FSRPTH11','PC_FFRSALES07','PCT_NSLP12','PCT_FREE_LUNCH10','PCT_REDUCED_LUNCH10','PCT_SBP12','PCT_SFSP12','PCT_WIC12','PCT_WICINFANTCHILD14',
    'PCT_WICWOMEN14','PCT_CACFP12','FDPIR12','FMRKTPTH13','VEG_ACRESPTH07','FRESHVEG_ACRESPTH07','ORCHARD_ACRESPTH12','BERRY_ACRESPTH07','SLHOUSE07','GHVEG_SQFTPTH07',
    'AGRITRSM_OPS07','PCT_DIABETES_ADULTS08','PCT_OBESE_ADULTS12','RECFACPTH11','PCT_NHWHITE10','PCT_NHBLACK10','PCT_HISP10','PCT_NHASIAN10','PCT_NHNA10','PCT_NHPI10',
    'PCT_65OLDER10','PCT_18YOUNGER10','PERPOV10','METRO13','POPLOSS10'
]

# Drop NA values to maintain data quality
data = df[columns].dropna()

# Define treatment properly using median
median_snap = data['PCH_RECFACPTH_11_16'].median()
data['High_fitness'] = (data['PCH_RECFACPTH_11_16'] > median_snap).astype(int)

# Outcome and Treatment
Y = data['PCT_DIABETES_ADULTS13'].values
T = data['High_fitness'].values

# Confounders (covariates)
X = data.drop(columns=['PCH_RECFACPTH_11_16', 'PCT_DIABETES_ADULTS13', 'High_fitness']).values

# Standardize confounders
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

### 🌟 **1. Propensity Score Matching (PSM)**

causal = CausalModel(Y, T, X_std)
causal.est_propensity()

print("\nBefore Matching Balance:")
print(causal.summary_stats)

causal.est_via_matching()

print("\nMatching Estimates:")
print(causal.estimates)

print("\nAfter Matching Balance:")
print(causal.summary_stats)

### 🌟 **2. Inverse Probability of Treatment Weighting (IPTW)**

# Logistic Regression for Propensity Scores
ps_model = LogisticRegression(max_iter=1000).fit(X_std, T)
propensity_scores = ps_model.predict_proba(X_std)[:, 1]

# Ensure stable weights (clipping extreme propensity scores)
eps = 1e-6
propensity_scores = np.clip(propensity_scores, eps, 1 - eps)
weights = T / propensity_scores + (1 - T) / (1 - propensity_scores)

# Weighted regression to estimate causal effects
X_treatment = sm.add_constant(T)
iptw_model = sm.WLS(Y, X_treatment, weights=weights)
iptw_results = iptw_model.fit()

print("\nIPTW Regression Results:")
print(iptw_results.summary())



## 📌 **Additional Robust Causal Inference Methods**

# Include these methods to strengthen your analysis:

### 🌟 **3. Doubly Robust Estimation (Recommended)**
seed = 12345

from sklearn.ensemble import RandomForestRegressor
from econml.dr import LinearDRLearner

regressor = RandomForestRegressor(random_state=seed, n_jobs=-1)
propensity = LogisticRegression(max_iter=1000, random_state=seed)

dr_model = LinearDRLearner(model_regression=regressor,
                           model_propensity=propensity)
dr_model.fit(Y, T, X=X_std)
treatment_effect = dr_model.ate(X_std)
print(f"Doubly Robust ATE: {treatment_effect:.4f}")


#%%
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
# Assuming df is your original dataframe loaded already
columns = [
    'PCH_FMRKTPTH_13_18', 'FOODINSEC_15_17', 'PCT_LACCESS_POP10','PCT_LACCESS_LOWI10',
    'PCT_LACCESS_HHNV10','PCT_LACCESS_CHILD10','PCT_LACCESS_SENIORS10','GROCPTH11','SUPERCPTH11','CONVSPTH11','SPECSPTH11','SNAPSPTH12',
    'WICSPTH11','FSRPTH11','PC_FFRSALES07','PCT_NSLP12','PCT_FREE_LUNCH10','PCT_REDUCED_LUNCH10','PCT_SBP12','PCT_SFSP12','PCT_WIC12','PCT_WICINFANTCHILD14',
    'PCT_WICWOMEN14','PCT_CACFP12','FDPIR12','FMRKTPTH13','VEG_ACRESPTH07','FRESHVEG_ACRESPTH07','ORCHARD_ACRESPTH12','BERRY_ACRESPTH07','SLHOUSE07','GHVEG_SQFTPTH07',
    'AGRITRSM_OPS07','PCT_DIABETES_ADULTS08','PCT_OBESE_ADULTS12','RECFACPTH11','PCT_NHWHITE10','PCT_NHBLACK10','PCT_HISP10','PCT_NHASIAN10','PCT_NHNA10','PCT_NHPI10',
    'PCT_65OLDER10','PCT_18YOUNGER10','PERPOV10','METRO13','POPLOSS10'
]

# Drop NA values to maintain data quality
data = df[columns].dropna()

# Define treatment properly using median
median_snap = data['PCH_FMRKTPTH_13_18'].median()
data['High_Farmer_summer'] = (data['PCH_FMRKTPTH_13_18'] > median_snap).astype(int)
# 1️⃣ Descriptive statistics by group
summary = data.groupby('High_Farmer_summer')['FOODINSEC_15_17'].agg(
    count='count',
    mean='mean',
    median='median',
    std='std',
    min='min',
    max='max'
)
print("Descriptive stats:\n", summary)

# 2️⃣ Independent t‑test (with variance check)
levene_p = stats.levene(
    data.loc[data.High_Farmer_summer==1, 'FOODINSEC_15_17'],
    data.loc[data.High_Farmer_summer==0, 'FOODINSEC_15_17']
).pvalue
equal_var = True if levene_p > 0.05 else False

t_stat, t_p = stats.ttest_ind(
    data.loc[data.High_Farmer_summer==1, 'FOODINSEC_15_17'],
    data.loc[data.High_Farmer_summer==0, 'FOODINSEC_15_17'],
    equal_var=equal_var
)
print(f"\nLevene’s test p-value = {levene_p:.4f} → equal_var={equal_var}")
print(f"T‑test: t = {t_stat:.3f}, p = {t_p:.4f}")

# 3️⃣ Cohen’s d
mean_diff = summary.loc[1,'mean'] - summary.loc[0,'mean']
pooled_sd = np.sqrt(((summary.loc[1,'count']-1)*summary.loc[1,'std']**2 +
                     (summary.loc[0,'count']-1)*summary.loc[0,'std']**2) /
                    (summary.loc[1,'count'] + summary.loc[0,'count'] - 2))
cohens_d = mean_diff / pooled_sd
print(f"Cohen’s d = {cohens_d:.3f}")

# 4️⃣ Mann–Whitney U test (non-parametric)
u_stat, u_p = stats.mannwhitneyu(
    data.loc[data.High_Farmer_summer==1, 'FOODINSEC_15_17'],
    data.loc[data.High_Farmer_summer==0, 'FOODINSEC_15_17'],
    alternative='two-sided'
)
print(f"Mann–Whitney U: U = {u_stat:.3f}, p = {u_p:.4f}")

# 5️⃣ Visualizations
plt.figure()
data.boxplot(column='FOODINSEC_15_17', by='High_Farmer_summer')
plt.title('Food Insecurity by Farmer Market Level')
plt.suptitle('')
plt.xlabel('High Farmer Market (1 = above median)')
plt.ylabel('Food Insecurity (%)')
plt.show()

plt.figure()
data[data.High_Farmer_summer==1]['FOODINSEC_15_17'].hist(alpha=0.7, label='High Farmer')
data[data.High_Farmer_summer==0]['FOODINSEC_15_17'].hist(alpha=0.7, label='Low FM')
plt.legend()
plt.title('Distribution of Food Insecurity by Farmer Market  Level')
plt.xlabel('Food Insecurity (%)')
plt.ylabel('Frequency')
plt.show()


#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from causalinference import CausalModel
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm



# Assuming df is your original dataframe loaded already
# columns = [
#     'PCH_RECFACPTH_11_16', 'FOODINSEC_15_17', 'PCT_LACCESS_POP10','PCT_LACCESS_LOWI10',
#     'PCT_LACCESS_HHNV10','PCT_LACCESS_CHILD10','PCT_LACCESS_SENIORS10','GROCPTH11','SUPERCPTH11','CONVSPTH11','SPECSPTH11','SNAPSPTH12',
#     'WICSPTH11','FSRPTH11','PC_FFRSALES07','PCT_NSLP12','PCT_FREE_LUNCH10','PCT_REDUCED_LUNCH10','PCT_SBP12','PCT_SFSP12','PCT_WIC12','PCT_WICINFANTCHILD14',
#     'PCT_WICWOMEN14','PCT_CACFP12','FDPIR12','FMRKTPTH13','VEG_ACRESPTH07','FRESHVEG_ACRESPTH07','ORCHARD_ACRESPTH12','BERRY_ACRESPTH07','SLHOUSE07','GHVEG_SQFTPTH07',
#     'AGRITRSM_OPS07','PCT_DIABETES_ADULTS08','PCT_OBESE_ADULTS12','RECFACPTH11','PCT_NHWHITE10','PCT_NHBLACK10','PCT_HISP10','PCT_NHASIAN10','PCT_NHNA10','PCT_NHPI10',
#     'PCT_65OLDER10','PCT_18YOUNGER10','PERPOV10','METRO13','POPLOSS10'
# ]

# # Drop NA values to maintain data quality
# data = df[columns].dropna()

# # Define treatment properly using median
# median_snap = data['PCH_FMRKTPTH_13_18'].median()
# data['High_fitness'] = (data['PCH_FMRKTPTH_13_18'] > median_snap).astype(int)

# Outcome and Treatment
Y = data['FOODINSEC_15_17'].values
T = data['High_Farmer_summer'].values

# Confounders (covariates)
X = data.drop(columns=['PCH_FMRKTPTH_13_18', 'FOODINSEC_15_17', 'High_Farmer_summer']).values

# Standardize confounders
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

### 🌟 **1. Propensity Score Matching (PSM)**

causal = CausalModel(Y, T, X_std)
causal.est_propensity()

print("\nBefore Matching Balance:")
print(causal.summary_stats)

causal.est_via_matching()

print("\nMatching Estimates:")
print(causal.estimates)

print("\nAfter Matching Balance:")
print(causal.summary_stats)

### 🌟 **2. Inverse Probability of Treatment Weighting (IPTW)**

# Logistic Regression for Propensity Scores
ps_model = LogisticRegression(max_iter=1000).fit(X_std, T)
propensity_scores = ps_model.predict_proba(X_std)[:, 1]

# Ensure stable weights (clipping extreme propensity scores)
eps = 1e-6
propensity_scores = np.clip(propensity_scores, eps, 1 - eps)
weights = T / propensity_scores + (1 - T) / (1 - propensity_scores)

# Weighted regression to estimate causal effects
X_treatment = sm.add_constant(T)
iptw_model = sm.WLS(Y, X_treatment, weights=weights)
iptw_results = iptw_model.fit()

print("\nIPTW Regression Results:")
print(iptw_results.summary())



## 📌 **Additional Robust Causal Inference Methods**

# Include these methods to strengthen your analysis:

### 🌟 **3. Doubly Robust Estimation (Recommended)**
seed = 12345

from sklearn.ensemble import RandomForestRegressor
from econml.dr import LinearDRLearner

regressor = RandomForestRegressor(random_state=seed, n_jobs=-1)
propensity = LogisticRegression(max_iter=1000, random_state=seed)

dr_model = LinearDRLearner(model_regression=regressor,
                           model_propensity=propensity)
dr_model.fit(Y, T, X=X_std)
treatment_effect = dr_model.ate(X_std)
print(f"Doubly Robust ATE: {treatment_effect:.4f}")

#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from causalinference import CausalModel
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm



# Assuming df is your original dataframe loaded already

columns = [
    'PCH_FMRKTPTH_13_18', 'PCT_OBESE_ADULTS17', 'PCT_LACCESS_POP10','PCT_LACCESS_LOWI10',
    'PCT_LACCESS_HHNV10','PCT_LACCESS_CHILD10','PCT_LACCESS_SENIORS10','GROCPTH11','SUPERCPTH11','CONVSPTH11','SPECSPTH11','SNAPSPTH12',
    'WICSPTH11','FSRPTH11','PC_FFRSALES07','PCT_NSLP12','PCT_FREE_LUNCH10','PCT_REDUCED_LUNCH10','PCT_SBP12','PCT_SFSP12','PCT_WIC12','PCT_WICINFANTCHILD14',
    'PCT_WICWOMEN14','PCT_CACFP12','FDPIR12','FMRKTPTH13','VEG_ACRESPTH07','FRESHVEG_ACRESPTH07','ORCHARD_ACRESPTH12','BERRY_ACRESPTH07','SLHOUSE07','GHVEG_SQFTPTH07',
    'AGRITRSM_OPS07','PCT_DIABETES_ADULTS08','PCT_OBESE_ADULTS12','RECFACPTH11','PCT_NHWHITE10','PCT_NHBLACK10','PCT_HISP10','PCT_NHASIAN10','PCT_NHNA10','PCT_NHPI10',
    'PCT_65OLDER10','PCT_18YOUNGER10','PERPOV10','METRO13','POPLOSS10'
]

# Drop NA values to maintain data quality
data = df[columns].dropna()

# Define treatment properly using median
median_snap = data['PCH_FMRKTPTH_13_18'].median()
data['High_Farmer_summer'] = (data['PCH_FMRKTPTH_13_18'] > median_snap).astype(int)

# Outcome and Treatment
Y = data['PCT_OBESE_ADULTS17'].values
T = data['High_Farmer_summer'].values

# Confounders (covariates)
X = data.drop(columns=['PCH_FMRKTPTH_13_18', 'PCT_OBESE_ADULTS17', 'High_Farmer_summer']).values

# Standardize confounders
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

### 🌟 **1. Propensity Score Matching (PSM)**

causal = CausalModel(Y, T, X_std)
causal.est_propensity()

print("\nBefore Matching Balance:")
print(causal.summary_stats)

causal.est_via_matching()

print("\nMatching Estimates:")
print(causal.estimates)

print("\nAfter Matching Balance:")
print(causal.summary_stats)

### 🌟 **2. Inverse Probability of Treatment Weighting (IPTW)**

# Logistic Regression for Propensity Scores
ps_model = LogisticRegression(max_iter=1000).fit(X_std, T)
propensity_scores = ps_model.predict_proba(X_std)[:, 1]

# Ensure stable weights (clipping extreme propensity scores)
eps = 1e-6
propensity_scores = np.clip(propensity_scores, eps, 1 - eps)
weights = T / propensity_scores + (1 - T) / (1 - propensity_scores)

# Weighted regression to estimate causal effects
X_treatment = sm.add_constant(T)
iptw_model = sm.WLS(Y, X_treatment, weights=weights)
iptw_results = iptw_model.fit()

print("\nIPTW Regression Results:")
print(iptw_results.summary())



## 📌 **Additional Robust Causal Inference Methods**

# Include these methods to strengthen your analysis:

### 🌟 **3. Doubly Robust Estimation (Recommended)**
seed = 12345

from sklearn.ensemble import RandomForestRegressor
from econml.dr import LinearDRLearner

regressor = RandomForestRegressor(random_state=seed, n_jobs=-1)
propensity = LogisticRegression(max_iter=1000, random_state=seed)

dr_model = LinearDRLearner(model_regression=regressor,
                           model_propensity=propensity)
dr_model.fit(Y, T, X=X_std)
treatment_effect = dr_model.ate(X_std)
print(f"Doubly Robust ATE: {treatment_effect:.4f}")

#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from causalinference import CausalModel
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm



# Assuming df is your original dataframe loaded already

columns = [
    'PCH_FMRKTPTH_13_18', 'PCT_DIABETES_ADULTS13', 'PCT_LACCESS_POP10','PCT_LACCESS_LOWI10',
    'PCT_LACCESS_HHNV10','PCT_LACCESS_CHILD10','PCT_LACCESS_SENIORS10','GROCPTH11','SUPERCPTH11','CONVSPTH11','SPECSPTH11','SNAPSPTH12',
    'WICSPTH11','FSRPTH11','PC_FFRSALES07','PCT_NSLP12','PCT_FREE_LUNCH10','PCT_REDUCED_LUNCH10','PCT_SBP12','PCT_SFSP12','PCT_WIC12','PCT_WICINFANTCHILD14',
    'PCT_WICWOMEN14','PCT_CACFP12','FDPIR12','FMRKTPTH13','VEG_ACRESPTH07','FRESHVEG_ACRESPTH07','ORCHARD_ACRESPTH12','BERRY_ACRESPTH07','SLHOUSE07','GHVEG_SQFTPTH07',
    'AGRITRSM_OPS07','PCT_DIABETES_ADULTS08','PCT_OBESE_ADULTS12','RECFACPTH11','PCT_NHWHITE10','PCT_NHBLACK10','PCT_HISP10','PCT_NHASIAN10','PCT_NHNA10','PCT_NHPI10',
    'PCT_65OLDER10','PCT_18YOUNGER10','PERPOV10','METRO13','POPLOSS10'
]

# Drop NA values to maintain data quality
data = df[columns].dropna()

# Define treatment properly using median
median_snap = data['PCH_FMRKTPTH_13_18'].median()
data['High_Farmer_summer'] = (data['PCH_FMRKTPTH_13_18'] > median_snap).astype(int)

# Outcome and Treatment
Y = data['PCT_DIABETES_ADULTS13'].values
T = data['High_Farmer_summer'].values

# Confounders (covariates)
X = data.drop(columns=['PCH_FMRKTPTH_13_18', 'PCT_DIABETES_ADULTS13', 'High_Farmer_summer']).values

# Standardize confounders
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

### 🌟 **1. Propensity Score Matching (PSM)**

causal = CausalModel(Y, T, X_std)
causal.est_propensity()

print("\nBefore Matching Balance:")
print(causal.summary_stats)

causal.est_via_matching()

print("\nMatching Estimates:")
print(causal.estimates)

print("\nAfter Matching Balance:")
print(causal.summary_stats)

### 🌟 **2. Inverse Probability of Treatment Weighting (IPTW)**

# Logistic Regression for Propensity Scores
ps_model = LogisticRegression(max_iter=1000).fit(X_std, T)
propensity_scores = ps_model.predict_proba(X_std)[:, 1]

# Ensure stable weights (clipping extreme propensity scores)
eps = 1e-6
propensity_scores = np.clip(propensity_scores, eps, 1 - eps)
weights = T / propensity_scores + (1 - T) / (1 - propensity_scores)

# Weighted regression to estimate causal effects
X_treatment = sm.add_constant(T)
iptw_model = sm.WLS(Y, X_treatment, weights=weights)
iptw_results = iptw_model.fit()

print("\nIPTW Regression Results:")
print(iptw_results.summary())



## 📌 **Additional Robust Causal Inference Methods**

# Include these methods to strengthen your analysis:

### 🌟 **3. Doubly Robust Estimation (Recommended)**
seed = 12345

from sklearn.ensemble import RandomForestRegressor
from econml.dr import LinearDRLearner

regressor = RandomForestRegressor(random_state=seed, n_jobs=-1)
propensity = LogisticRegression(max_iter=1000, random_state=seed)

dr_model = LinearDRLearner(model_regression=regressor,
                           model_propensity=propensity)
dr_model.fit(Y, T, X=X_std)
treatment_effect = dr_model.ate(X_std)
print(f"Doubly Robust ATE: {treatment_effect:.4f}")

