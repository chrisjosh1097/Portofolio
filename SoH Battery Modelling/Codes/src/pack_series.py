import pandas as pd

def preprocess_and_flatten_with_soh(df):
    # Extract the group_id (G1, G2, etc.)
    df['group_id'] = df['cell_id'].str.extract(r'(G\d+)')
    
    trimmed_groups = []
    
    # Trim each group
    for group_id, group_df in df.groupby('group_id'):
        # Find maximum rpt_index per cell
        max_rpt_per_cell = group_df.groupby('cell_id')['rpt_index'].max()
        # Find the minimum across the 4 cells
        min_max_rpt = max_rpt_per_cell.min()
        # Trim group's data
        trimmed_group_df = group_df[group_df['rpt_index'] <= min_max_rpt]
        trimmed_groups.append(trimmed_group_df)
    
    # Combine trimmed groups
    trimmed_df = pd.concat(trimmed_groups).reset_index(drop=True)
    
    pivoted = []
    
    for group_id, group_df in trimmed_df.groupby('group_id'):
        # Extract cell number
        group_df['cell_no'] = group_df['cell_id'].str.extract(r'C(\d)')
        
        # Select feature columns correctly from group_df
        feature_cols = [col for col in group_df.columns if col not in ['cell_id', 'group_id', 'cell_no', 'rpt_index']]
        
        # Pivot based on cell_no
        group_pivot = group_df.pivot(index=['group_id', 'rpt_index'], columns='cell_no', values=feature_cols)
        
        # Flatten multi-level columns
        group_pivot.columns = [f'C{cell_no}_{feature}' for feature, cell_no in group_pivot.columns]
        
        group_pivot = group_pivot.reset_index()  # Reset index to bring back group_id, rpt_index as columns
        
        # Create SoH columns list
        soh_cols = [col for col in group_pivot.columns if '_soh' in col.lower()]
        
        # Create the lowest SoH column
        group_pivot['lowest_soh'] = group_pivot[soh_cols].min(axis=1)

        
        
        pivoted.append(group_pivot)
    
    # Combine all groups
    final_df = pd.concat(pivoted).reset_index(drop=True)
    final_df = final_df.drop(columns=['C1_soh','C2_soh','C3_soh','C4_soh'])
    return final_df

# Example
df = pd.read_csv('Datasets/isu_Interp_flat.csv')
df_flattened = preprocess_and_flatten_with_soh(df)
print(df_flattened)
print(df_flattened.iloc[:, 0:10])


# selected_cols = [col for col in df_flattened.columns if col in ['group_id','rpt_index','' 'C1_soh','C2_soh','C3_soh','C4_soh','lowest_soh']]
# print(selected_cols)