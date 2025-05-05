import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

def synthesize_pack_dataset_advanced(df, rated_capacity_col='rated_capacity', expected_cells=4):
    df = df.copy()
    df.sort_values(['group', 'cycle_index', 'cell_id'], inplace=True)

    # Assign pack UID from group + cycle
    df['pack_uid'] = df['group'].astype(str) + '_c' + df['cycle_index'].astype(str)

    # Identify columns
    dqdt_cols = [col for col in df.columns if col.startswith('dqdt_')]
    didt_cols = [col for col in df.columns if col.startswith('didt_')]
    q_interp_cols = [col for col in df.columns if col.startswith('q_interp_')]
    i_interp_cols = [col for col in df.columns if col.startswith('i_interp_')]

    group_keys = ['group', 'cycle_index']
    grouped = list(df.groupby(group_keys))

    def aggregate_pack(group_key, group):
        group = group.copy()

        # Pad if not enough cells
        actual_cells = len(group)
        is_complete = actual_cells >= expected_cells
        if not is_complete:
            deficit = expected_cells - actual_cells
            pad = group.iloc[[-1] * deficit].copy()
            pad['cell_id'] += '_pad'
            group = pd.concat([group, pad], ignore_index=True)

        # Compute SoH metrics
        soh_values = group['capacity'] / group[rated_capacity_col]
        soh_pack = group['capacity'].sum() / group[rated_capacity_col].sum()

        result = {
            'group': group_key[0],
            'cycle_index': group_key[1],
            'pack_uid': group['pack_uid'].iloc[0],
            'soh': soh_pack,
            'soh_mean': soh_values.mean(),
            'soh_std': soh_values.std(),
            'soh_min': soh_values.min(),
            'soh_max': soh_values.max(),
            'cell_id_percel': group['cell_id'].tolist(),
            'capacity_percel': group['capacity'].tolist(),
            'rated_capacity_percel': group[rated_capacity_col].tolist(),
            'is_complete': is_complete
        }

        for col_group in [dqdt_cols, didt_cols]:
            for col in col_group:
                result[col + '_mean'] = group[col].mean()
                result[col + '_std'] = group[col].std()
                result[col + '_min'] = group[col].min()
                result[col + '_max'] = group[col].max()

        for col_group in [q_interp_cols, i_interp_cols]:
            for col in col_group:
                result[col + '_mean'] = group[col].mean()
                result[col + '_std'] = group[col].std()
                result[col + '_min'] = group[col].min()
                result[col + '_max'] = group[col].max()

        return result

    results = Parallel(n_jobs=-1, backend='loky')(
        delayed(aggregate_pack)(key, group) for key, group in tqdm(grouped)
    )

    return pd.DataFrame(results)
