import json
import scipy.io
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback


# ========== NASA INGESTION ==========

def nasa_find_all_mat_files(base_folder):
    return list(Path(base_folder).rglob("*.mat"))

def nasa_load_mat_file(file_path):
    mat = scipy.io.loadmat(file_path, struct_as_record=False, squeeze_me=True)
    key = [k for k in mat.keys() if not k.startswith("__")][0]
    return mat[key].cycle

def nasa_extract_discharge_cycles(cycle_data):
    return [i for i, c in enumerate(cycle_data) if getattr(c, "type", None) == "discharge"]

def nasa_extract_raw_signals(cycle):
    try:
        data = cycle.data
        V = np.atleast_1d(data.Voltage_measured).flatten()
        I = -np.atleast_1d(data.Current_measured).flatten()
        T = np.atleast_1d(data.Time).flatten()
        Temp = np.atleast_1d(data.Temperature_measured).flatten()
        cap = np.atleast_1d(getattr(data, "Capacity", []))
        if cap.size == 0:
            return None
        return T.tolist(), V.tolist(), I.tolist(), Temp.tolist(), float(cap[0])
    except Exception:
        return None

def nasa_compute_charge(T, I):
    if len(T) < 2 or len(T) != len(I):
        return None
    order = np.argsort(T)
    T_sorted = np.array(T)[order]
    I_sorted = np.array(I)[order]
    dt = np.diff(T_sorted, append=T_sorted[-1])
    Q = np.cumsum(I_sorted * dt) / 3600
    return Q.tolist()

def nasa_process_single_file(file_path, rated_capacity=2.0):
    all_rows = []
    cell_id = Path(file_path).stem
    try:
        cycles = nasa_load_mat_file(file_path)
        discharge_indices = nasa_extract_discharge_cycles(cycles)
        for idx in discharge_indices:
            raw = nasa_extract_raw_signals(cycles[idx])
            if raw is None:
                continue
            T, V, I, Temp, cap = raw
            Q = nasa_compute_charge(T, I)
            if Q is None:
                continue
            soh = cap / rated_capacity

            all_rows.append({
                "source": "nasa",
                "cell_id": cell_id,
                "cycle_index": int(idx),
                "timestamp": T,
                "voltage": V,
                "current": I,
                "charge": Q,
                "temperature": Temp,
                "capacity": float(cap),
                "rated_capacity": float(rated_capacity),
                "soh": float(soh)
            })
    except Exception as e:
        print(f"Failed to process {cell_id}: {e}")
        traceback.print_exc()
    return all_rows

def nasa_process_all_files_parallel(base_folder, rated_capacity=2.0, max_workers=10):
    mat_files = nasa_find_all_mat_files(base_folder)
    print(f"Found {len(mat_files)} .mat files")

    all_results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(nasa_process_single_file, str(file), rated_capacity): file.stem
            for file in mat_files
        }

        for i, future in enumerate(as_completed(futures), 1):
            cell_id = futures[future]
            try:
                result = future.result()
                if result:
                    all_results.extend(result)
                print(f"[{i}/{len(mat_files)}] ✓ {cell_id}")
            except Exception as e:
                print(f"[{i}/{len(mat_files)}] ✗ {cell_id} - Failed: {e}")

    df = pd.DataFrame(all_results)
    print(f"Final DataFrame contains {len(df)} rows from {len(df['cell_id'].unique())} unique cells.")
    return df

def nasa_main():
    df = nasa_process_all_files_parallel("../Datasets/5.+Battery+Data+Set/5. Battery Data Set", rated_capacity=2.0, max_workers=10)
    df.to_pickle("../Datasets/nasa_raw.pkl")
    print(f"Saved nasa_raw.pkl with {len(df)} rows.")


# ========== ISU INGESTION ==========

# Static cell batch info
ISU_BATCH2_CELLS = {
    'G57C1','G57C2','G57C3','G57C4','G58C1',
    'G26C3','G49C1','G49C2','G49C3','G49C4',
    'G50C1','G50C3','G50C4'
}

def isu_load_json_file(base_path, cell_id):
    subfolder = 'Release 2.0' if cell_id in ISU_BATCH2_CELLS else 'Release 1.0'
    json_path = Path(f'{base_path}/{subfolder}/{cell_id}.json')
    with json_path.open('r') as file:
        data = json.loads(json.load(file))

    for i, t_list in enumerate(data['QV_discharge_C_5']['t']):
        if len(t_list) > 0:
            data['QV_discharge_C_5']['t'][i] = list(map(np.datetime64, t_list))

    return data

def isu_extract_raw_signals(data, rpt_index):
    try:
        t_raw = np.array(data["QV_discharge_C_5"]["t"][rpt_index], dtype='datetime64[ns]')
        v = np.array(data["QV_discharge_C_5"]["V"][rpt_index])
        i = np.array(data["QV_discharge_C_5"]["I"][rpt_index])
        q = np.array(data["QV_discharge_C_5"]["Q"][rpt_index])
        e = np.array(data["QV_discharge_C_5"]["E"][rpt_index])

        if not all(len(arr) == len(t_raw) for arr in [v, i, q, e]):
            return None

        # Drop duplicate timestamps — keep first
        _, unique_indices = np.unique(t_raw, return_index=True)
        t_raw = t_raw[unique_indices]
        v = v[unique_indices]
        i = i[unique_indices]
        q = q[unique_indices]
        e = e[unique_indices]

        t_seconds = (t_raw - t_raw[0]).astype('timedelta64[ms]').astype(float) / 1000  # ms → s
        return t_seconds.tolist(), v.tolist(), (-i).tolist(), q.tolist(), e.tolist()
    except:
        return None


def isu_process_single_cell(cell_id, base_path, rated_capacity=0.25):
    all_rows = []
    try:
        data = isu_load_json_file(base_path, cell_id)
        capacities = data.get('capacity_discharge_C_5', [])

        for i, cap in enumerate(capacities):
            if isinstance(cap, list):  # skip invalid
                continue

            raw = isu_extract_raw_signals(data, i)
            if raw is None:
                continue

            T, V, I, Q, E = raw
            soh = cap / rated_capacity

            all_rows.append({
                "source": "isu",
                "cell_id": cell_id,
                "cycle_index": i,
                "timestamp": T,
                "voltage": V,
                "current": I,
                "charge": Q,
                "temperature": [30] * len(T),  # ambient only
                "energy": E,
                "capacity": cap,
                "rated_capacity": rated_capacity,
                "soh": soh
            })
    except Exception as e:
        print(f"Failed to process {cell_id}: {e}")
        traceback.print_exc()
    return all_rows

def isu_process_all_files_parallel(base_path, valid_cells_path='valid_cells.csv', rated_capacity=0.25, max_workers=10):
    valid_cells = pd.read_csv(valid_cells_path).values.flatten().tolist()
    print(f"Found {len(valid_cells)} valid ISU cells")

    all_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(isu_process_single_cell, cell_id, base_path, rated_capacity): cell_id
            for cell_id in valid_cells
        }

        for i, future in enumerate(as_completed(futures), 1):
            cell_id = futures[future]
            try:
                result = future.result()
                if result:
                    all_results.extend(result)
                print(f"[{i}/{len(futures)}] ✓ {cell_id}")
            except Exception as e:
                print(f"[{i}/{len(futures)}] ✗ {cell_id} - Failed: {e}")

    df = pd.DataFrame(all_results)
    print(f"Final DataFrame contains {len(df)} rows from {len(df['cell_id'].unique())} unique cells.")
    return df

def isu_main():
    df = isu_process_all_files_parallel(
        base_path="path/to/isu_data",
        valid_cells_path="valid_cells.csv",
        rated_capacity=0.25,
        max_workers=10
    )
    df.to_pickle("../Datasets/isu_raw.pkl")
    print(f"Saved isu_raw.pkl with {len(df)} rows.")


# ========== OXFORD INGESTION ==========

def oxford_check_keys(obj):
    if isinstance(obj, scipy.io.matlab.mat_struct):
        return oxford_todict(obj)
    elif isinstance(obj, np.ndarray):
        return [oxford_check_keys(el) for el in obj]
    else:
        return obj

def oxford_todict(matobj):
    return {f: oxford_check_keys(getattr(matobj, f)) for f in matobj._fieldnames}

def oxford_load_mat_file(mat):
    return {k: oxford_check_keys(v) for k, v in mat.items() if not k.startswith("__")}

def oxford_extract_raw_signals(dc_data):
    try:
        q = np.array(dc_data.get("q", []))
        t_raw = np.array(dc_data.get("t", []))
        v = np.array(dc_data.get("v", []))
        temp = np.array(dc_data.get("T", []))

        if len(q) < 2 or not all(len(arr) == len(t_raw) for arr in [q, v, temp]):
            return None

        t_seconds = (t_raw - t_raw[0]) * 86400  # convert from days to seconds
        q_ah = (-q / 1000).tolist()  # uAh → Ah
        cap = abs(min(q)) / 1000
        return t_seconds.tolist(), v.tolist(), q_ah, temp.tolist(), cap
    except:
        return None

def oxford_compute_current(Q, T):
    Q = np.array(Q)
    T = np.array(T)
    if len(Q) < 2 or len(Q) != len(T):
        return None
    order = np.argsort(T)
    Q_sorted = Q[order]
    T_sorted = T[order]
    dt = np.gradient(T_sorted)
    dQ = np.gradient(Q_sorted)
    I = dQ / dt  # A
    return I.tolist()

def oxford_process_single_file(mat_file, rated_capacity=0.74):
    all_rows = []
    file_id = Path(mat_file).stem
    try:
        mat = scipy.io.loadmat(mat_file, struct_as_record=False, squeeze_me=True)
        mat_dict = oxford_load_mat_file(mat)

        for cell_id, cell_data in mat_dict.items():
            if not isinstance(cell_data, dict):
                continue
            for cycle_key, content in cell_data.items():
                if not isinstance(content, dict) or "C1dc" not in content:
                    continue

                raw = oxford_extract_raw_signals(content["C1dc"])
                if raw is None:
                    continue

                T, V, Q, Temp, cap = raw
                I = oxford_compute_current(Q, T)
                if I is None:
                    continue
                soh = cap / rated_capacity

                all_rows.append({
                    "source": "oxford",
                    "cell_id": cell_id,
                    "cycle_index": int(cycle_key.replace("cyc", "")),
                    "timestamp": T,
                    "voltage": V,
                    "current": I,
                    "charge": Q,
                    "temperature": Temp,
                    "capacity": cap,
                    "rated_capacity": rated_capacity,
                    "soh": soh
                })
    except Exception as e:
        print(f"Failed to process {file_id}: {e}")
        traceback.print_exc()
    return all_rows

def oxford_process_all_files_parallel(base_folder, rated_capacity=0.74, max_workers=10):
    mat_files = list(Path(base_folder).glob("*.mat"))
    print(f"Found {len(mat_files)} Oxford .mat files")

    all_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(oxford_process_single_file, str(file), rated_capacity): file.stem
            for file in mat_files
        }

        for i, future in enumerate(as_completed(futures), 1):
            file_id = futures[future]
            try:
                result = future.result()
                if result:
                    all_results.extend(result)
                print(f"[{i}/{len(futures)}] ✓ {file_id}")
            except Exception as e:
                print(f"[{i}/{len(futures)}] ✗ {file_id} - Failed: {e}")

    df = pd.DataFrame(all_results)
    if df.empty:
        print("No data was ingested.")
    else:
        print(f"Final DataFrame contains {len(df)} rows from {len(df['cell_id'].unique())} unique cells.")
    return df

def oxford_main():
    df = oxford_process_all_files_parallel("path/to/oxford_data", rated_capacity=0.74, max_workers=10)
    df.to_pickle("../Datasets/oxford_raw.pkl")
    print(f"Saved oxford_raw.pkl with {len(df)} rows.")


# ========== INTERPOLATION ==========

import numpy as np
from scipy.interpolate import splrep, splev
from sklearn.metrics import r2_score
from tqdm.notebook import tqdm


DEFAULT_GRID_POINTS = 500

method_config = {
    "nasa": "sample",
    "oxford": "spline",
    "isu": "spline"
}

v_range_config = {
    "nasa": (3.0, 4.2),
    "oxford": (3.0, 4.2),
    "isu": (3.0, 4.2)
}

def interpolate_qv_row(row_dict):
    source = row_dict["source"]
    method = method_config.get(source, "spline")
    vmin, vmax = v_range_config.get(source, (3.0, 4.2))
    s_start, s_max = 1e-3, 1e3
    grid_points = DEFAULT_GRID_POINTS

    status = "nan"

    # Raw values
    V_raw = np.array(row_dict["voltage"])
    Q_raw = np.array(row_dict["charge"])
    I_raw = np.array(row_dict["current"])

    # Mask valid voltage range
    mask = (V_raw >= vmin) & (V_raw <= vmax)
    if np.count_nonzero(mask) < 3:
        q_interp = [np.nan] * grid_points
        i_interp = [np.nan] * grid_points
        status = "low_points"
    else:
        V_masked = V_raw[mask]
        Q_masked = Q_raw[mask]
        I_masked = I_raw[mask]

        # Sort by voltage
        v_order = np.argsort(V_masked)
        V_sorted = V_masked[v_order]
        Q_sorted = Q_masked[v_order]
        I_sorted = I_masked[v_order]
        V_grid = np.linspace(V_sorted.min(), V_sorted.max(), grid_points)

        # Check for valid charge signal
        if len(V_sorted) < 3 or np.allclose(Q_sorted, Q_sorted[0], atol=1e-6):
            q_interp = [Q_sorted[0]] * grid_points if len(Q_sorted) > 0 else [np.nan] * grid_points
            i_interp = [I_sorted[0]] * grid_points if len(I_sorted) > 0 else [np.nan] * grid_points
            status = "flat"
        else:
            try:
                if method == "sample":
                    if len(np.unique(V_sorted)) < 2:
                        q_interp = [np.nan] * grid_points
                        i_interp = [np.nan] * grid_points
                        status = "fail"
                    else:
                        q_interp = np.interp(V_grid, V_sorted, Q_sorted)
                        i_interp = np.interp(V_grid, V_sorted, I_sorted)
                        status = "ok"

                elif method == "spline":
                    def safe_spline(x, y, grid, s_init=1e-3):
                        best_interp = None
                        best_r2 = -np.inf
                        s = s_init
                        while s <= s_max:
                            try:
                                tck = splrep(x, y, s=s)
                                y_fit = splev(x, tck)
                                r2 = r2_score(y, y_fit)
                                if r2 > best_r2:
                                    best_r2 = r2
                                    best_interp = splev(grid, tck)
                            except:
                                pass
                            s *= 10
                        return best_interp if best_interp is not None else [np.nan] * len(grid)

                    q_interp = safe_spline(V_sorted, Q_sorted, V_grid)
                    i_interp = safe_spline(V_sorted, I_sorted, V_grid)
                    status = "ok" if not np.isnan(q_interp).all() else "fail"

                else:
                    q_interp = [np.nan] * grid_points
                    i_interp = [np.nan] * grid_points
                    status = "fail"
            except:
                q_interp = [np.nan] * grid_points
                i_interp = [np.nan] * grid_points
                status = "fail"

    # Build result row
    row_out = row_dict.copy()
    for i in range(grid_points):
        row_out[f"q_interp_{i}"] = q_interp[i]
        row_out[f"i_interp_{i}"] = i_interp[i]

    row_out["interp_status"] = status
    return row_out

def interpolate_qv_parallel(df, max_workers=8, save_path=None, verbose=True):

    rows = df.to_dict("records")
    results = []

    if verbose:
        print(f"Interpolating {len(rows)} rows using {max_workers} workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(interpolate_qv_row, row) for row in rows]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Interpolating", leave=True):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                if verbose:
                    print(f"Error: {e}")

    df_interp = pd.DataFrame(results)

    if save_path:
        df_interp.to_pickle(save_path)
        if verbose:
            print(f"Saved interpolated results to {save_path}")

    return df_interp

# ========== GRADIENTS ==========
def compute_single_gradient(row_dict):
    try:
        T = np.array(row_dict["timestamp"])
        Q = np.array(row_dict["charge"])
        V = np.array(row_dict["voltage"])
        I = np.array(row_dict["current"])

        if len(T) < 3 or np.any(np.diff(T) <= 0):
            return {
                "index": row_dict["index"],
                "dqdt_min": np.nan, "dqdt_max": np.nan, "dqdt_mean": np.nan, "dqdt_std": np.nan,
                "dvdt_min": np.nan, "dvdt_max": np.nan, "dvdt_mean": np.nan, "dvdt_std": np.nan,
                "didt_min": np.nan, "didt_max": np.nan
            }

        dQdt = np.gradient(Q, T)
        dVdt = np.gradient(V, T)
        dIdt = np.gradient(I, T)

        return {
            "index": row_dict["index"],
            "dqdt_min": np.min(dQdt), "dqdt_max": np.max(dQdt),
            "dqdt_mean": np.mean(dQdt), "dqdt_std": np.std(dQdt),
            "dvdt_min": np.min(dVdt), "dvdt_max": np.max(dVdt),
            "dvdt_mean": np.mean(dVdt), "dvdt_std": np.std(dVdt),
            "didt_min": np.min(dIdt), "didt_max": np.max(dIdt)
        }

    except Exception:
        return {
            "index": row_dict["index"],
            "dqdt_min": np.nan, "dqdt_max": np.nan, "dqdt_mean": np.nan, "dqdt_std": np.nan,
            "dvdt_min": np.nan, "dvdt_max": np.nan, "dvdt_mean": np.nan, "dvdt_std": np.nan,
            "didt_min": np.nan, "didt_max": np.nan
        }

def compute_gradient_features_parallel(df, max_workers=8):
    df = df.reset_index(drop=True).copy()
    row_dicts = df.to_dict("records")
    for i, row in enumerate(row_dicts):
        row["index"] = i

    results = []

    print(f"Calculating gradients {len(row_dicts)} rows using {max_workers} workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(compute_single_gradient, row) for row in row_dicts]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Computing Gradients", leave=True):
            results.append(future.result())

    grad_df = pd.DataFrame(results).sort_values("index").drop(columns=["index"]).reset_index(drop=True)
    return pd.concat([df.reset_index(drop=True), grad_df], axis=1)