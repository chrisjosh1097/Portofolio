from abc import ABC, abstractmethod
from typing import List, NamedTuple

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from split_utils import BatteryWiseSplit
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from plot_utils import plot_comparison_boxplots, plot_comparison_lineplots

class CVResult(NamedTuple):
    mse: List[float]
    mae: List[float]
    r2: List[float]
    avg_mse: float
    avg_mae: float
    avg_r2: float

class BatteryTrainerBase(ABC):
    def __init__(self, df, target_cols, verbose=True, **kwargs):
        self.df = df.copy()
        self.target_cols = target_cols
        self.verbose = verbose

        self._prepare_data()

    def _prepare_data(self):
        splitter = BatteryWiseSplit(n_splits=5, random_state=42, column_group="source")
        self.df = self.df.reset_index(drop=True)
        train_idx, test_idx = next(splitter.split(self.df))

        self.train_idx = train_idx
        self.test_idx = test_idx

        train_df = self.df.iloc[train_idx].copy()
        test_df = self.df.iloc[test_idx].copy()

        feature_cols = [
            c for c in train_df.columns
            if (
                c.startswith("q_interp_") or
                c.startswith("i_interp_") or
                c.startswith("dqdt_") or
                c.startswith("didt_") or
                # c in [
                #     "dqdt_min", "dqdt_max", "dqdt_mean", "dqdt_std",
                #     "dvdt_min", "dvdt_max", "dvdt_mean", "dvdt_std",
                #     "didt_min", "didt_max", "didt_mean", "didt_std"
                # ] or
                c == "cycle_index"
            )
        ]
        self.feature_cols = [col for col in feature_cols if col not in ["capacity", "soh"]]


        self.train_df = train_df
        self.test_df = test_df

    def cross_validate(self, n_splits=5, random_state=42, reset_cache=False, **train_kwargs) -> CVResult:

        if reset_cache or (not hasattr(self, "_fold_indices")):
            splitter = BatteryWiseSplit(
                n_splits=n_splits, random_state=random_state, column_group="source"
            )
            self._fold_indices = list(splitter.split(self.df))

        self.fold_info = []
        mse_scores, mae_scores, r2_scores = [], [], []

        for fold, (train_idx, val_idx) in enumerate(self._fold_indices, 1):
            print(f"\n=== Fold {fold} ===")
            train_df = self.df.iloc[train_idx].copy()
            val_df = self.df.iloc[val_idx].copy()

            train_df, val_df = self.normalize_per_source_split(train_df, val_df)

            model, y_val, y_pred = self.train(
                train_df, val_df, feature_cols=self.feature_cols, **train_kwargs
            )

            mse = mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)

            mse_scores.append(mse)
            mae_scores.append(mae)
            r2_scores.append(r2)

            print(f"  MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

            self.fold_info.append(
                {
                    "fold": fold,
                    "val_cell_ids": val_df["cell_id"].unique(),
                    "val_sources": val_df["source"].unique(),
                    "val_idx": val_idx,
                }
            )

        self.mse_scores = mse_scores
        self.mae_scores = mae_scores
        self.r2_scores = r2_scores

        print("\n=== Average CV Performance ===")
        print(f"Avg MSE: {np.mean(mse_scores):.4f}")
        print(f"Avg MAE: {np.mean(mae_scores):.4f}")
        print(f"Avg R² : {np.mean(r2_scores):.4f}")

        return CVResult(
            mse=mse_scores,
            mae=mae_scores,
            r2=r2_scores,
            avg_mse=np.mean(mse_scores),
            avg_mae=np.mean(mae_scores),
            avg_r2=np.mean(r2_scores)
        )


    def normalize_per_source_split(
        self, train_df, test_df, exclude_cols=["capacity", "soh"]
    ):
        all_train, all_test = [], []
        # self.scalers_by_source = {}

        exclude_cols = list(set(exclude_cols) | {"capacity", "soh"})

        for source in train_df["source"].unique():
            train_src = train_df[train_df["source"] == source].copy()
            test_src = test_df[test_df["source"] == source].copy()

            numeric_cols = train_src.select_dtypes(include="number").columns.tolist()
            columns_base = [col for col in numeric_cols if col not in exclude_cols]
            columns_to_normalize = [
                col for col in columns_base if train_src[col].isna().mean() < 0.5
            ]

            if self.verbose:
                print(f"[{source}] Normalizing {len(columns_to_normalize)} features")

            scaler = StandardScaler()
            train_src[columns_to_normalize] = scaler.fit_transform(
                train_src[columns_to_normalize]
            )
            test_src[columns_to_normalize] = scaler.transform(
                test_src[columns_to_normalize]
            )
            # self.scalers_by_source[source] = scaler

            all_train.append(train_src)
            all_test.append(test_src)

        return pd.concat(all_train), pd.concat(all_test)

    def run_optuna_tuning(self, n_trials=10):
        sampler = optuna.samplers.TPESampler(
            seed=42
        )  # ✅ fixed seed for reproducibility
        study = optuna.create_study(
            direction="minimize", sampler=sampler
        )  # ✅ maximize R²
        study.optimize(
            lambda trial: self.optuna_objective(
                trial
            ),
            n_trials=n_trials,
        )

        print("\nBest trial:")
        best_trial = study.best_trial
        print(f"  MSE: {best_trial.value:.4f}")
        print("  Params:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
        self.best_params = best_trial.params

        return study
    
    def train_base_model(self):
        print("\nTraining base model with default hyperparameters")
        model, _, _ = self.train(
            self.train_df,
            self.test_df,
            feature_cols=self.feature_cols
        )
        self.base_model = model
        return model
    
    def train_tuned_model(self):
        if not hasattr(self, "best_params"):
            raise ValueError("No best_params found. Run run_optuna_tuning() first.")

        print("\nTraining with best Optuna params:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")

        model, _, _ = self.train(
            self.train_df,
            self.test_df,
            feature_cols=self.feature_cols,
            **self.best_params
        )
        self.tuned_model = model
        return model

    def compare_base_vs_tuned_cv(self):
        print("\nRunning cross-validation for base model...")
        base = self.cross_validate()

        print("\nRunning cross-validation for tuned model...")
        tuned = self.cross_validate(**self.best_params)

        base_scores = {"MSE": base.mse, "MAE": base.mae, "R²": base.r2}
        tuned_scores = {"MSE": tuned.mse, "MAE": tuned.mae, "R²": tuned.r2}

        plot_comparison_boxplots(base_scores, tuned_scores)
        plot_comparison_lineplots(base_scores, tuned_scores)

    @abstractmethod
    def optuna_objective(self):
        """
        Should return: mse
        """
        pass

    @abstractmethod
    def train(self, train_df, val_df, feature_cols=None, **kwargs):
        """
        Should return: model, y_val, y_pred
        """
        pass


def build_lstm_sequences(df, feature_cols, target_col="capacity"):
    X_seq, y_seq = {}, {}
    for cell_id in df["cell_id"].unique():
        df_cell = df[df["cell_id"] == cell_id].sort_values("cycle_index")
        X_seq[cell_id] = df_cell[feature_cols].values.astype(np.float32)
        y_seq[cell_id] = df_cell[target_col].values.astype(np.float32)
    return X_seq, y_seq


# ============================
# Dataset for LSTM
# ============================
class BatterySeqDataset(Dataset):
    def __init__(self, X_seq_dict, y_seq_dict):
        self.X = list(X_seq_dict.values())
        self.y = list(y_seq_dict.values())

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


def pad_collate_fn(batch):
    x_seqs, y_seqs = zip(*batch)

    x_padded = pad_sequence(x_seqs, batch_first=True)
    y_padded = pad_sequence(y_seqs, batch_first=True)

    return x_padded, y_padded
