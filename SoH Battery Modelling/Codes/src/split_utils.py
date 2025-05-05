import numpy as np
from sklearn.model_selection import BaseCrossValidator


class BatteryWiseSplit(BaseCrossValidator):
    def __init__(self, n_splits=5, random_state=None, column_group='source'):
        self.n_splits = n_splits
        self.random_state = random_state
        self.column_group = column_group

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        df = X.copy()
        if 'cell_id' not in df.columns or self.column_group not in df.columns:
            raise ValueError("DataFrame must contain 'cell_id' and column group.")

        rng = np.random.RandomState(self.random_state)
        unique_batteries = df[["cell_id", self.column_group]].drop_duplicates()
        batteries_by_source = unique_batteries.groupby(self.column_group)["cell_id"].apply(list)
        folds = [[] for _ in range(self.n_splits)]

        for source, battery_list in batteries_by_source.items():
            rng.shuffle(battery_list)
            for i, cell_id in enumerate(battery_list):
                folds[i % self.n_splits].append(cell_id)

        for i in range(self.n_splits):
            val_batteries = set(folds[i])
            train_batteries = set(unique_batteries["cell_id"]) - val_batteries
            train_idx = df[df["cell_id"].isin(train_batteries)].index.values
            val_idx = df[df["cell_id"].isin(val_batteries)].index.values
            yield train_idx, val_idx
