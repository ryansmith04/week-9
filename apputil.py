import pandas as pd
import numpy as np


class GroupEstimate(object):
    """Predict a value for new data based on the group it falls into."""

    def __init__(self, estimate="mean"):
        """Store which estimate to use: "mean" or "median"."""
        if estimate not in ["mean", "median"]:
            raise ValueError("estimate must be 'mean' or 'median'")
        self.estimate = estimate
        self.columns = None
        self.group_estimates = None

    def fit(self, X, y):
        """Calculate the mean or median of `y` for each group in `X`."""
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        df = pd.DataFrame(X).reset_index(drop=True)
        df["y"] = np.asarray(y)
        if df["y"].isna().any():
            raise ValueError("y should not contain missing values")
        self.columns = list(pd.DataFrame(X).columns)
        self.group_estimates = (
            df.groupby(self.columns, observed=True)["y"]
            .agg(self.estimate)
            .reset_index()
        )
        return self

    def predict(self, X_):
        """Predict the value for each row in `X_` based on its group."""
        if self.group_estimates is None:
            raise ValueError("Call .fit() first.")
        x_ = pd.DataFrame(X_, columns=self.columns)
        # left merge keeps every row of x_ in the same order
        merged = x_.merge(self.group_estimates, on=self.columns, how="left")
        predictions = merged["y"].to_numpy()
        n_missing = int(np.isnan(predictions).sum())
        if n_missing > 0:
            print(f"{n_missing} observation(s) belong to groups that were "
                  "not in the original data.")
        return predictions
