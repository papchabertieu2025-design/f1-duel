"""Parts II-IV - the methods of the course, each with one job.

  II   Linear regression   how many places is a grid slot worth?          (finish position)
  III  k-means             car = team points + reliability; 3 families     (top / midfield / back)
  IV   Logistic regression three questions about the same race:
                             - will this driver finish in the top 10?      (binary logistic)
                             - will driver A finish ahead of driver B?     (logistic on differences)
                             - who wins?                                   (softmax over the 20 drivers)

Every model is trained only on races that happened BEFORE the races it predicts.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SEED = 2026
TIERS = ["Top car", "Midfield car", "Unreliable car"]

# Features. "Before the weekend" = what we know on Monday; "after qualifying" = Saturday night.
CAR = ["car_perf", "tier_top", "tier_unrel"]          # reliability is used by k-means, not as its own predictor
BEFORE_WEEKEND = CAR + ["drv_avg_finish", "drv_top10"]
AFTER_QUALI = ["grid", "quali_gap", "practice_gap"] + CAR


# --------------------------------------------------------------------------- III
CAR_STATS = ["team_points", "team_mech_dnf"]          # points per race (last 5 races), mechanical failure rate (last 10)


class CarProfile:
    """Two simple team statistics describe the car, then k-means groups the cars into families.

    car_perf  = team points per race over the last 5 races, standardised (higher = faster car)
    car_rel   = minus the team's mechanical-failure rate over the last 10 races, standardised (higher = more reliable)
    tier      = k-means cluster on (car_perf, car_rel), ordered 0 = top car (fastest), 1 = midfield, 2 = unreliable car (most mechanical failures)
    """

    def __init__(self, k=3):
        self.k = k

    def fit(self, train):
        teams = train.drop_duplicates(["raceId", "team"])[CAR_STATS]
        self.scaler = StandardScaler().fit(teams)
        z = self._z(teams)
        self.kmeans = KMeans(self.k, n_init=20, random_state=SEED).fit(z)
        perf = [z[self.kmeans.labels_ == c, 0].mean() for c in range(self.k)]
        rel = [z[self.kmeans.labels_ == c, 1].mean() for c in range(self.k)]
        top = int(np.argmax(perf))
        unrel = min((c for c in range(self.k) if c != top), key=lambda c: rel[c])
        mid = [c for c in range(self.k) if c not in (top, unrel)]
        self.rank = {top: 0, **{c: 1 for c in mid}, unrel: self.k - 1}
        return self

    def _z(self, x):
        return self.scaler.transform(x[CAR_STATS]) * np.array([1.0, -1.0])

    def transform(self, df):
        df = df.copy()
        z = self._z(df)
        df["car_perf"], df["car_rel"] = z[:, 0], z[:, 1]
        df["tier"] = [self.rank[c] for c in self.kmeans.predict(z)]
        df["tier_top"] = (df.tier == 0).astype(int)
        df["tier_unrel"] = (df.tier == self.k - 1).astype(int)
        df["log_grid"] = np.log(df.grid)
        return df


# --------------------------------------------------------------------------- IV
def fit_top10(train, features):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(train[features], train.top10)


def race_pairs(df, features, scaler, return_race=False):
    """All pairs (i, j) of drivers in the same race. x = features_i - features_j, y = 1 if i finished ahead.
    Pairs where both drivers retired are dropped (nobody finished ahead)."""
    X, y, grid_says, race = [], [], [], []
    for rid, g in df.groupby("raceId"):
        fin = g.finish_pos.fillna(99).to_numpy()
        x = scaler.transform(g[features])
        gr = g.grid.to_numpy()
        i, j = np.triu_indices(len(g), 1)
        keep = ~((fin[i] == 99) & (fin[j] == 99))
        i, j = i[keep], j[keep]
        X.append(x[i] - x[j]); y.append((fin[i] < fin[j]).astype(int)); grid_says.append((gr[i] < gr[j]).astype(int))
        race.append(np.full(len(i), rid))
    if return_race:
        return np.vstack(X), np.concatenate(y), np.concatenate(grid_says), np.concatenate(race)
    return np.vstack(X), np.concatenate(y), np.concatenate(grid_says)


class DuelModel:
    """P(A finishes ahead of B) = sigmoid(w . (x_A - x_B)).  No intercept: the order of A and B does not matter.
    Each driver therefore gets one 'race score' s = w . x, and P(A ahead of B) = sigmoid(s_A - s_B)."""

    def fit(self, train, features):
        self.features = features
        self.scaler = StandardScaler().fit(train[features])
        X, y, _ = race_pairs(train, features, self.scaler)
        self.lr = LogisticRegression(fit_intercept=False, max_iter=3000).fit(X, y)
        return self

    def score(self, df):
        return self.scaler.transform(df[self.features]) @ self.lr.coef_[0]

    def prob(self, a_row, b_row):
        s = self.score(pd.DataFrame([a_row, b_row]))
        return float(1 / (1 + np.exp(-(s[0] - s[1]))))


class WinnerModel:
    """Softmax over the drivers of one race: P(i wins) = exp(w . x_i) / sum_j exp(w . x_j).
    It is multiclass logistic regression where the 'classes' are the ~20 drivers of the race.
    Fitted by maximising the likelihood of the real winners (L2 penalty, gradient computed by hand)."""

    def fit(self, train, features, l2=1.0):
        self.features = features
        self.scaler = StandardScaler().fit(train[features])
        races = [(self.scaler.transform(g[features]), int(g.win.to_numpy().argmax()))
                 for _, g in train.groupby("raceId") if g.win.sum() == 1]

        def loss_and_grad(w):
            loss, grad = l2 * w @ w / 2, l2 * w
            for X, k in races:
                s = X @ w
                p = np.exp(s - s.max()); p /= p.sum()
                loss -= np.log(p[k])
                grad += X.T @ p - X[k]          # d(-log p_k)/dw = sum_j p_j x_j - x_k
            return loss, grad

        self.w = minimize(loss_and_grad, np.zeros(len(features)), jac=True, method="L-BFGS-B").x
        return self

    def prob(self, race_df):
        s = self.scaler.transform(race_df[self.features]) @ self.w
        p = np.exp(s - s.max())
        return p / p.sum()


# --------------------------------------------------------------------------- helpers
def pick10(g, col, ascending=False):
    """1 for the 10 drivers a method names as points scorers in race g."""
    return (g[col].rank(ascending=ascending, method="first") <= 10).astype(int)


def scorers_named(df, col, ascending=False):
    """Per race: how many of the 10 drivers picked by `col` really finished in the top 10."""
    return df.groupby("raceId").apply(lambda g: int(g.top10[pick10(g, col, ascending) == 1].sum()),
                                      include_groups=False)
