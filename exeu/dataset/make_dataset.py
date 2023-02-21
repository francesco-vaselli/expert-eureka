import numpy as np
import pandas as pd


def make_dataset(seed: int = 0, N: int = 100000) -> pd.DataFrame:
    """Make a dataset for training and testing."""

    np.random.seed(seed)
    N = N

    x0 = np.random.normal(loc=0, scale=1, size=N)
    y0 = np.random.normal(loc=0, scale=1, size=N)
    x1 = np.random.normal(loc=0, scale=1, size=N)
    y1 = np.random.normal(loc=0, scale=1, size=N)

    rho0 = np.hypot(x0, y0)
    rho1 = np.hypot(x1, y1)
    phi0 = np.arctan2(y0, x0)
    phi1 = np.arctan2(y1, x1)

    rhoT = np.hypot(x0 + x1, y0 + y1)
    phiT = np.arctan2(y0 + y1, x0 + x1)

    quad = np.zeros(N)
    quad[(0 < phiT) & (phiT < np.pi / 2)] = 0
    quad[(np.pi / 2 < phiT) & (phiT < np.pi)] = 1
    quad[(-np.pi < phiT) & (phiT < -np.pi / 2)] = 2
    quad[(-np.pi / 2 < phiT) & (phiT < 0)] = 3

    poisson = np.random.poisson(lam=rhoT, size=N)

    df = pd.DataFrame(
        {
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "rho0": rho0,
            "phi0": phi0,
            "rho1": rho1,
            "phi1": phi1,
            "rhoT": rhoT,
            "phiT": phiT,
            "quad": quad,
            "poisson": poisson,
        },
        dtype=np.float32,
    )

    return df


if __name__ == "__main__":
    df = make_dataset(seed=0, N=1000000)
    print(df.poisson.describe())
    df.to_csv("data.csv", index=False)
