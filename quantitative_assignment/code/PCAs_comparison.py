from fred_query import get_all_series
import matplotlib.pyplot as plt
# import tensorflow as tf
import xpca
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

PLOT = True
TICKERS = ["GS"+str(n) for n in [1, 2, 3, 5, 7, 10]]
N = len(TICKERS)
df = get_all_series(TICKERS, as_ret=True)

print(df)


# load models
classical_pca = PCA()
ewmpca = xpca.EWMPCA(alpha=0.9305)
ipca = xpca.IPCA()


# demeaning
de_meaned = df.apply(lambda x: x - x.mean())
val = de_meaned.values
print("val:")
print(val)

# pca
print("------PCA------")
classical_pca.fit(val)
components = classical_pca.components_
print("components:")
print(components)

print("explained_variance_ratio:")
print(classical_pca.explained_variance_ratio_)

transformed = classical_pca.transform(val)
df_pca = pd.DataFrame(transformed)
print()
print("------IPCA------")
# ipca
window = 100
end = int(np.ceil(len(df) / window) * window)
Z_periods_ipca = []
for start in range(0, end, window):
    print(f"period [{start}-{min(start+window, len(val))}]:")
    print()
    period = val[start:min(start+window, len(val))]
    zs_ipca = ipca.fit(period)
    print("ipca components:")
    print(ipca.components_)
    period_transformed = ipca.transform(period)
    Z_periods_ipca.append(period_transformed)
    print()
    print()
Z_periods_ipca = np.vstack(Z_periods_ipca)

df_ipca = pd.DataFrame(Z_periods_ipca)

# ewmpca
ewmpca = xpca.EWMPCA(alpha=0.9)
transformed = ewmpca.add_all(val)
df_ewmpca = pd.DataFrame(transformed)


# Residual analysis

n_component_ = 4
X = df_pca.iloc[:, :2]
res = []
for col in df:
    LR = LinearRegression()
    y = df[col]
    LR.fit(X, y)
    error = y - LR.predict(X)
    res.append(error)

residual = np.vstack(res)
df_residual_pca = pd.DataFrame(residual.T)

X = df_ipca.iloc[:, :2]
res = []
for col in df:
    LR = LinearRegression()
    y = df[col]
    LR.fit(X, y)
    error = y - LR.predict(X)
    res.append(error)
residual = np.vstack(res)
df_residual_ipca = pd.DataFrame(residual.T)

X = df_ewmpca.iloc[:, :2]
res = []
for col in df:
    LR = LinearRegression()
    y = df[col]
    LR.fit(X, y)
    error = y - LR.predict(X)
    res.append(error)
residual = np.vstack(res)
df_residual_ewmpca = pd.DataFrame(residual.T)

if PLOT:
    fig, ax = plt.subplots(2, 3, figsize=[20, 20])
    df_pca.plot(ax=ax[0, 0])
    df_ipca.plot(ax=ax[0, 1])
    df_ewmpca.plot(ax=ax[0, 2])
    df_residual_pca.plot(ax=ax[1, 0])
    df_residual_ipca.plot(ax=ax[1, 1])
    df_residual_ewmpca.plot(ax=ax[1, 2])
    for a in ax:
        for b in a:
            b.grid()
    plt.show()
