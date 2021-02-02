#!/usr/bin/env python3

from contextlib import contextmanager
import pandas as pd
import numpy as np
import random
import torch
import time
import os
import argparse
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

@contextmanager
def timeit(name: str) -> None:
    before = time.time()
    try:
        yield
    finally:
        duration = time.time() - before
        print("%s: %.3f sec." % (name, duration))

parser = argparse.ArgumentParser(description='Linear Regression')
parser.add_argument('csv_file', metavar='csv-file', type=str, nargs=1)
parser.add_argument('target', metavar='target-column', type=str, nargs=1)
parser.add_argument('exclude', metavar='excluded-columns', type=str, nargs='*')
parser.add_argument('-testratio', metavar='ratio', type=float, default=0.5, nargs=None)
parser.add_argument('-epochs', metavar='epochs', type=int, default=1, nargs=None)
parser.add_argument('-batchsize', metavar='batch size', type=int, default=256, nargs=None)
parser.add_argument('-lr', metavar='learning rate', type=float, default=0.001, nargs=None)
parser.add_argument('-decay', metavar='weight decay', type=float, default=0.0, nargs=None)
parser.add_argument('-momentum', metavar='gradient momentum', type=float, default=0.1, nargs=None)
parser.add_argument('-sep', metavar='separator', type=str, default=",", nargs=None)
args = parser.parse_args()

target_col = args.target[0]

with timeit("CSV parsing"):
    excluded = set(args.exclude)
    df = pd.read_csv(args.csv_file[0], sep=args.sep[0], header=0, na_values=["", " ", "NA", "-"])
    numerical = [
        col for col, t in df.dtypes.iteritems()
        if t in (np.int64, np.float64) and t not in excluded
    ]
    categorical = [
        col for col, t in df.dtypes.iteritems()
        if t == np.object and t not in excluded
    ]
    numerical.remove(target_col)
    df[categorical] = df[categorical].astype(str)  # required for one-hot

with timeit("set split"):
    train_set, test_set = train_test_split(df, shuffle=True, test_size=args.testratio)

with timeit("training+running imputer"):
    X_num = train_set[numerical].values  # already makes a copy
    imputer = SimpleImputer(copy=False)
    X_num = imputer.fit_transform(X_num)

with timeit("training+running scaler"):
    scaler = StandardScaler(copy=False)
    X_num = scaler.fit_transform(X_num)

# with timeit("hash encoding"):
#     X_cat = df[categorical]
#     hash = HashingEncoder(n_components=32).fit(X_cat)
#     X_cat = hash.transform(X_cat)

if len(categorical) > 0:
    with timeit("one-hot encoding"):
        X_cat = train_set[categorical].values
        #cat_imputer = SimpleImputer(copy=False, strategy='most_frequent')
        #X_cat = cat_imputer.fit_transform(X_cat)
        one_hot = OneHotEncoder(sparse=True, handle_unknown='ignore')
        X_cat = one_hot.fit_transform(X_cat)
    dim = X_cat.shape[1] + X_num.shape[1]
else:
    dim = X_num.shape[1]

print("dimensions:", dim)

y_true = train_set[args.target[0]].values.astype(np.float32)
y_scale = y_true.std()
y_true /= y_scale
regressor = torch.nn.Linear(dim, 1)
torch.nn.init.kaiming_normal_(regressor.weight, nonlinearity='linear')
optimizer = torch.optim.SGD(
    regressor.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay
)

with timeit("training"):
    indices = list(range(len(X_num)))
    n_sections = 1 + len(X_num) // args.batchsize
    for epoch in range(args.epochs):
        print("epoch", epoch)
        random.shuffle(indices)
        for idx in np.array_split(indices, n_sections):
            y_batch = torch.Tensor(y_true[idx])
            num = torch.Tensor(X_num[idx, :])
            if len(categorical) > 0:
                cat = torch.Tensor(X_cat[idx, :].todense())
                batch = torch.cat([num, cat], dim=1)
            else:
                batch = num
            optimizer.zero_grad()
            y_pred = regressor(batch).squeeze(1)
            loss = (y_batch - y_pred).pow(2).sum()
            loss.backward()
            optimizer.step()

    regressor.eval()

with timeit("running imputer on testing data"):
    X_num = test_set[numerical].values
    X_num = imputer.transform(X_num)

with timeit("running scaler on testing data"):
    X_num = scaler.transform(X_num)

if len(categorical) > 0:
    with timeit("running one-hot on testing data"):
        X_cat = test_set[categorical].values
        X_cat = one_hot.transform(X_cat)

with timeit("predicting"):
    batch_size = 4096
    y = []
    for i in range(0, len(X_num), batch_size):
        end = min(len(X_num), i+batch_size)
        num = torch.Tensor(X_num[i:end, :])
        if len(categorical) > 0:
            cat = torch.Tensor(X_cat[i:end, :].todense())
            batch = torch.cat([num, cat], dim=1)
        else:
            batch = num
        y += regressor(batch).squeeze(1).tolist()

    y = np.array(y) * y_scale
    y_true = test_set[args.target[0]].values.astype(np.float32)
    mae = np.abs(y_true - y).mean()
    print("MAE", mae)
    ref = np.abs(y_true - y_true.mean()).mean()
    print("Baseline", ref)


outdir = "outdir"

# with timeit("writing"):
#     batch_size = 1024
#     for j, i in enumerate(range(0, len(X_num), batch_size)):
#         d = X_cat[i:i+batch_size, :].todense()
#         X = np.concatenate([X_num[i:i+batch_size], d], axis=1)
#         print("X dim", X.shape[1])
#         pd.DataFrame(X).to_csv("%s/output%i.csv" % (outdir, j), index=False)

with timeit("reading again"):
    n = 0
    for filename in os.listdir(outdir):
        df = pd.read_csv(os.path.join(outdir, filename))
        n += len(df)
    print("number of rows:", n)
