import numpy as np
import pandas as pd
import jax

import xgboost

from jaxgboost.trees.tree import GHTree


def get_col(df):
    n_estimators = df["Tree"].nunique()
    num_parallel_trees = 1
    max_depth = 1 + int(df["depth"].max())
    num_nodes = 2 ** (max_depth + 1) - 1

    col = np.zeros((n_estimators, num_parallel_trees, num_nodes))
    for i, row in df.iterrows():
        if row["Feature"] != "Leaf":
            col[row["Tree"], 0, row["Node"]] = int(row["Feature"].replace("f", ""))

    return col


def get_thr(df):
    n_estimators = df["Tree"].nunique()
    num_parallel_trees = 1
    max_depth = 1 + int(df["depth"].max())
    num_nodes = 2 ** (max_depth + 1) - 1

    thr = np.zeros((n_estimators, num_parallel_trees, num_nodes))
    for i, row in df.iterrows():
        if row["Feature"] != "Leaf":
            split = row["Split"]
            if np.isnan(split):
                split = -np.inf
            thr[row["Tree"], 0, row["Node"]] = split

    return thr


def get_val(df):
    n_estimators = df["Tree"].nunique()
    num_parallel_trees = 1
    max_depth = 1 + int(df["depth"].max())
    num_nodes = 2 ** (max_depth + 1) - 1

    val = np.zeros((n_estimators, num_parallel_trees, num_nodes))
    for i, row in df.iterrows():
        if row["Feature"] == "Leaf":
            val[row["Tree"], 0, row["Node"]] = row["Gain"]

    return val


def get_depth(df):
    n_estimators = df["Tree"].nunique()
    num_parallel_trees = 1
    max_depth = 1 + int(df["depth"].max())
    num_nodes = 2 ** (max_depth + 1) - 1

    depth = np.zeros((n_estimators, num_parallel_trees, num_nodes)) - 1
    for i, row in df.iterrows():
        depth[row["Tree"], 0, row["Node"]] = row["depth"]

    return depth


def get_l_child(df):
    n_estimators = df["Tree"].nunique()
    num_parallel_trees = 1
    max_depth = 1 + int(df["depth"].max())
    num_nodes = 2 ** (max_depth + 1) - 1

    l_child = np.zeros((n_estimators, num_parallel_trees, num_nodes))
    for i, row in df.iterrows():
        if row["Feature"] != "Leaf":
            l_child[row["Tree"], 0, row["Node"]] = int(row["Yes"].split("-")[-1])

    return l_child


def get_r_child(df):
    n_estimators = df["Tree"].nunique()
    num_parallel_trees = 1
    max_depth = 1 + int(df["depth"].max())
    num_nodes = 2 ** (max_depth + 1) - 1

    r_child = np.zeros((n_estimators, num_parallel_trees, num_nodes))
    for i, row in df.iterrows():
        if row["Feature"] != "Leaf":
            r_child[row["Tree"], 0, row["Node"]] = int(row["No"].split("-")[-1])

    return r_child


def get_is_leaf(df):
    n_estimators = df["Tree"].nunique()
    num_parallel_trees = 1
    max_depth = 1 + int(df["depth"].max())
    num_nodes = 2 ** (max_depth + 1) - 1

    is_leaf = np.zeros((n_estimators, num_parallel_trees, num_nodes))
    for i, row in df.iterrows():
        is_leaf[row["Tree"], 0, row["Node"]] = row["Feature"] == "Leaf"

    return is_leaf


def get_is_split(df):
    n_estimators = df["Tree"].nunique()
    num_parallel_trees = 1
    max_depth = 1 + int(df["depth"].max())
    num_nodes = 2 ** (max_depth + 1) - 1

    is_split = np.zeros((n_estimators, num_parallel_trees, num_nodes))
    for i, row in df.iterrows():
        is_split[row["Tree"], 0, row["Node"]] = row["Feature"] != "Leaf"

    return is_split


def trees_to_dataframe(model):
    df = model.trees_to_dataframe()
    df = df.drop(columns=["Cover", "Category"])
    df["depth"] = -1

    is_root = ~(df["ID"].isin(df["Yes"]) | df["ID"].isin(df["No"]))
    nodes = df.loc[is_root, "ID"].unique()

    d = 0
    while True:
        df.loc[df["ID"].isin(nodes), "depth"] = d
        nodes = pd.concat([df[df["ID"].isin(nodes)]["Yes"], df[df["ID"].isin(nodes)]["No"]]).unique()
        if nodes.shape[0] == 0:
            break
        d += 1

    return df


def from_xgboost(model: xgboost.XGBModel | xgboost.Booster):
    if not isinstance(model, xgboost.Booster):
        model = model.get_booster()

    df = trees_to_dataframe(model)

    n_estimators = df["Tree"].nunique()
    num_parallel_trees = 1
    max_depth = 1 + int(df["depth"].max())

    num_nodes = 2 ** (max_depth + 1) - 1

    col = get_col(df)
    thr = get_thr(df)
    val = get_val(df)
    is_leaf = get_is_leaf(df)
    is_split = get_is_split(df)
    l_child = get_l_child(df)
    r_child = get_r_child(df)
    depth = get_depth(df)

    ghtree = GHTree(
        depth=jax.numpy.array(depth),

        # split
        is_split=jax.numpy.array(is_split, dtype=jax.numpy.bool),
        col=jax.numpy.array(col, dtype=jax.numpy.int32),
        thr=jax.numpy.array(thr),
        gain=jax.numpy.zeros((n_estimators, num_parallel_trees, num_nodes,)),
        l_child_id=jax.numpy.array(l_child, dtype=jax.numpy.int32),
        r_child_id=jax.numpy.array(r_child, dtype=jax.numpy.int32),

        # leaf
        is_leaf=jax.numpy.array(is_leaf),
        gh_sum=jax.numpy.zeros((n_estimators, num_parallel_trees, num_nodes, 1, 2)),
        score=jax.numpy.zeros((n_estimators, num_parallel_trees, num_nodes,)),
        value=jax.numpy.array(val)
    )
    return ghtree