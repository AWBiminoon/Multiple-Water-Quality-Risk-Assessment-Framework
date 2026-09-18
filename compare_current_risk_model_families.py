from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
)
from sklearn.neural_network import BernoulliRBM, MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import build_fig3_graph_model_comparison as graphlib


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "current_risk_model_comparison_outputs"
OUT.mkdir(exist_ok=True)
SEED = 42
METRICS = ["Accuracy", "Balanced Acc.", "Macro-F1", "QWK", "±1 Acc.", "ARI", "MCC", "MAE"]
FAMILY_COLORS = {
    "Linear": "#7B8794",
    "Ensemble": "#B8965A",
    "Neural network": "#8D9ECF",
    "Graph model": "#2F7D6D",
}


def evaluate(name, family, truth, pred, comparison="family"):
    return {
        "Model": name,
        "Family": family,
        "Comparison": comparison,
        "Accuracy": accuracy_score(truth, pred),
        "Balanced Acc.": balanced_accuracy_score(truth, pred),
        "Macro-F1": f1_score(truth, pred, average="macro", zero_division=0),
        "QWK": cohen_kappa_score(truth, pred, weights="quadratic"),
        "±1 Acc.": np.mean(np.abs(truth - pred) <= 1),
        "ARI": adjusted_rand_score(truth, pred),
        "MCC": matthews_corrcoef(truth, pred),
        "MAE": mean_absolute_error(truth, pred),
    }


def relation_matrices(edges, station_ids):
    smap = {station: i for i, station in enumerate(station_ids)}
    n = len(station_ids)
    directed = np.zeros((n, n), dtype=np.float32)
    for _, edge in edges.iterrows():
        source, target = smap[edge["source"]], smap[edge["target"]]
        directed[target, source] = float(edge.get("weight", 1.0))
    identity = np.eye(n, dtype=np.float32)
    no_relation = identity
    partial = identity + directed
    full = identity + directed + directed.T
    normalize = lambda matrix: matrix / np.maximum(matrix.sum(1, keepdims=True), 1e-8)
    return normalize(no_relation), normalize(partial), normalize(full), full > 0


def train_graph(name, model, graph_input, x, y, train_t, val_t, test_t):
    pred, info = graphlib.train_graph_model(model, x, y, train_t, val_t, test_t, graph_input, name)
    row = evaluate(name, "Graph model", y[test_t].reshape(-1), pred.reshape(-1))
    row["Best epoch"] = info.iloc[0, 1] if hasattr(info, "iloc") else info.get("最佳轮次", np.nan)
    return row, pred


def normalized(values, metric_names):
    colors = np.zeros_like(values)
    for col, metric in enumerate(metric_names):
        v = values[:, col]
        low, high = v.min(), v.max()
        score = np.ones(len(v)) if high == low else (v - low) / (high - low)
        colors[:, col] = 1 - score if metric == "MAE" else score
    return colors


def draw_heatmap(data, filename, family_markers=True, note=""):
    ordered = data.sort_values("Macro-F1", ascending=False).reset_index(drop=True)
    values = ordered[METRICS].to_numpy(float)
    colors = normalized(values, METRICS)
    cmap = LinearSegmentedColormap.from_list("est", ["#F4F7F9", "#C7D7E5", "#4D77A7"])
    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    ax.pcolormesh(np.arange(9), np.arange(len(ordered)+1), colors, cmap=cmap, vmin=0, vmax=1,
                  edgecolors="white", linewidth=.8, shading="flat")
    ax.set_xlim(0,8); ax.set_ylim(len(ordered),0)
    ax.set_xticks(np.arange(8)+.5, METRICS, rotation=35, ha="left")
    ax.tick_params(top=True,bottom=False,labeltop=True,labelbottom=False,length=0)
    ax.set_yticks(np.arange(len(ordered))+.5, ordered["Model"])
    if family_markers:
        for label, family in zip(ax.get_yticklabels(), ordered["Family"]):
            label.set_color(FAMILY_COLORS[family])
    for r in range(len(ordered)):
        for c in range(8):
            ax.text(c+.5,r+.5,f"{values[r,c]:.3f}",ha="center",va="center",
                    color="white" if colors[r,c]>=.72 else "#222")
    for spine in ax.spines.values(): spine.set_linewidth(.8)
    if note:
        ax.text(1,-.09,note,transform=ax.transAxes,ha="right",va="top",fontsize=9,color="#555")
    fig.subplots_adjust(left=.40 if not family_markers else .22,right=.98,bottom=.14,top=.80)
    svg,png=OUT/f"{filename}.svg",OUT/f"{filename}.png"
    fig.savefig(svg,format="svg",facecolor="white");fig.savefig(png,dpi=300,facecolor="white");plt.close(fig)
    return svg,png


def main():
    np.random.seed(SEED); torch.manual_seed(SEED)
    x,y,features,nodes,edges,station_ids,time_ids,train_t,val_t,test_t,_,_=graphlib.load_data()
    x_train=x[train_t].reshape(-1,x.shape[-1]); y_train=y[train_t].reshape(-1)
    x_test=x[test_t].reshape(-1,x.shape[-1]); y_test=y[test_t].reshape(-1)
    models = [
        ("LR", "Linear", Pipeline([("scale",StandardScaler()),("model",LogisticRegression(max_iter=2500,class_weight="balanced",random_state=SEED))])),
        ("SGD", "Linear", Pipeline([("scale",StandardScaler()),("model",SGDClassifier(loss="log_loss",alpha=2e-4,max_iter=2500,class_weight="balanced",random_state=SEED))])),
        ("RF", "Ensemble", RandomForestClassifier(n_estimators=450,max_depth=14,min_samples_leaf=2,class_weight="balanced_subsample",random_state=SEED,n_jobs=-1)),
        ("LightGBM", "Ensemble", LGBMClassifier(n_estimators=350,learning_rate=.035,num_leaves=25,max_depth=-1,
                                                  class_weight="balanced",random_state=SEED,verbosity=-1,n_jobs=-1)),
        ("ANN", "Neural network", Pipeline([("scale",StandardScaler()),("model",MLPClassifier(hidden_layer_sizes=(96,48),alpha=2e-4,
                                                              learning_rate_init=8e-4,max_iter=700,early_stopping=True,random_state=SEED))])),
        ("RBM", "Neural network", Pipeline([("scale",MinMaxScaler()),("rbm",BernoulliRBM(n_components=72,learning_rate=.025,
                                                      batch_size=24,n_iter=60,random_state=SEED)),
                                             ("model",LogisticRegression(max_iter=2000,class_weight="balanced",random_state=SEED))])),
    ]
    family_rows=[]
    for name,family,model in models:
        print(f"Training {name}...",flush=True);model.fit(x_train,y_train);pred=model.predict(x_test)
        family_rows.append(evaluate(name,family,y_test,pred))

    no_adj,partial_adj,full_adj,full_mask=relation_matrices(edges,station_ids)
    graph_specs=[
        ("GraphSAGE",graphlib.GraphSAGE(x.shape[-1]),full_adj),
        ("GAT",graphlib.GAT(x.shape[-1]),full_mask.astype(bool)),
        ("GCN",graphlib.GCN(x.shape[-1]),full_adj),
    ]
    graph_predictions={}
    for name,model,graph_input in graph_specs:
        print(f"Training {name}...",flush=True);torch.manual_seed(SEED)
        if name == "GraphSAGE": model = graphlib.GraphSAGE(x.shape[-1])
        elif name == "GAT": model = graphlib.GAT(x.shape[-1])
        elif name == "GCN": model = graphlib.GCN(x.shape[-1])
        row,pred=train_graph(name,model,graph_input,x,y,train_t,val_t,test_t)
        family_rows.append(row);graph_predictions[name]=pred

    relation_rows=[]
    for label,adj in [("GraphSAGE: no relations",no_adj),("GraphSAGE: directed relations",partial_adj),
                      ("GraphSAGE: full bidirectional",full_adj)]:
        print(f"Training {label}...",flush=True);torch.manual_seed(SEED)
        row,_=train_graph(label,graphlib.GraphSAGE(x.shape[-1]),adj,x,y,train_t,val_t,test_t)
        row["Comparison"]="GraphSAGE relation ablation";relation_rows.append(row)

    family=pd.DataFrame(family_rows).sort_values("Macro-F1",ascending=False)
    relation=pd.DataFrame(relation_rows).sort_values("Macro-F1",ascending=False)
    family.to_csv(OUT/"current_risk_model_family_metrics.csv",index=False,encoding="utf-8-sig")
    relation.to_csv(OUT/"graphsage_relation_ablation_metrics.csv",index=False,encoding="utf-8-sig")
    print("MODEL FAMILIES\n",family.to_string(index=False));print("RELATION ABLATION\n",relation.to_string(index=False))
    for p in (*draw_heatmap(family,"current_risk_model_family_comparison_4x3",True,"Current-month GBDT-calibrated CWQI risk target."),
              *draw_heatmap(relation,"graphsage_relation_ablation_4x3",False,"No relations = self nodes only; directed = physical flow; full = bidirectional messages.")):
        print(p)


if __name__=="__main__": main()
