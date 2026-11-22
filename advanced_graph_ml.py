\
"""Train a Node2Vec-style graph embedding + XGBoost demo model.
Uses synthetic labels from the demo dataset; scores are not real-world fraud probabilities.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib

BASE = Path(__file__).resolve().parent
TX_PATH = BASE / "transactions.csv"


def build_graph(tx):
    g = nx.DiGraph()
    for r in tx.itertuples(index=False):
        g.add_edge(str(r.source_account), str(r.target_account), weight=float(r.amount))
    return g


def random_walk(g, start, walk_length=12, p=1.0, q=1.0, rng=None):
    rng = rng or np.random.default_rng(42)
    walk = [start]
    while len(walk) < walk_length:
        cur = walk[-1]
        neighbors = list(g.neighbors(cur))
        if not neighbors:
            break
        if len(walk) == 1:
            nxt = rng.choice(neighbors)
        else:
            prev = walk[-2]
            weights = []
            for n in neighbors:
                if n == prev:
                    weights.append(1.0 / p)
                elif g.has_edge(prev, n) or g.has_edge(n, prev):
                    weights.append(1.0)
                else:
                    weights.append(1.0 / q)
            weights = np.asarray(weights, dtype=float)
            weights /= weights.sum()
            nxt = rng.choice(neighbors, p=weights)
        walk.append(nxt)
    return walk


def node2vec_style_embeddings(g, dim=8, walks_per_node=20, walk_length=12, window=3):
    """Create Node2Vec-style embeddings from biased walks and a PPMI/SVD factorization.
    This is a lightweight educational implementation, not the gensim Node2Vec package.
    """
    nodes = sorted(g.nodes())
    idx = {n:i for i,n in enumerate(nodes)}
    co = np.zeros((len(nodes), len(nodes)), dtype=np.float64)
    rng = np.random.default_rng(42)
    for node in nodes:
        for _ in range(walks_per_node):
            walk = random_walk(g, node, walk_length=walk_length, rng=rng)
            ids = [idx[x] for x in walk]
            for i, center in enumerate(ids):
                lo, hi = max(0, i-window), min(len(ids), i+window+1)
                for j in range(lo, hi):
                    if i != j:
                        co[center, ids[j]] += 1.0 / abs(i-j)
    total = co.sum()
    if total == 0:
        return nodes, np.zeros((len(nodes), dim))
    row = co.sum(axis=1, keepdims=True)
    col = co.sum(axis=0, keepdims=True)
    expected = row @ col / total
    ppmi = np.maximum(np.log((co * total + 1e-9) / (expected + 1e-9)), 0.0)
    k = min(dim, max(1, len(nodes)-1))
    emb = TruncatedSVD(n_components=k, random_state=42).fit_transform(ppmi)
    if k < dim:
        emb = np.pad(emb, ((0,0),(0,dim-k)))
    return nodes, emb


def main():
    tx = pd.read_csv(TX_PATH)
    required = {'source_account','target_account','amount','synthetic_fraud_flag'}
    if not required.issubset(tx.columns):
        raise ValueError(f"transactions.csv must contain {sorted(required)}")
    g = build_graph(tx)
    nodes, emb = node2vec_style_embeddings(g)
    und = g.to_undirected()
    pr = nx.pagerank(g, weight='weight')
    personalized = nx.pagerank(g, personalization={n: 1/len(nodes) for n in nodes}, weight='weight')
    between = nx.betweenness_centrality(und, normalized=True)
    degree = dict(g.degree())
    in_degree, out_degree = dict(g.in_degree()), dict(g.out_degree())
    # Synthetic entity label: account appears in a planted synthetic flagged transfer.
    positive = set(tx.loc[tx.synthetic_fraud_flag.astype(int)==1, 'source_account'].astype(str)) | set(tx.loc[tx.synthetic_fraud_flag.astype(int)==1, 'target_account'].astype(str))
    rows=[]
    for i,n in enumerate(nodes):
        row={'account':n,'degree':degree.get(n,0),'in_degree':in_degree.get(n,0),'out_degree':out_degree.get(n,0),
             'pagerank':pr.get(n,0),'personalized_pagerank':personalized.get(n,0),'betweenness':between.get(n,0),
             'synthetic_label':int(n in positive)}
        for j,v in enumerate(emb[i]): row[f'embedding_{j+1}']=float(v)
        rows.append(row)
    features=pd.DataFrame(rows)
    feature_cols=[c for c in features.columns if c not in ('account','synthetic_label')]
    X=features[feature_cols].replace([np.inf,-np.inf],0).fillna(0)
    y=features.synthetic_label
    if y.nunique()<2 or y.value_counts().min()<2:
        raise ValueError('Need at least two accounts in each synthetic label class to train/test the demo model.')
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=42,stratify=y)
    model=Pipeline([('scale',StandardScaler()),('xgb',XGBClassifier(n_estimators=120,max_depth=3,learning_rate=0.05,
        subsample=0.9,colsample_bytree=0.9,reg_lambda=2,objective='binary:logistic',eval_metric='logloss',
        random_state=42,n_jobs=2,scale_pos_weight=max(1,(y_train==0).sum()/max(1,(y_train==1).sum()))))])
    model.fit(X_train,y_train)
    prob=model.predict_proba(X)[:,1]
    pred=model.predict(X_test)
    test_prob=model.predict_proba(X_test)[:,1]
    metrics={'model':'XGBoostClassifier','embedding':'Node2Vec-style biased random walks + PPMI + TruncatedSVD',
             'accounts':int(len(nodes)),'positive_synthetic_accounts':int(y.sum()),'negative_synthetic_accounts':int((1-y).sum()),
             'test_rows':int(len(y_test)),'roc_auc':float(roc_auc_score(y_test,test_prob)) if y_test.nunique()==2 else None,
             'average_precision':float(average_precision_score(y_test,test_prob)),
             'classification_report':classification_report(y_test,pred,output_dict=True,zero_division=0),
             'warning':'Synthetic demonstration only. Scores are model outputs, not calibrated real-world fraud probabilities.'}
    features['model_score']=np.round(prob*100,1)
    features['model_risk_level']=pd.cut(features.model_score,bins=[-1,25,55,100.0001],labels=['Low','Medium','High']).astype(str)
    # Transparent combined review score: model score + centrality signals; still a ranking aid.
    for c in ['pagerank','betweenness','degree']:
        mx=features[c].max(); features[c+'_norm']=features[c]/mx if mx else 0
    features['combined_review_score']=np.round(0.70*features.model_score+30*(0.4*features.degree_norm+0.3*features.pagerank_norm+0.3*features.betweenness_norm),1)
    features['combined_risk_level']=pd.cut(features.combined_review_score,bins=[-1,25,55,101],labels=['Low','Medium','High']).astype(str)
    features.sort_values('combined_review_score',ascending=False).to_csv(BASE/'advanced_account_risk_scores.csv',index=False)
    (BASE/'model_metrics.json').write_text(json.dumps(metrics,indent=2))
    joblib.dump({'model':model,'feature_columns':feature_cols},BASE/'fraud_xgb_model.joblib')
    print('Advanced graph ML complete.')
    print(f"Accounts: {len(nodes)} | Synthetic positive labels: {int(y.sum())} | Test ROC-AUC: {metrics['roc_auc']}")
    print(features[['account','model_score','model_risk_level','combined_review_score','combined_risk_level']].sort_values('combined_review_score',ascending=False).head(10).to_string(index=False))

if __name__=='__main__': main()
