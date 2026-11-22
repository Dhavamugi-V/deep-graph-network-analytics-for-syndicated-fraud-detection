import json
from pathlib import Path
import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parent
st.set_page_config(page_title="Graph Fraud Analytics", layout="wide")
st.title("Syndicated Fraud Detection – Graph Analytics")
st.caption("Synthetic demonstration only. Model scores are review signals, not proof of fraud.")
tx=pd.read_csv(BASE/"transactions.csv")
risk_path=BASE/"advanced_account_risk_scores.csv"
if risk_path.exists():
    risk=pd.read_csv(risk_path)
    score_col='combined_review_score' if 'combined_review_score' in risk else 'risk_score'
    level_col='combined_risk_level' if 'combined_risk_level' in risk else 'risk_level'
else:
    risk=pd.read_csv(BASE/"account_risk_scores.csv")
    score_col='risk_score'; level_col='risk_level'

c1,c2,c3,c4=st.columns(4)
c1.metric("Transactions",len(tx)); c2.metric("Accounts",len(risk))
c3.metric("Synthetic flagged transactions",int(tx.get('synthetic_fraud_flag',pd.Series(dtype=int)).sum()))
c4.metric("High review-risk accounts",int((risk[level_col].astype(str)=='High').sum()))

st.subheader("Highest-risk accounts")
show=[c for c in ['account','model_score','model_risk_level','combined_review_score','combined_risk_level','degree','pagerank','betweenness','community','risk_score','risk_level'] if c in risk.columns]
st.dataframe(risk[show].head(20),use_container_width=True)

st.subheader("Transaction network")
G=nx.from_pandas_edgelist(tx,'source_account','target_account',edge_attr='amount',create_using=nx.DiGraph())
fig,ax=plt.subplots(figsize=(11,7)); pos=nx.spring_layout(G,seed=42)
nx.draw_networkx(G,pos,ax=ax,node_size=300,font_size=7,arrows=True,with_labels=True)
ax.set_axis_off(); st.pyplot(fig)

st.subheader("Account risk lookup")
account=st.selectbox("Choose an account",sorted(risk.account.astype(str).unique()))
selected=risk[risk.account.astype(str)==account].iloc[0]
a,b=st.columns(2); a.metric("Combined review score",f"{float(selected[score_col]):.1f}/100"); b.metric("Risk band",str(selected[level_col]))

st.subheader("Filter transactions")
min_amount=st.slider("Minimum transaction amount",0.0,float(tx.amount.max()),0.0)
st.dataframe(tx[tx.amount>=min_amount].sort_values('amount',ascending=False),use_container_width=True)

metrics_path=BASE/'model_metrics.json'
if metrics_path.exists():
    with st.expander("Model evaluation (synthetic holdout)"):
        metrics=json.loads(metrics_path.read_text())
        st.write({k:v for k,v in metrics.items() if k!='classification_report'})
        st.json(metrics.get('classification_report',{}))
