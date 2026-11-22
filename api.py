"""FastAPI service for graph fraud review scores (synthetic demo)."""
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException

BASE=Path(__file__).resolve().parent
app=FastAPI(title="Graph Fraud Risk Scoring API", version="1.0.0",
            description="Synthetic demo API. Scores are review signals, not proof of fraud.")

@app.get("/health")
def health():
    return {"status":"ok","dataset":"synthetic demonstration data"}

@app.get("/risk/{account_id}")
def get_risk(account_id: str):
    path=BASE/"advanced_account_risk_scores.csv"
    if not path.exists():
        raise HTTPException(status_code=503, detail="Run python advanced_graph_ml.py first")
    df=pd.read_csv(path)
    row=df[df.account.astype(str).str.upper()==account_id.upper()]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Account {account_id} not found")
    r=row.iloc[0]
    return {"account":str(r.account),"model_score":float(r.model_score),
            "model_risk_level":str(r.model_risk_level),
            "combined_review_score":float(r.combined_review_score),
            "combined_risk_level":str(r.combined_risk_level),
            "pagerank":float(r.pagerank),"betweenness":float(r.betweenness),
            "community":int(r.community) if 'community' in r.index and pd.notna(r.community) else None,
            "disclaimer":"Synthetic demo score; not a calibrated fraud probability or proof of fraud."}

@app.get("/accounts")
def list_accounts(limit: int=20):
    path=BASE/"advanced_account_risk_scores.csv"
    if not path.exists():
        raise HTTPException(status_code=503, detail="Run python advanced_graph_ml.py first")
    df=pd.read_csv(path).sort_values('combined_review_score',ascending=False).head(max(1,min(limit,100)))
    return df[['account','model_score','model_risk_level','combined_review_score','combined_risk_level']].to_dict(orient='records')
