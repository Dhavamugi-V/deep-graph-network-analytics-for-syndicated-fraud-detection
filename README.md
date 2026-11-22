# Graph Analytics for Syndicated Fraud Detection

## Project scope
This is a one-day, beginner/intermediate-friendly prototype inspired by Project 3 in the supplied project brief. It uses **synthetic transaction data** and graph analytics to flag structurally unusual accounts. It is not a production fraud system and does not establish that any real person/account committed fraud.

## Included
- `transactions.csv`: generated synthetic transaction data
- `generate_data.py`: regenerate the data
- `analyze_fraud.py`: build a transaction graph, calculate degree, PageRank, betweenness, communities and a heuristic risk score
- `app.py`: Streamlit dashboard
- `account_risk_scores.csv`: generated account-level results

## Setup (Windows PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python analyze_fraud.py
streamlit run app.py
```

If PowerShell blocks activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

## GitHub
Create a repository named `graph-fraud-detection-analytics`, then:
```bash
git init
git add .
git commit -m "Build initial graph fraud analytics prototype"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

## Important methodology note
The fraud labels are synthetic demonstration labels. The risk score is a transparent heuristic combining normalized degree, PageRank and betweenness centrality. It is **not** a calibrated probability. A real deployment would require validated labels, leakage controls, temporal validation, model evaluation, privacy/security review and investigator review.
