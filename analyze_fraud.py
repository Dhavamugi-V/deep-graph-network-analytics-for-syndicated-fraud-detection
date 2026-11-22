import pandas as pd
import networkx as nx
from networkx.algorithms import community

df = pd.read_csv("transactions.csv")
required = {"source_account", "target_account", "amount"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

G = nx.DiGraph()
for row in df.itertuples(index=False):
    G.add_edge(row.source_account, row.target_account,
               weight=float(row.amount),
               amount=float(row.amount))

# Structural features
undirected = G.to_undirected()
degree = dict(G.degree())
pagerank = nx.pagerank(G, weight="weight")
betweenness = nx.betweenness_centrality(undirected, normalized=True)
try:
    communities = list(community.greedy_modularity_communities(undirected, weight="weight"))
except Exception:
    communities = [{n} for n in G.nodes()]

community_id = {}
for idx, group in enumerate(communities):
    for node in group:
        community_id[node] = idx

risk = pd.DataFrame({
    "account": list(G.nodes()),
    "degree": [degree[n] for n in G.nodes()],
    "pagerank": [pagerank[n] for n in G.nodes()],
    "betweenness": [betweenness[n] for n in G.nodes()],
    "community": [community_id.get(n, -1) for n in G.nodes()],
})
# Transparent heuristic score for this prototype, not a validated fraud probability
for col in ["degree", "pagerank", "betweenness"]:
    max_val = risk[col].max()
    risk[col + "_norm"] = risk[col] / max_val if max_val else 0
risk["risk_score"] = (
    0.4 * risk["degree_norm"] +
    0.3 * risk["pagerank_norm"] +
    0.3 * risk["betweenness_norm"]
).mul(100).round(1)
risk["risk_level"] = pd.cut(
    risk["risk_score"], bins=[-1, 25, 55, 101],
    labels=["Low", "Medium", "High"]
)
risk = risk.sort_values("risk_score", ascending=False)
risk.to_csv("account_risk_scores.csv", index=False)

# Export a graph edge list for visualization
df.to_csv("analyzed_transactions.csv", index=False)
print("Analysis complete.")
print(risk.head(10).to_string(index=False))
print(f"\nNodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()} | "
      f"Communities: {len(communities)}")
