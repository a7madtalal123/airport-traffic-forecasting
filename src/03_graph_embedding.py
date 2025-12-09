# src/03_graph_embedding.py
import os
import pandas as pd
import networkx as nx
from node2vec import Node2Vec

DATA_PATH   = "./data/flights_RUH.csv"   
OUTPUT_PATH = "./outputs/node2vec_embeddings.csv"

df = pd.read_csv(DATA_PATH)
RUH_IATA, RUH_ICAO = "RUH", "OERK"

is_origin_RUH = (df.get("origin_airport_iata","").astype(str).str.upper()==RUH_IATA) | \
                (df.get("origin_airport_icao","").astype(str).str.upper()==RUH_ICAO)
is_dest_RUH   = (df.get("destination_airport_iata","").astype(str).str.upper()==RUH_IATA) | \
                (df.get("destination_airport_icao","").astype(str).str.upper()==RUH_ICAO)

outbound = df[is_origin_RUH].copy()
inbound  = df[is_dest_RUH].copy()

G = nx.DiGraph()

def add_edges(sub, s, t):
    grp = sub.groupby([s,t]).size().reset_index(name="w")
    for _, r in grp.iterrows():
        S, T, W = str(r[s]), str(r[t]), int(r["w"])
        if S not in G: G.add_node(S)
        if T not in G: G.add_node(T)
        if G.has_edge(S,T): G[S][T]["w"] += W
        else: G.add_edge(S,T, w=W)

add_edges(outbound, "origin_airport_iata", "destination_airport_iata")
add_edges(inbound,  "origin_airport_iata", "destination_airport_iata")

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Node2Vec
node2vec = Node2Vec(G, dimensions=32, walk_length=20, num_walks=200, workers=2, weight_key="w")
model = node2vec.fit(window=10, min_count=1, batch_words=64)

emb = {n: model.wv[n] for n in G.nodes() if n in model.wv}
emb_df = pd.DataFrame({"airport": list(emb.keys()), "vec": list(emb.values())})

vec_cols = [f"n2v_{i}" for i in range(32)]
emb_expanded = pd.DataFrame(emb_df["vec"].tolist(), columns=vec_cols)
final = pd.concat([emb_df[["airport"]].reset_index(drop=True), emb_expanded], axis=1)

os.makedirs("./outputs", exist_ok=True)
final.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Saved: {OUTPUT_PATH}")
