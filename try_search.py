import sys

from app import db
from app.retrieval.search import search

query = " ".join(sys.argv[1:]) or "what does fillfactor do"
db.open_pool()

for mode in ("vector", "keyword", "hybrid"):
    res = search(query, mode=mode, rerank=False, top_k=5)
    print(f"\n=== {mode} ===")
    for h in res.hits:
        print(f"  {h.score:.3f}  {h.section[:90]}")

res = search(query, mode="hybrid", rerank=True, top_k=5)
print("\n=== hybrid + rerank ===")
for h in res.hits:
    print(f"  {h.score:.3f}  {h.section[:90]}")
print("\ntimings (ms):", res.timings_ms)

search(query, mode="hybrid", rerank=True)  # warm-up: loads the reranker model
res = search(query, mode="hybrid", rerank=True, top_k=5)

db.close_pool()