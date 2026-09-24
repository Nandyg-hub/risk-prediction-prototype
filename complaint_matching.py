"""
complaint_matching.py
----------------------------------
WEEK 2: Complaint -> Risk (FMEA) traceability matching, plus the novel
"unknown risk" detection layer.

Part A (standard traceability -- what existing tools like Greenlight Guru
already do): match each complaint to the closest known FMEA risk item for
that product, using text similarity (TF-IDF + cosine similarity -- a
lightweight, fully offline stand-in for a full embedding model).

Part B (the novel contribution): if a complaint's best match is still
weak (below a confidence threshold), don't force it onto that risk item.
Instead, mark it "unmatched" and pool it. If enough unmatched complaints
for a product turn out to be similar TO EACH OTHER (via clustering), that
is flagged as a possible emerging / undocumented failure mode -- something
no existing "known-risk-matching" tool would surface.

Output:
- outputs/complaint_matches.csv      (every complaint + its best match + confidence)
- outputs/emerging_risk_clusters.csv (clusters of unmatched, similar complaints)
- outputs/match_confidence_hist.png  (chart: distribution of match confidence)
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MATCH_THRESHOLD = 0.35  # below this similarity, we call it "unmatched" rather than force a weak match

# ---------------------------------------------------------------
# 1. Load complaints and FMEA risk items
# ---------------------------------------------------------------
complaints = pd.read_csv("data/complaints.csv")
fmea = pd.read_csv("data/fmea_risk_items.csv")

print(f"Loaded {len(complaints)} complaints and {len(fmea)} known FMEA risk items.\n")

# ---------------------------------------------------------------
# 2. For each product, match its complaints against its own FMEA items only
#    (a complaint should only be compared to risks relevant to that product)
# ---------------------------------------------------------------
match_rows = []

for product_id, comp_group in complaints.groupby("product_id"):
    fmea_group = fmea[fmea["product_id"] == product_id].reset_index(drop=True)

    if fmea_group.empty:
        # no known risks at all for this product -> everything is unmatched
        for _, c in comp_group.iterrows():
            match_rows.append({
                "complaint_id": c["complaint_id"], "product_id": product_id,
                "complaint_text": c["description"], "best_match_risk_id": None,
                "best_match_text": None, "confidence": 0.0, "status": "unmatched"
            })
        continue

    # TF-IDF over this product's complaint texts + FMEA texts together,
    # so they share the same vocabulary space
    corpus = list(comp_group["description"]) + list(fmea_group["description"])
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(corpus)

    n_complaints = len(comp_group)
    complaint_vecs = tfidf[:n_complaints]
    fmea_vecs = tfidf[n_complaints:]

    sims = cosine_similarity(complaint_vecs, fmea_vecs)  # shape: (n_complaints, n_fmea_items)

    for i, (_, c) in enumerate(comp_group.iterrows()):
        best_idx = np.argmax(sims[i])
        best_score = sims[i][best_idx]
        status = "matched" if best_score >= MATCH_THRESHOLD else "unmatched"
        match_rows.append({
            "complaint_id": c["complaint_id"],
            "product_id": product_id,
            "complaint_text": c["description"],
            "best_match_risk_id": fmea_group.iloc[best_idx]["risk_id"] if status == "matched" else None,
            "best_match_text": fmea_group.iloc[best_idx]["description"] if status == "matched" else None,
            "confidence": round(float(best_score), 3),
            "status": status,
        })

matches = pd.DataFrame(match_rows)
matches.to_csv("outputs/complaint_matches.csv", index=False)

n_matched = (matches["status"] == "matched").sum()
n_unmatched = (matches["status"] == "unmatched").sum()
print(f"Matched to a known risk:   {n_matched} complaints ({n_matched/len(matches):.0%})")
print(f"Unmatched (no good match): {n_unmatched} complaints ({n_unmatched/len(matches):.0%})")
print(f"(Match threshold = {MATCH_THRESHOLD} cosine similarity)\n")

# ---------------------------------------------------------------
# 3. NOVEL PART: cluster the unmatched complaints per product to find
#    emerging / undocumented failure modes
# ---------------------------------------------------------------
cluster_summary_rows = []

for product_id, group in matches[matches["status"] == "unmatched"].groupby("product_id"):
    if len(group) < 3:
        continue  # need at least a few unmatched complaints to look for a pattern

    vectorizer = TfidfVectorizer(stop_words="english")
    vecs = vectorizer.fit_transform(group["complaint_text"])

    # DBSCAN groups points that are close together; noise (isolated, one-off
    # complaints) gets label -1 and is ignored -- we only care about complaints
    # that resemble OTHER unmatched complaints, not random singletons
    clustering = DBSCAN(eps=0.6, min_samples=3, metric="cosine").fit(vecs.toarray())
    group = group.copy()
    group["cluster"] = clustering.labels_

    for cluster_id in sorted(set(clustering.labels_)):
        if cluster_id == -1:
            continue  # noise, not a real pattern
        cluster_complaints = group[group["cluster"] == cluster_id]
        cluster_summary_rows.append({
            "product_id": product_id,
            "cluster_id": f"{product_id}-EMERGING-{cluster_id}",
            "num_complaints": len(cluster_complaints),
            "sample_complaint_ids": ", ".join(cluster_complaints["complaint_id"].head(5)),
            "sample_text": cluster_complaints["complaint_text"].iloc[0],
        })

emerging = pd.DataFrame(cluster_summary_rows)
emerging.to_csv("outputs/emerging_risk_clusters.csv", index=False)

print("=== Emerging risk clusters (unmatched complaints that resemble each other) ===")
if emerging.empty:
    print("None found with current threshold/data.")
else:
    for _, row in emerging.iterrows():
        print(f"- {row['product_id']}: {row['num_complaints']} similar unmatched complaints "
              f"(possible undocumented failure mode)")
        print(f"  Example: \"{row['sample_text']}\"")

# ---------------------------------------------------------------
# 4. Chart: distribution of match confidence, with threshold line
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(matches["confidence"], bins=25, color="steelblue", edgecolor="white")
ax.axvline(MATCH_THRESHOLD, color="crimson", linestyle="--", linewidth=2,
           label=f"Match threshold ({MATCH_THRESHOLD})")
ax.set_xlim(0, 1)
ax.set_title("Complaint-to-Risk Match Confidence Distribution")
ax.set_xlabel("Cosine similarity to best-matching known risk item")
ax.set_ylabel("Number of complaints")
ax.legend()
fig.tight_layout()
fig.savefig("outputs/match_confidence_hist.png", dpi=150)
print("\nSaved outputs/match_confidence_hist.png")
print("\nDone. See outputs/complaint_matches.csv and outputs/emerging_risk_clusters.csv")
