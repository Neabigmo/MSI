"""Create supplementary assets from the authoritative formal outputs."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "generated"
TAB = ROOT / "results" / "generated"


def enrichment() -> None:
    stability = pd.read_csv(OUT / "compact_gene_stability.csv")
    query = stability.loc[stability.selection_frequency >= 0.20, "gene"].tolist()
    manifest = pd.read_csv(TAB / "cohort_manifest.csv")
    # The reference universe is the frozen 11,316-gene input space, not the
    # post-screen compact panel.
    background = pd.read_csv(ROOT / "data" / "processed" / "coad_msi_probe_mean" /
                              "TCGA_COAD_shared_expression.tsv", sep="\t", nrows=1).columns.tolist()[1:]
    payload = {"organism": "hsapiens", "query": query, "background": background,
               "sources": ["GO:BP", "REAC", "KEGG"], "user_threshold": 1.0,
               "domain_scope": "custom", "significance_threshold_method": "fdr", "no_evidences": True}
    response = requests.post("https://biit.cs.ut.ee/gprofiler/api/gost/profile/", json=payload, timeout=120)
    response.raise_for_status()
    body = response.json()
    result = pd.DataFrame(body.get("result", []))
    if not result.empty:
        result["significant"] = result["p_value"] < 0.05
    keep = [c for c in ["source", "native", "name", "p_value", "significant", "term_size",
                        "query_size", "intersection_size", "effective_domain_size", "intersections"]
            if c in result]
    result[keep].to_csv(TAB / "stable_gene_pathway_enrichment.csv", index=False)
    (TAB / "stable_gene_pathway_enrichment_provenance.json").write_text(json.dumps({
        "service": "g:Profiler", "endpoint": "https://biit.cs.ut.ee/gprofiler/api/gost/profile/",
        "metadata": body.get("meta", {}), "query_rule": "selection_frequency >= 0.20",
        "query_genes": query, "background_size": len(background), "multiple_testing": "FDR 0.05"
    }, indent=2), encoding="utf-8")


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    enrichment()
    print("Wrote formal enrichment assets to", TAB)


if __name__ == "__main__":
    main()

