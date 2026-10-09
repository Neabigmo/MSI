"""Public COAD-MSI GEO ingestion helpers.

This module contains only the input preparation used by the released MSI
benchmark. GEO probes are mapped with a fixed GPL annotation and multiple
probes for one gene are averaged within each sample. No cohort-level probe
mean is used to select a representative probe.

The release deliberately keeps raw matrices and clinical files outside the
package. Run this module in a working copy with public GEO inputs when a
fresh processed matrix is required, then pass the resulting matrices and
label files to ``src/run_formal.py``.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import re
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
COAD_EXTERNAL_MATRIX_URLS = {
    "GSE13294": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE13nnn/GSE13294/matrix/GSE13294_series_matrix.txt.gz",
    "GSE13067": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE13nnn/GSE13067/matrix/GSE13067_series_matrix.txt.gz",
    "GSE18088": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE18nnn/GSE18088/matrix/GSE18088_series_matrix.txt.gz",
    "GSE26682": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE26nnn/GSE26682/matrix/GSE26682-GPL570_series_matrix.txt.gz",
    "GSE39084": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE39nnn/GSE39084/matrix/GSE39084_series_matrix.txt.gz",
}
GPL570_ANNOTATION_URL = "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL570/annot/GPL570.annot.gz"


def fetch_text(url: str) -> str:
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    payload = response.content
    if url.endswith(".gz"):
        payload = gzip.decompress(payload)
    return payload.decode("utf-8", errors="replace")


def parse_series_matrix_expression_table(text: str) -> pd.DataFrame:
    lines = text.splitlines()
    start = next((i + 1 for i, line in enumerate(lines)
                  if line.startswith("!series_matrix_table_begin")), None)
    end = next((i for i, line in enumerate(lines)
                if line.startswith("!series_matrix_table_end")), None)
    if start is None or end is None or end <= start:
        raise ValueError("Series matrix expression table not found")
    frame = pd.read_csv(StringIO("\n".join(lines[start:end])), sep="\t")
    if "ID_REF" not in frame.columns:
        raise ValueError("Series matrix table missing ID_REF")
    return frame


def parse_series_matrix_sample_metadata(text: str) -> pd.DataFrame:
    values: dict[str, list[str]] = {}
    for line in text.splitlines():
        if not line.startswith("!Sample_"):
            continue
        fields = next(csv.reader([line], delimiter="\t"))
        if len(fields) >= 2:
            values[fields[0]] = [item.strip().strip('"') for item in fields[1:]]
    sample_ids = values.get("!Sample_geo_accession", [])
    if not sample_ids:
        raise ValueError("GEO sample accession metadata not found")
    n = len(sample_ids)
    def padded(key: str) -> list[str]:
        return (values.get(key, []) + [""] * n)[:n]
    characteristics = values.get("!Sample_characteristics_ch1", [])
    if characteristics and len(characteristics) != n:
        characteristics = characteristics[:n] + [""] * max(0, n - len(characteristics))
    return pd.DataFrame({
        "sample_id": sample_ids,
        "sample_title": padded("!Sample_title"),
        "source_name": padded("!Sample_source_name_ch1"),
        "characteristics": characteristics or [""] * n,
    })


def infer_msi_label(accession: str, row: pd.Series) -> str:
    raw = " | ".join(str(row.get(key, "")) for key in
                     ("sample_title", "source_name", "characteristics"))
    if accession in {"GSE13294", "GSE13067"}:
        if re.search(r"\bmsi(?:[- ]?high)?\b(?![- ]?(?:l|low)\b)", raw, re.I):
            return "MSI"
        if re.search(r"\bmss\b", raw, re.I):
            return "MSS"
    elif accession == "GSE18088":
        if re.search(r"microsatellite status:\s*msi[- ]?high", raw, re.I):
            return "MSI"
        if re.search(r"microsatellite status:\s*mss\b", raw, re.I):
            return "MSS"
    elif accession == "GSE26682":
        if re.search(r"stable\s*\[mss\]", raw, re.I):
            return "MSS"
        if re.search(r"high\s*\[msi-h\]", raw, re.I):
            return "MSI"
    elif accession == "GSE39084":
        if re.search(r"msi\.status.*?:\s*high\b", raw, re.I):
            return "MSI"
        if re.search(r"msi\.status.*?:\s*no\b", raw, re.I):
            return "MSS"
    return "EXCLUDE_UNKNOWN"


def fetch_gpl570_probe_to_gene_map() -> dict[str, str]:
    text = fetch_text(GPL570_ANNOTATION_URL)
    lines = text.splitlines()
    header = next((i for i, line in enumerate(lines) if line.startswith("ID\t")), None)
    if header is None:
        raise ValueError("GPL570 annotation header not found")
    annot = pd.read_csv(StringIO("\n".join(lines[header:])), sep="\t")
    mapping: dict[str, str] = {}
    for _, row in annot.iterrows():
        probe = str(row.get("ID", "")).strip()
        symbol = str(row.get("Gene symbol", "")).strip()
        if not probe or not symbol or symbol in {"---", "nan"}:
            continue
        symbol = symbol.split("///")[0].strip()
        if symbol and symbol != "---":
            mapping[probe] = symbol
    return mapping


def collapse_probe_matrix_to_gene_matrix(
    expression_frame: pd.DataFrame,
    sample_columns: list[str],
    probe_to_gene: dict[str, str],
) -> pd.DataFrame:
    """Average all mapped probes for each gene separately within each sample."""
    working = expression_frame[["ID_REF", *sample_columns]].copy()
    working["gene_symbol"] = working["ID_REF"].map(probe_to_gene).fillna("")
    working = working[working["gene_symbol"] != ""].copy()
    working[sample_columns] = working[sample_columns].apply(pd.to_numeric, errors="coerce")
    gene_matrix = working.groupby("gene_symbol", sort=True)[sample_columns].mean().T
    gene_matrix.index.name = "sample_id"
    return gene_matrix.sort_index(axis=1)


def materialize_geo_labels(accession: str, output_dir: Path) -> Path:
    if accession not in COAD_EXTERNAL_MATRIX_URLS:
        raise ValueError(f"Unsupported COAD accession: {accession}")
    metadata = parse_series_matrix_sample_metadata(fetch_text(COAD_EXTERNAL_MATRIX_URLS[accession]))
    metadata["label"] = metadata.apply(lambda row: infer_msi_label(accession, row), axis=1)
    metadata["included"] = metadata.label.isin(["MSI", "MSS"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{accession}_samples.csv"
    metadata.to_csv(output, index=False)
    return output


def materialize_geo_expression(
    accession: str,
    labels_path: Path,
    output_dir: Path,
    probe_to_gene: dict[str, str],
    force: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{accession}_expression.tsv"
    if output.exists() and output.stat().st_size > 0 and not force:
        return output
    labels = pd.read_csv(labels_path)
    retained = labels.loc[labels.label.isin(["MSI", "MSS"]), ["sample_id", "label"]].copy()
    if retained.sample_id.astype(str).duplicated().any():
        raise ValueError(f"Duplicate retained sample IDs in frozen {accession} labels")
    sample_ids = retained.sample_id.astype(str).tolist()
    frame = parse_series_matrix_expression_table(fetch_text(COAD_EXTERNAL_MATRIX_URLS[accession]))
    missing_samples = sorted(set(sample_ids) - set(frame.columns.astype(str)))
    if missing_samples:
        raise ValueError(f"{accession} series matrix is missing {len(missing_samples)} frozen samples: " + ", ".join(missing_samples[:8]))
    collapsed = collapse_probe_matrix_to_gene_matrix(frame, sample_ids, probe_to_gene)
    if collapsed.index.astype(str).tolist() != sample_ids:
        raise ValueError(f"{accession} expression rows do not preserve frozen-manifest order")
    collapsed.to_csv(output, sep="\t")
    return output


def build_geo_inputs(
    accessions: list[str], force: bool = False, output_root: Path | None = None,
    frozen_gene_list: Path | None = None,
) -> list[str]:
    processed_dir = (output_root or ROOT / "data" / "processed" / "coad_msi_probe_mean").resolve()
    metadata_dir = processed_dir
    interim_dir = processed_dir.parent / f"{processed_dir.name}_interim"
    mapping = fetch_gpl570_probe_to_gene_map()
    outputs: list[str] = []
    matrices = {}
    for accession in accessions:
        labels = materialize_geo_labels(accession, metadata_dir)
        published_labels = ROOT / "data" / "metadata" / "coad_msi_labels" / f"{accession}_labels.csv"
        if published_labels.exists():
            # The publication manifest is authoritative; inferred metadata are
            # retained only as an audit aid and never redefine the analysis set.
            label_frame = pd.read_csv(published_labels)
            label_frame.to_csv(labels, index=False)
        expression = materialize_geo_expression(accession, labels, interim_dir, mapping, force)
        frame = pd.read_csv(expression, sep="\t", index_col=0)
        label_frame = pd.read_csv(labels).set_index("sample_id")
        expected_ids = label_frame.index[label_frame.label.isin(["MSI", "MSS"])].astype(str).tolist()
        actual_ids = frame.index.astype(str).tolist()
        if set(actual_ids) != set(expected_ids):
            missing_ids = sorted(set(expected_ids) - set(actual_ids))
            extra_ids = sorted(set(actual_ids) - set(expected_ids))
            raise ValueError(f"{accession} sample set differs from frozen labels: {len(missing_ids)} missing, {len(extra_ids)} unexpected")
        matrices[accession] = frame.loc[expected_ids]
        outputs.extend([str(labels), str(expression)])
    frozen_path = frozen_gene_list or ROOT / "data" / "metadata" / "shared_gene_set.csv"
    if not frozen_path.exists():
        raise FileNotFoundError(
            f"Frozen six-cohort gene list not found: {frozen_path}. "
            "The publication workflow must not infer a new GEO-only intersection."
        )
    frozen_frame = pd.read_csv(frozen_path)
    shared = frozen_frame.iloc[:, 0].astype(str).tolist()
    missing = {name: sorted(set(shared) - set(frame.columns)) for name, frame in matrices.items()}
    missing = {name: genes for name, genes in missing.items() if genes}
    if missing:
        summary = "; ".join(f"{name}: {len(genes)} missing" for name, genes in missing.items())
        raise ValueError(f"Inputs do not satisfy the frozen 11,316-gene interface ({summary})")
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "shared_gene_set.csv").write_text("gene\n" + "\n".join(shared) + "\n", encoding="utf-8")
    for accession, frame in matrices.items():
        path = processed_dir / f"{accession}_shared_expression.tsv"
        aligned = frame.loc[:, shared]
        if aligned.columns.astype(str).tolist() != shared or aligned.shape[1] != 11316:
            raise ValueError(f"{accession} failed the ordered 11,316-gene interface assertion")
        aligned.to_csv(path, sep="\t")
        outputs.append(str(path))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fixed-mapping, sample-wise GEO COAD-MSI inputs.")
    parser.add_argument("--accessions", nargs="+", default=list(COAD_EXTERNAL_MATRIX_URLS))
    parser.add_argument("--force", action="store_true", help="Re-fetch and overwrite existing GEO expression outputs.")
    parser.add_argument("--output-root", type=Path,
                        help="Directory for rebuilt labels and shared expression matrices.")
    parser.add_argument("--frozen-gene-list", type=Path,
                        help="Published six-cohort gene list; defaults to data/metadata/shared_gene_set.csv.")
    args = parser.parse_args()
    for path in build_geo_inputs(args.accessions, args.force, args.output_root, args.frozen_gene_list):
        print(path)


if __name__ == "__main__":
    main()
