# Bibliography verification

The 35-entry `references.bib` was checked on 2026-10-09 with [Altman-conquer/bibtex-verifier](https://github.com/Altman-conquer/bibtex-verifier), version 0.2.0, using Crossref/DataCite DOI lookup and OpenAlex fallback search.

The final run returned 32 `OK`, 0 `WARNING`, 0 `NOT_FOUND`, 0 `UNVERIFIED`, and 3 `ERROR` statuses. The three flagged entries are manually resolved metadata issues or parser limitations, not unresolved references:

- `vancalster2019_calibration`: the article is cited with Ben Van Calster as first author followed by the named authors and the STRATOS Topic Group collaboration. The BibTeX entry preserves the collaboration as a group author.
- `tcga2012_crc`: Crossref lists the group author “The Cancer Genome Atlas Network”. The BibTeX entry preserves this group author.
- `alba2017_discrimination_calibration`: the verifier compares against the shortened Crossref title, whereas the BibTeX entry uses the full official PubMed/JAMA title, including “Users' Guides to the Medical Literature”.

For `alba2017_discrimination_calibration`, the title was checked against the official PubMed/JAMA title, including the subtitle “Users' Guides to the Medical Literature”. The machine-readable JSON and Markdown output from the verifier are included beside this note.

The tool checks bibliographic metadata only; it does not determine whether a cited paper supports the specific sentence in which it is cited.
