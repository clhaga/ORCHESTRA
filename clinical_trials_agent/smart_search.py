import sys
import time
import xml.etree.ElementTree as ET
import requests
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm

from .llm_client import LLMClient
from .config import get_pubmed_config, get_llm_config
from .utils import chunked, clean_text, parse_year_from_pubdate, save_json
from .query_builder import build_pubmed_query
from .llm_rerank import rerank_records_with_llm_batched


EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


@dataclass
class SearchResult:
    """
    Represents a single PubMed study with metadata.
    Extended with LLM-derived fields for transparency.
    """
    pmid: str
    pmcid: str
    title: str
    abstract: str
    journal: str
    year: Optional[int]
    authors: List[str]
    doi: str
    mesh_headings: List[str]
    publication_types: List[str]
    license: dict
    # LLM-added fields
    relevance: int = 95
    relevance_reason: str = "No LLM scoring"


class SmartPubMedSearcher:
    """
    A natural language-powered PubMed search engine.
    Translates plain queries into expert-level PubMed searches using LLM,
    fetches full metadata, reranks with LLM, and filters by relevance threshold.

    Designed for integration with clinical_trials_agent/orchestra.py
    """

    def __init__(self):
        # Load PubMed credentials
        config = get_pubmed_config()
        self.email = config["email"]
        self.api_key = config["api_key"]
        self.session = requests.Session()
        self.rate_limit = 0.1  

        # Initialize LLM for query building
        llm_config = get_llm_config()
        self.llm = LLMClient(model=llm_config.get("model", "gemini-2.5-flash"))

    def search(
        self,
        query: str,
        max_results: int = 100,
        pmc_only: bool = True,
        only_trials: bool = True,
        llm_rerank: bool = True,
        relevance_threshold: int = 70,
        results_dir: Optional[Path] = None,
        from_year: Optional[int] = None,
        to_year: Optional[int] = None
    ) -> Tuple[List[str], Dict[str, Dict]]:
        """
        Execute end-to-end search: natural language → filtered, scored PMCIDs.
        """
        print(f"🗣️  Interpreting natural language query: '{query}'", file=sys.stderr)

        # Step 1: Use LLM to convert natural language → optimized PubMed query
        try:
            plan = build_pubmed_query(self.llm, query, from_year, to_year, only_trials)
            pubmed_query = plan["pubmed_query"]
            print(f"🧠 LLM-generated PubMed query: {pubmed_query}", file=sys.stderr)
        except Exception as e:
            print(f"⚠️  LLM query builder failed: {e}. Falling back to direct search.", file=sys.stderr)
            pubmed_query = query

        # Apply PMC-only filter
        if pmc_only:
            pubmed_query = f"({pubmed_query}) AND pubmed pmc[sb]"
        # Ensure English
        if "english[lang]" not in pubmed_query.lower():
            pubmed_query = f"({pubmed_query}) AND english[lang]"

        # Save query for transparency and debugging
        if results_dir:
            results_dir.mkdir(exist_ok=True)
            query_log = results_dir / "pubmed_query.txt"
            query_log.write_text(
                f"Natural Language Query:\n{query}\n\n"
                f"LLM-Generated PubMed Query:\n{pubmed_query}\n\n"
                f"Filters Applied:\n"
                f"- PMC full text: {pmc_only}\n"
                f"- Clinical trials only: {only_trials}\n"
                f"- Relevance threshold: ≥{relevance_threshold}\n"
            )
            print(f"📄 Saved search logic to: {query_log}", file=sys.stderr)

        # Step 2: Fetch PMIDs via ESearch
        print("🔍 Fetching study IDs from PubMed...", file=sys.stderr)
        ids = self._fetch_ids(pubmed_query, max_results)
        if not ids:
            print("❌ No studies found matching your query.", file=sys.stderr)
            return [], {}

        # Step 3: Fetch summaries (ESummary)
        print("📋 Fetching study metadata...", file=sys.stderr)
        summaries = self._fetch_summaries(ids)

        # Step 4: Fetch abstracts and details (EFetch XML)
        print("📝 Fetching abstracts and MeSH terms...", file=sys.stderr)
        details = self._fetch_abstracts(ids)

        # Step 5: Assemble records
        records = []
        for pmid in ids:
            if pmid not in summaries or pmid not in details:
                continue
            s = summaries[pmid]
            d = details[pmid]
            records.append(SearchResult(
                pmid=pmid,
                pmcid=d["pmcid"],
                title=s.get("title", "No title"),
                abstract=d["abstract"],
                journal=s.get("source", ""),
                year=d["year"] or (int(s.get("pubdate", "0000")[:4]) if s.get("pubdate", "").startswith("1") else None),
                authors=[a.get("name", "") for a in s.get("authorlist", [])],
                doi=s.get("epub", {}).get("doi", "") or s.get("uids", {}).get("doi", ""),
                mesh_headings=d["mesh_headings"],
                publication_types=d["publication_types"],
                license=d["license"]
            ).__dict__)

        if llm_rerank:
            print(f"🧠 LLM reranking {len(records)} studies...", file=sys.stderr)
            try:
                ranked = rerank_records_with_llm_batched(
                    cleaned_query=pubmed_query,
                    records=records,
                    llm=self.llm,
                    batch_size=20,
                    abstract_length=500
                )
                # Apply relevance threshold
                filtered_records = [r for r in ranked if r["relevance"] >= relevance_threshold]
                print(f"🎯 Relevance filter (≥{relevance_threshold}): {len(ranked)} → {len(filtered_records)}", file=sys.stderr)
            except Exception as e:
                print(f"⚠️  LLM reranking failed: {e}. Using fallback.", file=sys.stderr)
                for r in records:
                    r["relevance"] = 50
                    r["relevance_reason"] = "Reranking failed"
                filtered_records = sorted(records, key=lambda x: -(x["year"] or 0))
                ranked = filtered_records  # For exclusion tracking
        else:
            for r in records:
                r["relevance"] = 50
                r["relevance_reason"] = "LLM reranking disabled"
            filtered_records = sorted(records, key=lambda x: -(x["year"] or 0))
            ranked = filtered_records

        # Step 7: Extract PMCIDs and build metadata dict
        pmcid_list = []
        studies_metadata = {}
        excluded_studies = []

        for record in filtered_records:
            pmcid = record["pmcid"]

            if not pmcid or not pmcid.startswith("PMC"):
                excluded_studies.append({
                    "pmid": record["pmid"],
                    "pmcid": pmcid,
                    "title": record["title"],
                    "relevance": record["relevance"],
                    "reason": "No PMC full text available"
                })
                continue

            pmcid_list.append(pmcid)
            studies_metadata[pmcid] = record

        # Add studies that failed relevance threshold
        filtered_pmcids = {studies_metadata[pmcid]["pmid"] for pmcid in studies_metadata}
        for record in ranked:
            if record["pmid"] not in filtered_pmcids:
                if record["pmid"] not in [r["pmid"] for r in excluded_studies]:
                    excluded_studies.append({
                        "pmid": record["pmid"],
                        "pmcid": record.get("pmcid", "Unknown"),
                        "title": record["title"],
                        "relevance": record["relevance"],
                        "relevance_reason": record["relevance_reason"]
                        #"reason": f"Below relevance threshold ({record['relevance']} < {relevance_threshold})"
                    })

        # Save metadata with LLM scores
        if results_dir:
            meta_path = results_dir / "studies_metadata.json"
            save_json(studies_metadata, meta_path)
            print(f"📄 studies_metadata.json saved ({len(studies_metadata)} studies)", file=sys.stderr)

            # Save excluded studies
            excluded_path = results_dir / "excluded_studies.json"
            save_json(excluded_studies, excluded_path)
            print(f"📋 Excluded studies saved to: {excluded_path} ({len(excluded_studies)} studies)", file=sys.stderr)

        return pmcid_list, studies_metadata

    def _fetch_ids(self, pubmed_query: str, max_results: int) -> List[str]:
        """Fetch PMIDs using ESearch."""
        url = f"{EUTILS_BASE}/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": pubmed_query,
            "retmax": 200,
            "retmode": "json",
            "sort": "relevance"
        }
        ids = []
        retstart = 0
        with tqdm(desc="Fetching PMIDs", unit="id", leave=False, file=sys.stderr) as pbar:
            while len(ids) < max_results:
                time.sleep(self.rate_limit)
                params["retstart"] = retstart
                try:
                    r = self.session.get(url, params=params, timeout=30)
                    r.raise_for_status()
                    data = r.json()["esearchresult"]
                    batch = data.get("idlist", [])
                    if not batch:
                        break
                    ids.extend(batch)
                    pbar.update(len(batch))
                    retstart += len(batch)
                    if len(batch) < 200:  # Last page
                        break
                except Exception as e:
                    print(f"⚠️  ESearch failed: {e}", file=sys.stderr)
                    break
        return ids[:max_results]

    def _fetch_summaries(self, pmids: List[str]) -> Dict[str, dict]:
        """Fetch metadata via ESummary."""
        url = f"{EUTILS_BASE}/esummary.fcgi"
        summaries = {}
        for batch in chunked(pmids, 200):
            time.sleep(self.rate_limit)
            try:
                r = self.session.get(url, params={
                    "db": "pubmed",
                    "id": ",".join(batch),
                    "retmode": "json"
                }, timeout=30)
                r.raise_for_status()
                data = r.json().get("result", {})
                for pid in batch:
                    if pid in data and "exception" not in data[pid]:
                        summaries[pid] = data[pid]
            except Exception as e:
                print(f"⚠️  ESummary failed: {e}", file=sys.stderr)
        return summaries

    def _fetch_abstracts(self, pmids: List[str]) -> Dict[str, dict]:
        """Fetch abstracts, MeSH, PMCIDs via EFetch (XML)."""
        url = f"{EUTILS_BASE}/efetch.fcgi"
        out = {}
        namespaces = {'ali': 'http://www.niso.org/schemas/ali/1.0/'}

        for batch in chunked(pmids, 200):
            time.sleep(self.rate_limit)
            try:
                r = self.session.get(url, params={
                    "db": "pubmed",
                    "id": ",".join(batch),
                    "retmode": "xml"
                }, timeout=30)
                r.raise_for_status()
                root = ET.fromstring(r.text)
                for art in root.findall(".//PubmedArticle"):
                    pmid = self._get_text(art, ".//MedlineCitation/PMID")
                    if not pmid or pmid not in pmids:
                        continue

                    # Extract PMC ID
                    pmcid = "NA"
                    for aid in art.findall(".//ArticleId"):
                        if aid.get("IdType") == "pmc":
                            val = "".join(aid.itertext()).strip()
                            if val.startswith("PMC"):
                                pmcid = val
                                break

                    # Not yet implemented...need to switch the db over to PMC. 
                    allows_text_mining = False
                    for elem in art.findall(".//ali:license_ref", namespaces):
                        if elem.get("specific-use") == "textmining":
                            allows_text_mining = True
                            break

                    record = {
                        "abstract": self._extract_abstract(art),
                        "publication_types": [pt.text.strip() for pt in art.findall(".//PublicationType") if pt.text],
                        "mesh_headings": [mh.text.strip() for mh in art.findall(".//DescriptorName") if mh.text],
                        "year": self._extract_year(art),
                        "pmcid": pmcid,
                        "license": {"allows_text_mining": allows_text_mining}
                    }
                    out[pmid] = record
            except Exception as e:
                print(f"⚠️  EFetch failed: {e}", file=sys.stderr)
        return out

    def _get_text(self, node: ET.Element, path: str) -> str:
        el = node.find(path)
        return clean_text(el.text if el is not None and el.text else "")

    def _extract_abstract(self, art: ET.Element) -> str:
        texts = []
        for ab in art.findall(".//AbstractText"):
            label = ab.get("Label") or ab.get("NlmCategory") or ""
            content = "".join(ab.itertext()).strip()
            if content:
                texts.append(f"{label}: {content}" if label else content)
        return clean_text(" ".join(texts))

    def _extract_year(self, art: ET.Element) -> Optional[int]:
        y = self._get_text(art, ".//ArticleDate/Year")
        if y.isdigit():
            return int(y)
        y = self._get_text(art, ".//PubDate/Year")
        if y.isdigit():
            return int(y)
        md = self._get_text(art, ".//MedlineDate")
        return parse_year_from_pubdate(md)


# CLI for testing
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Smart PubMed Search CLI")
    parser.add_argument("--query", required=True, help="Natural language query")
    parser.add_argument("--max-results", type=int, default=20)
    parser.add_argument("--threshold", type=int, default=70, help="Relevance threshold (1-100)")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--no-rerank", action="store_true", help="Skip LLM reranking")
    args = parser.parse_args()

    searcher = SmartPubMedSearcher()
    pmcids, meta = searcher.search(
        query=args.query,
        max_results=args.max_results,
        llm_rerank=not args.no_rerank,
        relevance_threshold=args.threshold,
        results_dir=args.out_dir
    )
    print(f"🎯 Final relevant PMCIDs: {len(pmcids)}")