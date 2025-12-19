import sys
import textwrap
from typing import List, Dict, Any
from tqdm import tqdm
import time
from .llm_client import LLMClient
from .utils import chunked

def rerank_records_with_llm_batched(
    cleaned_query: str,
    records: List[Dict[str, Any]],
    llm: LLMClient,
    batch_size: int = 5,
    abstract_length: int = 500
) -> List[Dict[str, Any]]:
    ranked = []

    system_prompt = '''
You are a clinical research expert that serves as a matching engine for clinical trial relevance.

Your task: Score each study on a 1–100 scale based **only** on how closely the intervention in the study matches the intervention in the query.

Rules:
- The **core treatment modality** defines relevance (e.g., cell therapy, gene therapy, biologic, small molecule).
- If the study uses a **subtype, variant, or source-specific version** of the queried intervention, it is still a strong match.
  - Example: "adipose-derived stem cells" is a variant of "mesenchymal stem cells" → high relevance.
  - Example: "autologous CAR-T" is a variant of "CAR-T therapy" → high relevance.
- Do NOT penalize for specifying delivery method, dose, or source — these are refinements, not mismatches.
- Relevance is determined by **intervention class**, not by exact keyword match.
- If the intervention is conceptually different (e.g., physical therapy vs. cell therapy), score low.Score relevance of each paper to the query on a 1–100 scale.

Scoring:
- 90–100: Direct match (PICO)
- 70–89: High relevance
- 50–69: Partial
- 30–49: Tangential
- 1–29: Not relevant

Return a JSON object with field "results" containing an array of objects.
Each object must have:
- "pmid": string
- "relevance": integer (1–100)
- "reason": string

Example:
{"results": [{"pmid": "123456", "relevance": 90, "reason": "Direct PICO match"}]}
'''

    print(f"🔍 Scoring {len(records)} papers in batches of {batch_size}...", file=sys.stderr)
    scored_count = 0

    for batch in tqdm(list(chunked(records, batch_size)), desc="LLM Scoring", unit="batch"):
        user_prompt = f"QUERY: {cleaned_query}\n\n"
        for rec in batch:
            title = rec.get("title", "No title")
            abstract = rec.get("abstract") or "No abstract"
            mesh = ", ".join([m for m in rec.get("mesh_headings", [])[:5] if m]) or "None"
            types = ", ".join([t for t in rec.get("publication_types", []) if t]) or "Unknown"

            user_prompt += f"""
--- PMID {rec['pmid']} ---
Title: {textwrap.shorten(title, 180, placeholder='...')}
Abstract: {textwrap.shorten(abstract, abstract_length, placeholder='...')}
MeSH: {mesh}
Types: {types}
"""

        user_prompt += '\n\nReturn a JSON object with a "results" field containing the array.'

        try:
            resp = llm.complete_json(system_prompt, user_prompt, max_tokens=300000)

            if isinstance(resp, dict):
                items = resp.get("results", [])
            elif isinstance(resp, list):
                items = resp
            else:
                raise ValueError("Invalid response format")

            if not isinstance(items, list):
                raise ValueError("Results is not a list")

            result_map = {
                r["pmid"]: r
                for r in items
                if isinstance(r, dict) and "pmid" in r and "relevance" in r
            }

            for rec in batch:
                item = result_map.get(rec["pmid"])
                if item:
                    relevance = max(1, min(100, item["relevance"]))
                    reason = item["reason"]
                else:
                    relevance, reason = 50, "LLM did not score"
                rec2 = dict(rec)
                rec2["relevance"] = relevance
                rec2["relevance_reason"] = reason
                ranked.append(rec2)
                scored_count += 1

        except Exception as e:
            print(f"⚠️ Batch failed: {e}. Falling back to individual.", file=sys.stderr)
            for rec in batch:
                try:
                    item = _make_individual_scorer(llm, cleaned_query)(rec)
                except:
                    item = {"relevance": 50, "reason": "Scoring failed"}
                rec2 = dict(rec)
                rec2["relevance"] = item["relevance"]
                rec2["relevance_reason"] = item["reason"]
                ranked.append(rec2)
                scored_count += 1

        #time.sleep(0.2)  # Avoid rate limits

    print(f"📊 Scored {scored_count} studies", file=sys.stderr)
    ranked.sort(key=lambda r: -r["relevance"])
    return ranked

def _make_individual_scorer(llm: LLMClient, cleaned_query: str):
    def score(rec: Dict[str, Any]) -> Dict[str, Any]:
        system = 'Return JSON: {"relevance": int, "reason": str}'
        user = f"""
Query: {cleaned_query}
Title: {rec.get('title', '')}
Abstract: {textwrap.shorten(rec.get('abstract', ''), 800)}
MeSH: {', '.join(rec.get('mesh_headings', [])[:5])}
Return: {{"relevance": 1–100, "reason": "..."}}
"""
        resp = llm.complete_json(system, user, max_tokens=300000)
        return {
            "relevance": max(1, min(100, resp.get("relevance", 50))),
            "reason": resp.get("reason", "No reason")
        }
    return score
