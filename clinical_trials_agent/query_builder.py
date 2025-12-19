from typing import Dict, Any
from .llm_client import LLMClient

QUERY_BUILDER_SYSTEM = """
You are a biomedical search expert. Convert the user's natural language query into a precise PubMed query.
Use synonyms, and Boolean logic.
Return ONLY JSON: {"pubmed_query": str, "notes": str, "inferred_filters": {}}
"""

def build_pubmed_query(
    llm: LLMClient,
    cleaned_query: str,
    from_year: int = None,
    to_year: int = None,
    require_trials: bool = True
) -> Dict[str, Any]:
    constraints = []
    if from_year or to_year:
        constraints.append(f"published between {from_year or 'earliest'} and {to_year or 'latest'}")
    if require_trials:
        constraints.append("clinical trials only")

    constraints_str = "; ".join(constraints)
    user_prompt = f"""
Natural Language Query: "{cleaned_query}"
Constraints: {constraints_str}

Guidelines:
- Identify the core concept. DO NOT USE [tiab] for core concepts or terms.
- Generate all relevant synonyms, spelling variants, and closely related terms.
- Avoid using abbreviations by themselves as core concepts or terms. 
- DO NOT USE MeSH terms [mh].
- Exclude the words "therapy", "treatment", and similar synonyms, focusing just on the core concept or term.    
- Expand abbreviations
- Prefer sensitivity + precision
- Append (english[lang]) AND (Therapy/Broad[filter]) AND (clinical trial[pt]) AND (humans[mh]) at the end of the query.  

Return JSON only.
"""

    try:
        return llm.complete_json(QUERY_BUILDER_SYSTEM, user_prompt)
    except Exception as e:
        return {
            "pubmed_query": f"{cleaned_query} AND (english[lang]) AND (Therapy/Broad[filter]) AND (clinical trial[pt])",
            "notes": f"Fallback due to error: {e}",
            "inferred_filters": {}
        }
        
        
#- Use [mh] for MeSH terms (include synonyms)        