import json
import logging
import re
import sys
import asyncio
from typing import Dict, Any, List
from .llm_client import LLMClient
from .config import get_bias_criteria
from .utils import extract_json_from_text

# Create a specific logger for analysis result details
analysis_logger = logging.getLogger('analysis_results')
analysis_logger.setLevel(logging.DEBUG)

# Prevent propagation to avoid double-logging
if not analysis_logger.handlers:
    analysis_logger.propagate = False

    # File handler
    fh = logging.FileHandler('analysis_results.log', mode='a')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    analysis_logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    analysis_logger.addHandler(ch)

# Module-level logger for llm_utils
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# TOOL_REGISTRY maps synthesis question types to tool definitions

TOOL_REGISTRY = {
    "binary": {
        "description": "For yes/no questions about study methods, reporting, or presence/absence of elements.",
        "schema": {
            "pmcid": "str",
            "value": "str (yes / no / not reported)",
            "numerical value": "str (1 / 0)",
            "evidence": "str (short excerpt from study)"
        },
        "instructions": """
- Classify each study's value as 'yes', 'no', or 'not reported'.
- If 'yes', set numerical value to '1'; if 'no', set to '0'; if 'not reported', set to '0'.
- Provide a short evidence excerpt.
- Be factual and conservative.
- If uncertain, use 'not reported' for the value.
"""
    },
    "numeric": {
        "description": "For extracting measurable quantities (age, dose, cell count, p-value, etc.).",
        "schema": {
            "pmcid": "str",
            "value": "str (number, range, or 'not reported')",
            "unit": "str (unit of measurement or 'unknown')",
            "evidence": "str"
        },
        "instructions": """
- Extract exact values or ranges. 
- Include units (e.g., years, cells/kg, ng/mL, %).
- Do not estimate or impute.
- If not reported, set value to 'not reported' and unit to 'unknown'.
"""
    },
    "descriptive": {
        "description": "For qualitative descriptions of methods, populations, or interventions.",
        "schema": {
            "pmcid": "str",
            "value": "str (short description or 'not reported')",
            "evidence": "str"
        },
        "instructions": """
- Provide a concise description backed by the answers.
- Use direct quotes or close paraphrasing.
- If not reported, use 'not reported'.
- Avoid interpretation.
"""
    },
    "categorical": {
        "description": "For categorizing items into predefined or open-ended categories.",
        "schema": {
            "pmcid": "str",
            "value": "str (category label or 'not reported')",
            "unit": "str ('categorical')",
            "evidence": "str"
        },
        "instructions": """
- Assign one category label per study.
- Use exact terms from the text if possible.
- If no category applies, use 'not reported'.
- Do NOT invent categories.
- Set unit='categorical'.
"""
    },
    "summarize": {
        "description": "For cross-study synthesis with analysis, comparison, and conclusion.",
        "schema": {
            "entries": "list[dict: { " 
                "conclusion_entry: dict: { " # Use a clearer name to indicate the single complex object
                    "supporting_studies: list[dict: {pmcid: str, evidence: str}], "
                    "conflicting_studies: list[dict: {pmcid: str, evidence: str}], "
                    "neutral_or_unreported_studies: list[dict: {pmcid: str, evidence: str}] "
                "} "
            "}]"
        },
        "instructions": """
- For EACH study, classify as:
  - supporting: CLEAR, DIRECT evidence that aligns with the conclusion (e.g., "MELD score significantly improved")
  - conflicting: CLEAR evidence that contradicts (e.g., "No improvement in liver function")
  - neutral_or_unreported: 
      * Answer is "Not reported", 
      * Evidence is weak/indirect (e.g., "tendency for improvement" without stats),
      * Study design limits interpretation (e.g., single-arm, no control),
      * Or question is not addressed
- NEVER assume support from vague answers like "Yes" without evidence.
- The evidence field is a synthesis of all relevant evidence from the study addressing the synthesis question.
- Do not merely repeat the answers that are found in the A fields in the evidence field.
- Provide comprehensive, short summary of evidence excerpt for each that uses all answers for each PMCID given. Include a brief reasoning for classification. 
- Do not use A1, A2, etc. in the evidence field. Provide the evidence as a coherent summary.
- If a study is neutral, explain the reasoning why it is neutral using evidence from the study. 
- Do NOT omit any study.
- Be specific and evidence-based.
- Return JSON with exactly ONE entry in the 'entries' array.
"""
    }
}

def parse_questions(questions_text: str) -> list:
    """Parse questions from text."""
    if not questions_text.strip():
        return []
    lines = questions_text.strip().split('\n')
    questions = []
    for line in lines:
        cleaned = re.sub(r'^[\s\-\*\d\.\)]+\s*', '', line).strip()
        if cleaned:
            questions.append(cleaned)
    return questions


# Load bias criteria using the centralized config system
BIAS_CRITERIA = get_bias_criteria()


def select_tool_type(synthesis_question: str, llm_client: LLMClient) -> str:
    prompt = f"""You are an expert system for classifying clinical research synthesis questions. Your goal is to determine the primary analytical tool required to answer a given question.

Classify the question based on the fundamental nature of the data being sought and the complexity of the synthesis required.

---

### **Tool Type Definitions**

**1. binary:**
- **Core Concept:** A dichotomous (two-outcome) classification across a set of studies.
- **What it seeks:** A count or percentage of studies based on a simple YES/NO, PRESENT/ABSENT, or INCLUDES/EXCLUDES status.
- **The options are always a mutually exclusive pair.**
- **Keywords:** "Did...", "Was...", "What percentage of studies reported/did/used...", "Count of studies that have/do..."

**2. numeric:**
- **Core Concept:** Extraction of a quantifiable measurement or a raw count.
- **What it seeks:** A specific numerical value or set of values *from within* the studies, or a raw count of studies meeting a criterion.
- **This is for the number itself, not a percentage of the total.**
- **Includes:** means, medians, ranges, standard deviations, p-values, doses, durations, sample sizes, and raw counts of studies.
- **Keywords:** "What was the mean/median/range...", "Extract the...", "How many studies...", "Report the count of..."

**3. descriptive:**
- **Core Concept:** Extraction of qualitative, non-numerical information.
- **What it seeks:** Text-based descriptions, explanations, protocols, or rationale.
- **The result is prose, not a number or a simple category.**
- **Keywords:** "Describe...", "Explain...", "What are the...", "Detail the...", "Summarize the methods/protocol..."

**4. categorical:**
- **Core Concept:** Distribution across multiple, mutually exclusive categories.
- **What it seeks:** A count or percentage of studies classified into three or more non-numeric options.
- **The categories are distinct and do not overlap.**
- **Keywords:** "What percentage of studies used A, B, or C...", "Categorize the studies by...", "Distribution of study designs...", "Breakdown of intervention types..."

**5. summarize:**
- **Core Concept:** High-level synthesis requiring interpretation and judgment.
- **What it seeks:** A conclusion, comparison, or resolution of conflicting evidence across studies.
- **This goes beyond simple extraction to create new understanding.**
- **Keywords:** "Is X effective for Y?", "Compare the outcomes...", "What is the overall conclusion...", "Resolve the conflict between..."

---

### **Classification Decision Process**

Follow this sequence to determine the correct tool type:

**Step 1: Synthesis vs. Extraction?**
- Does the question require weighing evidence, comparing results, or forming a conclusion to resolve a complex issue?
  - **YES** -> Classify as **`summarize`**. Stop here.
- **NO** (It's asking for specific information) -> Proceed to Step 2.

**Step 2: What is the fundamental data type being extracted?**
- Is the answer a prose description or explanation?
  - **YES** -> Classify as **`descriptive`**. Stop here.
- **NO** (The answer is a number or a category) -> Proceed to Step 3.

**Step 3: Is the primary output a number or a category?**
- Is the answer a specific numerical value (e.g., a mean, a dose, a raw count)?
  - **YES** -> Classify as **`numeric`**. Stop here.
- **NO** (The answer is a count or percentage of studies distributed into options) -> Proceed to Step 4.

**Step 4: How many mutually exclusive options are there?**
- Are there exactly two options (e.g., YES/NO, PRESENT/ABSENT)?
  - **YES** -> Classify as **`binary`**.
- Are there three or more options?
  - **YES** -> Classify as **`categorical`**.

---

### **Critical Distinctions**

- **`numeric` vs. `binary`/`categorical`**: If the final answer is a raw number (like "5 studies" or "a mean of 25.4"), it is `numeric`. If the final answer is a percentage or distribution *across a set of studies* based on a classification, it is `binary` or `categorical`.
- **Compound Questions**: If a question has multiple parts, classify it based on the *primary* or most complex requirement. A question asking for a count AND a description should be classified based on which part is the main focus.

---

**QUESTION TO CLASSIFY:**
{synthesis_question}

INSTRUCTIONS:
Return ONLY the tool type: binary / numeric / descriptive / categorical / summarize
"""

    try:
        raw = llm_client.generate_text(prompt)
        tool = raw.strip().lower()
        if tool in ["binary", "numeric", "descriptive", "categorical", "summarize"]:
            return tool
        else:
            q = synthesis_question.lower()
            if any(kw in q for kw in ["yes/no", "used", "present", "whether", "if ", "did ", "does ", "is ", "are "]):
                return "binary"
            elif any(kw in q for kw in ["how many", "what proportion", "percentage", "mean", "median", "average", "count"]):
                return "numeric"
            elif any(kw in q for kw in ["describe", "what was", "how were", "type of", "classify", "categorize"]):
                return "categorical" if any(kw in q for kw in ["type of", "classify", "categorize"]) else "descriptive"
            else:
                return "summarize"
    except Exception as e:
        logger.error(f"Tool selection failed: {e}")
        return "summarize"
    

def chunk_list(lst, n):
    """Split a list into chunks of size n."""
    return [lst[i:i + n] for i in range(0, len(lst), n)]
        
def _create_tool_direct(
    synthesis_question: str,
    tool_type: str,
    study_answers: Dict[str, Dict[str, str]],
    all_pmcids: List[str],
    llm_client: LLMClient,
    tool_def: Dict
) -> Dict:
    """Create tool directly (no batching)"""
    
    relevant_questions = set()
    for pmcid in all_pmcids:
        relevant_questions.update(study_answers[pmcid].keys())
    relevant_questions = sorted(relevant_questions)

    if relevant_questions:
        question_index = {q: f"A{i+1}" for i, q in enumerate(relevant_questions)}
        question_header = "\n".join([f"{i+1}. {q}" for i, q in enumerate(relevant_questions)])
        
        study_blocks = []
        for pmcid in all_pmcids:
            answer_lines = []
            for q in relevant_questions:
                ans = study_answers[pmcid].get(q, "Not reported")
                answer_lines.append(f"- {question_index[q]}: {ans}")
            study_blocks.append(f"[{pmcid}]\n" + "\n".join(answer_lines))
        
        context = f"""RELEVANT QUESTIONS:
{question_header}

STUDY RESPONSES:
{"\n\n".join(study_blocks)}"""
    else:
        # Fallback to old format if no questions
        entries = []
        for pmcid in all_pmcids:
            ans = "\n".join([f"  - {a}" for a in study_answers[pmcid].values()])
            entries.append(f"[{pmcid}]\n{ans if ans else '  - No relevant data reported'}")
        context = "\n\n".join(entries)

    # Adjust instructions based on tool type
    if tool_type == "summarize":
        extra_instructions = """
- Return JSON with top-level keys: "supporting_studies", "conflicting_studies", "neutral_or_unreported_studies".
- Do NOT use an "entries" array.
- Each study must appear in exactly one of the three lists.
"""
    else:
        extra_instructions = "- Return JSON with an 'entries' array containing one object per study."

    prompt = f"""
You are a clinical research data architect.
Create a '{tool_type}' tool to answer:

SYNTHESIS QUESTION:
{synthesis_question}

{context}

TOOL DESCRIPTION:
{tool_def['description']}

SCHEMA:
{json.dumps(tool_def['schema'], indent=2)}

INSTRUCTIONS:
{tool_def['instructions']}

FORMAT NOTES:
- Questions are listed once at the top as "1.", "2.", etc.
- Each study uses "A1", "A2", etc. to answer the corresponding question.
- Example: 
[PMC123]
- A1: Adipose
- A2: Yes

{extra_instructions}
- Include ALL studies.
- Return ONLY JSON. No explanations.
"""

    try:
        raw = llm_client.generate_text(prompt)
        cleaned_raw = raw.replace('```json', '').replace('```', '').strip()
        result = extract_json_from_text(cleaned_raw)
        
        if not result:
            logger.warning(f"Failed to parse tool for: {synthesis_question}")
            return {
                "synthesis_question": synthesis_question,
                "tool_type": tool_type,
                "status": "parsing_failed",
                "raw_response": json.dumps(cleaned_raw)[:1000],
                "entries": []
            }

        # === HANDLE SUMMARIZE ===
        if tool_type == "summarize":
            supporting = result.get("supporting_studies", [])
            conflicting = result.get("conflicting_studies", [])
            neutral = result.get("neutral_or_unreported_studies", [])

            # Deduplicate
            def dedup(lst):
                seen = set()
                out = []
                for item in lst:
                    if isinstance(item, dict) and "pmcid" in item:
                        if item["pmcid"] not in seen:
                            out.append(item)
                            seen.add(item["pmcid"])
                return out

            supporting = dedup(supporting)
            conflicting = dedup(conflicting)
            neutral = dedup(neutral)

            # Ensure all PMCIDs covered
            covered = {item["pmcid"] for lst in [supporting, conflicting, neutral] for item in lst if isinstance(item, dict) and "pmcid" in item}
            for pmcid in all_pmcids:
                if pmcid not in covered:
                    neutral.append({"pmcid": pmcid, "evidence": "No relevant data found."})

            # Build flat entries
            final_entries = []
            for s in supporting:
                if isinstance(s, dict) and "pmcid" in s:
                    final_entries.append({
                        "pmcid": s["pmcid"],
                        "value": "supported",
                        "evidence": s.get("evidence", "No evidence")
                    })
            for s in conflicting:
                if isinstance(s, dict) and "pmcid" in s:
                    final_entries.append({
                        "pmcid": s["pmcid"],
                        "value": "conflicting",
                        "evidence": s.get("evidence", "No evidence")
                    })
            for s in neutral:
                if isinstance(s, dict) and "pmcid" in s:
                    final_entries.append({
                        "pmcid": s["pmcid"],
                        "value": "neutral",
                        "evidence": s.get("evidence", "No evidence")
                    })

            return {
                "synthesis_question": synthesis_question,
                "tool_type": tool_type,
                "supporting_studies": supporting,
                "conflicting_studies": conflicting,
                "neutral_or_unreported_studies": neutral,
                "entries": final_entries
            }

        # === HANDLE NON-SUMMARIZE ===
        else:
            entries = result.get("entries", [])
            entry_dict = {}
            
            for e in entries:
                if isinstance(e, dict) and "pmcid" in e:
                    pmcid = e["pmcid"]
                    # Enforce schema
                    if tool_type == "binary":
                        val = str(e.get("value", "")).strip().lower()
                        if val in ["yes", "1"]:
                            e.update({"value": "yes", "numerical value": "1"})
                        elif val in ["no", "0"]:
                            e.update({"value": "no", "numerical value": "0"})
                        else:
                            e.update({"value": "not reported", "numerical value": "0"})
                        e.setdefault("evidence", "Not reported")
                    elif tool_type == "numeric":
                        e.setdefault("value", "not reported")
                        e.setdefault("unit", "unknown")
                        e.setdefault("evidence", "Not reported")
                    elif tool_type == "categorical":
                        e.setdefault("value", "not reported")
                        e["unit"] = "categorical"
                        e.setdefault("evidence", "Not reported")
                    elif tool_type == "descriptive":
                        e.setdefault("value", "not reported")
                        e.setdefault("evidence", "Not reported")
                    entry_dict[pmcid] = e

            # Ensure all PMCIDs present
            final_entries = []
            for pmcid in all_pmcids:
                if pmcid in entry_dict:
                    final_entries.append(entry_dict[pmcid])
                else:
                    if tool_type == "binary":
                        final_entries.append({
                            "pmcid": pmcid,
                            "value": "not reported",
                            "numerical value": "0",
                            "evidence": "Not reported"
                        })
                    elif tool_type == "numeric":
                        final_entries.append({
                            "pmcid": pmcid,
                            "value": "not reported",
                            "unit": "unknown",
                            "evidence": "Not reported"
                        })
                    elif tool_type == "categorical":
                        final_entries.append({
                            "pmcid": pmcid,
                            "value": "not reported",
                            "unit": "categorical",
                            "evidence": "Not reported"
                        })
                    else:  # descriptive
                        final_entries.append({
                            "pmcid": pmcid,
                            "value": "not reported",
                            "evidence": "Not reported"
                        })

            return {
                "synthesis_question": synthesis_question,
                "tool_type": tool_type,
                "entries": final_entries
            }

    except Exception as e:
        logger.error(f"Tool creation failed: {e}")
        safe_raw = json.dumps(str(raw)[:2000]).replace('\n', '\\n')
        return {
            "synthesis_question": synthesis_question,
            "tool_type": tool_type,
            "status": "error",
            "error": str(e),
            "raw_response": safe_raw,
            "entries": []
        }
        
def normalize_entry_for_tool_type(entry: dict, tool_type: str) -> dict:
    """Normalize entry based on tool type."""
    if not isinstance(entry, dict):
        return entry

    val = str(entry.get("value", "")).strip()
    unit = str(entry.get("unit", "")).strip()

    # Binary: force 0/1
    if tool_type == "binary":
        if val.lower() in ["yes", "true", "used", "present"]:
            entry["value"] = "1"
        elif val.lower() in ["no", "false", "not used", "absent"]:
            entry["value"] = "0"
        else:
            entry["value"] = "not reported"
        entry["unit"] = "binary"

    # Categorical: ensure unit is 'categorical'
    elif tool_type == "categorical":
        entry["unit"] = "categorical"
        if not val or val.lower() in ["not reported", "nr", "unknown"]:
            entry["value"] = "not reported"

    # Numeric: let LLM decide unit, but validate
    elif tool_type == "numeric":
        if not unit:
            entry["unit"] = "unknown"  # or infer from context if possible

    # Summarize: skip (handled separately)
    elif tool_type == "summarize":
        pass

    return entry

# async def create_tool_for_synthesis_question(
#     synthesis_question: str,
#     tool_type: str,
#     analyses_by_question: Dict[str, Dict[str, str]],
#     relevance_map: Dict[str, List[str]],
#     all_pmcids: List[str],
#     llm_client: LLMClient,
#     batch_size: int = 25
# ) -> Dict:
#     def normalize(q: str) -> str:
#         return re.sub(r'[^\w\s]', '', q.lower().strip())

#     # === 1. Get relevant questions from relevance map ===
#     raw_entry = relevance_map.get(synthesis_question, {})
#     if not raw_entry:
#         return {
#             "synthesis_question": synthesis_question,  
#             "tool_type": tool_type,
#             "status": "no_relevant_data",
#             "entries": []
#         }

#     tool_type_from_map = raw_entry.get("tool_type", tool_type)
#     relevant_q_list = raw_entry.get("relevant_questions", [])

#     if not relevant_q_list:
#         logger.warning(f"No valid analysis questions matched for: {synthesis_question}")
#         return {
#             "synthesis_question": synthesis_question,  
#             "tool_type": tool_type_from_map,
#             "status": "no_matching_questions",
#             "entries": []
#         }

#     # === 2. Gather answers for all PMCIDs ===
#     study_answers = {pmcid: {} for pmcid in all_pmcids}
#     normalized_analyses_keys = {
#         normalize(q): q for q in analyses_by_question.keys()
#     }

#     for analysis_q in relevant_q_list:
#         norm_analysis_q = normalize(analysis_q)
#         original_q = normalized_analyses_keys.get(norm_analysis_q)
#         if not original_q:
#             logger.warning(f"Question not found: '{analysis_q}'")
#             continue
#         for pmcid, answer in analyses_by_question[original_q].items():
#             if pmcid in study_answers:
#                 study_answers[pmcid][original_q] = answer

#     # === 3. If few studies, process directly ===
#     if len(all_pmcids) <= batch_size:
#         result = _create_tool_direct(
#             synthesis_question=synthesis_question,
#             tool_type=tool_type_from_map,
#             study_answers=study_answers,
#             all_pmcids=all_pmcids,
#             llm_client=llm_client,
#             tool_def=TOOL_REGISTRY.get(tool_type_from_map, TOOL_REGISTRY["summarize"])
#         )
#         result["synthesis_question"] = synthesis_question 
#         return result

#     # === 4. ASYNC BATCHED PROCESSING ===
#     batches = chunk_list(all_pmcids, batch_size)
#     tool_def = TOOL_REGISTRY.get(tool_type_from_map, TOOL_REGISTRY["summarize"])

#     async def process_batch(batch_pmcids, batch_idx):
#         # Get relevant questions for this synthesis question
#         raw_entry = relevance_map.get(synthesis_question, {})
#         relevant_questions = raw_entry.get("relevant_questions", [])
        
#         if not relevant_questions:
#             # Fallback: collect all questions present in this batch
#             relevant_questions = set()
#             for pmcid in batch_pmcids:
#                 relevant_questions.update(study_answers[pmcid].keys())
#             relevant_questions = sorted(relevant_questions)

#         # Create question index
#         question_index = {q: f"A{i+1}" for i, q in enumerate(relevant_questions)}
#         question_header = "\n".join([f"{i+1}. {q}" for i, q in enumerate(relevant_questions)])

#         # Build study blocks
#         study_blocks = []
#         for pmcid in batch_pmcids:
#             answer_lines = []
#             for q in relevant_questions:
#                 ans = study_answers[pmcid].get(q, "Not reported")
#                 answer_lines.append(f"- {question_index[q]}: {ans}")
#             study_blocks.append(f"[{pmcid}]\n" + "\n".join(answer_lines))

#         context = f"""RELEVANT QUESTIONS:
# {question_header}

# STUDY RESPONSES:
# {"\n\n".join(study_blocks)}"""

#         prompt = f"""
# You are a clinical research data architect.
# Create a '{tool_type_from_map}' tool to answer:

# SYNTHESIS QUESTION:
# {synthesis_question}

# {context}

# TOOL DESCRIPTION:
# {tool_def['description']}

# SCHEMA:
# {json.dumps(tool_def['schema'], indent=2)}

# INSTRUCTIONS:
# {tool_def['instructions']}

# FORMAT NOTES:
# - Questions are listed once at the top as "1.", "2.", etc.
# - Each study uses "A1", "A2", etc. to answer the corresponding question.
# - Example: 
# [PMC123]
# - A1: Adipose
# - A2: Yes

# - Include ALL studies in this batch.
# - Return JSON with ONE entry in 'entries' array.  
# - Return ONLY JSON. No explanations.
# """

#         try:
#             raw = await llm_client.generate_text_async(prompt)
#             analysis_logger.debug(f"🔧 Raw LLM response for batch {batch_idx+1}: {raw[:500]}...")
#             cleaned_raw = raw.replace('```json', '').replace('```', '').strip()
#             tool = extract_json_from_text(cleaned_raw)
#             return tool if tool and isinstance(tool.get("entries"), list) else None
#         except Exception as e:
#             logger.error(f"Batch {batch_idx} failed: {e}")
#             return None

#     tasks = [process_batch(batch, i) for i, batch in enumerate(batches)]
#     results = await asyncio.gather(*tasks, return_exceptions=True)
#     partial_tools = [r for r in results if r is not None and not isinstance(r, Exception)]

#     if not partial_tools:
#         return {
#             "synthesis_question": synthesis_question, 
#             "tool_type": tool_type_from_map,
#             "status": "batch_failed",
#             "entries": []
#         }

#     # === 5. FINAL MERGE BASED ON TOOL TYPE ===
#     if tool_type_from_map == "summarize":
#         supporting_studies = []
#         conflicting_studies = []
#         neutral_or_unreported_studies = []

#         for pt in partial_tools:
#             for entry in pt.get("entries", []):
#                 if not isinstance(entry, dict):
#                     continue
                    
#                 # Deduplicate WITHIN this batch entry first
#                 def dedup_list(lst):
#                     seen = set()
#                     unique = []
#                     for item in lst:
#                         if isinstance(item, dict) and "pmcid" in item:
#                             pmcid = item["pmcid"]
#                             if pmcid not in seen:
#                                 unique.append(item)
#                                 seen.add(pmcid)
#                     return unique

#                 if "supporting_studies" in entry:
#                     supporting_studies.extend(dedup_list(entry["supporting_studies"]))
#                 if "conflicting_studies" in entry:
#                     conflicting_studies.extend(dedup_list(entry["conflicting_studies"]))
#                 if "neutral_or_unreported_studies" in entry:
#                     neutral_or_unreported_studies.extend(dedup_list(entry["neutral_or_unreported_studies"]))

#         # NOW deduplicate globally
#         def deduplicate(studies):
#             seen = set()
#             unique = []
#             for item in studies:
#                 if isinstance(item, dict) and "pmcid" in item:
#                     pmcid = item["pmcid"]
#                     if pmcid not in seen:
#                         unique.append(item)
#                         seen.add(pmcid)
#             return unique

#         supporting_studies = deduplicate(supporting_studies)
#         conflicting_studies = deduplicate(conflicting_studies)
#         neutral_or_unreported_studies = deduplicate(neutral_or_unreported_studies)

#         # Ensure all PMCIDs are covered
#         covered_pmcids = set()
#         for lst in [supporting_studies, conflicting_studies, neutral_or_unreported_studies]:
#             for item in lst:
#                 if isinstance(item, dict) and "pmcid" in item:
#                     covered_pmcids.add(item["pmcid"])

#         for pmcid in all_pmcids:
#             if pmcid not in covered_pmcids:
#                 neutral_or_unreported_studies.append({
#                     "pmcid": pmcid,
#                     "evidence": "No relevant data found for this study."
#                 })

#         # Build final entries
#         final_entries = []
#         for s in supporting_studies:
#             if isinstance(s, dict) and "pmcid" in s:
#                 final_entries.append({
#                     "pmcid": s["pmcid"],
#                     "value": "supported",
#                     "evidence": s.get("evidence", "No evidence provided")
#                 })
#         for s in conflicting_studies:
#             if isinstance(s, dict) and "pmcid" in s:
#                 final_entries.append({
#                     "pmcid": s["pmcid"],
#                     "value": "conflicting",
#                     "evidence": s.get("evidence", "No evidence provided")
#                 })
#         for s in neutral_or_unreported_studies:
#             if isinstance(s, dict) and "pmcid" in s:
#                 final_entries.append({
#                     "pmcid": s["pmcid"],
#                     "value": "neutral",
#                     "evidence": s.get("evidence", "No evidence provided")
#                 })

#         return {
#             "synthesis_question": synthesis_question,
#             "tool_type": tool_type_from_map,
#             "supporting_studies": supporting_studies,
#             "conflicting_studies": conflicting_studies,
#             "neutral_or_unreported_studies": neutral_or_unreported_studies
#         }

#     else:
#         # === FIXED NON-summarize MERGE (matches your new schemas) ===
#         all_entries = []
#         for pt in partial_tools:
#             all_entries.extend(pt.get("entries", []))

#         # Build map from pmcid to entry
#         entry_dict = {}
#         for e in all_entries:
#             if isinstance(e, dict) and "pmcid" in e:
#                 pmcid = e["pmcid"]
#                 if tool_type_from_map == "binary":
#                     val = str(e.get("value", "")).strip().lower()
#                     if val in ["yes", "1"]:
#                         e.update({
#                             "value": "yes",
#                             "numerical value": "1"
#                         })
#                     elif val in ["no", "0"]:
#                         e.update({
#                             "value": "no",
#                             "numerical value": "0"
#                         })
#                     else:
#                         e.update({
#                             "value": "not reported",
#                             "numerical value": "0"
#                         })
#                     e.setdefault("evidence", "Not reported")
#                 elif tool_type_from_map == "numeric":
#                     e.setdefault("value", "not reported")
#                     e.setdefault("unit", "unknown")
#                     e.setdefault("evidence", "Not reported")
#                 elif tool_type_from_map == "categorical":
#                     e.setdefault("value", "not reported")
#                     e["unit"] = "categorical"
#                     e.setdefault("evidence", "Not reported")
#                 elif tool_type_from_map == "descriptive":
#                     e.setdefault("value", "not reported")
#                     e.setdefault("evidence", "Not reported")
#                 entry_dict[pmcid] = e

#         # Ensure all PMCIDs are present with proper defaults
#         final_entries = []
#         for pmcid in all_pmcids:
#             if pmcid in entry_dict:
#                 final_entries.append(entry_dict[pmcid])
#             else:
#                 if tool_type_from_map == "binary":
#                     default_entry = {
#                         "pmcid": pmcid,
#                         "value": "not reported",
#                         "numerical value": "0",
#                         "evidence": "Not reported"
#                     }
#                 elif tool_type_from_map == "numeric":
#                     default_entry = {
#                         "pmcid": pmcid,
#                         "value": "not reported",
#                         "unit": "unknown",
#                         "evidence": "Not reported"
#                     }
#                 elif tool_type_from_map == "categorical":
#                     default_entry = {
#                         "pmcid": pmcid,
#                         "value": "not reported",
#                         "unit": "categorical",
#                         "evidence": "Not reported"
#                     }
#                 else:  # descriptive
#                     default_entry = {
#                         "pmcid": pmcid,
#                         "value": "not reported",
#                         "evidence": "Not reported"
#                     }
#                 final_entries.append(default_entry)

#         return {
#             "synthesis_question": synthesis_question,  
#             "tool_type": tool_type_from_map,
#             "entries": final_entries
#         }

async def create_tool_for_synthesis_question(
    synthesis_question: str,
    tool_type: str,
    analyses_by_question: Dict[str, Dict[str, str]],
    relevance_map: Dict[str, Dict[str, Any]],
    all_pmcids: List[str],
    llm_client: LLMClient,
    batch_size: int = 25
) -> Dict:
    def normalize(q: str) -> str:
        return re.sub(r'[^\w\s]', '', q.lower().strip())

    # === 1. Get relevant questions from relevance map ===
    raw_entry = relevance_map.get(synthesis_question, {})
    if not raw_entry:
        return {
            "synthesis_question": synthesis_question, 
            "tool_type": tool_type,
            "status": "no_relevant_data",
            "entries": []
        }

    tool_type_from_map = raw_entry.get("tool_type", tool_type)
    relevant_q_list = raw_entry.get("relevant_questions", [])

    if not relevant_q_list:
        logger.warning(f"No valid analysis questions matched for: {synthesis_question}")
        return {
            "synthesis_question": synthesis_question, 
            "tool_type": tool_type_from_map,
            "status": "no_matching_questions",
            "entries": []
        }

    # === 2. Gather answers for all PMCIDs ===
    study_answers = {pmcid: {} for pmcid in all_pmcids}
    normalized_analyses_keys = {
        normalize(q): q for q in analyses_by_question.keys()
    }

    for analysis_q in relevant_q_list:
        norm_analysis_q = normalize(analysis_q)
        original_q = normalized_analyses_keys.get(norm_analysis_q)
        if not original_q:
            logger.warning(f"Question not found: '{analysis_q}'")
            continue
        for pmcid, answer in analyses_by_question.get(original_q, {}).items():
            if pmcid in study_answers:
                study_answers[pmcid][original_q] = answer

    # === 3. If few studies, process directly ===
    if len(all_pmcids) <= batch_size:
        # Note: This relies on your custom _create_tool_direct
        tool_def = TOOL_REGISTRY.get(tool_type_from_map, TOOL_REGISTRY["summarize"])
        batches = [all_pmcids]
        
    # === 4. ASYNC BATCHED PROCESSING (If not handled by direct call) ===
    else:
        batches = chunk_list(all_pmcids, batch_size)
        
    tool_def = TOOL_REGISTRY.get(tool_type_from_map, TOOL_REGISTRY["summarize"])

    async def process_batch(batch_pmcids, batch_idx):
        raw_entry = relevance_map.get(synthesis_question, {})
        relevant_questions = raw_entry.get("relevant_questions", [])
        
        # Fallback: collect all questions present in this batch
        if not relevant_questions:
            relevant_questions = set()
            for pmcid in batch_pmcids:
                relevant_questions.update(study_answers[pmcid].keys())
            relevant_questions = sorted(relevant_questions)

        # Create question index and header
        question_index = {q: f"A{i+1}" for i, q in enumerate(relevant_questions)}
        question_header = "\n".join([f"A{i+1}. {q}" for i, q in enumerate(relevant_questions)])

        # Build study blocks
        study_blocks = []
        for pmcid in batch_pmcids:
            answer_lines = []
            for q in relevant_questions:
                ans = study_answers[pmcid].get(q, "Not reported")
                answer_lines.append(f"- {question_index[q]}: {ans}")
            study_blocks.append(f"[{pmcid}]\n" + "\n".join(answer_lines))

        context = f"""RELEVANT QUESTIONS:
{question_header}

STUDY RESPONSES:
{"\n\n".join(study_blocks)}"""

        # --- Customized Instructions based on tool_type_from_map ---
        per_pmcid_instruction = ""
        example_value = ""
        
        if tool_type_from_map == "summarize":
            per_pmcid_instruction = (
                "2. **THE GOAL IS BATCH SYNTHESIS.** You must return exactly ONE entry in the 'entries' array. \n"
                "3. This single entry must contain the three study lists: supporting_studies, conflicting_studies, and neutral_or_unreported_studies. \n"
                "4. For EACH study listed in the 'STUDY RESPONSES', classify it into one of the three lists. \n"
                "5. **CRITICAL:** The 'evidence' field must be a comprehensive, short summary of **all relevant** study answers (A1, A2, etc.) for that PMCID. **DO NOT simply repeat the A# statements.** Condense the information and briefly explain the reasoning for your classification (supporting, conflicting, or neutral/unreported)."
            )

            example_value = (
                '"supporting_studies": [\n'
                '      {"pmcid": "PMC123", "evidence": Study utilized autologous cells with donor age 44 years (range 18-73) across both sexes. This is supporting because key demographic data are provided."}\n'
                '    ],\n'
                '    "conflicting_studies": [\n'
                '      {"pmcid": "PMC456", "evidence": "Study reported toxicity."}\n'
                '    ],'
            )
            
        elif tool_type_from_map == "numeric":
            per_pmcid_instruction = (
                "2. **DO NOT SUMMARIZE THE BATCH.** You must process the data study-by-study, providing one entry per PMCID.\n"
                "3. If the Synthesis Question requires multiple data points from the study responses (A1, A2, etc.), **combine all relevant extracted data into the single 'value' field** for that PMCID using a **pipe-separated format** (e.g., 'Age|45.2 years; Sex|Male; Weight|75 kg').\n"
                "4. Set the 'unit' field to 'extracted_values' when combining multiple metrics.\n"
                "5. **CRITICAL:** The 'evidence' field must be a comprehensive, short summary of **all relevant** study answers (A1, A2, etc.) for that PMCID. **DO NOT simply repeat the A# statements.** Condense the information, and omit reporting 'Not reported' unless it is essential for context."
            )
            example_value = '"value": "Age|45.2 years; Sex|Male; Weight|75 kg", "unit": "extracted_values"'
        
        elif tool_type_from_map == "binary":
            per_pmcid_instruction = (
                "2. **DO NOT SUMMARIZE THE BATCH.** You must process the data study-by-study, providing one entry per PMCID.\n"
                "3. The 'value' field must be a simple 'yes', 'no', or 'not reported'."
            )
            example_value = '"value": "yes", "numerical value": "1"'
            
        else: 
            per_pmcid_instruction = (
                "2. **DO NOT SUMMARIZE THE BATCH.** You must process the data study-by-study, providing one entry per PMCID.\n"
                "3. The 'value' field should contain the single, extracted category or description that directly answers the Synthesis Question."
            )
            example_value = '"value": "Adipose-derived MSCs", "unit": "categorical"'


        prompt = f"""
You are a clinical research data architect.
Your task is to transform the raw data from the 'STUDY RESPONSES' into a list of structured JSON entries for the '{tool_type_from_map}' tool.

SYNTHESIS QUESTION:
{synthesis_question}

{context}

TOOL DESCRIPTION:
{tool_def['description']}

SCHEMA:
{json.dumps(tool_def['schema'], indent=2)}

INSTRUCTIONS:
1. **The primary goal is to return ONE entry in the 'entries' array for every PMCID in the 'STUDY RESPONSES'.**
{per_pmcid_instruction}
5. The 'evidence' field contains a brief summary of the relevant data from the study responses that supports your answer.

FORMAT NOTES:
- Questions are listed once at the top as "A1.", "A2.", etc.
- **Example of expected output structure (must be adhered to):**
  "entries": [
    {{
      "pmcid": "PMC1234567",
      {example_value},
      "evidence": "A brief summary of the relevant data from the study responses."
    }},
    // ... continue for all other PMCIDs in the batch
  ]
- Return ONLY JSON. No explanations.
"""

        try:
            raw = await llm_client.generate_text_async(prompt)
            analysis_logger.debug(f"🔧 Raw LLM response for batch {batch_idx+1}: {raw[:500]}...")
            cleaned_raw = raw.replace('```json', '').replace('```', '').strip()
            tool = extract_json_from_text(cleaned_raw)
            
            if tool_type_from_map == "summarize":
                # The LLM often ignores the 'entries' array instruction and returns the lists at the top level.
                summary_keys = ["supporting_studies", "conflicting_studies", "neutral_or_unreported_studies"]
                
                # Check if the lists exist at the TOP LEVEL of the returned JSON
                is_top_level_summary = any(key in tool and isinstance(tool.get(key), list) for key in summary_keys)

                if is_top_level_summary:
                    # If lists are at the top, wrap them in the expected 'entries' structure
                    summary_data = {key: tool.get(key) for key in summary_keys}
                    
                    # Ensure the tool has the expected 'entries' structure for the merge.
                    tool["entries"] = [summary_data]
                    
                    # Remove the top-level keys to prevent conflicts during merging
                    for key in summary_keys:
                        if key in tool:
                            del tool[key] 
                
                # If 'entries' is missing or empty after the fix, it's a failure
                if not tool.get("entries") or not isinstance(tool["entries"], list):
                    logger.error(f"Summarize batch {batch_idx} failed to produce a usable 'entries' list.")
                    return None
            # Critical validation: Check entry count
            if tool.get("entries") and isinstance(tool["entries"], list):
                if tool_type_from_map != "summarize" and len(tool["entries"]) != len(batch_pmcids):
                    logger.error(f"Batch {batch_idx} returned incorrect entry count: {len(tool.get('entries', []))} vs {len(batch_pmcids)}")
                    return None
                
                # For summarize, the length must be exactly 1 (the complex entry)
                if tool_type_from_map == "summarize" and len(tool["entries"]) != 1:
                    logger.error(f"Summarize batch {batch_idx} returned incorrect entry count: {len(tool.get('entries', []))} vs 1 (Expected one complex entry).")
                    return None
                
                return tool
            
            logger.error(f"Batch {batch_idx} failed validation: 'entries' list is missing or malformed.")
            return None

        except Exception as e:
            logger.error(f"Batch {batch_idx} failed: {e}")
            return None

    # This handles both the small batch (len(all_pmcids) <= batch_size) and large batches
    tasks = [process_batch(batch, i) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    partial_tools = [r for r in results if r is not None and not isinstance(r, Exception)]

    if not partial_tools:
        return {
            "synthesis_question": synthesis_question, 
            "tool_type": tool_type_from_map,
            "status": "batch_failed",
            "entries": []
        }

    # === 5. FINAL MERGE BASED ON TOOL TYPE ===
    if tool_type_from_map == "summarize":
        # Initialize final lists
        global_supporting_studies = []
        global_conflicting_studies = []
        global_neutral_or_unreported_studies = []
        
        # Helper list for easy iteration
        summary_keys = [
            "supporting_studies", 
            "conflicting_studies", 
            "neutral_or_unreported_studies"
        ]

        # Define or ensure global deduplicate is available here
        def deduplicate(studies):
            seen = set()
            unique = []
            if not isinstance(studies, list):
                 return []
            for item in studies:
                if isinstance(item, dict) and "pmcid" in item:
                    pmcid = item["pmcid"]
                    if pmcid not in seen:
                        unique.append(item)
                        seen.add(pmcid)
            return unique

        for pt in partial_tools:
            # Check if 'entries' exists and has at least one item
            if not pt.get("entries") or not isinstance(pt["entries"], list) or not pt["entries"]:
                logger.warning(f"Batch returned no 'entries' or malformed 'entries' list. Skipping batch.")
                continue
            
            # Get the *single* complex entry returned by the LLM for this batch
            summary_entry = pt["entries"][0]
            
            if not isinstance(summary_entry, dict):
                logger.warning(f"Summarize entry for batch was not a dictionary: {summary_entry}")
                continue

            if "conclusion_entry" in summary_entry and isinstance(summary_entry["conclusion_entry"], dict):
                data_source = summary_entry["conclusion_entry"]
            else:
                data_source = summary_entry

            # Iterate through the expected keys and safely extend the global lists
            for key in summary_keys:
                batch_list = data_source.get(key)
                if isinstance(batch_list, list):
                    if key == "supporting_studies":
                        global_supporting_studies.extend(batch_list)
                    elif key == "conflicting_studies":
                        global_conflicting_studies.extend(batch_list)
                    elif key == "neutral_or_unreported_studies":
                        global_neutral_or_unreported_studies.extend(batch_list)
                else:
                     # This warning is now much more meaningful
                     logger.warning(f"Batch entry for '{key}' was not a list in data_source.")


        # Apply global deduplication
        supporting_studies = deduplicate(global_supporting_studies)
        conflicting_studies = deduplicate(global_conflicting_studies)
        neutral_or_unreported_studies = deduplicate(global_neutral_or_unreported_studies)

        # Ensure all PMCIDs are covered
        covered_pmcids = set()
        for lst in [supporting_studies, conflicting_studies, neutral_or_unreported_studies]:
            for item in lst:
                if isinstance(item, dict) and "pmcid" in item:
                    covered_pmcids.add(item["pmcid"])

        for pmcid in all_pmcids:
            if pmcid not in covered_pmcids:
                # Add default entry for uncovered studies
                neutral_or_unreported_studies.append({
                    "pmcid": pmcid,
                    "evidence": "No relevant data found for this study."
                })
                covered_pmcids.add(pmcid) 

        # The final return structure for 'summarize' is unique
        return {
            "synthesis_question": synthesis_question,
            "tool_type": tool_type_from_map,
            "supporting_studies": supporting_studies,
            "conflicting_studies": conflicting_studies,
            "neutral_or_unreported_studies": neutral_or_unreported_studies
        }

    else:
        # === FIXED NON-summarize MERGE ===
        all_entries = []
        for pt in partial_tools:
            # Note: We rely on the LLM to have returned entries for *only* the PMCIDs in the batch
            # and that they have the correct schema fields.
            all_entries.extend(pt.get("entries", []))

        # Build map from pmcid to entry (for deduplication and coverage check)
        entry_dict = {}
        for e in all_entries:
            if isinstance(e, dict) and "pmcid" in e:
                pmcid = e["pmcid"]
                
                # Apply normalization and default values on the merged list
                if tool_type_from_map == "binary":
                    val = str(e.get("value", "")).strip().lower()
                    if val in ["yes", "1"]:
                        e.update({
                            "value": "yes",
                            "numerical value": "1"
                        })
                    elif val in ["no", "0"]:
                        e.update({
                            "value": "no",
                            "numerical value": "0"
                        })
                    else:
                        e.update({
                            "value": "not reported",
                            "numerical value": "0"
                        })
                    e.setdefault("evidence", "Not reported")
                    
                elif tool_type_from_map == "numeric":
                    # For numeric, we rely on the LLM's structured pipe-separated value if multiple are extracted
                    # No complex merge logic is needed here as entries are merged via the dict.
                    e.setdefault("value", "not reported")
                    # Set default unit based on the special case instruction
                    if "|" in str(e.get("value", "")):
                        e.setdefault("unit", "extracted_values")
                    else:
                        e.setdefault("unit", "unknown")
                    e.setdefault("evidence", "Not reported")
                    
                elif tool_type_from_map == "categorical":
                    e.setdefault("value", "not reported")
                    e["unit"] = "categorical"
                    e.setdefault("evidence", "Not reported")
                    
                elif tool_type_from_map == "descriptive":
                    e.setdefault("value", "not reported")
                    e.setdefault("evidence", "Not reported")
                    
                # Use the last entry seen for a PMCID (simple deduplication)
                entry_dict[pmcid] = e

        # Ensure all PMCIDs are present with proper defaults
        final_entries = []
        for pmcid in all_pmcids:
            if pmcid in entry_dict:
                final_entries.append(entry_dict[pmcid])
            else:
                # Create a default "not reported" entry if missing from all batches
                default_entry = {"pmcid": pmcid, "evidence": "Not reported"}
                if tool_type_from_map == "binary":
                    default_entry.update({"value": "not reported", "numerical value": "0"})
                elif tool_type_from_map == "numeric":
                    default_entry.update({"value": "not reported", "unit": "unknown"})
                elif tool_type_from_map == "categorical":
                    default_entry.update({"value": "not reported", "unit": "categorical"})
                elif tool_type_from_map == "descriptive":
                    default_entry.update({"value": "not reported"})
                
                final_entries.append(default_entry)

        return {
            "synthesis_question": synthesis_question, 
            "tool_type": tool_type_from_map,
            "entries": final_entries
        }

def build_relevance_map(
    analysis_questions: List[str],           
    synthesis_questions: List[str],
    llm_client: LLMClient,
    max_relevant_per_synthesis: int = 5
) -> Dict[str, List[str]]:
    """
    Maps each synthesis question to the per-study analysis questions 
    that would be needed to answer it — without any answer data.
    
    This must be called BEFORE any study processing.
    """
    if not analysis_questions:
        logger.error("No analysis questions provided to build_relevance_map.")
        return {q: [] for q in synthesis_questions}

    def normalize(q: str) -> str:
        return re.sub(r'[^\w\s]', '', q.lower().strip())

    lower_to_original = {normalize(q): q for q in analysis_questions}

    prompt = f"""
You are a clinical research information architect.
Your task is to determine which per-study analysis questions are needed 
to answer each high-level synthesis question.

PER-STUDY ANALYSIS QUESTIONS:
{chr(10).join([f"{i+1}. {q}" for i, q in enumerate(analysis_questions)])}

SYNTHESIS QUESTIONS:
{chr(10).join([f"{chr(65+i)}. {q}" for i, q in enumerate(synthesis_questions)])}

INSTRUCTIONS:
- Return a JSON object: {{ "Synthesis Question": ["Exact Analysis Q 1", "Exact Analysis Q 2", ...] }}
- Use the **exact, full text** of the analysis questions — copy them verbatim.
- Do NOT use numbers, letters (A., B.), or abbreviate.
- Include only analysis questions that provide **direct evidence** for answering the synthesis question.
- Max {max_relevant_per_synthesis} relevant analysis questions per synthesis question.
- If no analysis question is relevant, return an empty list.
- In the JSON, use only the **full synthesis question text** as the key — 
  do NOT include the letter/label prefix (e.g., use 
  "Summarize the most common..." instead of "A. Summarize...").

EXAMPLE:
{{
  "Which tissue source yielded better outcomes?": [
    "What was the tissue source of the MSCs?",
    "What were the primary and secondary outcomes?",
    "Was the primary endpoint achieved?"
  ]
}}
""".strip()

    try:
        raw = llm_client.generate_text(prompt)
        # logger.debug(f"Relevance Map LLM Prompt:\n{prompt}")
        # logger.debug(f"Relevance Map Raw Response:\n{raw}")
        # print(raw)
        
        raw_clean = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip(), flags=re.MULTILINE)
        json_match = re.search(r'\{.*\}', raw_clean, re.DOTALL)
        if not json_match:
            logger.warning("No JSON found in cleaned response")
            return {q: [] for q in synthesis_questions}

        raw_map = json.loads(json_match.group())

        # Step 3: Clean LLM keys by removing "A. ", "B. ", etc.
        cleaned_raw_map = {}
        for key, value in raw_map.items():
            if isinstance(key, str):
                # Remove leading letter/bracket + dot + space
                clean_key = re.sub(r'^[A-Z\[\]\\]\.]\s*', '', key)
                cleaned_raw_map[clean_key] = value
            else:
                logger.warning(f"Non-string key in LLM response: {key}")
                
        print(cleaned_raw_map)

        result = {}
        for synth_q in synthesis_questions:
            suggested_list = cleaned_raw_map.get(synth_q, [])
            if not isinstance(suggested_list, list):
                suggested_list = []

            tool_type = select_tool_type(synth_q, llm_client)  

            matched_questions = []
            for suggested_q in suggested_list:
                if not isinstance(suggested_q, str) or not suggested_q.strip():
                    continue
                norm_suggested = normalize(suggested_q)
                if norm_suggested in lower_to_original:
                    matched_questions.append(lower_to_original[norm_suggested])
                else:
                    words_suggested = set(norm_suggested.split())
                    best_match = None
                    max_overlap = 0
                    for norm_orig, orig_q in lower_to_original.items():
                        words_orig = set(norm_orig.split())
                        overlap = len(words_suggested & words_orig)
                        if overlap > max_overlap and overlap >= 2:
                            max_overlap = overlap
                            best_match = orig_q
                    if best_match:
                        matched_questions.append(best_match)
                    else:
                        logger.debug(f"⚠️ No match for suggested Q: '{suggested_q}'")

            seen = set()
            unique_questions = []
            for q in matched_questions:
                if q not in seen:
                    unique_questions.append(q)
                    seen.add(q)

            result[synth_q] = {
                "tool_type": tool_type,
                "relevant_questions": unique_questions[:max_relevant_per_synthesis]
    }

        # Log results
        for synth_q, matches in result.items():
            if matches:
                logger.info(f"🔗 '{synth_q}' → {len(matches)} relevant analysis questions")
            else:
                logger.warning(f"❌ '{synth_q}' → no relevant analysis questions found")

        return result

    except Exception as e:
        logger.error(f"Relevance mapping failed: {e}")
        logger.debug(f"Raw response: {raw}")
        return {q: [] for q in synthesis_questions}
    


def synthesize_from_tools(
    evidence_tools: Dict[str, Dict],
    bias_by_pmcid: Dict[str, Dict],
    llm_client: LLMClient
) -> Dict[str, Any]:
    """
    Generate scientifically styled, manuscript-ready summaries for each synthesis question.
    ALL tool types are handled by the LLM — no hardcoded math or logic.
    Output includes: answer, confidence, evidence_summary, bias_assessment.
    Citations include PMCIDs with brief bias notes.
    """
    full_result = {}

    # Precompute overall bias context for reference
    total_studies = len(bias_by_pmcid)
    bias_counts = {"Low": 0, "Moderate": 0, "Some concerns": 0, "Serious": 0, "High": 0, "Critical": 0, "Not Scored": 0}
    for ba in bias_by_pmcid.values():
        # Ensure we are getting the judgment, with 'Not Scored' as a default if missing
        judgment = ba.get("judgment", "Not Scored")
        # Capitalize or normalize to match keys, assuming original keys were correct.
        judgment = judgment.strip().title() if isinstance(judgment, str) else "Not Scored" 
        
        # Mapping to the defined categories, being robust to slight variations
        if 'low' in judgment.lower():
             bias_counts['Low'] += 1
        elif 'moderate' in judgment.lower():
             bias_counts['Moderate'] += 1
        elif 'some concerns' in judgment.lower():
             bias_counts['Some concerns'] += 1
        elif 'serious' in judgment.lower():
             bias_counts['Serious'] += 1
        elif 'high' in judgment.lower():
             bias_counts['High'] += 1
        elif 'critical' in judgment.lower():
             bias_counts['Critical'] += 1
        else:
             bias_counts['Not Scored'] += 1

    overall_bias_summary = (
        f"Overall Bias Distribution Across {total_studies} Studies:\n"
        f"  Low: {bias_counts['Low']}\n"
        f"  Moderate: {bias_counts['Moderate']}\n"
        f"  Some concerns: {bias_counts['Some concerns']}\n"
        f"  Serious: {bias_counts['Serious']}\n"
        f"  High: {bias_counts['High']}\n"
        f"  Critical: {bias_counts['Critical']}\n"
        f"  Not Scored: {bias_counts['Not Scored']}\n\n"
    )

    for synth_q, tool in evidence_tools.items():
        tool_type = tool.get("tool_type", "summarize")
        
        # Default lengths
        min_words_summary = 600
        initial_answer_desc = "Begin with a very brief, one-sentence answer to the synthesis question."
        
        # --- Common Citation Formatting Helper ---
        def format_citations(studies_list):
            if not studies_list:
                return []
            return [
                f"[{s.get('pmcid', 'Unknown')}] (bias: {bias_by_pmcid.get(s.get('pmcid'), {}).get('judgment', 'Unknown')}) "
                f"Evidence: {s.get('evidence', 'No evidence provided')}"
                for s in studies_list if isinstance(s, dict)
            ]

        # --- Tool-Specific Data Extraction and Prompt Setup ---
        tool_evidence_section = ""
        
        if tool_type == "summarize":
            supporting = tool.get("supporting_studies", [])
            conflicting = tool.get("conflicting_studies", [])
            neutral = tool.get("neutral_or_unreported_studies", [])
            
            supporting_citations = format_citations(supporting)
            conflicting_citations = format_citations(conflicting)
            neutral_citations = format_citations(neutral)
            
            tool_evidence_section = f"""
SUPPORTING EVIDENCE (N={len(supporting)}):
{chr(10).join(supporting_citations) if supporting_citations else "None."}

CONFLICTING EVIDENCE (N={len(conflicting)}):
{chr(10).join(conflicting_citations) if conflicting_citations else "None."}

NEUTRAL/UNREPORTED EVIDENCE (N={len(neutral)}):
{chr(10).join(neutral_citations) if neutral_citations else "None."}
"""
            initial_answer_desc = "Begin with a very brief, one-sentence answer to the synthesis question."
            min_words_summary = 600
            
        elif tool_type == "descriptive":
            entries = tool.get("entries", [])
            citations = [
                f"[{e.get('pmcid', 'Unknown')}] (bias: {bias_by_pmcid.get(e.get('pmcid'), {}).get('judgment', 'Unknown')}) "
                f"Description: {e.get('value', 'Not reported')} (Evidence: {e.get('evidence', 'No evidence provided')})"
                for e in entries if isinstance(e, dict)
            ] if entries else ["No data available."]
            
            tool_evidence_section = f"""
DESCRIPTIVE EVIDENCE (N={len(entries)}):
{chr(10).join(citations)}
"""
            initial_answer_desc = "Begin with a very brief, one-sentence thematic summary of the descriptions."
            min_words_summary = 600
            
        elif tool_type in ["binary", "numeric", "categorical"]:
            entries = tool.get("entries", [])
            
            # Binary/Numeric/Categorical tools need a brief *numerical* answer first
            min_words_summary = 200
            
            if tool_type == "binary":
                initial_answer_desc = "Begin with a clear, brief, numerical answer (e.g., 'X studies out of Y used Z' or 'X% of studies used Z')."
                citations = [
                    f"[{e.get('pmcid', 'Unknown')}] (bias: {bias_by_pmcid.get(e.get('pmcid'), {}).get('judgment', 'Unknown')}) "
                    f"Value: {e.get('value', 'Not reported')} (Evidence: {e.get('evidence', 'No evidence provided')})"
                    for e in entries if isinstance(e, dict)
                ]
            
            elif tool_type == "numeric":
                initial_answer_desc = "Begin with a clear, brief, numerical answer (e.g., 'The mean value was X ± Y' or 'The range was A to B')."
                citations = [
                    f"[{e.get('pmcid', 'Unknown')}] (bias: {bias_by_pmcid.get(e.get('pmcid'), {}).get('judgment', 'Unknown')}) "
                    f"Value: {e.get('value', 'Not reported')} {e.get('unit', '')} (Evidence: {e.get('evidence', 'No evidence provided')})"
                    for e in entries if isinstance(e, dict)
                ]
                
            elif tool_type == "categorical":
                initial_answer_desc = "Begin with a clear, brief, numerical answer (e.g., 'The most common category was X, reported in Y studies')."
                citations = [
                    f"[{e.get('pmcid', 'Unknown')}] (bias: {bias_by_pmcid.get(e.get('pmcid'), {}).get('judgment', 'Unknown')}) "
                    f"Category: {e.get('value', 'Not reported')} (Evidence: {e.get('evidence', 'No evidence provided')})"
                    for e in entries if isinstance(e, dict)
                ]

            tool_evidence_section = f"""
EVIDENCE (N={len(entries)}):
{chr(10).join(citations) if citations else "No data available."}
"""
        
        else: # Unknown tool type fallback
            tool_evidence_section = "\nEVIDENCE:\nNo structured evidence available."
            initial_answer_desc = "Begin with a clear, brief answer to the synthesis question based on available information."
            min_words_summary = 600

        # --- Build Universal LLM Prompt ---
        prompt = f"""
You are a clinical research author writing a scientific manuscript.
Generate a comprehensive, evidence-based summary for the following synthesis question.
Write the summary in **continuous prose**, suitable for direct inclusion in a peer-reviewed journal.
Cite specific PMCIDs inline with brief bias annotations (e.g., [PMC1234567] (bias: Low)).
Do NOT use bullet points, numbered lists, or markdown formatting *within the prose summary*.

SYNTHESIS QUESTION:
{synth_q}

OVERALL BIAS CONTEXT:
{overall_bias_summary}

{tool_evidence_section}

INSTRUCTIONS FOR PROSE SUMMARY:
- {initial_answer_desc}
- Then provide a detailed narrative synthesizing the evidence.
- Discuss patterns, contradictions, strengths, and limitations.
- Cite PMCIDs inline with brief bias notes (e.g., [PMC1234567] (bias: Low)).
- Do NOT make bias the central theme — it is a contextual footnote.
- Ensure the **prose summary** is at least {min_words_summary} words long.
- Write as if this will be copied directly into a scientific manuscript.

INSTRUCTIONS FOR STRUCTURED FIELDS (MUST APPEAR AT THE END):
After your continuous prose summary, you MUST include the following three fields on their own lines, exactly as shown, for machine parsing:

[SYNTHESIS_ANSWER]Provide the final, brief, numerical (if binary/numeric/categorical) or one-sentence (if descriptive/summarize) answer here.[/SYNTHESIS_ANSWER]
[CONFIDENCE]Assess the confidence (e.g., High, Moderate, Low, Very Low) based on data patterns (consistency, magnitude, sample size, statistical strength).[/CONFIDENCE]
[BIAS_ASSESSMENT]Evaluate how the bias levels of the supporting studies affect the reliability of the answer. Use the data in OVERALL BIAS CONTEXT and the bias notes in the EVIDENCE section to inform this short paragraph.[/BIAS_ASSESSMENT]
"""

        # === CALL LLM ===
        try:
            raw = llm_client.generate_text(prompt)

            cleaned_raw = raw.strip() 
            
            # Extract structured fields (as per prompt)
            answer_match = re.search(r'\[SYNTHESIS_ANSWER\](.*?)\[/SYNTHESIS_ANSWER\]', cleaned_raw, re.IGNORECASE | re.DOTALL)
            conf_match = re.search(r'\[CONFIDENCE\](.*?)\[/CONFIDENCE\]', cleaned_raw, re.IGNORECASE)
            bias_match = re.search(r'\[BIAS_ASSESSMENT\](.*?)\[/BIAS_ASSESSMENT\]', cleaned_raw, re.IGNORECASE | re.DOTALL)
            
            # Assign values
            answer = answer_match.group(1).strip() if answer_match else "See detailed summary below."
            confidence = conf_match.group(1).strip() if conf_match else "Moderate"
            bias_assessment = bias_match.group(1).strip() if bias_match else "No bias assessment provided."
            evidence_summary = cleaned_raw
            if bias_match:
                 start_of_fields = min([m.start() for m in [answer_match, conf_match, bias_match] if m])
                 if start_of_fields > 0:
                     evidence_summary = cleaned_raw[:start_of_fields].strip()
            
            # Basic fallback 
            if len(evidence_summary.split()) < min_words_summary * 0.5: # Check for gross failure (50% of min_words)
                 evidence_summary += "\n\n[Elaboration required] The synthesis needs further detail. The collective evidence, when analyzed for heterogeneity and consistency, suggests a need for cautious interpretation. Outliers and studies with high risk of bias introduce uncertainty, tempering the final conclusion."

            full_result[synth_q] = {
                "answer": answer,
                "confidence": confidence,
                "evidence_summary": evidence_summary,
                "bias_assessment": bias_assessment
            }

        except Exception as e:
            logger.error(f"LLM synthesis failed for '{synth_q}': {e}")
            full_result[synth_q] = {
                "answer": "Insufficient evidence",
                "confidence": "Very Low",
                "evidence_summary": f"Error: LLM call failed or response was unparseable. Details: {str(e)}",
                "bias_assessment": "Error occurred during synthesis; bias assessment not available."
            }

    return full_result

def process_study_comprehensive(
    pmcid: str,
    markdown_content: str,
    inclusion_criteria: str,
    exclusion_criteria: str,
    analysis_questions: list,
    publication_types: list,
    llm_client: LLMClient
) -> dict:
    context = markdown_content[:800000]
    is_rct = any(pt.lower() in ['randomized controlled trial', 'randomized clinical trial'] for pt in publication_types)
    
    # === 1. Load correct bias framework ===
    if is_rct:
        framework = "Cochrane RoB 2"
        rules = get_bias_criteria()["ROB2_SIGNALING_QUESTIONS"]
        from clinical_trials_agent.bias_scoring_rob_2 import score_rob2_domain_scores, get_overall_rob2_judgment
    else:
        framework = "Cochrane ROBINS-I"
        rules = get_bias_criteria()["ROBINS_I_SIGNALING_QUESTIONS"]
        from clinical_trials_agent.bias_scoring_robins_i import score_robins_i_domain_scores, get_overall_robins_i_judgment
        
    if is_rct:
        # === RoB 2: Response Definitions ===
        response_instructions = """
    ### A. For Signaling Questions (RoB 2 - RCTs):
    Use **ONLY** these responses:

    - **Yes**: The study clearly reports that the criterion was met (e.g., "allocation was concealed using sealed envelopes").
    - **Probably Yes**: It is very likely the criterion was met, even if not explicitly stated (e.g., study from a reputable trials unit with no red flags).
    - **Probably No**: It is likely the criterion was **not** met, but not definitively (e.g., no mention of blinding in a subjective outcome study).
    - **No**: The study clearly fails the criterion (e.g., "participants were aware of treatment allocation").
    - **No Information**: The study does **not provide enough detail** to judge. Do **NOT** use "Not reported", "Unknown", or "NR".

    These responses assess whether the **randomized trial design** was properly implemented and analyzed.
    """
    else:
        # === ROBINS-I V2: Response Definitions ===
        response_instructions = """
    ### A. For Signaling Questions (ROBINS-I V2 - Non-RCTs):
    Use **ONLY** these responses:

    - **Y / PY / N / PN / NI**: As defined in RoB 2 for general clarity.
    - **WY**: Yes, but the impact was **not substantial** → Moderate risk
    - **SY**: Yes, and the impact was **substantial** → Critical risk
    - **WN**: No, but the extent of the problem was **not substantial** → Moderate risk
    - **SN**: No, and the extent of the problem was **substantial** → Serious/Critical risk

    These special codes distinguish **degree of bias** in non-randomized studies:
    - Use `WY`/`WN` when a flaw exists but is minor
    - Use `SY`/`SN` when a flaw is severe and likely biased the result
    - Example: "Were confounders measured reliably?" → `SN` if poorly measured and likely distorted results

    Do **NOT** use "Not reported", "Unknown", or "NR". Use `NI` instead.
    """

    # === 2. Extract signaling questions from rules ===
    signaling_questions = {}
    for domain_info in rules["bias_domains"].values():
        if "signaling_questions" in domain_info:
            signaling_questions.update(domain_info["signaling_questions"])
        elif "components" in domain_info:
            for comp_info in domain_info["components"].values():
                signaling_questions.update(comp_info["signaling_questions"])

    # === 3. Build prompt with Q1, Q2, Q3... for analysis questions ===
    # Create mapping: Q1 → original question
    question_map = {}
    formatted_analysis_questions = []
    for i, q in enumerate(analysis_questions, 1):
        q_key = f"Q{i}"
        formatted_analysis_questions.append(f"{q_key}: {q}")
        question_map[q_key] = q  # store for later mapping back

    formatted_signaling_questions = "\n".join([f"{qid}: {qtext}" for qid, qtext in signaling_questions.items()])
    formatted_analysis_questions_str = "\n".join(formatted_analysis_questions)

    prompt = f"""
You are a clinical research expert. Extract answers to the following questions from the study.

## TASKS
1. **Eligibility**: Does the study meet inclusion/exclusion criteria?
2. **Signaling Questions**: Answer using the strict RoB 2 or ROBINS-I response set.
3. **Analysis Questions**: Answer each per-study question using ONLY the keys Q1, Q2, Q3, ... as specified below.

---

### STUDY (PMCID: {pmcid})
{context}

---

### INCLUSION CRITERIA
{inclusion_criteria if inclusion_criteria.strip() else '(None)'}

### EXCLUSION CRITERIA
{exclusion_criteria if exclusion_criteria.strip() else '(None)'}

### ANALYSIS QUESTIONS
{formatted_analysis_questions_str}

### {framework} SIGNALING QUESTIONS
{formatted_signaling_questions}

---

## INSTRUCTIONS

{response_instructions}

### B. For Analysis Questions:
- Answer fully and factually.
- Use **ONLY** these exact keys in the `"answers"` object: `Q1`, `Q2`, `Q3`, ..., up to `Q{len(analysis_questions)}`.
- Do **NOT** use the full question text as the key.
- Use **"Not reported"** only if the information is missing.
- Example: `"Q1": "Adults aged 18–65 with type 2 diabetes..."`

---

Return a single JSON object with this structure:
{{
  "eligibility": {{
    "eligible": true|false,
    "reason": "Brief justification"
  }},
  "signaling_answers": {{
    "Q1.1": "...", "Q1.2": "...", ..., 
    "Q7.4": "..."
  }},
  "answers": {{
    "Q1": "...",
    "Q2": "...",
    ...
  }}
}}

- Be factual. Use "Not reported" if missing.
- Return ONLY valid JSON. No explanations.
"""

    try:
        raw_response = llm_client.generate_text(prompt)
        parsed = extract_json_from_text(raw_response)
        if not parsed:
            raise ValueError("Failed to parse LLM response")

        # === CHECK ELIGIBILITY FIRST ===
        eligibility = parsed.get("eligibility", {"eligible": False, "reason": "Missing"})
        if not eligibility.get("eligible"):
            # ❌ Study is ineligible — skip all further processing
            return {
                "metadata": {"pmcid": pmcid, "status": "completed"},
                "eligibility": eligibility,
                "bias_assessment": {},  # empty
                "answers": {}  # empty
            }

        # ✅ Eligible — proceed with bias scoring and full output
        signaling_answers = parsed.get("signaling_answers", {})

        # --- Normalize analysis answers using question_map ---
        parsed_answers = parsed.get("answers", {})
        normalized_answers = {}

        for q_key, answer in parsed_answers.items():
            # Clean key: handle case/spaces (e.g., "q1", "Q 1", "Q1 ")
            clean_key = q_key.strip().upper().replace(' ', '')
            if clean_key in question_map:
                normalized_answers[question_map[clean_key]] = answer
            else:
                logger.warning(f"LLM returned unknown key: '{q_key}' for {pmcid}")

        # Ensure all original questions are present (even if LLM skipped some)
        for orig_q in analysis_questions:
            if orig_q not in normalized_answers:
                normalized_answers[orig_q] = "Not reported"

        # Replace raw answers with normalized ones
        parsed["answers"] = normalized_answers

        # === Bias Scoring ===
        if is_rct:
            domain_scores = score_rob2_domain_scores(signaling_answers)
            total_bias_score = get_overall_rob2_judgment(domain_scores)
        else:
            domain_scores = score_robins_i_domain_scores(signaling_answers)
            total_bias_score = get_overall_robins_i_judgment(domain_scores)

        # Map judgment to numeric score for downstream use
        score_map = {"Low": 1, "Moderate": 2, "Some concerns": 2, "Serious": 3, "High": 3, "Critical": 4}
        numeric_score = score_map.get(total_bias_score, 3)

        return {
            "metadata": {"pmcid": pmcid, "status": "completed"},
            "eligibility": eligibility,
            "bias_assessment": {
                "framework_used": framework,
                "signaling_answers": signaling_answers,
                "domain_scores": domain_scores,
                "judgment": total_bias_score,
                "total_bias_score": numeric_score
            },
            "answers": parsed.get("answers", {})  # already normalized
        }

    except Exception as e:
        return {
            "metadata": {"pmcid": pmcid, "status": "error", "error": str(e)},
            "eligibility": {"eligible": False, "reason": f"Processing failed: {str(e)}"},
            "bias_assessment": {},
            "answers": {}
        }


# def process_study_comprehensive(
#     pmcid: str,
#     markdown_content: str,
#     inclusion_criteria: str,
#     exclusion_criteria: str,
#     analysis_questions: list,
#     publication_types: list,
#     llm_client: LLMClient
# ) -> dict:
#     context = markdown_content[:800000]
#     is_rct = any(pt.lower() in ['randomized controlled trial', 'randomized clinical trial'] for pt in publication_types)
    
#     # === 1. Load correct bias framework ===
#     if is_rct:
#         framework = "Cochrane RoB 2"
#         rules = get_bias_criteria()["ROB2_SIGNALING_QUESTIONS"]
#         from clinical_trials_agent.bias_scoring_rob_2 import score_rob2_domain_scores, get_overall_rob2_judgment
#     else:
#         framework = "Cochrane ROBINS-I"
#         rules = get_bias_criteria()["ROBINS_I_SIGNALING_QUESTIONS"]
#         from clinical_trials_agent.bias_scoring_robins_i import score_robins_i_domain_scores, get_overall_robins_i_judgment
        
#     if is_rct:
#         # === RoB 2: Response Definitions ===
#         response_instructions = """
#     ### A. For Signaling Questions (RoB 2 - RCTs):
#     Use **ONLY** these responses:

#     - **Yes**: The study clearly reports that the criterion was met (e.g., "allocation was concealed using sealed envelopes").
#     - **Probably Yes**: It is very likely the criterion was met, even if not explicitly stated (e.g., study from a reputable trials unit with no red flags).
#     - **Probably No**: It is likely the criterion was **not** met, but not definitively (e.g., no mention of blinding in a subjective outcome study).
#     - **No**: The study clearly fails the criterion (e.g., "participants were aware of treatment allocation").
#     - **No Information**: The study does **not provide enough detail** to judge. Do **NOT** use "Not reported", "Unknown", or "NR".

#     These responses assess whether the **randomized trial design** was properly implemented and analyzed.
#     """
#     else:
#         # === ROBINS-I V2: Response Definitions ===
#         response_instructions = """
#     ### A. For Signaling Questions (ROBINS-I V2 - Non-RCTs):
#     Use **ONLY** these responses:

#     - **Y / PY / N / PN / NI**: As defined in RoB 2 for general clarity.
#     - **WY**: Yes, but the impact was **not substantial** → Moderate risk
#     - **SY**: Yes, and the impact was **substantial** → Critical risk
#     - **WN**: No, but the extent of the problem was **not substantial** → Moderate risk
#     - **SN**: No, and the extent of the problem was **substantial** → Serious/Critical risk

#     These special codes distinguish **degree of bias** in non-randomized studies:
#     - Use `WY`/`WN` when a flaw exists but is minor
#     - Use `SY`/`SN` when a flaw is severe and likely biased the result
#     - Example: "Were confounders measured reliably?" → `SN` if poorly measured and likely distorted results

#     Do **NOT** use "Not reported", "Unknown", or "NR". Use `NI` instead.
#     """

#     # === 2. Extract signaling questions from rules ===
#     signaling_questions = {}
#     for domain_info in rules["bias_domains"].values():
#         if "signaling_questions" in domain_info:
#             signaling_questions.update(domain_info["signaling_questions"])
#         elif "components" in domain_info:
#             for comp_info in domain_info["components"].values():
#                 signaling_questions.update(comp_info["signaling_questions"])

#     # === 3. Build prompt for fact extraction only ===
#     formatted_analysis_questions = "\n".join([f"- {q}" for q in analysis_questions])
#     formatted_signaling_questions = "\n".join([f"{qid}: {qtext}" for qid, qtext in signaling_questions.items()])

#     prompt = f"""
# You are a clinical research expert. Extract answers to the following questions from the study.

# ## TASKS
# 1. **Eligibility**: Does the study meet inclusion/exclusion criteria?
# 2. **Signaling Questions**: Answer using the strict RoB 2 or ROBINS-I response set.
# 3. **Analysis Questions**: Answer each per-study question.

# ---

# ### STUDY (PMCID: {pmcid})
# {context}

# ---

# ### INCLUSION CRITERIA
# {inclusion_criteria if inclusion_criteria.strip() else '(None)'}

# ### EXCLUSION CRITERIA
# {exclusion_criteria if exclusion_criteria.strip() else '(None)'}

# ### ANALYSIS QUESTIONS
# {formatted_analysis_questions}

# ### {framework} SIGNALING QUESTIONS
# {formatted_signaling_questions}

# ---

# ## INSTRUCTIONS

# {response_instructions}

# ### B. For Analysis Questions:
# - Answer fully and factually.
# - Use **"Not reported"** only if the information is missing.

# ---

# Return a single JSON object with this structure:
# {{
#   "eligibility": {{
#     "eligible": true|false,
#     "reason": "Brief justification"
#   }},
#   "signaling_answers": {{
#     "Q1.1": "...", "Q1.2": "...", ..., 
#     "Q7.4": "..."
#   }},
#   "answers": {{
#     "What was the patient population?": "...",
#     ...
#   }}
# }}

# - Use exact question strings as keys.
# - Be factual. Use "Not reported" if missing.
# - Return ONLY valid JSON. No explanations.
# """

#     try:
#         raw_response = llm_client.generate_text(prompt)
#         parsed = extract_json_from_text(raw_response)
#         if not parsed:
#             raise ValueError("Failed to parse LLM response")

#         # === CHECK ELIGIBILITY FIRST ===
#         eligibility = parsed.get("eligibility", {"eligible": False, "reason": "Missing"})
#         if not eligibility.get("eligible"):
#             # ❌ Study is ineligible — skip all further processing
#             return {
#                 "metadata": {"pmcid": pmcid, "status": "completed"},
#                 "eligibility": eligibility,
#                 "bias_assessment": {},  # empty
#                 "answers": {}  # empty
#             }

#         # ✅ Eligible — proceed with bias scoring and full output
#         signaling_answers = parsed.get("signaling_answers", {})

#         if is_rct:
#             domain_scores = score_rob2_domain_scores(signaling_answers)
#             total_bias_score = get_overall_rob2_judgment(domain_scores)
#         else:
#             domain_scores = score_robins_i_domain_scores(signaling_answers)
#             total_bias_score = get_overall_robins_i_judgment(domain_scores)

#         # Map judgment to numeric score for downstream use
#         score_map = {"Low": 1, "Moderate": 2, "Some concerns": 2, "Serious": 3, "High": 3, "Critical": 4}
#         numeric_score = score_map.get(total_bias_score, 3)

#         return {
#             "metadata": {"pmcid": pmcid, "status": "completed"},
#             "eligibility": eligibility,
#             "bias_assessment": {
#                 "framework_used": framework,
#                 "signaling_answers": signaling_answers,
#                 "domain_scores": domain_scores,
#                 "judgment": total_bias_score,
#                 "total_bias_score": numeric_score
#             },
#             "answers": parsed.get("answers", {})
#         }

#     except Exception as e:
#         return {
#             "metadata": {"pmcid": pmcid, "status": "error", "error": str(e)},
#             "eligibility": {"eligible": False, "reason": f"Processing failed: {str(e)}"},
#             "bias_assessment": {},
#             "answers": {}
#         }

# def process_study_comprehensive(
#     pmcid: str,
#     markdown_content: str,
#     inclusion_criteria: str,
#     exclusion_criteria: str,
#     analysis_questions: list,
#     publication_types: list,
#     llm_client: LLMClient
# ) -> dict:
#     context = markdown_content[:800000]
#     is_rct = any(pt.lower() in ['randomized controlled trial', 'randomized clinical trial'] for pt in publication_types)
    
#     # === 1. Load correct bias framework ===
#     if is_rct:
#         framework = "Cochrane RoB 2"
#         rules = get_bias_criteria()["ROB2_SIGNALING_QUESTIONS"]
#         from clinical_trials_agent.bias_scoring_rob_2 import score_rob2_domain_scores, get_overall_rob2_judgment
#     else:
#         framework = "Cochrane ROBINS-I"
#         rules = get_bias_criteria()["ROBINS_I_SIGNALING_QUESTIONS"]
#         from clinical_trials_agent.bias_scoring_robins_i import score_robins_i_domain_scores, get_overall_robins_i_judgment
        
#     if is_rct:
#         # === RoB 2: Response Definitions ===
#         response_instructions = """
#     ### A. For Signaling Questions (RoB 2 - RCTs):
#     Use **ONLY** these responses:

#     - **Yes**: The study clearly reports that the criterion was met (e.g., "allocation was concealed using sealed envelopes").
#     - **Probably Yes**: It is very likely the criterion was met, even if not explicitly stated (e.g., study from a reputable trials unit with no red flags).
#     - **Probably No**: It is likely the criterion was **not** met, but not definitively (e.g., no mention of blinding in a subjective outcome study).
#     - **No**: The study clearly fails the criterion (e.g., "participants were aware of treatment allocation").
#     - **No Information**: The study does **not provide enough detail** to judge. Do **NOT** use "Not reported", "Unknown", or "NR".

#     These responses assess whether the **randomized trial design** was properly implemented and analyzed.
#     """
#     else:
#         # === ROBINS-I V2: Response Definitions ===
#         response_instructions = """
#     ### A. For Signaling Questions (ROBINS-I V2 - Non-RCTs):
#     Use **ONLY** these responses:

#     - **Y / PY / N / PN / NI**: As defined in RoB 2 for general clarity.
#     - **WY**: Yes, but the impact was **not substantial** → Moderate risk
#     - **SY**: Yes, and the impact was **substantial** → Critical risk
#     - **WN**: No, but the extent of the problem was **not substantial** → Moderate risk
#     - **SN**: No, and the extent of the problem was **substantial** → Serious/Critical risk

#     These special codes distinguish **degree of bias** in non-randomized studies:
#     - Use `WY`/`WN` when a flaw exists but is minor
#     - Use `SY`/`SN` when a flaw is severe and likely biased the result
#     - Example: "Were confounders measured reliably?" → `SN` if poorly measured and likely distorted results

#     Do **NOT** use "Not reported", "Unknown", or "NR". Use `NI` instead.
#     """

#     # === 2. Extract signaling questions from rules ===
#     signaling_questions = {}
#     for domain_info in rules["bias_domains"].values():
#         if "signaling_questions" in domain_info:
#             signaling_questions.update(domain_info["signaling_questions"])
#         elif "components" in domain_info:
#             for comp_info in domain_info["components"].values():
#                 signaling_questions.update(comp_info["signaling_questions"])

#     # === 3. Build prompt for fact extraction only ===
#     formatted_analysis_questions = "\n".join([f"- {q}" for q in analysis_questions])
#     formatted_signaling_questions = "\n".join([f"{qid}: {qtext}" for qid, qtext in signaling_questions.items()])

#     prompt = f"""
# You are a clinical research expert. Extract answers to the following questions from the study.

# ## TASKS
# 1. **Eligibility**: Does the study meet inclusion/exclusion criteria?
# 2. **Signaling Questions**: Answer using the strict RoB 2 or ROBINS-I response set.
# 3. **Analysis Questions**: Answer each per-study question.

# ---

# ### STUDY (PMCID: {pmcid})
# {context}

# ---

# ### INCLUSION CRITERIA
# {inclusion_criteria if inclusion_criteria.strip() else '(None)'}

# ### EXCLUSION CRITERIA
# {exclusion_criteria if exclusion_criteria.strip() else '(None)'}

# ### ANALYSIS QUESTIONS
# {formatted_analysis_questions}

# ### {framework} SIGNALING QUESTIONS
# {formatted_signaling_questions}

# ---

# ## INSTRUCTIONS

# {response_instructions}

# ### B. For Analysis Questions:
# - Answer fully and factually.
# - Use **"Not reported"** only if the information is missing.

# ---

# Return a single JSON object with this structure:
# {{
#   "eligibility": {{
#     "eligible": true|false,
#     "reason": "Brief justification"
#   }},
#   "signaling_answers": {{
#     "Q1.1": "...", "Q1.2": "...", ..., 
#     "Q7.4": "..."
#   }},
#   "answers": {{
#     "What was the patient population?": "...",
#     ...
#   }}
# }}

# - Use exact question strings as keys.
# - Be factual. Use "Not reported" if missing.
# - Return ONLY valid JSON. No explanations.
# """

#     try:
#         raw_response = llm_client.generate_text(prompt)
#         parsed = extract_json_from_text(raw_response)
#         if not parsed:
#             raise ValueError("Failed to parse LLM response")

#         # === 4. Apply deterministic bias scoring ===
#         signaling_answers = parsed.get("signaling_answers", {})

#         if is_rct:
#             domain_scores = score_rob2_domain_scores(signaling_answers)
#             total_bias_score = get_overall_rob2_judgment(domain_scores)
#         else:
#             domain_scores = score_robins_i_domain_scores(signaling_answers)
#             total_bias_score = get_overall_robins_i_judgment(domain_scores)

#         # Map judgment to numeric score for downstream use
#         score_map = {"Low": 1, "Moderate": 2, "Some concerns": 2, "Serious": 3, "High": 3, "Critical": 4}
#         numeric_score = score_map.get(total_bias_score, 3)

#         return {
#             "metadata": {"pmcid": pmcid, "status": "completed"},
#             "eligibility": parsed.get("eligibility", {"eligible": False, "reason": "Missing"}),
#             "bias_assessment": {
#                 "framework_used": framework,
#                 "signaling_answers": signaling_answers,
#                 "domain_scores": domain_scores,
#                 "judgment": total_bias_score,
#                 "total_bias_score": numeric_score
#             },
#             "answers": parsed.get("answers", {})
#         }

#     except Exception as e:
#         return {
#             "metadata": {"pmcid": pmcid, "status": "error", "error": str(e)},
#             "eligibility": {"eligible": False, "reason": f"Processing failed: {str(e)}"},
#             "bias_assessment": {},
#             "answers": {}
#         }