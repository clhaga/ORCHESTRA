import asyncio
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import time
from datetime import timedelta

if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
logger = logging.getLogger(__name__)

from clinical_trials_agent.smart_search import SmartPubMedSearcher
from clinical_trials_agent.downloader import process_single_pmc
from clinical_trials_agent.llm_utils import (
    parse_questions,
    build_relevance_map,
    select_tool_type,
    synthesize_from_tools,
    create_tool_for_synthesis_question,
    process_study_comprehensive,
    # Import ProgressTracker if defined in llm_utils, otherwise define here
)
from clinical_trials_agent.utils import load_prompt, save_json
from clinical_trials_agent.config import load_config, get_llm_config
from clinical_trials_agent.llm_client import LLMClient

# Load LLM clients
llm_config = get_llm_config()
study_client = LLMClient(model=llm_config["models"]["study_processing"], temperature=0.0)
synth_client = LLMClient(model=llm_config["models"]["synthesis"], temperature=0.0)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
ARTICLES_DIR = PROJECT_ROOT / "articles"
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIG_DIR = PROJECT_ROOT / "config"


def setup_directories():
    """Create necessary directories if they don't exist."""
    ARTICLES_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)


def load_configuration() -> Dict[str, str]:
    """Load all prompt configurations."""
    config = load_config()
    return {
        "per_study_questions": load_prompt(CONFIG_DIR / "per_study_questions.txt"),
        "synthesis_questions": load_prompt(CONFIG_DIR / "synthesis_questions.txt"),
        "inclusion_criteria": load_prompt(CONFIG_DIR / "inclusion_criteria.txt"),
        "exclusion_criteria": load_prompt(CONFIG_DIR / "exclusion_criteria.txt"),
        "gemini_api_key": config.get("llm", {}).get("api_key")
    }


async def search_pubmed(user_query: str, max_results: int) -> tuple[List[str], Dict[str, Dict]]:
    """Search PubMed using SmartPubMedSearcher."""
    logger.info("1️⃣ SEARCHING PubMed for clinical trials...")
    searcher = SmartPubMedSearcher()
    pmcids, studies_metadata = searcher.search(
        query=user_query,
        max_results=max_results,
        pmc_only=True,
        only_trials=True,
        llm_rerank=True,
        relevance_threshold=60,
        results_dir=RESULTS_DIR
    )

    if not pmcids:
        logger.info("❌ No studies found.")
        return [], {}

    logger.info(f"✅ Found {len(pmcids)} eligible PMC articles.")
    logger.info(f"📄 Metadata saved to {RESULTS_DIR / 'studies_metadata.json'}")
    return pmcids, studies_metadata


async def download_articles(pmcids: List[str], articles_dir: Path):
    """Download full texts."""
    logger.info("2️⃣ DOWNLOADING full texts and supplementary materials...")
    for pmcid in pmcids:
        await process_single_pmc(pmcid, articles_dir)
    logger.info(f"✅ Downloaded {len(pmcids)} studies.\n")


class ProgressTracker:
    """Track processing progress across studies."""
    def __init__(self, total_studies: int):
        self.total = total_studies
        self.completed = 0
        self.eligible = 0
        self.ineligible = 0
        self.errors = 0
        self.bias_counts = {"Low": 0, "Moderate": 0, "Some concerns": 0, "Serious": 0, "High": 0, "Critical": 0, "Not Scored": 0}

    def update(self, result: dict):
        self.completed += 1

        status = result.get("metadata", {}).get("status")
        eligible = result.get("eligibility", {}).get("eligible")
        judgment = result.get("bias_assessment", {}).get("judgment", "Unknown")

        if status == "error":
            self.errors += 1
        elif eligible is True:
            self.eligible += 1
        elif eligible is False:
            self.ineligible += 1

        if judgment in self.bias_counts:
            self.bias_counts[judgment] += 1
        else:
            self.bias_counts["Not Scored"] += 1

        percent = self.completed / self.total * 100 if self.total > 0 else 0
        logger.info(
            f"✅ {self.completed}/{self.total} ({percent:.1f}%) | "
            f"Eligible: {self.eligible} | Ineligible: {self.ineligible} | Errors: {self.errors}"
        )

def process_studies_comprehensive(
    pmcids: List[str],
    articles_dir: Path,
    inclusion_criteria: str,
    exclusion_criteria: str,
    per_study_questions: List[str],
    studies_metadata: Dict[str, Dict],
    force_reanalyze: bool = False
) -> Tuple[List[Dict], Dict[str, Dict], Dict[str, Dict]]:
    """
    Process multiple studies with real-time progress tracking and ETA.
    
    Returns:
        all_results, bias_by_pmcid, analyses_by_question
    """
    all_results = []
    analyses_by_question = {}
    bias_by_pmcid = {}

    total = len(pmcids)
    start_time = time.time()

    logger.info(f"🧠 Starting comprehensive analysis of {total} studies...")
    logger.info("=" * 60)

    # Initialize counters
    completed = 0
    eligible = 0
    errors = 0

    for idx, pmcid in enumerate(pmcids, start=1):
        # === Show current progress ===
        elapsed = time.time() - start_time
        avg_time_per_study = elapsed / idx
        estimated_total = avg_time_per_study * total
        remaining = estimated_total - elapsed

        logger.info(
            f"📌 [{idx}/{total}] Processing study: {pmcid} | "
            f"Elapsed: {str(timedelta(seconds=int(elapsed)))} | "
            f"ETA: {str(timedelta(seconds=int(remaining)))}"
        )

        article_dir = articles_dir / pmcid
        json_file = article_dir / f"{pmcid}_comprehensive.json"
        md_file = article_dir / "article.md"

        # --- Check if already processed ---
        if not force_reanalyze and json_file.exists():
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                all_results.append(result)
                logger.debug(f"⏭️  [{idx}/{total}] Skipped (cached): {pmcid}")
                
                # Update counters
                if result["metadata"].get("status") == "completed" and result["eligibility"].get("eligible"):
                    eligible += 1
                completed += 1
                continue
            except Exception as e:
                logger.warning(f"⚠️  Failed to load cached result for {pmcid}: {e}. Re-processing.")

        # --- Check if markdown exists ---
        if not md_file.exists():
            logger.error(f"❌ [{idx}/{total}] MISSING FILE: {md_file}")
            result = {
                "metadata": {"pmcid": pmcid, "status": "error", "error": "Markdown file missing"},
                "eligibility": {"eligible": False, "reason": "File missing"},
                "bias_assessment": {},
                "answers": {}
            }
            all_results.append(result)
            errors += 1
            completed += 1
            continue

        # --- Process study ---
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()

            publication_types = studies_metadata.get(pmcid, {}).get("publication_types", [])
            
            # Call your existing processing function
            result = process_study_comprehensive(
                pmcid=pmcid,
                markdown_content=content,
                inclusion_criteria=inclusion_criteria,
                exclusion_criteria=exclusion_criteria,
                analysis_questions=per_study_questions,
                publication_types=publication_types,
                llm_client=study_client  
            )
            
            # Save result
            save_json(result, json_file)

        except Exception as e:
            logger.exception(f"💥 [{idx}/{total}] FAILED to process {pmcid}: {str(e)}")
            result = {
                "metadata": {"pmcid": pmcid, "status": "error", "error": str(e)},
                "eligibility": {"eligible": False, "reason": "Processing failed"},
                "bias_assessment": {},
                "answers": {}
            }
            errors += 1

        # --- Collect results ---
        all_results.append(result)
        completed += 1

        # Update eligibility count
        if result.get("metadata", {}).get("status") == "completed":
            if result["eligibility"].get("eligible"):
                eligible += 1
            # Extract data for synthesis
            for q, ans in result["answers"].items():
                if q not in analyses_by_question:
                    analyses_by_question[q] = {}
                analyses_by_question[q][pmcid] = ans
            bias_by_pmcid[pmcid] = result["bias_assessment"]

        # --- Optional: Log completion status ---
        status = result["metadata"].get("status", "unknown")
        eligible_status = result["eligibility"].get("eligible", False)
        logger.debug(f"✅ [{idx}/{total}] {pmcid} | Status: {status} | Eligible: {eligible_status}")

    # === Final summary logging ===
    logger.info("=" * 60)
    logger.info(f"✅ COMPLETED PROCESSING {completed}/{total} STUDIES")
    logger.info(f"📊 Eligible: {eligible} | ❌ Ineligible: {completed - eligible - errors} | 💥 Errors: {errors}")
    final_elapsed = timedelta(seconds=int(time.time() - start_time))
    logger.info(f"⏱️  Total Runtime: {final_elapsed}")

    # === Save aggregated results ===
    save_json(all_results, RESULTS_DIR / "comprehensive_results.json")
    save_json(analyses_by_question, RESULTS_DIR / "analyses_by_question.json")
    save_json(bias_by_pmcid, RESULTS_DIR / "bias_by_pmcid.json")

    return all_results, bias_by_pmcid, analyses_by_question


async def run_clinical_trials_agent(
    user_query: str,
    max_results: int,
    force_reanalyze: bool = False,
    skip_to: str = None
):
    print(f"🔍 Starting analysis for: '{user_query}'")
    print("=" * 60)

    setup_directories()
    config = load_configuration()

    # Load synthesis questions (for reference/logging only)
    synthesis_questions = parse_questions(config["synthesis_questions"])
    if not synthesis_questions:
        logger.error("❌ No synthesis questions defined.")
        print("❌ No synthesis questions found. Please check 'synthesis_questions.txt'.")
        return

    print(f"📊 Loaded {len(synthesis_questions)} synthesis questions.")
    print()

    # === VALIDATE SKIP TARGET ===
    valid_steps = {"search", "download", "process", "relevance", "tools", "synthesize"}
    if skip_to and skip_to not in valid_steps:
        raise ValueError(f"Invalid --skip-to value. Must be one of: {valid_steps}")

    # === EARLY EXIT: SKIP TO SYNTHESIS ===
    if skip_to == "synthesize":
        print("⏭️  Skipping all prior steps — jumping directly to synthesis.")

        tools_file = RESULTS_DIR / "evidence_tools.json"
        bias_file = RESULTS_DIR / "bias_by_pmcid.json"

        if not tools_file.exists():
            print(f"❌ Required file missing: {tools_file}")
            return
        if not bias_file.exists():
            print(f"❌ Required file missing: {bias_file}")
            return

        print("📂 Loading evidence tools and bias data...")
        with open(tools_file, 'r', encoding='utf-8') as f:
            evidence_tools = json.load(f)
        with open(bias_file, 'r', encoding='utf-8') as f:
            bias_by_pmcid = json.load(f)

        print(f"✅ Loaded {len(evidence_tools)} evidence tools.")
        print(f"✅ Loaded bias assessments for {len(bias_by_pmcid)} studies.")

        # === RUN SYNTHESIS ===
        print("📈 Step 8: Generating final synthesis across all studies...")
        synthesis = synthesize_from_tools(
            evidence_tools=evidence_tools,
            bias_by_pmcid=bias_by_pmcid,
            llm_client=synth_client
        )
        save_json(synthesis, RESULTS_DIR / "synthesis.json")
        print(f"✅ Final synthesis saved to: {RESULTS_DIR / 'synthesis.json'}")

        # === FINAL SUMMARY ===
        print("\n" + "🎉 ANALYSIS COMPLETE!")
        print("=" * 60)
        print(f"❓ Synthesis Questions: {len(synthesis_questions)}")
        print(f"📁 Results saved in: {RESULTS_DIR}")
        print(f"📄 View synthesis.json for final answers.")

        # Bias Distribution
        print("\n📊 BIAS DISTRIBUTION:")
        bias_counts = {}
        for ba in bias_by_pmcid.values():
            judgment = ba.get("judgment", "Not Scored")
            bias_counts[judgment] = bias_counts.get(judgment, 0) + 1

        for level in ["Low", "Moderate", "Some concerns", "Serious", "High", "Critical", "Not Scored"]:
            count = bias_counts.get(level, 0)
            if count > 0:
                print(f"  {level}: {count}")

        return  

    # === OTHERWISE: RUN FULL PIPELINE (as before) ===
    pmcids = []
    studies_metadata = {}
    all_results = []
    analyses_by_question = {}
    bias_by_pmcid = {}
    relevance_map = {}
    evidence_tools = {}

    # === STEP 1: Search ===
    if not skip_to or skip_to == "search":
        pmcids, studies_metadata = await search_pubmed(user_query, max_results)
        if not pmcids:
            print("❌ No studies found.")
            return
    else:
        metadata_file = RESULTS_DIR / "studies_metadata.json"
        if not metadata_file.exists():
            print(f"❌ Cannot skip: {metadata_file} not found.")
            return
        with open(metadata_file, 'r', encoding='utf-8') as f:
            studies_metadata = json.load(f)
        pmcids = list(studies_metadata.keys())
        print(f"⏭️  Skipped to next step — loaded {len(pmcids)} PMCIDs")

    # === STEP 2: Download ===
    if not skip_to or skip_to in ["search", "download"]:
        await download_articles(pmcids, ARTICLES_DIR)
    else:
        print("⏭️  Skipped article download.")

    # === STEP 3 & 4: Process Studies ===
    eligible_pmcids = pmcids
    if not skip_to or skip_to in ["search", "download", "process"]:
        print("🧠 Step 4: Processing studies (eligibility, bias, analysis) — this may take a while...")
        all_results, bias_by_pmcid, analyses_by_question = process_studies_comprehensive(
            pmcids=eligible_pmcids,
            articles_dir=ARTICLES_DIR,
            inclusion_criteria=config["inclusion_criteria"],
            exclusion_criteria=config["exclusion_criteria"],
            per_study_questions=parse_questions(config["per_study_questions"]),
            studies_metadata=studies_metadata,
            force_reanalyze=force_reanalyze
        )
    else:
        comp_file = RESULTS_DIR / "comprehensive_results.json"
        analysis_file = RESULTS_DIR / "analyses_by_question.json"
        bias_file = RESULTS_DIR / "bias_by_pmcid.json"
        if not comp_file.exists() or not analysis_file.exists() or not bias_file.exists():
            print("❌ Required files missing for skipping.")
            return
        with open(comp_file, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
        with open(analysis_file, 'r', encoding='utf-8') as f:
            analyses_by_question = json.load(f)
        with open(bias_file, 'r', encoding='utf-8') as f:
            bias_by_pmcid = json.load(f)
        progress = ProgressTracker(total_studies=len(all_results))
        for r in all_results:
            progress.update(r)
        print("⏭️  Skipped to later step — loaded processed results.")

    # Filter eligible studies
    eligible_studies = [
        r["metadata"]["pmcid"]
        for r in all_results
        if r["metadata"]["status"] == "completed" and r["eligibility"]["eligible"]
    ]
    print(f"✅ Final number of eligible studies: {len(eligible_studies)}")
    print()

    # === STEP 5: Build Relevance Map ===
    if not skip_to or skip_to in ["search", "download", "process", "relevance"]:
        print("🔗 Step 5: Building relevance map...")
        relevance_map = build_relevance_map(
            analysis_questions=parse_questions(config["per_study_questions"]),
            synthesis_questions=synthesis_questions,
            llm_client=synth_client
        )
        save_json(relevance_map, RESULTS_DIR / "relevance_map.json")
        print(f"✅ Relevance map built for {len(relevance_map)} synthesis questions.")
    else:
        map_file = RESULTS_DIR / "relevance_map.json"
        if not map_file.exists():
            print(f"❌ Cannot skip: {map_file} not found.")
            return
        with open(map_file, 'r', encoding='utf-8') as f:
            relevance_map = json.load(f)
        print("⏭️  Skipped relevance map building.")

    # === STEP 7: Create Evidence Tools ===
    if not skip_to or skip_to in ["search", "download", "process", "relevance", "tools"]:
        print("🛠️  Step 7: Creating evidence tools for synthesis...")
        evidence_tools = {}
        for synth_q in synthesis_questions:
            tool_type = select_tool_type(synth_q, llm_client=synth_client)
            print(f"  ➕ '{synth_q[:50]}{'...' if len(synth_q) > 50 else ''}' → {tool_type.upper()}")
            tool = await create_tool_for_synthesis_question(
                synthesis_question=synth_q,
                tool_type=tool_type,
                analyses_by_question=analyses_by_question,
                relevance_map=relevance_map,
                all_pmcids=eligible_studies,
                llm_client=synth_client,
                batch_size=15
            )
            evidence_tools[synth_q] = tool
        save_json(evidence_tools, RESULTS_DIR / "evidence_tools.json")
        print(f"💾 Evidence tools saved to: {RESULTS_DIR / 'evidence_tools.json'}")
    else:
        tools_file = RESULTS_DIR / "evidence_tools.json"
        if not tools_file.exists():
            print(f"❌ Cannot skip: {tools_file} not found.")
            return
        with open(tools_file, 'r', encoding='utf-8') as f:
            evidence_tools = json.load(f)
        print("⏭️  Skipped tool creation.")

    # === STEP 8: Final Synthesis ===
    print("📈 Step 8: Generating final synthesis across all studies...")
    synthesis = synthesize_from_tools(
        evidence_tools=evidence_tools,
        bias_by_pmcid=bias_by_pmcid,
        llm_client=synth_client
    )
    save_json(synthesis, RESULTS_DIR / "synthesis.json")
    print(f"✅ Final synthesis saved to: {RESULTS_DIR / 'synthesis.json'}")

    # === FINAL SUMMARY ===
    print("\n" + "🎉 ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"📊 Eligible Studies: {len(eligible_studies)}")
    print(f"❌ Ineligible/Excluded: {len(pmcids) - len(eligible_studies)}")
    print(f"🔧 Total Processed: {len(all_results)}")
    print(f"💥 Errors: {sum(1 for r in all_results if r['metadata'].get('status') == 'error')}")
    print(f"❓ Synthesis Questions: {len(synthesis_questions)}")
    print(f"📁 Results saved in: {RESULTS_DIR}")
    print(f"📄 View synthesis.json for final answers.")

    # Bias Distribution
    print("\n📊 BIAS DISTRIBUTION:")
    bias_counts = {}
    for ba in bias_by_pmcid.values():
        judgment = ba.get("judgment", "Not Scored")
        bias_counts[judgment] = bias_counts.get(judgment, 0) + 1

    for level in ["Low", "Moderate", "Some concerns", "Serious", "High", "Critical", "Not Scored"]:
        count = bias_counts.get(level, 0)
        if count > 0:
            print(f"  {level}: {count}")

def main():
    parser = argparse.ArgumentParser(description="A clinical trials agent that analyzes medical studies.")
    parser.add_argument("--query", type=str, required=True, help="The user's query for searching clinical trials.")
    parser.add_argument("--max_results", type=int, default=1000, help="Max number of studies to retrieve.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument("--force-reanalyze", action="store_true", help="Reprocess all studies.")
    parser.add_argument("--skip-to", type=str, choices=[
        "search", "download", "process", "relevance", "tools", "synthesize"
    ], help="Skip to a specific phase.")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    asyncio.run(run_clinical_trials_agent(
        user_query=args.query,
        max_results=args.max_results,
        force_reanalyze=args.force_reanalyze,
        skip_to=args.skip_to
    ))


if __name__ == "__main__":
    main()