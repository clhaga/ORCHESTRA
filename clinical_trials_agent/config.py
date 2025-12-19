import os
import json
from pathlib import Path

CONFIG_PATH = Path("./config.json")
CONFIG_DIR = Path("./config")
# Paths to bias rule files
ROB2_RULES_PATH = CONFIG_DIR/ "rob2_rules.json"
ROBINS_I_RULES_PATH = CONFIG_DIR / "robinsI_rules.json"

def load_config():
    """Load configuration from JSON file."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
    
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)

def get_llm_config():
    """Get LLM configuration with validation and support for multiple models."""
    config = load_config()
    llm_config = config.get("llm", {})

    # API key: from config or env var
    api_key = llm_config.get("api_key", "").strip() or os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key or api_key == "GEMINI_API_KEY":
        raise ValueError(
            "LLM API key is missing or placeholder in config.json. "
            "Please set a real GEMINI_API_KEY in config.json or environment variable."
        )

    # Model resolution logic
    models = llm_config.get("models", {})
    
    # If 'models' is not defined, fall back to single 'model' or default
    study_model = models.get(
        "study_processing",
        llm_config.get("model", "gemini-2.5-flash-lite")  
    )
    synthesis_model = models.get(
        "synthesis",
        llm_config.get("model", "gemini-2.5-flash")
    )

    return {
        "api_key": api_key,
        "models": {
            "study_processing": study_model,
            "synthesis": synthesis_model
        }
    }

def get_pubmed_config():
    """Get PubMed configuration."""
    config = load_config()
    pubmed_config = config.get("pubmed", {})
    
    return {
        "email": pubmed_config.get("email") or os.getenv("NCBI_EMAIL"),
        "api_key": pubmed_config.get("api_key") or os.getenv("NCBI_API_KEY")
    }

def get_bias_criteria():
    """
    Load bias assessment criteria.
    Prioritizes rob2_rules.json and robinsI_rules.json.
    Falls back to legacy bias_criteria.json if needed.
    """
    criteria = {}

    # === 1. Load RoB 2 signaling questions ===
    if ROB2_RULES_PATH.exists():
        try:
            with open(ROB2_RULES_PATH, 'r', encoding='utf-8') as f:
                rob2_data = json.load(f)
            criteria["ROB2_SIGNALING_QUESTIONS"] = rob2_data
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {ROB2_RULES_PATH}: {e}")
    else:
        print(f"⚠️  RoB 2 rules not found at {ROB2_RULES_PATH}")

    # === 2. Load ROBINS-I V2 signaling questions ===
    if ROBINS_I_RULES_PATH.exists():
        try:
            with open(ROBINS_I_RULES_PATH, 'r', encoding='utf-8') as f:
                robins_i_data = json.load(f)
            criteria["ROBINS_I_SIGNALING_QUESTIONS"] = robins_i_data
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {ROBINS_I_RULES_PATH}: {e}")
    else:
        print(f"⚠️  ROBINS-I V2 rules not found at {ROBINS_I_RULES_PATH}")
    if not criteria:
        raise FileNotFoundError(
            "No bias criteria files found. "
            "Please ensure one of the following exists: "
            "config/rob2_rules.json and config/robinsI_rules.json"
        )

    return criteria