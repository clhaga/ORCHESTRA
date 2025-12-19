# ORCHESTRA: Clinical Trials Analysis Agent

Orchestra is a comprehensive Python tool for analyzing clinical trials literature. It automates the process of searching, downloading, and analyzing medical studies from PubMed Central (PMC), providing evidence-based synthesis for clinical research questions.

## Overview

Orchestra orchestrates a multi-step pipeline to:
1. Search PubMed for relevant clinical trials
2. Download full-text articles and supplementary materials
3. Process studies for eligibility, bias assessment, and analysis
4. Build relevance maps between study questions and synthesis questions
5. Create evidence tools for synthesis
6. Generate final synthesis across all eligible studies

## Features

- **Intelligent Search**: Uses SmartPubMedSearcher with LLM-based reranking to find the most relevant clinical trials
- **Comprehensive Analysis**: Evaluates study eligibility, performs bias assessment, and answers research questions
- **Progress Tracking**: Real-time progress monitoring with ETA calculations
- **Flexible Execution**: Skip to specific pipeline steps or force reanalysis of studies
- **Evidence Synthesis**: Generates evidence-based answers to clinical research questions
- **Bias Assessment**: Evaluates study quality using standardized bias assessment criteria

## Installation

### Prerequisites

- Python 3.8 or higher
- Required dependencies (see requirements.txt)

### Dependencies

Orchestra requires a Gemini API key for LLM processing. Set up your API key in the configuration file.

## Usage

### Basic Usage

```bash
python orchestra.py --query "stem cell therapy for heart disease" --max_results 100
```

### Command Line Arguments

- `--query` (required): Your research question for searching clinical trials
- `--max_results` (optional): Maximum number of studies to retrieve (default: 1000)
- `--verbose` (optional): Enable verbose logging
- `--force-reanalyze` (optional): Reprocess all studies, ignoring cached results
- `--skip-to` (optional): Skip to a specific pipeline phase
  - Choices: `search`, `download`, `process`, `relevance`, `tools`, `synthesize`

### Examples

1. Basic search and analysis:
```bash
python orchestra.py --query "mesenchymal stem cells for osteoarthritis"
```

2. Skip to synthesis step (requires previous results):
```bash
python orchestra.py --query "any" --skip-to synthesize
```

3. Force reanalysis of all studies:
```bash
python orchestra.py --query "CAR T cell therapy" --force-reanalyze
```

4. Verbose output with custom result limit:
```bash
python orchestra.py --query "immunotherapy for lung cancer" --max_results 500 --verbose
```

## Pipeline Steps

### Step 1: Search
Searches PubMed using SmartPubMedSearcher with LLM-based reranking to find relevant clinical trials.

### Step 2: Download
Downloads full-text articles and supplementary materials from PubMed Central.

### Step 3: Process Studies
Analyzes each study for:
- Eligibility based on inclusion/exclusion criteria
- Bias assessment using standardized criteria
- Answers to per-study analysis questions

### Step 4: Build Relevance Map
Creates mappings between study analysis questions and synthesis questions.

### Step 5: Create Evidence Tools
Generates specialized tools for each synthesis question based on the relevance map.

### Step 6: Final Synthesis
Produces evidence-based answers to synthesis questions using the evidence tools and bias assessments.

## Directory Structure

```
project/
├── orchestra.py              # Main orchestration script
├── articles/                 # Downloaded articles (created automatically)
│   └── {PMCID}/             # Individual article folders
├── results/                  # Analysis results (created automatically)
│   ├── studies_metadata.json
│   ├── comprehensive_results.json
│   ├── analyses_by_question.json
│   ├── bias_by_pmcid.json
│   ├── relevance_map.json
│   ├── evidence_tools.json
│   └── synthesis.json
└── config/                   # Configuration files
    ├── per_study_questions.txt
    ├── synthesis_questions.txt
    ├── inclusion_criteria.txt
    └── exclusion_criteria.txt
```

## Output Files

### Key Result Files

- `synthesis.json`: Final evidence-based answers to synthesis questions
- `comprehensive_results.json`: Detailed analysis of each study
- `bias_by_pmcid.json`: Bias assessments for all studies
- `evidence_tools.json`: Tools used for synthesis

### Bias Assessment Categories

Studies are assessed for bias and categorized as:
- Low
- Moderate
- Some concerns
- Serious
- High
- Critical
- Not Scored

## Configuration

### Required Configuration Files

The tool requires several configuration files in the `config/` directory:

1. `per_study_questions.txt`: Questions to ask about each individual study
2. `synthesis_questions.txt`: Questions for final synthesis across studies
3. `inclusion_criteria.txt`: Criteria for including studies in analysis
4. `exclusion_criteria.txt`: Criteria for excluding studies from analysis

### LLM Configuration

The tool uses two LLM clients:
- Study processing client: For analyzing individual studies
- Synthesis client: For generating final synthesis

## Logging

Orchestra provides detailed logging with:
- Progress indicators showing completion percentage
- ETA calculations for remaining work
- Error reporting and warnings
- Final summary with statistics

## Error Handling

The tool includes robust error handling:
- Skips studies with missing files
- Continues processing when individual studies fail
- Provides detailed error messages
- Maintains progress tracking despite errors

## Performance Considerations

- Processing time varies based on the number of studies and complexity of questions
- Average processing time is calculated and displayed during execution
- Results are cached to avoid reprocessing completed studies
- Batch processing is used for efficiency in certain operations

## Troubleshooting

### Common Issues

1. **No studies found**: Check if your query is specific enough and try different keywords
2. **Missing configuration files**: Ensure all required config files exist in the config/ directory
3. **API key issues**: Verify your LLM API key is properly configured
4. **Memory issues**: Reduce `--max_results` if processing too many studies at once

### Debug Mode

Use the `--verbose` flag to enable detailed logging for troubleshooting:
```bash
python orchestra.py --query "your query" --verbose
```

## Contributing

When contributing to Orchestra, please ensure:
- Code follows Python best practices
- Functions have appropriate docstrings
- Error handling is comprehensive
- Logging is informative but not excessive

## License

This project is licensed under the GNU General Public License v3.0.


## Citation

