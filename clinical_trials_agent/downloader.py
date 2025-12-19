import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from pathlib import Path
import re
import logging
import shutil
import tempfile
import zipfile
from PyPDF2 import PdfReader
import docx2txt
import pandas as pd
from playwright.async_api import async_playwright
import mimetypes
import numpy as np


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("downloader.log"), logging.StreamHandler()]
)

# ======================
# File Conversion
# ======================

async def convert_file_to_markdown(file_path: Path, output_dir: Path) -> Path:
    """
    Convert various file types to markdown. Skip large Excel/CSV files (>1MB) that likely contain genomic data.
    Returns path to .md file, or None if skipped/failed.
    """
    file_path = Path(file_path)
    output_md = output_dir / f"{file_path.stem}.md"
    
    try:
        ext = file_path.suffix.lower()

        # 🔍 Skip large Excel/CSV files (>1 MB)
        if ext in ['.xlsx', '.xls', '.csv']:
            file_size = file_path.stat().st_size
            if file_size > 600000:  # 1 MB in bytes
                logging.info(f"📏 Skipping large file ({file_size / 1e6:.1f} MB): {file_path.name}")
                return None

        # --- ZIP ---
        if ext == '.zip':
            content = await process_zip_file(file_path, output_dir)
            if content is None:
                return None

        # --- PDF ---
        elif ext == '.pdf':
            reader = PdfReader(file_path)
            content = ''.join([page.extract_text() or '' for page in reader.pages])

        # --- DOCX / DOC ---
        elif ext in ['.docx', '.doc']:
            content = docx2txt.process(file_path)

        # --- EXCEL / CSV ---
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
            content = df.to_markdown(index=False)
        elif ext == '.csv':
            df = pd.read_csv(file_path)
            content = df.to_markdown(index=False)

        # --- TEXT / MARKDOWN ---
        elif ext in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()

        # --- UNSUPPORTED ---
        else:
            logging.warning(f"⚠️  Unsupported file type: {ext} ({file_path.name})")
            return None

        # --- SAVE MARKDOWN ---
        with open(output_md, 'w', encoding='utf-8') as f:
            f.write(f"# Attached File: {file_path.name}\n\n")
            f.write(f"```\n{content[:800000]}{'...' if len(content) > 800000 else ''}\n```")

        logging.info(f"✅ Converted: {file_path.name}")
        return output_md

    except Exception as e:
        logging.error(f"❌ Conversion failed for {file_path.name}: {e}")
        return None

# async def convert_file_to_markdown(file_path: Path, output_dir: Path) -> Path:
#     """
#     Convert various file types to markdown. Skip genomic count data files.
#     Returns path to .md file, or None if skipped/failed.
#     """
#     file_path = Path(file_path)
#     output_md = output_dir / f"{file_path.stem}.md"
    
#     try:
#         ext = file_path.suffix.lower()

#         # --- ZIP ---
#         if ext == '.zip':
#             content = await process_zip_file(file_path, output_dir)
#             if content is None:
#                 return None

#         # --- PDF ---
#         elif ext == '.pdf':
#             reader = PdfReader(file_path)
#             content = ''.join([page.extract_text() or '' for page in reader.pages])

#         # --- DOCX / DOC ---
#         elif ext in ['.docx', '.doc']:
#             content = docx2txt.process(file_path)

#         # --- EXCEL (XLS/XLSX) - CHECK FOR GENOMIC DATA ---
#         elif ext in ['.xlsx', '.xls']:
#             if is_genomic_count_data(file_path):
#                 logging.info(f"🧬 Skipping genomic data file: {file_path.name}")
#                 return None
#             df = pd.read_excel(file_path)
#             content = df.to_markdown(index=False)

#         # --- CSV - CHECK FOR GENOMIC DATA ---
#         elif ext == '.csv':
#             if is_genomic_count_data(file_path):
#                 logging.info(f"🧬 Skipping genomic data file: {file_path.name}")
#                 return None
#             df = pd.read_csv(file_path)
#             content = df.to_markdown(index=False)

#         # --- TEXT / MARKDOWN ---
#         elif ext in ['.txt', '.md']:
#             with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
#                 content = f.read()

#         # --- UNSUPPORTED ---
#         else:
#             logging.warning(f"⚠️  Unsupported file type: {ext} ({file_path.name})")
#             return None

#         # --- SAVE MARKDOWN ---
#         with open(output_md, 'w', encoding='utf-8') as f:
#             f.write(f"# Attached File: {file_path.name}\n\n")
#             f.write(f"```\n{content[:800000]}{'...' if len(content) > 800000 else ''}\n```")

#         logging.info(f"✅ Converted: {file_path.name}")
#         return output_md

#     except Exception as e:
#         logging.error(f"❌ Conversion failed for {file_path.name}: {e}")
#         return None

# async def convert_file_to_markdown(file_path: Path, output_dir: Path) -> Path:
#     """Convert various file types to markdown and return path to .md file."""
#     file_path = Path(file_path)
#     output_md = output_dir / f"{file_path.stem}.md"
#     try:
#         ext = file_path.suffix.lower()
#         if ext == '.zip':
#             content = await process_zip_file(file_path, output_dir)
#         elif ext == '.pdf':
#             reader = PdfReader(file_path)
#             content = ''.join([page.extract_text() or '' for page in reader.pages])
#         elif ext in ['.docx', '.doc']:
#             content = docx2txt.process(file_path)
#         elif ext in ['.xlsx', '.xls']:
#              if is_genomic_count_data(file_path):
#                 logging.info(f"🧬 Skipping genomic data file: {file_path.name}")
#                 return None  # Skip processing
#             else:
#                 df = pd.read_excel(file_path)
#                 content = df.to_markdown(index=False)
#         elif ext == '.csv':
#             df = pd.read_csv(file_path)
#             content = df.to_markdown(index=False)
#         elif ext in ['.txt', '.md']:
#             with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
#                 content = f.read()
#         else:
#             logging.warning(f"Unsupported file type: {ext}")
#             return None

#         with open(output_md, 'w', encoding='utf-8') as f:
#             f.write(f"# Attached File: {file_path.name}\n\n")
#             f.write(f"```\n{content[:800000]}{'...' if len(content) > 800000 else ''}\n```")
#         return output_md

#     except Exception as e:
#         logging.error(f"Conversion failed for {file_path}: {e}")
#         return None


async def process_zip_file(zip_path: Path, output_dir: Path) -> str:
    """Extract and convert contents of a ZIP file."""
    temp_dir = Path(tempfile.mkdtemp())
    extract_dir = output_dir / f"{zip_path.stem}_extracted"
    extract_dir.mkdir(exist_ok=True)
    md_lines = [f"# Contents of {zip_path.name}\n## Files:\n"]

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(temp_dir)
        for file in temp_dir.rglob('*'):
            if file.is_file():
                rel_path = file.relative_to(temp_dir)
                dest = extract_dir / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(file, dest)
                md_lines.append(f"- {rel_path}")
                md_file = await convert_file_to_markdown(dest, extract_dir)
                if md_file:
                    md_lines.append(f"  - [Markdown]({rel_path.parent / md_file.name})")
        return "\n".join(md_lines)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def is_genomic_count_data(file_path: Path) -> bool:
    """
    Detect if an Excel/CSV file contains RNA-seq or microarray count/abundance data.
    """
    try:
        if file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path, nrows=10, engine='openpyxl')
        elif file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path, nrows=10)
        else:
            return False

        if df.empty or len(df) < 2:
            return False

        # Normalize column names
        cols = [str(c).lower() for c in df.columns]

        # Look for abundance-related keywords in column names
        abundance_keywords = [
            'gene', 'symbol', 'id', 'feature', 'transcript',
            'count', 'counts', 'expression', 'level', 'abundance', 'normalized',
            'fpkm', 'tpm', 'rpm', 'fragments', 'reads', 'read_count',
            'log2foldchange', 'lfc', 'pvalue', 'padj', 'adj_p',
            'intensity', 'probe', 'oligo',
            'mirna', 'pre-mirna', 'mir_name', 'mir_seq', 'mirid',
            'genomeid', 'chromosome', 'chr', 'start', 'end', 'strand',
            'sequence', 'seq', 'hairpin', 'dG', 'cg%', 'ss', 'structure',
            'snp', 'variant', 'genotype', 'allele', 'vcf', 'bam'
        ]
        abundance_matches = [kw for kw in abundance_keywords if any(kw in col for col in cols)]

        # Check first column for gene-like entries
        first_col_vals = df.iloc[:, 0].dropna().astype(str)

        has_gene_ids = first_col_vals.str.contains(
            r'^(?:ENS|NM_|NR_|XM_|XR_|AT[1-5]G\d+|IL\d+|HSA_|MIR-|MIRNA-|HSA-|MIR_|MIRNA_|TP53|GAPDH|ALB|FN1)', 
            case=False, 
            na=False
        ).any()

        # Check if values are numeric (abundance data)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        high_numeric_ratio = len(numeric_cols) >= 2

        # Decision logic
        if has_gene_ids and (len(abundance_matches) >= 1 or high_numeric_ratio):
            return True

        # If first row has "Gene Symbol" or similar
        if 'gene' in cols[0] or 'symbol' in cols[0]:
            return True

        return False

    except Exception as e:
        logging.warning(f"Could not analyze {file_path.name} for genomic content: {e}")
        return False


async def download_file(url: str, save_dir: Path) -> Path:
    """Use Playwright to download a binary file, handling Cloudflare PoW challenge."""
    browser = None
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = url.split('/')[-1]
        filepath = save_dir / filename

        logging.info(f"📥 Attempting to download: {url}")

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
                viewport={"width": 1920, "height": 1080},
                java_script_enabled=True,
                bypass_csp=True  
            )
            page = await context.new_page()

            try:
                async with page.expect_download() as download_info:
                    await page.goto(url, wait_until="networkidle", timeout=30000)
                    await page.wait_for_timeout(2000)  

                download = await download_info.value
                await download.save_as(filepath)

                logging.info(f"✅ Successfully downloaded: {filename}")
                return filepath

            except Exception as e:
                logging.warning(f"❌ Playwright navigation failed: {e}")
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                    content = await page.content()
                    if "<title>Preparing to download</title>" in content:
                        logging.warning(f"❌ PoW challenge not solved for {url}")
                        return None
                    binary = await page.evaluate("() => fetch(window.location.href).then(r => r.arrayBuffer()).then(b => b)")
                    with open(filepath, 'wb') as f:
                        f.write(binary)
                    logging.info(f"✅ Fallback binary download succeeded: {filename}")
                    return filepath
                except Exception as e:
                    logging.error(f"❌ Fallback download failed: {e}")
                    return None

    except Exception as e:
        logging.error(f"❌ Download failed {url}: {e}")
        return None
    finally:
        if browser:
            await browser.close()


def truncate_at_references(text: str) -> str:
    """Truncate text at '## References' (case-insensitive)."""
    match = re.search(r'##\s*References', text, re.IGNORECASE)
    return text[:match.end()] if match else text

def truncate_at_supplementary(text: str) -> str:
    """Truncate text at '## Supplementary Information' (case-insensitive)."""
    match = re.search(r'##\s*Supplementary', text, re.IGNORECASE)
    return text[:match.end()] if match else text


# ======================
# Main Download Function
# ======================

async def process_single_pmc(pmc_id: str, base_articles_dir: Path = Path("articles")):
    """
    Download PMC article and all supplementary materials.
    Only one informed consent form is downloaded (based on link text).
    All content is combined into a single article.md.
    """
    article_dir = base_articles_dir / pmc_id
    article_dir.mkdir(parents=True, exist_ok=True)
    markdown_file = article_dir / "article.md"

    # Skip if already exists
    if markdown_file.exists():
        logging.info(f"⏭️ Skipping {pmc_id} — already downloaded.")
        return

    base_url = "https://www.ncbi.nlm.nih.gov/pmc"
    url = f"{base_url}/articles/{pmc_id}/"

    # Accumulate all markdown content
    full_markdown = []

    async with AsyncWebCrawler(verbose=True) as crawler:
        try:
            # 1. Fetch main article
            config = CrawlerRunConfig(
                css_selector="section.body.main-article-body",
                word_count_threshold=10,
                exclude_external_images=True,
                exclude_external_links=True,
                magic=True,
            )
            result = await crawler.arun(url=url, config=config)
            if not result or not result.markdown.strip():
                logging.warning(f"No content extracted for {pmc_id}")
                return

            # Truncate main article at references
            truncated_main = truncate_at_references(result.markdown)
            truncated_main = truncate_at_supplementary(truncated_main)
            full_markdown.append(truncated_main)

            # 2. Find supplementary files
            supplement_pattern = r'<a[^>]*href="(/articles/instance/[^"]+\.(?:docx?|pdf|xlsx?|csv|zip|txt))"[^>]*>(.*?)</a>'
            matches = list(re.finditer(supplement_pattern, result.html, re.IGNORECASE))

            # Extract all consent form candidates
            consent_forms = []
            other_supplements = []

            for match in matches:
                url_path = match.group(1)
                description = match.group(2).strip()
                full_url = f"{base_url}{url_path}"
                if re.search(r'informed\s+consent|consent\s+form', description, re.IGNORECASE):
                    consent_forms.append((full_url, description))
                else:
                    other_supplements.append((full_url, description))

            # Decide: download only ONE consent form (the first one found)
            if consent_forms:
                full_url, description = consent_forms[0]  # Pick only the first
                logging.info(f"📥 Downloading ONE consent form: '{description}'")
                downloaded_file = await download_file(full_url, article_dir)
                if downloaded_file:
                    md_file = await convert_file_to_markdown(downloaded_file, article_dir)
                    if md_file:
                        with open(md_file, 'r', encoding='utf-8') as f:
                            full_markdown.append(f.read())
                    logging.info(f"✅ Added consent form: {description}")
                else:
                    logging.warning(f"❌ Failed to download consent form: {full_url}")
            else:
                logging.info(f"🔍 No consent forms found in supplementary materials.")

            # Process all other supplements (non-consent)
            processed_urls = set()
            for full_url, description in other_supplements:
                if full_url in processed_urls:
                    continue
                processed_urls.add(full_url)

                url_path = full_url.replace(f"{base_url}/articles/instance/", "")
                ext = Path(url_path).suffix.lower()

                # Use crawl4ai for HTML pages
                if ext in ['.html', '.htm']:
                    logging.info(f"🌐 Processing HTML supplement with crawl4ai: {description}")
                    try:
                        supp_result = await crawler.arun(
                            url=full_url,
                            config=CrawlerRunConfig(magic=True, word_count_threshold=10)
                        )
                        if supp_result and supp_result.markdown.strip():
                            header = f"\n## Supplementary Material: {description}\n"
                            full_markdown.append(header + supp_result.markdown)
                    except Exception as e:
                        logging.error(f"Failed to process HTML supplement {full_url}: {e}")
                    continue

                # For spreadsheets served as HTML tables
                if ext in ['.xlsx', '.xls', '.csv']:
                    mime, _ = mimetypes.guess_type(full_url)
                    if mime == 'text/html':
                        logging.info(f"📊 Processing table page with crawl4ai: {description}")
                        try:
                            supp_result = await crawler.arun(
                                url=full_url,
                                config=CrawlerRunConfig(table_extraction=True, magic=True)
                            )
                            if supp_result and supp_result.markdown.strip():
                                header = f"\n## Supplementary Table: {description}\n"
                                full_markdown.append(header + supp_result.markdown)
                        except Exception as e:
                            logging.error(f"Failed to process table {full_url}: {e}")
                        continue

                # Download binary files (PDF, DOCX, ZIP, etc.)
                logging.info(f"📥 Downloading supplementary file: {description}")
                downloaded_file = await download_file(full_url, article_dir)
                if downloaded_file:
                    # Check if it's genomic count data
                    if downloaded_file.suffix.lower() in ['.xlsx', '.xls', '.csv']:
                        if is_genomic_count_data(downloaded_file):
                            logging.info(f"🧬 Skipping genomic count data: {description} ({downloaded_file.name})")
                            downloaded_file.unlink()  # Delete the file
                            continue

                    # ✅ Otherwise, process normally
                    md_file = await convert_file_to_markdown(downloaded_file, article_dir)
                    if md_file:
                        with open(md_file, 'r', encoding='utf-8') as f:
                            full_markdown.append(f.read())
                    await asyncio.sleep(1)
                else:
                    logging.warning(f"❌ Failed to download: {full_url}")

            # 3. Final: Combine all content into single article.md
            final_content = "\n\n---\n\n".join(full_markdown).strip()
            with open(markdown_file, 'w', encoding='utf-8') as f:
                f.write(final_content)

            logging.info(f"✅ Successfully created single article: {markdown_file}")

        except Exception as e:
            logging.error(f"❌ Failed to process {pmc_id}: {e}")
            # Optional: clean up partial file
            if markdown_file.exists():
                markdown_file.unlink()