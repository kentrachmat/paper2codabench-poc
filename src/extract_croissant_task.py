#!/usr/bin/env python3
"""
Extract Croissant Task (cr:TaskProblem) from research papers using Azure OpenAI.

Usage:
    python extract_croissant_task.py papers/paper1.pdf
    python extract_croissant_task.py papers/paper2.pdf --paper-id paper2
"""
import argparse
import json
import sys
from pathlib import Path
import pymupdf  # PyMuPDF
from openai import OpenAI
from tqdm import tqdm
from config import Config
from croissant_schema import CroissantTaskProblem
from prompts import (
    CROISSANT_TASK_EXTRACTION_SYSTEM,
    create_croissant_task_extraction_prompt,
    CHUNK_SUMMARIZE_SYSTEM,
    create_chunk_summarize_prompt,
    COMBINE_SUMMARIES_SYSTEM,
    create_combine_summaries_prompt,
)


def extract_pdf_text(pdf_path: Path, max_pages: int = 50) -> tuple[str, list[str]]:
    """Extract text from PDF using PyMuPDF.

    Returns:
        Tuple of (full_text, page_texts) where page_texts is a list of per-page strings.
    """
    print(f"Extracting text from {pdf_path.name}...")

    try:
        doc = pymupdf.open(pdf_path)
        text_parts = []

        total_pages = len(doc)
        pages_to_extract = min(total_pages, max_pages)

        # Warn if paper is being truncated
        if total_pages > max_pages:
            print(f"  Paper has {total_pages} pages, extracting first {max_pages} only")
            print(f"  This may result in incomplete task specification")

        # Extract text with progress bar
        for page_num in tqdm(range(pages_to_extract), desc="Extracting pages", unit="page"):
            page = doc[page_num]
            text_parts.append(page.get_text())

        doc.close()

        full_text = "\n\n".join(text_parts)
        print(f"  Extracted {len(full_text)} characters from {pages_to_extract}/{total_pages} pages")
        return full_text, text_parts

    except Exception as e:
        print(f"  Error extracting PDF: {e}")
        raise


def call_azure_openai(prompt_messages: list, max_retries: int = 3, max_tokens: int = 16000) -> str:
    """Call Azure OpenAI API with retry logic"""
    # Validate configuration
    try:
        Config.validate()
    except ValueError as e:
        print(f"  Configuration error: {e}")
        sys.exit(1)

    # Use OpenAI client with base_url for Azure
    client = OpenAI(
        base_url=Config.AZURE_OPENAI_ENDPOINT,
        api_key=Config.AZURE_OPENAI_KEY
    )

    # Progress bar for API call attempts
    with tqdm(total=1, desc="Calling Azure OpenAI", unit="call") as pbar:
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=Config.AZURE_OPENAI_DEPLOYMENT,
                    messages=prompt_messages,
                    max_tokens=max_tokens,
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )

                content = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason

                if finish_reason == "length":
                    print(f"\n  Warning: Response truncated due to token limit!")
                    print(f"  Consider reducing paper length or splitting into chunks")

                pbar.update(1)
                pbar.set_description(f"  Received response ({len(content)} chars)")
                return content

            except Exception as e:
                if attempt < max_retries - 1:
                    pbar.set_description(f"  Attempt {attempt + 1}/{max_retries} failed, retrying")
                else:
                    pbar.set_description("  All retries failed")
                    print(f"\n  Error: {e}")
                    raise


def call_azure_openai_text(prompt_messages: list, max_retries: int = 3, max_tokens: int = 4000) -> str:
    """Call Azure OpenAI API for plain text responses (no JSON mode)"""
    try:
        Config.validate()
    except ValueError as e:
        print(f"  Configuration error: {e}")
        sys.exit(1)

    client = OpenAI(
        base_url=Config.AZURE_OPENAI_ENDPOINT,
        api_key=Config.AZURE_OPENAI_KEY
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=Config.AZURE_OPENAI_DEPLOYMENT,
                messages=prompt_messages,
                max_tokens=max_tokens,
                temperature=0.3,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                continue
            raise


# Threshold for chunked summarization (number of pages)
CHUNK_PAGE_THRESHOLD = 6
# Number of pages per chunk when summarizing
PAGES_PER_CHUNK = 3


def summarize_long_paper(page_texts: list[str]) -> str:
    """Summarize a long paper using map-reduce: summarize chunks, then combine.

    Args:
        page_texts: List of per-page text strings.

    Returns:
        Combined summary string to use for Croissant Task extraction.
    """
    num_pages = len(page_texts)
    print(f"\n  Paper has {num_pages} pages (>{CHUNK_PAGE_THRESHOLD}), using chunked summarization")

    # Build chunks of PAGES_PER_CHUNK pages each
    chunks = []
    for i in range(0, num_pages, PAGES_PER_CHUNK):
        chunk_pages = page_texts[i:i + PAGES_PER_CHUNK]
        chunks.append("\n\n".join(chunk_pages))

    total_chunks = len(chunks)
    print(f"  Split into {total_chunks} chunks of ~{PAGES_PER_CHUNK} pages each")

    # Map phase: summarize each chunk
    summaries = []
    for i, chunk in enumerate(tqdm(chunks, desc="Summarizing chunks", unit="chunk")):
        messages = [
            {"role": "system", "content": CHUNK_SUMMARIZE_SYSTEM},
            {"role": "user", "content": create_chunk_summarize_prompt(chunk, i, total_chunks)}
        ]
        summary = call_azure_openai_text(messages, max_tokens=4000)
        summaries.append(summary)

    # Reduce phase: combine all summaries
    print("  Combining chunk summaries...")
    combine_messages = [
        {"role": "system", "content": COMBINE_SUMMARIES_SYSTEM},
        {"role": "user", "content": create_combine_summaries_prompt(summaries)}
    ]
    combined_summary = call_azure_openai_text(combine_messages, max_tokens=8000)

    print(f"  Combined summary: {len(combined_summary)} characters")
    return combined_summary


def log_fill_in_blanks(croissant_task: dict):
    """Report which fields have [FILL IN THE BLANK] placeholders"""
    fill_in_blanks = croissant_task.get('fill_in_the_blank', [])
    if fill_in_blanks:
        print(f"\n  [FILL IN THE BLANK] placeholders found ({len(fill_in_blanks)}):")
        for item in fill_in_blanks:
            print(f"    - {item}")
        print("  These fields need to be manually filled before deployment.\n")
    else:
        print("  No [FILL IN THE BLANK] placeholders found.\n")


def extract_croissant_task(pdf_path: Path, paper_id: str) -> dict:
    """
    Main extraction function.

    Args:
        pdf_path: Path to PDF file
        paper_id: Identifier for the paper (e.g., 'paper1')

    Returns:
        Extracted Croissant Task as dictionary
    """
    print(f"\n{'='*60}")
    print(f"Extracting Croissant Task from {pdf_path.name}")
    print(f"Paper ID: {paper_id}")
    print(f"{'='*60}\n")

    # Step 1: Extract PDF text
    paper_text, page_texts = extract_pdf_text(pdf_path)

    # Save extracted PDF text
    pdf_text_output_path = Config.CROISSANT_DIR / f"{paper_id}.pdf_text.txt"
    with tqdm(total=1, desc="Saving PDF text", unit="file") as pbar:
        with open(pdf_text_output_path, 'w', encoding='utf-8') as f:
            f.write(paper_text)
        pbar.update(1)

    # Step 2: For long papers, use chunked map-reduce summarization
    num_pages = len(page_texts)
    if num_pages > CHUNK_PAGE_THRESHOLD:
        paper_text = summarize_long_paper(page_texts)
    else:
        # Short paper: check if text is still too long for context window
        if len(paper_text) > 100000:
            print(f"  Paper text is very long ({len(paper_text)} chars)")
            print(f"  Truncating to first 100k characters to fit context window")
            paper_text = paper_text[:100000]

    # Step 3: Prepare prompt
    messages = [
        {"role": "system", "content": CROISSANT_TASK_EXTRACTION_SYSTEM},
        {"role": "user", "content": create_croissant_task_extraction_prompt(paper_text)}
    ]

    # Step 4: Call Azure OpenAI
    raw_response = call_azure_openai(messages, max_tokens=16000)

    # Step 5: Parse JSON response
    with tqdm(total=1, desc="Parsing JSON response", unit="task") as pbar:
        try:
            croissant_data = json.loads(raw_response)
            # Ensure paper_id is set correctly
            croissant_data["paper_id"] = paper_id
            pbar.update(1)
            pbar.set_description("  Successfully parsed JSON")
        except json.JSONDecodeError as e:
            pbar.set_description("  Failed to parse JSON")
            print(f"\n  Error: {e}")
            print(f"Raw response:\n{raw_response}")
            raise

    # Step 6: Save raw response
    raw_output_path = Config.CROISSANT_DIR / f"{paper_id}.raw.json"
    with tqdm(total=1, desc="Saving raw response", unit="file") as pbar:
        with open(raw_output_path, 'w') as f:
            json.dump({"raw_response": raw_response}, f, indent=2)
        pbar.update(1)

    # Step 7: Validate with Pydantic
    with tqdm(total=1, desc="Validating Croissant Task schema", unit="task") as pbar:
        try:
            validated = CroissantTaskProblem(**croissant_data)
            croissant_dict = validated.model_dump(by_alias=True)
            pbar.update(1)
            pbar.set_description("  Croissant Task is valid")
        except Exception as e:
            pbar.update(1)
            pbar.set_description("  Validation warnings")
            print(f"\n  {e}")
            print("Continuing with unvalidated Croissant Task...")
            croissant_dict = croissant_data

    # Step 8: Log fill-in-the-blank placeholders
    log_fill_in_blanks(croissant_dict)

    # Step 9: Save validated Croissant Task
    croissant_output_path = Config.CROISSANT_DIR / f"{paper_id}.croissant_task.json"
    with tqdm(total=1, desc="Saving Croissant Task", unit="file") as pbar:
        with open(croissant_output_path, 'w') as f:
            json.dump(croissant_dict, f, indent=2)
        pbar.update(1)

    print(f"\n{'='*60}")
    print(f"Croissant Task extraction complete!")
    print(f"   PDF text: {pdf_text_output_path}")
    print(f"   Raw output: {raw_output_path}")
    print(f"   Croissant Task: {croissant_output_path}")
    print(f"{'='*60}\n")

    return croissant_dict


def main():
    parser = argparse.ArgumentParser(
        description="Extract Croissant Task from research paper PDF"
    )
    parser.add_argument(
        "pdf_path",
        type=Path,
        help="Path to PDF file (e.g., papers/paper1.pdf)"
    )
    parser.add_argument(
        "--paper-id",
        type=str,
        help="Paper identifier (default: derived from filename)"
    )

    args = parser.parse_args()

    # Validate PDF path
    if not args.pdf_path.exists():
        print(f"  Error: PDF file not found: {args.pdf_path}")
        sys.exit(1)

    # Derive paper_id from filename if not provided
    if args.paper_id:
        paper_id = args.paper_id
    else:
        paper_id = args.pdf_path.stem

    # Extract Croissant Task
    try:
        croissant_task = extract_croissant_task(args.pdf_path, paper_id)
        print(f"  Task: {croissant_task.get('name', 'N/A')}")
        evaluation = croissant_task.get('cr:evaluation', {})
        print(f"  Metric: {evaluation.get('primaryMetric', 'N/A')}")
    except Exception as e:
        print(f"\n  Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
