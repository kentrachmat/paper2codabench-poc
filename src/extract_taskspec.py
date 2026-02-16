#!/usr/bin/env python3
"""
Extract TaskSpec from research papers using Azure OpenAI.

Usage:
    python extract_taskspec.py papers/paper1.pdf
    python extract_taskspec.py papers/paper2.pdf --paper-id paper2
"""
import argparse
import json
import sys
from pathlib import Path
import pymupdf  # PyMuPDF
from openai import OpenAI
from tqdm import tqdm
from config import Config
from taskspec_schema import TaskSpec
from prompts import (
    TASKSPEC_EXTRACTION_SYSTEM,
    create_taskspec_extraction_prompt
)


def extract_pdf_text(pdf_path: Path, max_pages: int = 50) -> str:
    """Extract text from PDF using PyMuPDF"""
    print(f"📄 Extracting text from {pdf_path.name}...")

    try:
        doc = pymupdf.open(pdf_path)
        text_parts = []

        total_pages = len(doc)
        pages_to_extract = min(total_pages, max_pages)

        # Warn if paper is being truncated
        if total_pages > max_pages:
            print(f"⚠️  Paper has {total_pages} pages, extracting first {max_pages} only")
            print(f"   This may result in incomplete task specification")

        # Extract text with progress bar
        for page_num in tqdm(range(pages_to_extract), desc="Extracting pages", unit="page"):
            page = doc[page_num]
            text_parts.append(page.get_text())

        doc.close()

        full_text = "\n\n".join(text_parts)
        print(f"✓ Extracted {len(full_text)} characters from {pages_to_extract}/{total_pages} pages")
        return full_text

    except Exception as e:
        print(f"✗ Error extracting PDF: {e}")
        raise


def call_azure_openai(prompt_messages: list, max_retries: int = 3, max_tokens: int = 16000) -> str:
    """Call Azure OpenAI API with retry logic"""
    # Validate configuration
    try:
        Config.validate()
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
        sys.exit(1)

    # Use OpenAI client with base_url for Azure
    client = OpenAI(
        base_url=Config.AZURE_OPENAI_ENDPOINT,
        api_key=Config.AZURE_OPENAI_KEY
    )

    # Progress bar for API call attempts
    with tqdm(total=1, desc="🤖 Calling Azure OpenAI", unit="call") as pbar:
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=Config.AZURE_OPENAI_DEPLOYMENT,
                    messages=prompt_messages,
                    max_tokens=max_tokens,  # Increased from 4000 to support longer papers
                    temperature=0.3,  # Lower temperature for more deterministic output
                    response_format={"type": "json_object"}  # Force JSON response
                )

                content = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason

                # Check if response was cut off due to token limits
                if finish_reason == "length":
                    print(f"\n⚠️  Warning: Response truncated due to token limit!")
                    print(f"   Consider reducing paper length or splitting into chunks")
                    # Still return the content, validation will catch issues

                pbar.update(1)
                pbar.set_description(f"✓ Received response ({len(content)} chars)")
                return content

            except Exception as e:
                if attempt < max_retries - 1:
                    pbar.set_description(f"⚠️  Attempt {attempt + 1}/{max_retries} failed, retrying")
                else:
                    pbar.set_description("✗ All retries failed")
                    print(f"\n✗ Error: {e}")
                    raise


def extract_taskspec(pdf_path: Path, paper_id: str) -> dict:
    """
    Main extraction function.

    Args:
        pdf_path: Path to PDF file
        paper_id: Identifier for the paper (e.g., 'paper1')

    Returns:
        Extracted TaskSpec as dictionary
    """
    print(f"\n{'='*60}")
    print(f"Extracting TaskSpec from {pdf_path.name}")
    print(f"Paper ID: {paper_id}")
    print(f"{'='*60}\n")

    # Step 1: Extract PDF text
    paper_text = extract_pdf_text(pdf_path)

    # Save extracted PDF text
    pdf_text_output_path = Config.TASKSPEC_DIR / f"{paper_id}.pdf_text.txt"
    with tqdm(total=1, desc="💾 Saving PDF text", unit="file") as pbar:
        with open(pdf_text_output_path, 'w', encoding='utf-8') as f:
            f.write(paper_text)
        pbar.update(1)

    # Check if paper is extremely long and might need truncation
    paper_char_count = len(paper_text)
    if paper_char_count > 100000:  # ~25k tokens for GPT-4
        print(f"⚠️  Paper is very long ({paper_char_count} chars)")
        print(f"   Truncating to first 100k characters to fit context window")
        paper_text = paper_text[:100000]

    # Step 2: Prepare prompt
    messages = [
        {"role": "system", "content": TASKSPEC_EXTRACTION_SYSTEM},
        {"role": "user", "content": create_taskspec_extraction_prompt(paper_text)}
    ]

    # Step 3: Call Azure OpenAI with increased token limit
    raw_response = call_azure_openai(messages, max_tokens=16000)

    # Step 4: Parse JSON response
    with tqdm(total=1, desc="📝 Parsing JSON response", unit="task") as pbar:
        try:
            taskspec_data = json.loads(raw_response)
            # Ensure paper_id is set correctly
            taskspec_data["paper_id"] = paper_id
            pbar.update(1)
            pbar.set_description("✓ Successfully parsed JSON")
        except json.JSONDecodeError as e:
            pbar.set_description("✗ Failed to parse JSON")
            print(f"\n✗ Error: {e}")
            print(f"Raw response:\n{raw_response}")
            raise

    # Step 5: Save raw response
    raw_output_path = Config.TASKSPEC_DIR / f"{paper_id}.raw.json"
    with tqdm(total=1, desc="💾 Saving raw response", unit="file") as pbar:
        with open(raw_output_path, 'w') as f:
            json.dump({"raw_response": raw_response}, f, indent=2)
        pbar.update(1)

    # Step 6: Validate with Pydantic (basic validation)
    with tqdm(total=1, desc="🔍 Validating TaskSpec schema", unit="task") as pbar:
        try:
            validated_taskspec = TaskSpec(**taskspec_data)
            taskspec_dict = validated_taskspec.model_dump()
            pbar.update(1)
            pbar.set_description("✓ TaskSpec is valid")
        except Exception as e:
            pbar.update(1)
            pbar.set_description("⚠️  Validation warnings")
            print(f"\n⚠️  {e}")
            print("Continuing with unvalidated TaskSpec...")
            taskspec_dict = taskspec_data

    # Step 7: Save validated TaskSpec
    taskspec_output_path = Config.TASKSPEC_DIR / f"{paper_id}.taskspec.json"
    with tqdm(total=1, desc="💾 Saving TaskSpec", unit="file") as pbar:
        with open(taskspec_output_path, 'w') as f:
            json.dump(taskspec_dict, f, indent=2)
        pbar.update(1)

    print(f"\n{'='*60}")
    print(f"✅ TaskSpec extraction complete!")
    print(f"   PDF text: {pdf_text_output_path}")
    print(f"   Raw output: {raw_output_path}")
    print(f"   TaskSpec: {taskspec_output_path}")
    print(f"{'='*60}\n")

    return taskspec_dict


def main():
    parser = argparse.ArgumentParser(
        description="Extract TaskSpec from research paper PDF"
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
        print(f"✗ Error: PDF file not found: {args.pdf_path}")
        sys.exit(1)

    # Derive paper_id from filename if not provided
    if args.paper_id:
        paper_id = args.paper_id
    else:
        paper_id = args.pdf_path.stem  # e.g., 'paper1' from 'paper1.pdf'

    # Extract TaskSpec
    try:
        taskspec = extract_taskspec(args.pdf_path, paper_id)
        print(f"📋 Task: {taskspec.get('task_name', 'N/A')}")
        print(f"📋 Type: {taskspec.get('task_type', 'N/A')}")
        print(f"📋 Metric: {taskspec.get('evaluation', {}).get('primary_metric', 'N/A')}")
    except Exception as e:
        print(f"\n✗ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
