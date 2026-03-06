#!/usr/bin/env python3
"""
Scrape NeurIPS 2024-2025 Datasets & Benchmarks track papers from OpenReview
and identify which ones have associated Croissant metadata.

Output:
  - data/neurips_2024_db_papers.csv
  - data/neurips_2025_db_papers.csv
  - data/neurips_db_croissant_summary.json
"""

import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import openreview

# Project root
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# Venue configurations
VENUES = {
    2024: "NeurIPS.cc/2024/Datasets_and_Benchmarks_Track",
    2025: "NeurIPS.cc/2025/Datasets_and_Benchmarks_Track",
}

# Competition heuristic keywords
COMPETITION_KEYWORDS = re.compile(
    r"\b(competition|challenge|contest|shared.?task|retrospective)\b", re.IGNORECASE
)

# URL patterns for downloadable datasets
DATASET_URL_PATTERNS = re.compile(
    r"https?://(?:"
    r"huggingface\.co/datasets/[\w\-\.]+/[\w\-\.]+"
    r"|kaggle\.com/(?:datasets|competitions)/[\w\-\.]+"
    r"|github\.com/[\w\-\.]+/[\w\-\.]+"
    r"|zenodo\.org/record(?:s)?/\d+"
    r"|doi\.org/10\.\d{4,}/[\w\.\-]+"
    r"|drive\.google\.com/[\w\?=&/]+"
    r"|figshare\.com/[\w/\.\-]+"
    r")",
    re.IGNORECASE,
)


def extract_dataset_urls_from_text(text: str) -> list:
    """Find known dataset hosting URLs in text."""
    if not text:
        return []
    return list(set(DATASET_URL_PATTERNS.findall(text)))


def connect():
    """Connect to OpenReview API v2 (public, no auth needed)."""
    print("Connecting to OpenReview API v2...")
    client = openreview.api.OpenReviewClient(baseurl="https://api2.openreview.net")
    print("Connected.")
    return client


def discover_fields(client, venue_id):
    """Inspect a sample submission to discover available content fields."""
    notes = client.get_notes(
        content={"venueid": venue_id},
        limit=3,
    )
    if not notes:
        return set()
    fields = set()
    for note in notes[:3]:
        fields.update(note.content.keys())
    return fields


def check_croissant_in_field(value):
    """Check if a field value references Croissant metadata."""
    if value is None:
        return False
    text = str(value).lower()
    return "croissant" in text


def extract_text(content_field):
    """Extract text value from an OpenReview content field (may be dict with 'value')."""
    if content_field is None:
        return ""
    if isinstance(content_field, dict):
        return str(content_field.get("value", ""))
    return str(content_field)


def extract_attachment_url(content_field, note_id, client_baseurl="https://openreview.net"):
    """Extract download URL for an attachment field."""
    if content_field is None:
        return ""
    if isinstance(content_field, dict):
        val = content_field.get("value", "")
        if val and isinstance(val, str) and not val.startswith("http"):
            # It's a relative path — construct full URL
            return f"{client_baseurl}/attachment?id={note_id}&name={val}"
        return str(val) if val else ""
    return str(content_field) if content_field else ""


def is_competition_paper(title, keywords, abstract):
    """Heuristic: check if paper is competition/challenge related."""
    text = f"{title} {keywords} {abstract}"
    return bool(COMPETITION_KEYWORDS.search(text))


def fetch_papers(client, year, venue_id):
    """Fetch all accepted papers for a venue and extract metadata."""
    print(f"\n{'='*60}")
    print(f"Fetching NeurIPS {year} D&B Track papers...")
    print(f"Venue: {venue_id}")
    print(f"{'='*60}")

    # Discover available fields
    available_fields = discover_fields(client, venue_id)
    print(f"Available content fields: {sorted(available_fields)}")

    # Identify Croissant-related fields
    croissant_fields = [f for f in available_fields if "croissant" in f.lower()]
    supplementary_fields = [f for f in available_fields if "supplement" in f.lower()]
    dataset_fields = [f for f in available_fields if "dataset" in f.lower() or "data" in f.lower()]
    print(f"Croissant fields: {croissant_fields}")
    print(f"Supplementary fields: {supplementary_fields}")
    print(f"Dataset fields: {dataset_fields}")

    # Fetch all accepted papers
    print(f"\nFetching all accepted papers (this may take a moment)...")
    submissions = client.get_all_notes(
        content={"venueid": venue_id},
    )
    print(f"Retrieved {len(submissions)} papers.")

    papers = []
    for i, note in enumerate(submissions):
        if (i + 1) % 100 == 0:
            print(f"  Processing {i+1}/{len(submissions)}...")

        content = note.content
        title = extract_text(content.get("title"))
        authors_field = content.get("authors", {})
        if isinstance(authors_field, dict):
            authors_val = authors_field.get("value", [])
        else:
            authors_val = authors_field if authors_field else []
        authors = "; ".join(authors_val) if isinstance(authors_val, list) else str(authors_val)

        abstract = extract_text(content.get("abstract"))
        keywords_field = content.get("keywords", {})
        if isinstance(keywords_field, dict):
            keywords_val = keywords_field.get("value", [])
        else:
            keywords_val = keywords_field if keywords_field else []
        keywords = "; ".join(keywords_val) if isinstance(keywords_val, list) else str(keywords_val)

        # Build OpenReview URL
        openreview_url = f"https://openreview.net/forum?id={note.id}"

        # Check for Croissant metadata
        has_croissant = False
        croissant_url = ""

        # Check dedicated Croissant fields
        for field in croissant_fields:
            val = content.get(field)
            if val is not None:
                text_val = extract_text(val)
                if text_val and text_val.lower() not in ("", "none", "n/a"):
                    has_croissant = True
                    if text_val.startswith("http"):
                        croissant_url = text_val
                    elif isinstance(val, dict) and val.get("value"):
                        # It's an attachment
                        croissant_url = f"https://openreview.net/attachment?id={note.id}&name={field}"

        # Check supplementary materials for croissant references
        for field in supplementary_fields:
            val = content.get(field)
            if val is not None:
                text_val = extract_text(val)
                if check_croissant_in_field(text_val):
                    has_croissant = True
                    if not croissant_url:
                        croissant_url = f"https://openreview.net/attachment?id={note.id}&name={field}"

        # Check if supplementary material filename contains .json or croissant
        for field in supplementary_fields:
            val = content.get(field)
            if isinstance(val, dict):
                v = val.get("value", "")
                if isinstance(v, str) and (".json" in v.lower() or "croissant" in v.lower()):
                    has_croissant = True
                    if not croissant_url:
                        croissant_url = f"https://openreview.net/attachment?id={note.id}&name={field}"

        # Check abstract/keywords for croissant mentions (weaker signal)
        if not has_croissant:
            if "croissant" in abstract.lower() or "croissant" in keywords.lower():
                has_croissant = True  # Mentioned but no direct file

        # Dataset URL
        dataset_url = ""
        for field in dataset_fields:
            val = content.get(field)
            if val is not None:
                text_val = extract_text(val)
                if text_val.startswith("http"):
                    dataset_url = text_val
                    break

        # Also check common field names
        for field_name in ["dataset_url", "code", "repository", "url"]:
            if field_name in content and not dataset_url:
                text_val = extract_text(content[field_name])
                if text_val.startswith("http"):
                    dataset_url = text_val

        # Scan abstract + keywords for dataset URLs if dataset_url is still empty
        if not dataset_url:
            found_urls = extract_dataset_urls_from_text(f"{abstract} {keywords}")
            if found_urls:
                dataset_url = found_urls[0]

        has_downloadable_dataset = bool(dataset_url)

        # Competition heuristic
        is_comp = is_competition_paper(title, keywords, abstract)

        # PDF URL
        pdf_val = content.get("pdf")
        pdf_url = ""
        if pdf_val:
            text_val = extract_text(pdf_val)
            if text_val.startswith("http"):
                pdf_url = text_val
            elif text_val:
                pdf_url = f"https://openreview.net/pdf?id={note.id}"

        papers.append({
            "year": year,
            "paper_id": note.id,
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "keywords": keywords,
            "has_croissant": has_croissant,
            "croissant_url": croissant_url,
            "dataset_url": dataset_url,
            "has_downloadable_dataset": has_downloadable_dataset,
            "openreview_url": openreview_url,
            "pdf_url": pdf_url,
            "is_competition_paper": is_comp,
        })

    return papers


def save_csv(papers, filepath):
    """Save papers to CSV."""
    if not papers:
        print(f"No papers to save to {filepath}")
        return

    fieldnames = [
        "year", "paper_id", "title", "authors", "abstract", "keywords",
        "has_croissant", "croissant_url", "dataset_url", "has_downloadable_dataset",
        "openreview_url", "pdf_url", "is_competition_paper",
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(papers)
    print(f"Saved {len(papers)} papers to {filepath}")


def generate_summary(all_papers):
    """Generate aggregate summary statistics."""
    summary = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}

    for year in sorted(VENUES.keys()):
        year_papers = [p for p in all_papers if p["year"] == year]
        croissant_papers = [p for p in year_papers if p["has_croissant"]]
        competition_papers = [p for p in year_papers if p["is_competition_paper"]]
        comp_with_croissant = [p for p in competition_papers if p["has_croissant"]]
        dataset_papers = [p for p in year_papers if p.get("has_downloadable_dataset")]

        # Count dataset URL domains
        domain_counts = {}
        for p in dataset_papers:
            url = p.get("dataset_url", "")
            if "huggingface.co" in url:
                domain_counts["huggingface"] = domain_counts.get("huggingface", 0) + 1
            elif "kaggle.com" in url:
                domain_counts["kaggle"] = domain_counts.get("kaggle", 0) + 1
            elif "github.com" in url:
                domain_counts["github"] = domain_counts.get("github", 0) + 1
            elif "zenodo.org" in url:
                domain_counts["zenodo"] = domain_counts.get("zenodo", 0) + 1
            elif "doi.org" in url:
                domain_counts["doi"] = domain_counts.get("doi", 0) + 1
            elif "drive.google.com" in url:
                domain_counts["google_drive"] = domain_counts.get("google_drive", 0) + 1
            elif "figshare.com" in url:
                domain_counts["figshare"] = domain_counts.get("figshare", 0) + 1
            elif url:
                domain_counts["other"] = domain_counts.get("other", 0) + 1

        summary[f"neurips_{year}"] = {
            "total_accepted": len(year_papers),
            "with_croissant": len(croissant_papers),
            "croissant_percentage": round(100 * len(croissant_papers) / max(len(year_papers), 1), 1),
            "competition_papers": len(competition_papers),
            "competition_with_croissant": len(comp_with_croissant),
            "papers_with_dataset_url": len(dataset_papers),
            "dataset_url_percentage": round(100 * len(dataset_papers) / max(len(year_papers), 1), 1),
            "dataset_url_domains": domain_counts,
            "papers_with_croissant": [
                {
                    "paper_id": p["paper_id"],
                    "title": p["title"],
                    "croissant_url": p["croissant_url"],
                    "openreview_url": p["openreview_url"],
                    "is_competition_paper": p["is_competition_paper"],
                }
                for p in croissant_papers
            ],
        }

    return summary


def main():
    client = connect()
    all_papers = []

    for year, venue_id in VENUES.items():
        try:
            papers = fetch_papers(client, year, venue_id)
            all_papers.extend(papers)

            # Save per-year CSV
            csv_path = DATA_DIR / f"neurips_{year}_db_papers.csv"
            save_csv(papers, csv_path)

            # Small delay between years
            time.sleep(1)
        except Exception as e:
            print(f"\nError fetching {year}: {e}")
            print("This may be expected if the venue hasn't been created yet on OpenReview.")
            import traceback
            traceback.print_exc()

    # Generate and save summary
    summary = generate_summary(all_papers)
    summary_path = DATA_DIR / "neurips_db_croissant_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to {summary_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for year in sorted(VENUES.keys()):
        key = f"neurips_{year}"
        if key in summary:
            s = summary[key]
            print(f"\nNeurIPS {year} D&B Track:")
            print(f"  Total accepted papers: {s['total_accepted']}")
            print(f"  With Croissant metadata: {s['with_croissant']} ({s['croissant_percentage']}%)")
            print(f"  Competition papers: {s['competition_papers']}")
            print(f"  Competition + Croissant: {s['competition_with_croissant']}")

    total = len(all_papers)
    total_croissant = len([p for p in all_papers if p["has_croissant"]])
    print(f"\nOverall: {total_croissant}/{total} papers have Croissant metadata")


if __name__ == "__main__":
    main()
