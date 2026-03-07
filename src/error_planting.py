#!/usr/bin/env python3
"""
Error Planting for Hallucination Detection.

Plants deliberate modifications in paper text, then checks whether the LLM
reproduces them in its output — measuring hallucination vs. faithful extraction.

Plant types:
- metric_rename:  Change the primary metric name (e.g., "accuracy" → "precision@7")
- number_change:  Alter a salient number (runtime limit, threshold, etc.)
- dataset_rename: Change dataset name to a fictional one
- column_rename:  Rename a column in data format descriptions
"""
import re
import random
import json
from typing import Tuple, Optional


PLANT_TYPES = ["metric_rename", "number_change", "dataset_rename", "column_rename"]

# Fictional replacements
FAKE_METRICS = [
    "precision@7", "harmonic_recall_v2", "weighted_f3_score",
    "normalized_discounted_accuracy", "calibrated_brier_loss",
]

FAKE_DATASET_NAMES = [
    "SynthBench-2024", "NeuroEval-X", "CryptoGraph-500K",
    "PolyPhase-3D", "OmniTask-Lite",
]

FAKE_COLUMN_NAMES = [
    "zeta_score", "flux_index", "omega_label",
    "phase_id", "quantum_flag",
]


def _find_metric_in_text(text: str) -> Optional[str]:
    """Find a likely metric name in paper text."""
    common_metrics = [
        "accuracy", "f1.score", "f1-score", "precision", "recall",
        "auc", "roc.auc", "mean.average.precision", "map",
        "bleu", "rouge", "meteor", "perplexity",
        "rmse", "mae", "mse", "r.squared", "r2",
        "ndcg", "mrr", "iou", "dice",
    ]
    text_lower = text.lower()
    for metric in common_metrics:
        pattern = re.compile(re.escape(metric).replace(r"\.", r"[\s_\-]?"), re.IGNORECASE)
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


def _find_number_in_text(text: str) -> Optional[Tuple[str, int, int]]:
    """Find a salient number (3+ digits) in paper text. Returns (number_str, start, end)."""
    # Look for numbers like 600, 4096, 1000, etc.
    matches = list(re.finditer(r'\b(\d{3,6})\b', text))
    if matches:
        match = matches[len(matches) // 2]  # Pick a middle one
        return match.group(0), match.start(), match.end()
    return None


def _find_dataset_name(text: str) -> Optional[str]:
    """Find a likely dataset name in paper text (capitalized multi-word or hyphenated)."""
    patterns = [
        r'\b([A-Z][a-zA-Z]*(?:[-_][A-Z][a-zA-Z]*)+)\b',  # CamelCase-Hyphenated
        r'\b([A-Z]{2,}[\-_]?\d*[A-Za-z]*)\b',  # ACRONYM or ACRONYM-123
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        # Filter out very short or common words
        for m in matches:
            if len(m) >= 4 and m not in ("Python", "Table", "Figure", "Section", "Method"):
                return m
    return None


def _find_column_name(text: str) -> Optional[str]:
    """Find a likely column/field name in paper text."""
    # Look for patterns like "column_name", "field_name" in context of data description
    patterns = [
        r'\b((?:id|label|target|pred|score|class|category|feature)\w*)\b',
        r'\b(\w+_(?:id|label|score|pred|value|name))\b',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches:
            if len(m) >= 3:
                return m
    return None


def plant_error(paper_text: str, plant_type: str, seed: int = 42) -> Tuple[str, dict]:
    """
    Plant a deliberate error in paper text.

    Args:
        paper_text: Original paper text
        plant_type: One of PLANT_TYPES
        seed: Random seed for reproducibility

    Returns:
        (modified_text, plant_record) where plant_record contains:
            - plant_type: type of error planted
            - original: original value
            - replacement: planted value
            - planted: whether planting succeeded
    """
    rng = random.Random(seed)

    record = {
        "plant_type": plant_type,
        "original": None,
        "replacement": None,
        "planted": False,
    }

    if plant_type == "metric_rename":
        original = _find_metric_in_text(paper_text)
        if original:
            replacement = rng.choice(FAKE_METRICS)
            modified = paper_text.replace(original, replacement)
            if modified != paper_text:
                record["original"] = original
                record["replacement"] = replacement
                record["planted"] = True
                return modified, record

    elif plant_type == "number_change":
        result = _find_number_in_text(paper_text)
        if result:
            num_str, start, end = result
            original_num = int(num_str)
            # Change the number significantly
            replacement_num = original_num * 3 + 7
            replacement = str(replacement_num)
            modified = paper_text[:start] + replacement + paper_text[end:]
            record["original"] = num_str
            record["replacement"] = replacement
            record["planted"] = True
            return modified, record

    elif plant_type == "dataset_rename":
        original = _find_dataset_name(paper_text)
        if original:
            replacement = rng.choice(FAKE_DATASET_NAMES)
            modified = paper_text.replace(original, replacement)
            if modified != paper_text:
                record["original"] = original
                record["replacement"] = replacement
                record["planted"] = True
                return modified, record

    elif plant_type == "column_rename":
        original = _find_column_name(paper_text)
        if original:
            replacement = rng.choice(FAKE_COLUMN_NAMES)
            modified = paper_text.replace(original, replacement)
            if modified != paper_text:
                record["original"] = original
                record["replacement"] = replacement
                record["planted"] = True
                return modified, record

    # If planting failed, return original text
    return paper_text, record


def check_planted_error(plant_record: dict, llm_output: dict) -> dict:
    """
    Check if a planted error propagated into the LLM output.

    Args:
        plant_record: Record from plant_error()
        llm_output: Parsed Croissant Task JSON from LLM

    Returns:
        detection_result dict with:
            - plant_type: type of error
            - planted: whether error was successfully planted
            - propagated: whether the planted error appears in LLM output
            - original_found: whether the original value still appears
            - details: explanation
    """
    result = {
        "plant_type": plant_record["plant_type"],
        "planted": plant_record["planted"],
        "original": plant_record.get("original"),
        "replacement": plant_record.get("replacement"),
        "propagated": False,
        "original_found": False,
        "details": "",
    }

    if not plant_record["planted"]:
        result["details"] = "Error planting failed — could not find target in text"
        return result

    replacement = plant_record["replacement"]
    original = plant_record["original"]

    # Serialize the LLM output to search it
    output_text = json.dumps(llm_output, indent=2).lower()
    replacement_lower = replacement.lower()
    original_lower = original.lower()

    if replacement_lower in output_text:
        result["propagated"] = True
        result["details"] = f"Planted value '{replacement}' found in LLM output"
    else:
        result["details"] = f"Planted value '{replacement}' NOT found in LLM output"

    if original_lower in output_text:
        result["original_found"] = True
        result["details"] += f"; original '{original}' also found"

    return result
