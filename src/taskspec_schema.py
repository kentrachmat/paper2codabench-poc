"""
Pydantic schema for TaskSpec validation.
Matches the JSON schema defined in prompt.md.
"""
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class SubmissionFormat(BaseModel):
    """Submission format specification"""
    type: Literal["predictions_file", "code", "docker"]
    filename: str
    columns: Optional[List[str]] = None
    example_rows: Optional[List[List[str]]] = None


class Dataset(BaseModel):
    """Dataset availability information"""
    public_train_available: bool
    public_dev_available: bool
    test_is_hidden: bool
    notes: str


class Evaluation(BaseModel):
    """Evaluation metrics specification"""
    primary_metric: str
    metrics: List[str]
    higher_is_better: bool
    notes: str


class Constraints(BaseModel):
    """Resource and data constraints"""
    runtime_limit_sec: int = 600
    memory_limit_mb: int = 4096
    allowed_external_data: str = "unknown"


class CodebenchMapping(BaseModel):
    """Mapping to Codabench bundle components"""
    ingestion_io: str
    scoring_steps: List[str]
    edge_cases: List[str]


class TaskSpec(BaseModel):
    """Complete TaskSpec schema"""
    paper_id: str
    task_name: str
    task_type: Literal["classification", "ranking", "generation", "detection", "segmentation", "other"]
    problem_statement: str
    input_description: str
    output_description: str
    submission_format: SubmissionFormat
    dataset: Dataset
    evaluation: Evaluation
    constraints: Constraints
    codabench_mapping: CodebenchMapping
    open_questions: List[str] = Field(default_factory=list)

    class Config:
        # Allow extra fields for extensibility
        extra = "allow"
