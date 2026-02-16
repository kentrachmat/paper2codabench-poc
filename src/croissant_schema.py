"""
Pydantic schema for Croissant Task (cr:TaskProblem) JSON-LD validation.
Based on the MLCommons Croissant vocabulary from structure.pdf.

Validates the intermediate metadata format used between PDF extraction
and Codabench bundle generation.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class RecordSetField(BaseModel):
    """A field within a RecordSet (cr:Field)"""
    name: str = Field(..., alias="name")
    description: Optional[str] = None
    data_type: str = Field("sc:Text", alias="dataType")
    source: Optional[str] = None

    model_config = {"populate_by_name": True, "extra": "allow"}


class RecordSet(BaseModel):
    """A RecordSet defining tabular structure (cr:RecordSet)"""
    name: Optional[str] = None
    description: Optional[str] = None
    field: List[RecordSetField] = Field(default_factory=list)

    model_config = {"populate_by_name": True, "extra": "allow"}


class FileObject(BaseModel):
    """A file reference (sc:FileObject / cr:FileObject)"""
    name: Optional[str] = None
    content_url: Optional[str] = Field(None, alias="contentUrl")
    encoding_format: Optional[str] = Field(None, alias="encodingFormat")
    description: Optional[str] = None

    model_config = {"populate_by_name": True, "extra": "allow"}


class DatasetInput(BaseModel):
    """Dataset input specification (schema:Dataset used as cr:input)"""
    name: str
    description: Optional[str] = None
    url: Optional[str] = None
    distribution: Optional[List[FileObject]] = None
    record_set: Optional[RecordSet] = Field(None, alias="recordSet")

    model_config = {"populate_by_name": True, "extra": "allow"}


class OutputSpec(BaseModel):
    """Output specification (cr:OutputSpec for cr:output)"""
    name: Optional[str] = "predictions"
    description: Optional[str] = None
    schema_def: Optional[RecordSet] = Field(None, alias="cr:schema")

    model_config = {"populate_by_name": True, "extra": "allow"}


class EnvironmentSpec(BaseModel):
    """Environment specification within ImplementationSpec"""
    language: Optional[str] = "Python"
    packages: Optional[List[str]] = None
    docker_image: Optional[str] = Field(None, alias="dockerImage")

    model_config = {"populate_by_name": True, "extra": "allow"}


class TestSpec(BaseModel):
    """Test specification within ImplementationSpec"""
    name: Optional[str] = None
    description: Optional[str] = None
    input_file: Optional[str] = Field(None, alias="inputFile")
    expected_output: Optional[str] = Field(None, alias="expectedOutput")

    model_config = {"populate_by_name": True, "extra": "allow"}


class ImplementationSpec(BaseModel):
    """Implementation specification (cr:ImplementationSpec)"""
    environment: Optional[EnvironmentSpec] = Field(None, alias="cr:environment")
    tests: Optional[List[TestSpec]] = Field(None, alias="cr:tests")
    entry_point: Optional[str] = Field(None, alias="entryPoint")
    interface: Optional[str] = None

    model_config = {"populate_by_name": True, "extra": "allow"}


class EvaluationMethod(BaseModel):
    """Evaluation method specification (cr:EvaluationMethod)"""
    primary_metric: str = Field("accuracy", alias="primaryMetric")
    metrics: List[str] = Field(default_factory=lambda: ["accuracy"])
    higher_is_better: bool = Field(True, alias="higherIsBetter")
    notes: Optional[str] = None

    model_config = {"populate_by_name": True, "extra": "allow"}


class ExecutionInfo(BaseModel):
    """Execution constraints (cr:ExecutionInfo)"""
    runtime_limit_sec: int = Field(600, alias="runtimeLimitSec")
    memory_limit_mb: int = Field(4096, alias="memoryLimitMb")
    allowed_external_data: Optional[str] = Field("unknown", alias="allowedExternalData")

    model_config = {"populate_by_name": True, "extra": "allow"}


class CroissantTaskProblem(BaseModel):
    """
    Top-level Croissant Task Problem (cr:TaskProblem).

    JSON-LD format with @context, @type, @id for the intermediate
    metadata representation between PDF extraction and bundle generation.
    """
    context: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "@vocab": "http://schema.org/",
            "cr": "http://mlcommons.org/croissant/",
            "sc": "http://schema.org/"
        },
        alias="@context"
    )
    type: str = Field("cr:TaskProblem", alias="@type")
    id: Optional[str] = Field(None, alias="@id")
    name: str
    description: Optional[str] = None

    # Core Croissant Task fields
    input: Optional[List[DatasetInput]] = Field(None, alias="cr:input")
    output: Optional[OutputSpec] = Field(None, alias="cr:output")
    implementation: Optional[ImplementationSpec] = Field(None, alias="cr:implementation")
    evaluation: Optional[EvaluationMethod] = Field(None, alias="cr:evaluation")
    execution: Optional[ExecutionInfo] = Field(None, alias="cr:execution")

    # Extended fields for Paper2Codabench
    paper_id: Optional[str] = Field(None, alias="paper_id")
    open_questions: List[str] = Field(default_factory=list)
    fill_in_the_blank: List[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True, "extra": "allow"}
