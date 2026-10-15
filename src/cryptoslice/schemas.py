"""Pydantic contracts for CryptoSlice intermediate artifacts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

NodeType = Literal["definition", "equation", "structure", "constraint", "scope"]
EdgeType = Literal["du", "sc", "cm", "ord"]
Modality = Literal["text", "equation", "table", "pseudocode", "figure", "caption"]
TemplateName = Literal[
    "RoundTransition",
    "SBoxConstraint",
    "LinearLayer",
    "ObjectiveTerm",
    "BoundaryCondition",
    "PermutationLayer",
    "PropagationTable",
    "OperationModel",
    "PropagationRule",
    "MILPConstraint",
    "ComplexityClaim",
    "AttackProcedure",
    "ProblemDefinition",
    "ReductionClaim",
    "AlgorithmClaim",
]
MethodName = Literal[
    "cryptoslice",
    "visrag",
    "visrag_page",
    "bge_faiss",
    "raganything",
    "lightrag",
    "byaldi",
    "mveu_kb",
]


class StrictModel(BaseModel):
    """Base model with stable validation behavior."""

    model_config = ConfigDict(extra="forbid")


class Meta(StrictModel):
    """Required metadata for every generated JSON artifact."""

    generated_at: str
    generator: str
    git_commit: str
    config_hash: str
    input_files: list[str]


class Artifact(StrictModel):
    """Generic meta/data envelope."""

    meta: Meta
    data: dict[str, Any]


class Provenance(StrictModel):
    """Source provenance for a DOG node."""

    paper_id: str
    page: int
    bbox: tuple[float, float, float, float] | None = None
    modality: Modality
    section_path: list[str] = Field(default_factory=list)
    source_object_id: str


class TypeInfo(StrictModel):
    """Type and scope attributes used by closure checks."""

    bit_width: int | None = None
    domain: str | None = None
    role: str | None = None
    bit_order: Literal["MSB0", "LSB0"] | None = None
    word_order: Literal["BE", "LE"] | None = None
    round_range: tuple[int, int] | None = None
    primitive: str | None = None
    attack_model: Literal["single_key", "related_key", "chosen_key"] | None = None
    independence: bool | None = None


class Node(StrictModel):
    """A node in the document object graph."""

    id: str
    node_type: NodeType
    content: str
    provenance: Provenance
    annotations: dict[str, Any] = Field(default_factory=dict)
    type_info: TypeInfo = Field(default_factory=TypeInfo)


class Edge(StrictModel):
    """A typed edge in the document object graph."""

    src: str
    dst: str
    edge_type: EdgeType
    evidence: list[str] = Field(default_factory=list)


class DOG(StrictModel):
    """Document object graph contract."""

    paper_id: str
    nodes: list[Node]
    edges: list[Edge]


class Query(StrictModel):
    """Retrieval query q = (a, m, r, T, eta)."""

    query_id: str
    paper_id: str
    primitive: str
    analysis_model: str
    round_range: tuple[int, int]
    template: TemplateName
    symbol_anchors: list[str] = Field(default_factory=list)


class ClosureResult(StrictModel):
    """Three-layer closure result."""

    syn: bool
    type: bool
    sem: bool
    reasons: list[str] = Field(default_factory=list)


class MVEU(StrictModel):
    """Minimal verifiable evidence unit."""

    mveu_id: str
    paper_id: str
    seed_id: str
    template: str
    node_ids: list[str]
    edge_ids: list[tuple[str, str, EdgeType]]
    closure: ClosureResult
    size: int
    span_pages: int
    ambiguity_count: int
    canonical_phi: tuple[Any, ...]
    provenance: list[str]
    status: Literal["valid", "unresolved", "rejected"]
    reject_reason: str | None = None


class TopKItem(StrictModel):
    """One ranked retrieval item."""

    rank: int
    mveu_id_or_chunk_id: str
    score: float
    payload: dict[str, Any]


class TopKResult(StrictModel):
    """Unified top-k retrieval output for all methods."""

    query_id: str
    method: MethodName
    items: list[TopKItem]


class UnresolvedResult(StrictModel):
    """Explicit unresolved result for non-silent failure modes."""

    status: Literal["unresolved"] = "unresolved"
    reason: str
    details: dict[str, Any] = Field(default_factory=dict)
