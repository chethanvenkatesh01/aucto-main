from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
import uuid

# --- ADOS V4 PROTOCOL: THE UNIVERSAL SCHEMA ---

class UniversalContext(BaseModel):
    """
    Meta-data visible to the Operating System.
    The OS uses this for routing, logging, and access control.
    """
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    user_role: str
    trace_id: str

class ConstraintEnvelope(BaseModel):
    """
    Global Guardrails applied to ALL cartridges.
    The 'Budget Governor' and 'Risk Policy' inject these values.
    """
    budget_cap: Optional[float] = None
    risk_tolerance: Literal['low', 'medium', 'high'] = 'medium'
    required_approvals: List[str] = []

class DomainContext(BaseModel):
    """
    The Opaque Envelope.
    The OS passes this to the Cartridge but acts as a 'dumb pipe' regarding its contents.
    """
    domain: str  # e.g., "retail", "healthcare"
    payload: Dict[str, Any]  # The messy domain data (SKUs, patient records, etc.)

class DecisionRequest(BaseModel):
    universal_context: UniversalContext
    constraints: ConstraintEnvelope
    domain_context: DomainContext

class DecisionResponse(BaseModel):
    request_id: str
    status: Literal['executed', 'needs_approval', 'rejected']
    system_level: Literal[0, 1, 2]  # 0=Reflex, 1=Routine, 2=Strategic
    rationale: str
    actions: List[Dict[str, Any]]
    claims_ledger_id: str  # The Proof of Thought