from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import json
import uvicorn
import os
import sqlite3

# --- CORE SYSTEMS ---
from core.ingestion import ingestion_engine
from core.orchestrator import orchestrator
from core.feature_store import feature_store
from core.ledger import ledger
from core.policy_engine import policy_engine
from core.domain_model import domain_mgr
from core.sql_schema import init_db

# --- TRANSFORMATION ENGINE (Graceful Load) ---
try:
    from core.transformations import transform_engine
except ImportError:
    transform_engine = None
    print("⚠️ [KERNEL] Transformation Engine: OFFLINE")

# --- AI LAYER (Gemini) ---
try:
    import google.generativeai as genai
    GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
    if GEMINI_KEY:
        genai.configure(api_key=GEMINI_KEY)
        print("🤖 [KERNEL] Gemini AI: ONLINE")
    else:
        print("⚠️ [KERNEL] Gemini AI: OFFLINE (Missing API Key)")
except ImportError:
    pass

# --- ML ENGINE (Graceful Load) ---
try:
    from ml_engine import ml_engine
except ImportError:
    ml_engine = None
    print("⚠️ [KERNEL] ML Engine: OFFLINE (Running heuristics)")

app = FastAPI(
    title="Auctorian ADOS Kernel", 
    version="10.30",
    description="Retail OS: PlanSmart Architecture (Self-Correcting Edition)"
)

# [CRITICAL FIX] CORS Configuration for Local Development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# 0. GLOBAL STATE & MODELS
# ==============================================================================

ACTIVE_ONTOLOGY = {
    "PRODUCT": ["Division", "Department", "Category", "SKU"],
    "LOCATION": ["Region", "District", "Store"]
}

class OntologyUpdate(BaseModel):
    type: str
    levels: List[str]

class CouncilRequest(BaseModel):
    node_id: str
    mode: str

class PolicyUpdate(BaseModel):
    key: str
    value: float
    entity_id: str = "GLOBAL"

class MetricDerivation(BaseModel):
    target: str
    metric_a: str
    op: str
    metric_b: str

# ==============================================================================
# 1. ORCHESTRATION: PLANSMART & DECISION ENGINES
# ==============================================================================

@app.get("/orchestration/plan/tree")
async def get_planning_tree(
    hierarchy: str = Query(default='["Division", "Category"]'),
    start_year: int = Query(default=2025),
    horizon_years: int = Query(default=1)
):
    """
    [PLANSMART] Fetches the blended 5-Year Hierarchical Data Cube (Actuals + Forecast).
    """
    return orchestrator.get_planning_tree(hierarchy, start_year, horizon_years)

@app.get("/orchestration/financial_plan")
async def get_financial_plan_summary(compare: str = "LY", group_by: str = "category", time_bucket: str = "Month"):
    """Legacy summary view."""
    return []

@app.post("/orchestration/convene_council")
async def convene_council(req: CouncilRequest):
    return orchestrator.convene_council_for_node(req.node_id, req.mode)

# --- TACTICAL SUB-ENGINES ---

@app.post("/orchestration/run")
async def run_replenishment():
    return orchestrator.run_replenishment_simulation()

@app.post("/orchestration/price_optimize")
async def run_pricing():
    return orchestrator.run_pricing_simulation()

@app.post("/orchestration/markdown")
async def run_markdown():
    return orchestrator.run_markdown_simulation()

@app.get("/orchestration/assortment")
async def get_assortment():
    return orchestrator.run_assortment_simulation()

@app.get("/orchestration/allocation")
async def get_allocation():
    return orchestrator.run_allocation_simulation()

# ==============================================================================
# 2. ONTOLOGY & STRUCTURE
# ==============================================================================

@app.get("/ontology/structure")
def get_ontology_structure(type: str = "PRODUCT"):
    return {"levels": ACTIVE_ONTOLOGY.get(type, [])}

@app.post("/ontology/structure")
def define_ontology_structure(req: OntologyUpdate):
    ACTIVE_ONTOLOGY[req.type] = req.levels
    return {"status": "success", "levels": req.levels}

@app.get("/ontology/stats")
async def get_stats():
    stats = ledger.get_stats()
    graph_stats = domain_mgr.get_stats()
    return {**stats, **graph_stats}

# ==============================================================================
# 3. INGESTION ROUTES
# ==============================================================================

@app.post("/ingest/universal")
async def ingest_universal(file: UploadFile = File(...), config: str = Form(...)):
    try:
        from core.ingestion import ingest_file
        config_dict = json.loads(config)
        return ingest_file(file.file, config_dict, file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/sales")
async def ingest_sales(file: UploadFile = File(...), mapping: str = Form(...), prefix: str = Form("SALES")):
    try:
        from core.ingestion import ingest_file
        map_conf = json.loads(mapping)
        universal_conf = {
            "type": "EVENTS",
            "mapping": map_conf,
            "event_type_override": f"{prefix}_QTY"
        }
        return ingest_file(file.file, universal_conf, file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ingest/contracts")
def get_active_contracts():
    stats = domain_mgr.get_stats()
    contracts = []
    if stats.get('objects', 0) > 0:
        contracts.append({
            "id": "DC-MASTER-01", "datasetName": "Universal Master Data", "source": "CSV Ingestion",
            "frequency": "Ad-hoc", "lastIngested": "Active", "status": "Healthy",
            "qualityScore": 98, "schemaValid": True, "rowCount": stats['objects'], "detectedType": "OBJECT"
        })
    if stats.get('events', 0) > 0:
        contracts.append({
            "id": "DC-STREAM-01", "datasetName": "Transactional Events", "source": "Event Bus",
            "frequency": "Real-time", "lastIngested": "Active", "status": "Healthy",
            "qualityScore": 100, "schemaValid": True, "rowCount": stats['events'], "detectedType": "EVENT"
        })
    return contracts

# ==============================================================================
# 4. GRAPH EXPLORER
# ==============================================================================

@app.get("/graph/objects/{obj_type}")
async def get_objects(obj_type: str):
    return domain_mgr.get_objects(obj_type)

@app.get("/graph/events/{event_type}")
async def get_events(event_type: str, target: Optional[str] = None):
    try:
        with sqlite3.connect(domain_mgr.db_path) as conn:
            query = "SELECT * FROM universal_events WHERE event_type = ?"
            params = [event_type]
            if target:
                query += " AND primary_target_id = ?"
                params.append(target)
            query += " ORDER BY timestamp DESC LIMIT 100"
            import pandas as pd
            df = pd.read_sql(query, conn, params=params)
            return df.to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}

# ==============================================================================
# 5. GOVERNANCE & INTELLIGENCE (THE GLASS BOX)
# ==============================================================================

@app.get("/governance/policies")
async def get_policies():
    return policy_engine.get_all_policies()

@app.post("/governance/policies")
async def update_policy(req: PolicyUpdate):
    policy_engine.update_policy(req.key, req.value, req.entity_id)
    return {"status": "updated"}

# --- INTELLIGENCE ENDPOINTS ---

@app.post("/ml/train")
async def trigger_demand_modeling():
    """
    [GLASS BOX] Triggers the full Auditable & Self-Correcting Intelligence Pipeline.
    Cleanse -> Backtest -> Compete -> Vectorize -> Audit Log.
    """
    if not ml_engine: return {"status": "error", "message": "ML Engine missing"}
    # Note: Using run_demand_pipeline for the new Glass Box features
    return ml_engine.run_demand_pipeline()

@app.get("/ml/explain/{sku_id}")
async def explain_forecast(sku_id: str):
    """
    [GLASS BOX] Retrieves the audit trail (Data Health, Drivers, Tournament)
    for a specific SKU's forecast.
    """
    return orchestrator.get_forecast_explanation(sku_id)

@app.get("/ml/accuracy/{node_id}")
async def get_forecast_accuracy(node_id: str):
    """
    [SELF-CORRECTION] Retrieves the WMAPE Heatmap Scorecard for a specific Node
    and its children. Used for the 'Forecast Accuracy' widget.
    """
    # Handle the frontend sending 'null' or 'undefined' as string
    target = node_id if node_id and node_id not in ['null', 'undefined'] else "GLOBAL"
    return orchestrator.get_accuracy_context(target)

@app.get("/ml/metrics")
async def get_metrics():
    if not ml_engine: return {}
    return ml_engine.get_metrics()

@app.get("/ml/predict")
async def predict(sku: str, days: int = 7):
    if not ml_engine: return {}
    return ml_engine.generate_forecast(sku, days)

@app.post("/transform/derive_metric")
async def derive_metric(req: MetricDerivation):
    if not transform_engine: return {"error": "Transform Engine Offline"}
    return transform_engine.derive_metric(req.target, req.metric_a, req.op, req.metric_b)

# ==============================================================================
# 6. SYSTEM HEALTH
# ==============================================================================

@app.get("/")
def health_check():
    return {
        "status": "ONLINE", 
        "system": "Auctorian Kernel v10.30", 
        "architecture": "Self-Correcting AI (Glass Box)",
        "modules": {
            "ml": "ONLINE" if ml_engine else "OFFLINE",
            "genai": "ONLINE" if os.environ.get("GEMINI_API_KEY") else "OFFLINE",
            "db": domain_mgr.db_path
        }
    }

if __name__ == "__main__":
    # Ensure DB Schema is consistent on startup
    init_db(domain_mgr.db_path)
    
    print("🚀 Starting Auctorian Profit Kernel (Self-Correcting Edition)...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
