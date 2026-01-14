from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import json
import uvicorn
import os
import time
import logging

# --- SOVEREIGN CORE IMPORTS ---
from core.local_llm import sovereign_brain
from cartridges.retail.controller import FeasibilityAdapter

# --- LEGACY SYSTEM IMPORTS ---
from core.ingestion import ingestion_engine
from core.orchestrator import orchestrator
from core.feature_store import feature_store
from core.ledger import ledger
from core.policy_engine import policy_engine
from core.domain_model import domain_mgr
from core.sql_schema import init_db

# --- AUXILIARY ENGINES (Graceful Load) ---
try:
    from core.transformations import transform_engine
except ImportError:
    transform_engine = None

try:
    from ml_engine import ml_engine
except ImportError:
    ml_engine = None

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("AUCTORIAN_KERNEL")

# --- APP INITIALIZATION ---
app = FastAPI(
    title="Auctorian Sovereign Node",
    version="6.1.0-Monolith",
    description="The Autonomous Merchant Operating System (Local Inference Edition)"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# 1. THE V6.1 BOOT SEQUENCE
# ==============================================================================

@app.on_event("startup")
async def boot_sequence():
    """
    Auctorian V6.1 Startup Protocol:
    1. Initialize Database (SQL)
    2. Physics Check (Verify GPU/Ollama Access)
    3. Context Hydration (Load Retail Data to RAM)
    """
    logger.info("🟢 [SYSTEM] Initiating Auctorian Boot Sequence...")
    
    # 1. SQL Initialization
    init_db("ados_ledger.db")
    logger.info("💽 [STORAGE] SQL Ledger Initialized.")

    # 2. Physics Check (The "Beast" Wake-up Call)
    try:
        start_time = time.time()
        logger.info("🔌 [PHYSICS] Pinging Local Inference Node...")
        health_check = sovereign_brain.generate("Ping.", role="analyst")
        latency = time.time() - start_time
        logger.info(f"⚡ [PHYSICS] GPU Online. Latency: {latency:.2f}s. Response: {health_check}")
    except Exception as e:
        logger.critical(f"🔥 [PHYSICS FATAL] Sovereign Compute Node Unreachable: {e}")
        logger.warning("⚠️ System running in Headless Mode (No Intelligence). Check Ollama.")

    # 3. Context Hydration (The "Photographic Memory")
    try:
        logger.info("🧠 [MEMORY] Hydrating Retail Context...")
        adapter = FeasibilityAdapter()
        count = len(adapter._cache.get("products", []))
        logger.info(f"🧠 [MEMORY] Hydration Complete. Active Context: {count} SKUs.")
    except Exception as e:
        logger.error(f"❌ [MEMORY] Hydration Failed: {e}")

    logger.info("🚀 [SYSTEM] Auctorian Sovereign Node is ONLINE and READY.")


# ==============================================================================
# 2. DECISION ENDPOINTS (The "Soul")
# ==============================================================================

class DecisionRequest(BaseModel):
    type: str
    target: str
    params: Optional[Dict] = {}

@app.post("/orchestrator/decide")
async def orchestrate_decision(request: DecisionRequest):
    """
    The Primary Decision Pipeline.
    Routes traffic to the RetailCartridge (running on Local Silicon).
    """
    logger.info(f"📥 [INGRESS] Decision Request: {request.type} for {request.target}")
    try:
        decision = await orchestrator.process_decision(request.type, request.target, request.params)
        return decision
    except Exception as e:
        logger.error(f"❌ [KERNEL PANIC] Decision Failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/orchestrator/ledger")
async def get_ledger():
    return ledger.get_recent_claims(limit=50)


# ==============================================================================
# 3. INGESTION ENDPOINTS (The "Mouth")
# ==============================================================================

@app.post("/ingest/upload")
async def upload_file(file: UploadFile = File(...)):
    """Accepts CSV/Excel uploads for local processing."""
    try:
        file_location = f"data_lake/{file.filename}"
        os.makedirs("data_lake", exist_ok=True)
        with open(file_location, "wb+") as f:
            f.write(file.file.read())
        
        summary = ingestion_engine.preview_file(file_location)
        return {"status": "success", "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class MappingRequest(BaseModel):
    filename: str
    mapping: Dict[str, str]

@app.post("/ingest/map")
async def map_schema(req: MappingRequest):
    return ingestion_engine.apply_mapping(req.filename, req.mapping)


# ==============================================================================
# 4. ML & TRANSFORMATION ENDPOINTS (Restored)
# ==============================================================================

class MetricDerivation(BaseModel):
    target: str
    metric_a: str
    op: str
    metric_b: str

@app.post("/transform/derive_metric")
async def derive_metric(req: MetricDerivation):
    """Restored from V5: Allows deriving new metrics on the fly."""
    if not transform_engine: 
        return {"error": "Transform Engine Offline"}
    return transform_engine.derive_metric(req.target, req.metric_a, req.op, req.metric_b)

@app.get("/ml/predict")
async def predict(sku: str, days: int = 7):
    """Forecasting endpoint."""
    if not ml_engine: 
        return {"error": "ML Engine not loaded"}
    return ml_engine.generate_forecast(sku, days)

@app.get("/ml/metrics")
async def get_ml_metrics():
    """Restored from V5: Health stats of the ML models."""
    if not ml_engine: 
        return {}
    return ml_engine.get_metrics()


# ==============================================================================
# 5. SYSTEM HEALTH (The "Vitals")
# ==============================================================================

@app.get("/")
def health_check():
    """
    Reports the status of the Sovereign Node.
    """
    return {
        "status": "SOVEREIGN", 
        "system": "Auctorian Kernel v6.1-Monolith", 
        "architecture": "Local Inference (Llama-3)",
        "modules": {
            "physics_layer": "ONLINE",
            "memory_layer": "HYDRATED",
            "transform_engine": "ONLINE" if transform_engine else "OFFLINE",
            "cloud_dependency": "NONE"
        }
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

