import os
import json
import google.generativeai as genai
from typing import Dict, Any, List, Optional
from core.schema import ConstraintEnvelope

# --- ADOS V5 COMPONENT: RETAIL CARTRIDGE (Full Production) ---
# Features:
# 1. Multi-Agent Loop (Analyst/Strategist/Governance).
# 2. Feasibility Adapter (Read/Write to 'retail_db.json').
# 3. Batch Safety (Defaults to System 0 for unknown data).

class FeasibilityAdapter:
    """
    The Data Plane Adapter. 
    Manages the 'State of the World' (Inventory, Pricing, Sales).
    """
    def __init__(self, data_path="data/retail_db.json"):
        self.data_path = data_path
        self._cache = {}
        self._load_data()

    def _load_data(self):
        try:
            # Resolve absolute path for Docker stability
            base_path = os.getcwd() # backend/ is the workdir in Docker
            if 'backend' not in base_path: base_path = os.path.join(base_path, 'backend')
            
            full_path = os.path.join(base_path, self.data_path)
            
            if os.path.exists(full_path):
                with open(full_path, 'r') as f:
                    self._cache = json.load(f)
                print(f"[FEASIBILITY] Loaded {len(self._cache)} SKUs.")
            else:
                print(f"[FEASIBILITY] No DB found at {full_path}. Starting empty.")
                self._cache = {}
        except Exception as e:
            print(f"[ERROR] Feasibility Load Failed: {e}")

    def get_sku_context(self, sku: str) -> Dict:
        """Read Path: Retrieves live metrics."""
        return self._cache.get(sku, {"error": "SKU_NOT_FOUND_IN_ERP"})

    def upsert_sku_data(self, sku: str, data: Dict):
        """Write Path: Updates the ERP state."""
        if sku not in self._cache:
            self._cache[sku] = {}
        # Merge new data
        self._cache[sku].update(data)
        self._save_to_disk()

    def _save_to_disk(self):
        # In PROD, this would be an SQL COMMIT
        try:
            base_path = os.getcwd()
            if 'backend' not in base_path: base_path = os.path.join(base_path, 'backend')
            full_path = os.path.join(base_path, self.data_path)
            
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w') as f:
                json.dump(self._cache, f, indent=2)
        except Exception as e:
            print(f"[ERROR] DB Save Failed: {e}")

class RetailCartridge:
    def __init__(self):
        # Initialize Gemini
        api_key = os.environ.get("GEMINI_API_KEY")
        self.has_ai = False
        if api_key:
            genai.configure(api_key=api_key)
            self.has_ai = True
            # V5.5: Explicitly use stable model alias
            self.model_name = 'gemini-1.5-flash-latest' 
        
        # Initialize the Sensory Organ
        self.feasibility = FeasibilityAdapter()

    def assess_complexity(self, payload: Dict[str, Any]) -> int:
        """
        Determines if we need the AI Boardroom.
        V5 Fix: Defaults to System 0 (Reflex) to prevent Batch Crashes.
        """
        req_type = payload.get("type", "unknown").lower()
        value = payload.get("value", 0)
        
        # Explicit System 2 Triggers
        if req_type in ["new_product_launch", "competitor_response", "strategic_pivot"]:
            return 2
        
        # System 1 Triggers
        if req_type == "mark_down" and value < 5000: return 1
        if req_type == "restock_staples": return 0
        
        # DEFAULT SAFEGUARD: If unknown, treat as System 0 (Rule Based)
        # This prevents 1 billion rows from calling Gemini.
        return 0

    def execute_reflex(self, payload: Dict[str, Any]) -> Dict:
        """System 0: Deterministic Rules (No AI Cost)"""
        sku = payload.get("sku", "UNKNOWN")
        req_type = payload.get("type", "general_update")
        
        return {
            "rationale": f"System 0: Processed '{req_type}' via standard rules engine.",
            "actions": [{"action": "log_event", "sku": sku, "status": "processed"}]
        }

    async def execute_routine(self, payload: Dict[str, Any], constraints: ConstraintEnvelope) -> Dict:
        """System 1: Optimization Math (CPU only)"""
        sku = payload.get("sku")
        price = payload.get("current_price", 100)
        suggested_price = price * 0.90 # Simple elasticity heuristic
        return {
            "rationale": "System 1: Optimization Algo V2 executed.",
            "actions": [{"action": "update_price", "sku": sku, "price": suggested_price}]
        }

    async def deliberate(self, payload: Dict[str, Any], constraints: ConstraintEnvelope) -> Dict:
        """System 2: The Virtual Boardroom (Calls Gemini)"""
        if not self.has_ai:
            return {"rationale": "AI Offline", "actions": []}

        print(f"[RETAIL BOARDROOM] Session Started for: {payload.get('type')}")
        
        # --- ROUND 1: THE ANALYST (Context Gathering) ---
        analyst_context = await self._agent_analyst_gather(payload, constraints)
        print(f"[RETAIL BOARDROOM] Analyst Brief: {analyst_context[:100]}...")

        # --- ROUND 2: THE STRATEGIST (Planning) ---
        strategist_proposal = await self._agent_strategist_plan(payload, analyst_context, constraints)
        
        # --- ROUND 3: THE ANALYST (Governance Critique) ---
        final_decision = await self._agent_analyst_critique(strategist_proposal, constraints)
        
        return final_decision

    # --- AGENT DEFINITIONS (RESTORED) ---

    async def _agent_analyst_gather(self, payload: Dict, constraints: ConstraintEnvelope) -> str:
        """
        Role: Fact Checker. 
        Action: Queries FeasibilityAdapter for REAL data.
        """
        sku = payload.get("sku")
        
        # 1. TOOL CALL: Fetch Real Data
        if sku:
            live_data = self.feasibility.get_sku_context(sku)
        else:
            live_data = {"note": "No specific SKU provided. Using general context."}

        prompt = f"""
        ROLE: You are the Senior Retail Analyst (Auctorian System).
        TASK: specific factual briefing based ONLY on the provided live data.
        
        REQUEST:
        {json.dumps(payload, indent=2)}
        
        FEASIBILITY STORE (LIVE ERP DATA):
        {json.dumps(live_data, indent=2)}
        
        CONSTRAINTS:
        Risk Tolerance: {constraints.risk_tolerance}
        Budget Cap: {constraints.budget_cap}
        
        OUTPUT:
        Summarize the inventory health, price gaps, and margin room. 
        Highlight any immediate risks (e.g., "Inventory is 15% below safety stock").
        """
        
        model = genai.GenerativeModel(self.model_name, generation_config=genai.GenerationConfig(temperature=0.0))
        response = model.generate_content(prompt)
        return response.text

    async def _agent_strategist_plan(self, payload: Dict, analyst_brief: str, constraints: ConstraintEnvelope) -> Dict:
        """Role: Strategic Planner."""
        prompt = f"""
        ROLE: You are the Chief Retail Strategist.
        TASK: Propose optimal actions based on the Analyst's briefing.
        
        ANALYST BRIEFING:
        {analyst_brief}
        
        OBJECTIVE:
        Maximize margin without triggering a stockout.
        
        OUTPUT SCHEMA (JSON):
        {{
            "rationale": "Reasoning...",
            "actions": [{{"action": "name", "params": {{...}}}}]
        }}
        """
        model = genai.GenerativeModel(self.model_name, generation_config=genai.GenerationConfig(temperature=0.7))
        response = model.generate_content(prompt)
        return self._clean_json(response.text)

    async def _agent_analyst_critique(self, proposal: Dict, constraints: ConstraintEnvelope) -> Dict:
        """Role: Governance Auditor."""
        prompt = f"""
        ROLE: You are the Governance Auditor.
        TASK: Validate the Strategist's plan against constraints.
        
        PLAN:
        {json.dumps(proposal, indent=2)}
        
        CONSTRAINTS:
        Budget: {constraints.budget_cap}
        
        INSTRUCTIONS:
        If the plan violates 'Margin Floor' or 'Budget', rewrite the actions to be compliant.
        Return the FINAL JSON.
        """
        model = genai.GenerativeModel(self.model_name, generation_config=genai.GenerationConfig(temperature=0.0))
        response = model.generate_content(prompt)
        return self._clean_json(response.text)

    def _clean_json(self, text: str) -> Dict:
        try:
            text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except:
            return {"rationale": "JSON Parsing Error in Agent Loop", "actions": []}
