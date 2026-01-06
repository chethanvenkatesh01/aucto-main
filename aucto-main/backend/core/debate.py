import os
import json
import google.generativeai as genai
from typing import Dict, Any

class DebateEngine:
    """
    System 3: The Multi-Modal Council of Agents.
    Simulates domain-specific boardroom debates for ambiguous decisions.
    """
    
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.model = None
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-pro')
        else:
            print("[DEBATE] WARNING: No Gemini API Key. System 3 disabled.")

    def _get_personas(self, mode: str):
        """Returns the specific conflict agents for the given module."""
        matrix = {
            'REPLENISHMENT': {
                'A': {'role': 'VP of Sales', 'goal': 'Maximize Revenue & Availability (Never Stock Out)'},
                'B': {'role': 'CFO', 'goal': 'Preserve Cash Flow & Prevent Overstock'},
                'Judge': {'role': 'CEO', 'goal': 'Balance Growth with Solvency'}
            },
            'PRICING': {
                'A': {'role': 'Chief Revenue Officer', 'goal': 'Maximize Margin % and Yield'},
                'B': {'role': 'Customer Loyalty Director', 'goal': 'Prevent Churn & Protect Price Perception'},
                'Judge': {'role': 'Chief Commercial Officer', 'goal': 'Sustainable Profitability'}
            },
            'MARKDOWN': {
                'A': {'role': 'Inventory Controller', 'goal': 'Clear Dead Stock Immediately (Cash Recovery)'},
                'B': {'role': 'Brand Guardian', 'goal': 'Protect Brand Equity (Avoid Cheap Perception)'},
                'Judge': {'role': 'Merchandising Director', 'goal': 'Clean Exit Strategy'}
            },
            'ASSORTMENT': {
                'A': {'role': 'Efficiency Expert', 'goal': 'Rationalize SKU Count (Kill Low Performers)'},
                'B': {'role': 'Trend Hunter', 'goal': 'Maintain Strategic/Halo Items regardless of volume'},
                'Judge': {'role': 'Head of Product', 'goal': 'Curated & Profitable Mix'}
            },
            'ALLOCATION': {
                'A': {'role': 'Retail Operations', 'goal': 'Maximize On-Shelf Availability in All Stores'},
                'B': {'role': 'Logistics Manager', 'goal': 'Minimize Split Shipments & Transport Costs'},
                'Judge': {'role': 'COO', 'goal': 'Efficient Service Level'}
            }
        }
        return matrix.get(mode, matrix['REPLENISHMENT'])

    def convene_council(self, context: Dict[str, Any], mode: str = 'REPLENISHMENT') -> Dict[str, Any]:
        """
        Orchestrates the debate based on the mode.
        """
        if not self.model:
            return {"decision": "HOLD", "rationale": "AI Offline", "transcript": []}

        # 1. Setup Personas
        personas = self._get_personas(mode)
        item = context.get('name', 'Unknown Item')
        data_summary = f"Stats: {json.dumps(context.get('stats', {}))}"
        
        print(f"[DEBATE] Convening {mode} Council for {item}...")

        # 2. Agent A Argues (The Hawk)
        prompt_a = f"""
        You are the {personas['A']['role']}. Your goal: {personas['A']['goal']}.
        Context: Item '{item}'. {data_summary}.
        Situation: We need to make a bold move. 
        Task: Write 2 sentences arguing WHY we should take aggressive action (Buy More / Hike Price / Slash Price).
        """
        arg_a = self._invoke_agent(prompt_a)

        # 3. Agent B Argues (The Dove)
        prompt_b = f"""
        You are the {personas['B']['role']}. Your goal: {personas['B']['goal']}.
        Context: Item '{item}'. {data_summary}.
        Opposing View: "{arg_a}"
        Task: Write 2 sentences arguing WHY we should be conservative (Hold / Keep Price / Keep Item). 
        Critique the risk of the aggressive move.
        """
        arg_b = self._invoke_agent(prompt_b)

        # 4. The Judge Decides
        prompt_judge = f"""
        You are the {personas['Judge']['role']}. Goal: {personas['Judge']['goal']}.
        
        Item: {item}
        Data: {data_summary}
        
        Argument A ({personas['A']['role']}): "{arg_a}"
        Argument B ({personas['B']['role']}): "{arg_b}"
        
        Task: Issue a binding verdict.
        1. Action: Choose one (ACTION or HOLD).
        2. Modification: If ACTION, specify the % or Quantity (e.g., "Hike 3% instead of 5%" or "Buy 50 units").
        3. Rationale: Summarize why in 1 sentence.
        
        Return ONLY valid JSON: {{ "decision": "ACTION|HOLD", "value": <number>, "rationale": "<text>" }}
        """
        raw_verdict = self._invoke_agent(prompt_judge)
        
        # 5. Parse
        try:
            clean_json = raw_verdict.replace('```json', '').replace('```', '').strip()
            verdict = json.loads(clean_json)
            
            return {
                "decision": verdict.get("decision", "HOLD"),
                "value": verdict.get("value", 0),
                "rationale": verdict.get("rationale", "Council Deadlock"),
                "transcript": [
                    {"agent": personas['A']['role'], "arg": arg_a},
                    {"agent": personas['B']['role'], "arg": arg_b},
                    {"agent": personas['Judge']['role'], "verdict": verdict.get("rationale")}
                ]
            }
        except:
            return {"decision": "HOLD", "rationale": "Council Mistrial", "transcript": []}

    def _invoke_agent(self, prompt: str) -> str:
        try:
            return self.model.generate_content(prompt).text.strip()
        except:
            return "..."

debate_engine = DebateEngine()