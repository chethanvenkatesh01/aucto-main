import sqlite3
import json
from typing import Dict, Any, Tuple
from .domain_model import domain_mgr

class PolicyEngine:
    """
    The Safety Net (v1.0).
    Enforces dynamic business rules to prevent 'Rogue AI' actions.
    Supports Global Defaults with Entity-Specific Overrides.
    """
    
    def __init__(self):
        self.db_path = domain_mgr.db_path
        # Default Safety Nets (Hardcoded fallbacks if DB is empty)
        self.DEFAULTS = {
            "MAX_AUTO_SPEND": 5000.0,       # Max $ value for auto-orders
            "MIN_MARGIN_PCT": 20.0,         # Never price below 20% margin
            "MAX_PRICE_HIKE_PCT": 10.0,     # Don't hike price by >10% in one go
            "MAX_MARKDOWN_DEPTH": 40.0,     # Don't discount more than 40% without human approval
            "SYSTEM_3_TRIGGER": 60.0        # Confidence score < 60% triggers Debate
        }

    def _fetch_policy(self, key: str, entity_id: str = None) -> float:
        """
        Hierarchical Lookup:
        1. Check SKU-Specific Policy
        2. Check GLOBAL Policy
        3. Return Code Default
        """
        with sqlite3.connect(self.db_path) as conn:
            # 1. Try Specific Entity
            if entity_id:
                row = conn.execute(
                    "SELECT policy_value FROM governance_policies WHERE policy_key=? AND entity_id=?", 
                    (key, entity_id)
                ).fetchone()
                if row: return float(json.loads(row[0])['value'])

            # 2. Try Global
            row = conn.execute(
                "SELECT policy_value FROM governance_policies WHERE policy_key=? AND entity_id='GLOBAL'", 
                (key,)
            ).fetchone()
            if row: return float(json.loads(row[0])['value'])

        # 3. Fallback
        return self.DEFAULTS.get(key, 0.0)

    def validate_action(self, action_type: str, value: float, entity_id: str, context: Dict = None) -> Tuple[bool, str]:
        """
        Validates a proposed decision.
        Returns: (Approved: bool, Reason: str)
        """
        # RULE 1: Spending Limits (Replenishment)
        if action_type == 'ORDER':
            limit = self._fetch_policy("MAX_AUTO_SPEND", entity_id)
            cost = value * 50.0 # Mock cost, ideally passed in context
            if context and 'cost' in context: cost = context['cost']
            
            if cost > limit:
                return False, f"Cost ${cost:,.2f} exceeds Auto-Limit (${limit:,.2f})"

        # RULE 2: Price Safety (Pricing)
        elif action_type == 'HIKE':
            # value is the New Price. We need old price to calc % change.
            old_price = context.get('current_price', value)
            if old_price > 0:
                pct_change = ((value - old_price) / old_price) * 100
                max_hike = self._fetch_policy("MAX_PRICE_HIKE_PCT", entity_id)
                if pct_change > max_hike:
                    return False, f"Hike (+{pct_change:.1f}%) exceeds safety cap ({max_hike}%)"

        # RULE 3: Margin Protection (Markdown)
        elif action_type == 'MARKDOWN':
            # value is the % off
            max_depth = self._fetch_policy("MAX_MARKDOWN_DEPTH", entity_id)
            if value > max_depth:
                return False, f"Markdown ({value}%) exceeds max depth ({max_depth}%)"

        return True, "Approved"

    def get_all_policies(self):
        """Returns all active rules for the UI."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM governance_policies").fetchall()
            
            # Merge with defaults for display
            policies = []
            seen_keys = set()
            
            # DB Policies
            for r in rows:
                val = json.loads(r['policy_value'])
                policies.append({
                    "entity_id": r['entity_id'],
                    "key": r['policy_key'],
                    "value": val['value'],
                    "source": "DATABASE"
                })
                if r['entity_id'] == 'GLOBAL': seen_keys.add(r['policy_key'])
            
            # Default Policies (if not overridden globally)
            for k, v in self.DEFAULTS.items():
                if k not in seen_keys:
                    policies.append({
                        "entity_id": "GLOBAL",
                        "key": k,
                        "value": v,
                        "source": "CODE_DEFAULT"
                    })
            
            return policies

    def set_policy(self, key: str, value: float, entity_id: str = "GLOBAL"):
        """Updates or Creates a policy."""
        json_val = json.dumps({"value": value, "updated_at": "now"})
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO governance_policies (entity_id, policy_key, policy_value) VALUES (?,?,?)",
                (entity_id, key, json_val)
            )
            conn.commit()

policy_engine = PolicyEngine()