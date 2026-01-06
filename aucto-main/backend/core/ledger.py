import sqlite3
import uuid
import json
from datetime import datetime, timedelta
from .domain_model import domain_mgr

class Ledger:
    """
    The Immutable Audit Log (v3.1 - Restored Observability).
    Records every Human and AI decision forever.
    Acts as the 'Corporate Memory' for the Decision Engine.
    """
    def __init__(self):
        self.db_path = domain_mgr.db_path

    def log_execution(self, decision_payload: dict, mechanism="MANUAL_COMMIT"):
        """
        Writes a decision to the Claims Ledger.
        
        Args:
            decision_payload: {
                "node_id": str,
                "decision": str, (e.g., 'ORDER', 'MARKDOWN')
                "quantity": float,
                "rationale": str,
                "system_level": int (1=Auto, 2=Human, 3=Debate)
            }
            mechanism: str (e.g., 'SYSTEM_1_AUTO', 'MANUAL_COMMIT', 'COUNCIL_VOTE')
            
        Returns:
            tx_id (str): The unique transaction ID.
        """
        tx_id = f"TX-{uuid.uuid4().hex[:8].upper()}"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Default to System 2 (Human) if not specified
        sys_level = decision_payload.get('system_level', 2)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO claims_ledger 
                    (tx_id, timestamp, node_id, decision, quantity, rationale, system_level, status, mechanism)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tx_id,
                        timestamp,
                        decision_payload.get('node_id'),
                        decision_payload.get('decision'),
                        decision_payload.get('quantity', 0),
                        decision_payload.get('rationale', ''),
                        sys_level,
                        "COMMITTED",
                        mechanism
                    )
                )
                conn.commit()
                
            print(f"[LEDGER] Transaction {tx_id} committed via {mechanism}.")
            return tx_id
        except Exception as e:
            print(f"[LEDGER] Error logging execution: {e}")
            return None

    def get_recent_logs(self, limit=50):
        """
        Retrieves the last N transactions for the Audit View.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute("SELECT * FROM claims_ledger ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            print(f"[LEDGER] Error fetching logs: {e}")
            return []

    def get_transaction(self, tx_id: str):
        """
        Retrieves a specific transaction by ID.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute("SELECT * FROM claims_ledger WHERE tx_id = ?", (tx_id,)).fetchone()
                return dict(row) if row else None
        except Exception:
            return None

    # --- RESTORED OBSERVABILITY METHODS (Fixes 500 Error) ---

    def get_stats(self):
        """
        Returns high-level statistics for the Dashboard (Objects, Events, Decisions).
        Called by /ontology/stats
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 1. Object Count (Catalog Size)
                try:
                    obj_count = conn.execute("SELECT COUNT(*) FROM universal_objects").fetchone()[0]
                except sqlite3.OperationalError:
                    obj_count = 0

                # 2. Event Count (Signal Volume)
                try:
                    event_count = conn.execute("SELECT COUNT(*) FROM universal_events").fetchone()[0]
                except sqlite3.OperationalError:
                    event_count = 0
                
                # 3. Decision Count (Actions Taken)
                try:
                    decision_count = conn.execute("SELECT COUNT(*) FROM claims_ledger").fetchone()[0]
                except sqlite3.OperationalError:
                    decision_count = 0
                
                return {
                    "objects": obj_count,
                    "events": event_count,
                    "decisions": decision_count,
                    "status": "Active",
                    "system_health": "Good",
                    "last_updated": datetime.now().strftime("%H:%M:%S")
                }
        except Exception as e:
            print(f"[LEDGER] Stats Error: {e}")
            # Return safe defaults so UI doesn't crash
            return {"objects": 0, "events": 0, "decisions": 0, "status": "Error"}

    def get_daily_summary(self):
        """
        Returns decision counts grouped by System Level (Auto vs Human).
        Used for the 'Autonomy Score' chart.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute("""
                    SELECT system_level, COUNT(*) as count 
                    FROM claims_ledger 
                    GROUP BY system_level
                """).fetchall()
                
                summary = {1: 0, 2: 0, 3: 0} # 1=Auto, 2=Human, 3=Debate
                for r in rows:
                    if r[0] in summary:
                        summary[r[0]] = r[1]
                
                return summary
        except Exception:
            return {1: 0, 2: 0, 3: 0}

# Singleton Instance
ledger = Ledger()
