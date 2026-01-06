import sqlite3
import json
import uuid
from typing import List, Dict, Optional, Any
# FIX: Import the initialization function instead of the missing variable
from .sql_schema import init_db 

class DomainManager:
    """
    Guardian of the Universal Graph (v3.0).
    Manages the flexible Ontology (Objects, Events, Relationships).
    Includes Compatibility Adapters for v1 Frontend calls.
    """
    def __init__(self, db_path="ados_ledger.db"):
        self.db_path = db_path
        self._init_db()        
        self._ensure_indices()

    def _init_db(self):
        """Initializes the new Graph Schema."""
        # FIX: Delegate schema creation to the sql_schema module
        init_db(self.db_path)

    def _ensure_indices(self):
        """Performance optimizations for the Graph."""
        with sqlite3.connect(self.db_path) as conn:
            # Fast lookup for Time-Series aggregations
            conn.execute("CREATE INDEX IF NOT EXISTS idx_evt_agg ON universal_events(event_type, timestamp)")
            # Fast lookup for Object attributes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_obj_lookup ON universal_objects(obj_type, obj_id)")
            conn.commit()

    # =========================================================
    # 1. NEW UNIVERSAL GRAPH API (The Future)
    # =========================================================

    def get_objects(self, obj_type: str) -> List[Dict]:
        """Fetches Nouns (Products, Locations) from the Universal Store."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM universal_objects WHERE obj_type = ?", 
                (obj_type,)
            ).fetchall()
            
            results = []
            for r in rows:
                item = dict(r)
                # Unpack JSON attributes for easier consumption
                if item.get('attributes'):
                    try:
                        attrs = json.loads(item['attributes'])
                        item.update(attrs)
                    except: pass
                results.append(item)
            return results

    def get_events(self, event_type: str, target_id: str = None, limit: int = 100) -> List[Dict]:
        """Fetches Verbs (Sales, Prices) from the Event Store."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            query = "SELECT * FROM universal_events WHERE event_type = ?"
            params = [event_type]
            
            if target_id:
                query += " AND primary_target_id = ?"
                params.append(target_id)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def get_stats(self):
        """Telemetry for the Command Center."""
        with sqlite3.connect(self.db_path) as conn:
            try:
                objs = conn.execute("SELECT COUNT(*) FROM universal_objects").fetchone()[0]
                evts = conn.execute("SELECT COUNT(*) FROM universal_events").fetchone()[0]
                return {"objects": objs, "events": evts, "status": "Graph Active"}
            except:
                return {"objects": 0, "events": 0, "status": "Graph Empty"}

    # =========================================================
    # 2. LEGACY COMPATIBILITY LAYER (The Bridge)
    # Adapts the new Graph Schema to the old 'nodes'/'metrics' format
    # so the Frontend and Orchestrator don't break immediately.
    # =========================================================

    def get_table(self, level_name: str) -> List[Dict]:
        """
        Legacy Adapter: Maps 'universal_objects' -> old 'nodes' format.
        Called by Data Explorer.
        """
        # Map old types to new generic types if necessary, or just pass through
        obj_type = level_name  # e.g., 'PRODUCT', 'LOCATION'
        
        objects = self.get_objects(obj_type)
        
        # Transform to match old 'nodes' schema expected by UI
        adapted = []
        for o in objects:
            row = {
                "node_id": o['obj_id'],
                "name": o['name'],
                "type": o['obj_type'],
                "parent_id": o.get('parent_id'), # Might be missing in flat object store
                # Flatten attributes
                **{k: v for k, v in o.items() if k not in ['obj_id', 'obj_type', 'name', 'attributes']}
            }
            adapted.append(row)
        return adapted

    def get_metrics(self, limit=100, offset=0, metric_filter=None) -> List[Dict]:
        """
        Legacy Adapter: Maps 'universal_events' -> old 'node_metrics' format.
        Called by Data Plane View.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            query = "SELECT * FROM universal_events"
            params = []
            
            if metric_filter and metric_filter != 'TRANSACTIONS':
                query += " WHERE event_type LIKE ?"
                params.append(f"{metric_filter}%")
            
            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            rows = conn.execute(query, params).fetchall()
            
            adapted = []
            for r in rows:
                # FIX: Handle missing secondary_target_id in new schema
                loc_id = r['secondary_target_id'] if 'secondary_target_id' in r.keys() else None
                
                # Map Graph Event -> Old Metric Row
                adapted.append({
                    "node_id": r['primary_target_id'],
                    "location_id": loc_id,
                    "date": r['timestamp'],
                    "metric_type": r['event_type'],
                    "value": r['value']
                })
            return adapted

    def get_levels(self, tree_type: str = "PRODUCT") -> List[str]:
        """
        Mock Adapter: Returns standard levels since we don't use the old hierarchy table anymore.
        """
        if tree_type == "PRODUCT":
            return ["Category", "SubCategory", "SKU"]
        elif tree_type == "LOCATION":
            return ["Region", "District", "Store"]
        return ["Level 1", "Level 2"]

    # Helper for Orchestrator (Migration)
    def define_levels(self, levels, tree_type):
        pass # No-op in new schema

    def add_node(self, node_id, name, node_type, parent_id=None, scenario="LIVE"):
        """Legacy helper for single node insertion (used by old tests)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO universal_objects (obj_id, obj_type, name, attributes) VALUES (?,?,?,?)",
                (node_id, node_type, name, json.dumps({"parent_id": parent_id}))
            )
            conn.commit()

# Singleton Instance
domain_mgr = DomainManager()
