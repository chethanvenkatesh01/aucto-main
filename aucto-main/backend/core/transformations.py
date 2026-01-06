import sqlite3
import pandas as pd
import hashlib
import json
from .domain_model import domain_mgr

class TransformationEngine:
    """
    The Alchemist (v2.0).
    Derives new Events from existing Events in the Universal Graph.
    e.g. SALES_REVENUE = SALES_QTY * PRICE
    """
    def __init__(self):
        self.db_path = domain_mgr.db_path

    def derive_metric(self, target_metric: str, metric_a: str, op: str, metric_b: str):
        """
        Executes Vectorized Math on the Graph.
        target_metric: 'SALES_REV'
        metric_a: 'SALES_QTY'
        op: 'MULTIPLY'
        metric_b: 'PRICE' (or a static number)
        """
        with sqlite3.connect(self.db_path) as conn:
            # 1. Load Data Metric A
            query_a = "SELECT timestamp, primary_target_id, value as val_a FROM universal_events WHERE event_type=?"
            df_a = pd.read_sql(query_a, conn, params=(metric_a,))
            
            if df_a.empty:
                return {"status": "error", "message": f"Metric A ({metric_a}) not found."}

            # 2. Handle Metric B (could be a scalar or a time-series)
            try:
                # Case 1: Scalar Operation (e.g. Sales * 1.2)
                scalar_b = float(metric_b)
                df_a['val_b'] = scalar_b
                df_merged = df_a
            except ValueError:
                # Case 2: Time-Series Operation (e.g. Sales * Price)
                query_b = "SELECT timestamp, primary_target_id, value as val_b FROM universal_events WHERE event_type=?"
                df_b = pd.read_sql(query_b, conn, params=(metric_b,))
                
                # Merge on Date + Node
                df_merged = pd.merge(df_a, df_b, on=['timestamp', 'primary_target_id'], how='inner')

            if df_merged.empty:
                return {"status": "warning", "message": "No intersection found between metrics."}

            # 3. Compute
            if op == 'MULTIPLY' or op == '*':
                df_merged['result'] = df_merged['val_a'] * df_merged['val_b']
            elif op == 'DIVIDE' or op == '/':
                # Handle division by zero
                df_merged['result'] = df_merged.apply(lambda x: x['val_a'] / x['val_b'] if x['val_b'] != 0 else 0, axis=1)
            elif op == 'ADD' or op == '+':
                df_merged['result'] = df_merged['val_a'] + df_merged['val_b']
            elif op == 'SUBTRACT' or op == '-':
                df_merged['result'] = df_merged['val_a'] - df_merged['val_b']
            else:
                 return {"status": "error", "message": "Invalid Operator. Use ADD, SUBTRACT, MULTIPLY, DIVIDE."}

            # 4. Write Back as New Events
            events_to_insert = []
            for _, row in df_merged.iterrows():
                # Generate unique ID for this derived event
                raw_key = f"{target_metric}|{row['primary_target_id']}|GLOBAL|{row['timestamp']}"
                dedup_key = hashlib.md5(raw_key.encode()).hexdigest()
                event_id = f"CALC_{dedup_key[:12]}"
                
                # Metadata
                meta = json.dumps({"source": "DERIVED", "formula": f"{metric_a} {op} {metric_b}"})

                events_to_insert.append((
                    event_id, 
                    target_metric, 
                    row['timestamp'], 
                    row['primary_target_id'], 
                    'GLOBAL', 
                    row['result'], 
                    meta, 
                    dedup_key
                ))

            # Bulk Upsert
            sql = """
                INSERT OR REPLACE INTO universal_events 
                (event_id, event_type, timestamp, primary_target_id, secondary_target_id, value, meta, dedup_key) 
                VALUES (?,?,?,?,?,?,?,?)
            """
            conn.executemany(sql, events_to_insert)
            conn.commit()

            return {
                "status": "success", 
                "target_metric": target_metric,
                "rows_generated": len(events_to_insert)
            }

transform_engine = TransformationEngine()
