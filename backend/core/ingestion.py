import csv
import io
import re
import json
import hashlib
import sqlite3
from datetime import datetime
from typing import Dict, Any, List
from .domain_model import domain_mgr

class IngestionEngine:
    """
    Universal ETL Processor (v3.0 - Dynamic Ontology).
    Handles arbitrary CSVs by mapping them to Objects (Nouns) or Events (Verbs).
    Includes backward compatibility for v1/v2 API calls.
    """
    
    def __init__(self):
        self.BATCH_SIZE = 2000

    def _generate_dedup_key(self, ev_type, target, loc, time_str):
        """Creates a unique hash to enforce Idempotency."""
        raw = f"{ev_type}|{target}|{loc}|{time_str}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _clean_number(self, val: str) -> float:
        if not val: return 0.0
        # Robust cleaner: removes currency symbols, handles negative signs
        clean_val = re.sub(r'[^\d.-]', '', str(val))
        try:
            return float(clean_val)
        except ValueError:
            return 0.0

    def _standardize_date(self, val: str) -> str:
        """Parses MM/DD/YYYY, YYYY-MM-DD, etc. to ISO format."""
        if not val: return None
        val = val.strip()
        formats = [
            "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d", 
            "%d-%m-%Y", "%Y.%m.%d", "%d.%m.%Y"
        ]
        for fmt in formats:
            try:
                return datetime.strptime(val, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        return None

    def _get_reader(self, file_content: str):
        if file_content.startswith('\ufeff'): file_content = file_content[1:]
        first_line = file_content.split('\n')[0]
        delimiter = ','
        if '\t' in first_line: delimiter = '\t'
        elif ';' in first_line: delimiter = ';'
        elif '|' in first_line: delimiter = '|'
        return csv.DictReader(io.StringIO(file_content), delimiter=delimiter)

    def _match_header(self, row_keys, user_map_col):
        """Fuzzy matches user mapping to CSV header."""
        if not user_map_col: return None
        clean_target = user_map_col.strip().lower().replace('_', '')
        for key in row_keys:
            if key.strip().lower().replace('_', '') == clean_target:
                return key
        return None

    # --- CORE UNIVERSAL INGESTION ROUTER ---
    
    def process_generic_stream(self, file_content: str, config: Dict[str, Any]):
        """
        The One Function to Rule Them All.
        config = {
            'type': 'EVENT' | 'OBJECT',
            'entity_name': 'competitor_prices' | 'products',
            'mapping': { 'SKU': 'ItemCode', 'Price': 'RetailPrice', ... }
        }
        """
        if config.get('type') == 'OBJECT':
            return self._ingest_objects(file_content, config)
        else:
            return self._ingest_events(file_content, config)

    def _ingest_objects(self, content, config):
        reader = self._get_reader(content)
        mapping = config['mapping']
        obj_type = config['entity_name'].upper()
        
        batch = []
        rows_processed = 0
        
        # Identify Key Columns
        first_row = True
        key_col = None
        
        with sqlite3.connect(domain_mgr.db_path) as conn:
            for row in reader:
                if first_row:
                    key_col = self._match_header(row.keys(), mapping.get('ID'))
                    if not key_col: return {"status": "error", "message": "Primary ID column not found in CSV"}
                    first_row = False
                
                obj_id = row[key_col].strip()
                if not obj_id: continue
                
                # Pack everything else into JSON Attributes
                attributes = {}
                for csv_header, val in row.items():
                    if csv_header != key_col:
                        attributes[csv_header] = val.strip()
                
                # Use Name if mapped, else ID
                name_col = self._match_header(row.keys(), mapping.get('Name'))
                name = row[name_col].strip() if name_col and row.get(name_col) else obj_id

                batch.append((obj_id, obj_type, name, json.dumps(attributes)))
                rows_processed += 1
                
                if len(batch) >= self.BATCH_SIZE:
                    conn.executemany("INSERT OR REPLACE INTO universal_objects (obj_id, obj_type, name, attributes) VALUES (?,?,?,?)", batch)
                    conn.commit()
                    batch = []
            
            if batch:
                conn.executemany("INSERT OR REPLACE INTO universal_objects (obj_id, obj_type, name, attributes) VALUES (?,?,?,?)", batch)
                conn.commit()

        return {"status": "success", "count": rows_processed, "type": obj_type}

    def _ingest_events(self, content, config):
        reader = self._get_reader(content)
        mapping = config['mapping']
        event_type = config['entity_name'].upper() # e.g. 'COMP_PRICE'
        
        batch = []
        errors = []
        rows_processed = 0
        
        # Header Resolution
        first_row = True
        headers = {}
        
        with sqlite3.connect(domain_mgr.db_path) as conn:
            for row in reader:
                if first_row:
                    headers['target'] = self._match_header(row.keys(), mapping.get('Target_ID')) # SKU
                    headers['date'] = self._match_header(row.keys(), mapping.get('Date'))
                    headers['val'] = self._match_header(row.keys(), mapping.get('Value'))
                    headers['loc'] = self._match_header(row.keys(), mapping.get('Location_ID')) # Optional
                    
                    if not headers['target'] or not headers['date']:
                        return {"status": "error", "message": "Missing Target_ID or Date in mapping"}
                    first_row = False

                try:
                    # 1. Extract Core Data
                    target_id = row[headers['target']].strip()
                    date_str = self._standardize_date(row[headers['date']])
                    if not target_id or not date_str: continue
                    
                    val = 0.0
                    if headers.get('val') and row.get(headers['val']): 
                        val = self._clean_number(row[headers['val']])
                    
                    loc_id = "GLOBAL"
                    if headers.get('loc') and row.get(headers['loc']): 
                        loc_id = row[headers['loc']].strip()

                    # 2. Extract Meta Data (Anything not core)
                    meta = {}
                    for k, v in row.items():
                        if k not in headers.values():
                            meta[k] = v
                    
                    # 3. Generate Keys
                    dedup_key = self._generate_dedup_key(event_type, target_id, loc_id, date_str)
                    event_id = f"EVT_{dedup_key[:12]}" # Short ID

                    batch.append((event_id, event_type, date_str, target_id, loc_id, val, json.dumps(meta), dedup_key))
                    rows_processed += 1

                except Exception as e:
                    if len(errors) < 5: errors.append(f"Row {rows_processed}: {str(e)}")
                    continue

                if len(batch) >= self.BATCH_SIZE:
                    self._flush_events(conn, batch)
                    batch = []
            
            if batch:
                self._flush_events(conn, batch)

        return {"status": "success", "count": rows_processed, "errors": errors}

    def _flush_events(self, conn, batch):
        sql = """
            INSERT OR REPLACE INTO universal_events 
            (event_id, event_type, timestamp, primary_target_id, secondary_target_id, value, meta, dedup_key) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        conn.executemany(sql, batch)
        conn.commit()

    # --- COMPATIBILITY WRAPPERS (For Existing Main.py) ---

    def process_catalog_stream(self, file_content: str, mapping: Dict[str, str]):
        """Legacy wrapper for Product Catalog ingestion."""
        # Maps legacy mapping keys to new Universal Object keys
        new_mapping = {
            'ID': mapping.get('SKU', 'SKU'),
            'Name': mapping.get('Name', 'Name')
        }
        config = {
            'type': 'OBJECT',
            'entity_name': 'PRODUCT',
            'mapping': new_mapping
        }
        return self.process_generic_stream(file_content, config)

    def process_location_stream(self, file_content: str, mapping: Dict[str, str]):
        """Legacy wrapper for Location ingestion."""
        new_mapping = {
            'ID': mapping.get('Store', 'Store'),
            'Name': mapping.get('Name', 'Name')
        }
        config = {
            'type': 'OBJECT',
            'entity_name': 'LOCATION',
            'mapping': new_mapping
        }
        return self.process_generic_stream(file_content, config)

    def process_metric_stream(self, file_content: str, mapping: Dict[str, str], metric_prefix: str = 'SALES'):
        """Legacy wrapper for Sales/Inventory ingestion."""
        # Maps legacy mapping keys to new Universal Event keys
        new_mapping = {
            'Target_ID': mapping.get('SKU', 'SKU'),
            'Date': mapping.get('Date', 'Date'),
            'Value': mapping.get('Qty', 'Qty'),
            'Location_ID': mapping.get('Store', 'Store')
        }
        config = {
            'type': 'EVENT',
            'entity_name': f"{metric_prefix}_QTY", # e.g. SALES_QTY
            'mapping': new_mapping
        }
        return self.process_generic_stream(file_content, config)

# Singleton Instance
ingestion_engine = IngestionEngine()

# --- HELPER FOR MAIN.PY IMPORT ---
# This wrapper function allows main.py to call `ingest_file` directly
def ingest_file(file_obj, config, filename):
    """Wrapper to handle file reading and call the engine."""
    try:
        # Read SpooledTemporaryFile to string
        content = file_obj.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        return ingestion_engine.process_generic_stream(content, config)
    except Exception as e:
        return {"status": "error", "message": f"File read error: {str(e)}"}
