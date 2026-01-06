import csv
import io
import json
import uuid
from typing import List, Optional
from .schema import DecisionRequest, UniversalContext, DomainContext, ConstraintEnvelope

# --- ADOS V5 COMPONENT: DATA PROCESSOR ---
# Responsibility: Ingest raw files (CSV/JSON) and convert to Decision Envelopes.

class DataProcessor:
    @staticmethod
    def parse_csv_to_requests(file_content: bytes) -> List[DecisionRequest]:
        requests = []
        try:
            # Decode bytes to string
            text = file_content.decode('utf-8')
            reader = csv.DictReader(io.StringIO(text))
            
            for row in reader:
                # 1. Extract Constraints (if present in CSV)
                risk = row.pop("risk_tolerance", "medium")
                budget = float(row.pop("budget_cap", 10000.0))
                
                # 2. Extract Context (request_id, etc.)
                req_id = row.pop("request_id", str(uuid.uuid4()))
                
                # 3. Remainder is the Payload
                # Auto-convert numbers
                payload = {}
                for k, v in row.items():
                    try:
                        payload[k] = float(v) if '.' in v else int(v)
                    except:
                        payload[k] = v

                # 4. Construct Envelope
                req = DecisionRequest(
                    universal_context=UniversalContext(
                        request_id=req_id,
                        user_role="batch_uploader",
                        trace_id=str(uuid.uuid4())
                    ),
                    domain_context=DomainContext(
                        domain="retail", # Default to Retail for now
                        payload=payload
                    ),
                    constraints=ConstraintEnvelope(
                        risk_tolerance=risk,
                        budget_cap=budget
                    )
                )
                requests.append(req)
                
        except Exception as e:
            print(f"[PROCESSOR] CSV Parse Error: {e}")
            return []
            
        return requests
