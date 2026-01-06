import sqlite3
import pandas as pd
import numpy as np
import json
import random
import uuid
from datetime import datetime, timedelta
from core.domain_model import domain_mgr

# --- CONFIGURATION ---
DB_PATH = domain_mgr.db_path
DAYS_HISTORY = 400
START_DATE = datetime.now() - timedelta(days=DAYS_HISTORY)

print(f"🌍 INITIALIZING PUMA RETAIL UNIVERSE")
print(f"📅 Generating {DAYS_HISTORY} days of history from {START_DATE.date()}...")

# --- 1. THE CAST OF CHARACTERS (CATALOG) ---
# Each product represents a specific "Test Case" for the AI.

CATALOG = [
    # --- DOMAIN 1: REPLENISHMENT ---
    {
        "id": "PUMA-PALERMO-GRN",
        "name": "Puma Palermo (Viral Green)",
        "attributes": {
            "category": "Footwear", "brand": "Puma", "cost": 45.0, "lead_time_days": 14, "moq": 50,
            "dna": "VIRAL_SPIKE" # SCENARIO A: Viral TikTok Trend
        },
        "base_price": 90.0
    },
    {
        "id": "PUMA-VELOCITY-3",
        "name": "Velocity Nitro 3",
        "attributes": {
            "category": "Running", "brand": "Puma", "cost": 60.0, "lead_time_days": 60, "moq": 100,
            "dna": "SUPPLY_SHOCK" # SCENARIO B: Long Lead Time Risk
        },
        "base_price": 130.0
    },
    {
        "id": "PUMA-ESS-TEE-BLK",
        "name": "Essentials Logo Tee",
        "attributes": {
            "category": "Apparel", "brand": "Puma", "cost": 8.0, "lead_time_days": 7, "moq": 500,
            "dna": "PHANTOM" # SCENARIO C: Phantom Inventory (Theft)
        },
        "base_price": 25.0
    },

    # --- DOMAIN 2: PRICING ---
    {
        "id": "PUMA-SUEDE-CL",
        "name": "Suede Classic XXI",
        "attributes": {
            "category": "Classics", "brand": "Puma", "cost": 35.0, "lead_time_days": 14, "moq": 50,
            "dna": "LOSS_LEADER" # SCENARIO D: Competitor Pricing Trap
        },
        "base_price": 75.0
    },
    {
        "id": "PUMA-NO1-LOGO",
        "name": "No.1 Logo Tee",
        "attributes": {
            "category": "Apparel", "brand": "Puma", "cost": 10.0, "lead_time_days": 10, "moq": 200,
            "dna": "INELASTIC" # SCENARIO E: Margin Opportunity
        },
        "base_price": 30.0
    },

    # --- DOMAIN 3: MARKDOWN ---
    {
        "id": "PUMA-RSX-PREPPY",
        "name": "RS-X Preppy",
        "attributes": {
            "category": "Footwear", "brand": "Puma", "cost": 55.0, "lead_time_days": 21, "moq": 20,
            "dna": "BROKEN_SIZE" # SCENARIO F: Sizes 13 & 6 only
        },
        "base_price": 110.0
    },
    {
        "id": "FERRARI-RACE-JKT",
        "name": "Scuderia Ferrari Race Jacket",
        "attributes": {
            "category": "Motorsport", "brand": "Ferrari", "cost": 80.0, "lead_time_days": 45, "moq": 10,
            "dna": "BRAND_EQUITY" # SCENARIO G: Do not discount
        },
        "base_price": 180.0
    },

    # --- DOMAIN 4: ALLOCATION ---
    {
        "id": "MB-03-TOXIC",
        "name": "MB.03 Toxic (LaMelo)",
        "attributes": {
            "category": "Basketball", "brand": "Puma", "cost": 65.0, "lead_time_days": 30, "moq": 50,
            "dna": "OMNI_RESCUE" # SCENARIO H: DC Empty, Store Full
        },
        "base_price": 125.0
    }
]

def setup_catalog(conn):
    print("📦 Seeding Catalog...")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM universal_objects WHERE obj_type='PRODUCT'")
    for item in CATALOG:
        cursor.execute(
            "INSERT INTO universal_objects (obj_id, obj_type, name, attributes) VALUES (?, ?, ?, ?)",
            (item['id'], 'PRODUCT', item['name'], json.dumps(item['attributes']))
        )
    conn.commit()

# --- 2. THE SIMULATOR ENGINE ---

def generate_events(conn):
    print("📈 Generating 400 Days of Retail Reality...")
    events = []
    
    # Inventory State
    inv = {item['id']: 200 for item in CATALOG}
    # Specific starting states
    inv["PUMA-ESS-TEE-BLK"] = 5   # Phantom: stuck at 5
    inv["PUMA-RSX-PREPPY"] = 150  # Broken: high stock, no sales
    inv["FERRARI-RACE-JKT"] = 80  # Premium: slow mover
    inv["MB-03-TOXIC"] = 0        # Omni: DC Empty (simulated here as primary node empty)

    cursor = conn.cursor()
    cursor.execute("DELETE FROM universal_events")

    for day_offset in range(DAYS_HISTORY + 1):
        curr_date = START_DATE + timedelta(days=day_offset)
        date_str = curr_date.strftime("%Y-%m-%d")
        
        # Seasonality
        is_weekend = curr_date.weekday() >= 5
        is_summer = 5 <= curr_date.month <= 8
        days_from_now = (datetime.now() - curr_date).days

        for item in CATALOG:
            sku = item['id']
            dna = item['attributes']['dna']
            price = item['base_price']
            
            # --- SCENARIO LOGIC ---
            
            # 1. VIRAL SPIKE (Recent)
            if dna == "VIRAL_SPIKE":
                # Normal demand 10. Last 7 days -> 200!
                base_demand = np.random.normal(10, 2)
                if days_from_now < 7: base_demand = np.random.normal(150, 20) 
            
            # 2. SUPPLY SHOCK
            elif dna == "SUPPLY_SHOCK":
                base_demand = np.random.normal(25, 5) # Steady sales
                # Logic: Even if we reorder, stock stays low because lead time is 60d
            
            # 3. PHANTOM INVENTORY
            elif dna == "PHANTOM":
                # Demand exists (people want it), but sales are 0 because "System" thinks we have 5 but reality is 0?
                # Actually, simulation records SALES. If phantom, Sales = 0.
                base_demand = 0 
                # Note: We simulate the 'Inventory Snapshot' saying 5 later.
            
            # 4. LOSS LEADER TRAP
            elif dna == "LOSS_LEADER":
                base_demand = np.random.normal(30, 5)
                # Competitor drops price below cost ($35) sometimes
                if random.random() < 0.15:
                    comp_price = 30.0 # Danger!
                    events.append((str(uuid.uuid4()), sku, "COMP_PRICE", comp_price, date_str, json.dumps({"source":"CRAWLER"})))
            
            # 5. ELASTICITY TEST
            elif dna == "INELASTIC":
                # We hiked price 30 days ago. Sales didn't drop.
                if days_from_now < 30: price += 5.0 # Hike
                base_demand = np.random.normal(20, 2) # Sales steady despite hike
            
            # 6. BROKEN SIZE
            elif dna == "BROKEN_SIZE":
                base_demand = 0 # No one fits Size 13
                # Inventory stays high
            
            # 7. BRAND EQUITY
            elif dna == "BRAND_EQUITY":
                base_demand = np.random.poisson(0.5) # Very slow
                # Inventory piling up slowly
            
            # 8. OMNI RESCUE
            elif dna == "OMNI_RESCUE":
                base_demand = np.random.normal(40, 5) # High Demand
                # But Inventory is 0 at DC (Primary Node). 
                # This causes Lost Sales in simulation unless we had store nodes (simplified here).
            
            else:
                base_demand = 10

            # --- EXECUTE PHYSICS ---
            
            # Weekend Bump
            if is_weekend and dna not in ["PHANTOM", "BROKEN_SIZE"]:
                base_demand *= 1.4

            # Calculate Sales
            sales = max(0, int(base_demand))
            
            # Constraint: Inventory
            # Special case: Phantom thinks it has 5, but actually has 0. So Sales = 0.
            if dna == "PHANTOM":
                sales = 0 # Reality
            elif inv[sku] < sales:
                sales = inv[sku]
            
            inv[sku] -= sales
            
            # --- LOG EVENTS ---
            
            # 1. Price
            events.append((str(uuid.uuid4()), sku, "PRICE", price, date_str, json.dumps({"source":"SYSTEM"})))
            
            # 2. Sales
            if sales > 0:
                events.append((str(uuid.uuid4()), sku, "SALES_QTY", sales, date_str, json.dumps({"source":"POS"})))
            
            # 3. Inventory Snapshot
            # Phantom Logic: System sees 5 units forever
            snapshot_qty = 5 if dna == "PHANTOM" else inv[sku]
            events.append((str(uuid.uuid4()), sku, "INV_SNAPSHOT", snapshot_qty, date_str, json.dumps({"source":"WMS"})))

            # --- REPLENISHMENT ---
            # Basic reorder logic to keep simulation alive
            if dna not in ["PHANTOM", "BROKEN_SIZE", "BRAND_EQUITY", "OMNI_RESCUE"] and inv[sku] < 50:
                inv[sku] += 200 # Restock

        if day_offset % 30 == 0:
            print(f"   ... Processed {date_str}")

    print(f"💾 Committing {len(events)} events to database...")
    
    cursor.executemany(
        """
        INSERT INTO universal_events 
        (event_id, primary_target_id, event_type, value, timestamp, meta) 
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        events
    )
    conn.commit()

if __name__ == "__main__":
    try:
        with sqlite3.connect(DB_PATH) as conn:
            setup_catalog(conn)
            generate_events(conn)
        print("\n✅ SIMULATION COMPLETE. The World is Ready.")
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
