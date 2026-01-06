import sqlite3

INIT_STMTS = [
    # =========================================================
    # 1. CORE UNIVERSAL LEDGER (THE SOURCE OF TRUTH)
    # =========================================================
    """
    CREATE TABLE IF NOT EXISTS universal_objects (
        obj_id TEXT PRIMARY KEY,
        obj_type TEXT NOT NULL, -- 'PRODUCT', 'LOCATION', 'CUSTOMER'
        name TEXT,
        attributes JSON, -- { "category": "Footwear", "brand": "Puma" }
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS universal_events (
        event_id TEXT PRIMARY KEY,
        primary_target_id TEXT NOT NULL,
        event_type TEXT NOT NULL, -- 'SALES_QTY', 'PRICE', 'INV_SNAPSHOT'
        value REAL,
        timestamp TIMESTAMP,
        meta JSON, -- { "source": "POS", "is_promo": true }
        FOREIGN KEY(primary_target_id) REFERENCES universal_objects(obj_id)
    )
    """,
    
    # =========================================================
    # 2. FINANCIAL LEDGER (THE BANK)
    # =========================================================
    """
    CREATE TABLE IF NOT EXISTS ledger_entries (
        entry_id TEXT PRIMARY KEY,
        entity_id TEXT,
        account_type TEXT, -- 'REVENUE', 'COGS', 'OPEX'
        amount REAL,
        currency TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        meta JSON
    )
    """,
    
    # =========================================================
    # 3. GOVERNANCE (THE LAW)
    # =========================================================
    """
    CREATE TABLE IF NOT EXISTS policy_store (
        policy_key TEXT PRIMARY KEY, -- 'MIN_MARGIN_PCT'
        entity_id TEXT, -- 'GLOBAL' or Specific SKU
        value REAL,
        config JSON,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,

    # =========================================================
    # 4. INTELLIGENCE PLANE (THE BRAIN)
    # =========================================================
    
    # A. The Demand Hypercube (Fast Path)
    """
    CREATE TABLE IF NOT EXISTS demand_hypercube (
        id TEXT PRIMARY KEY,
        sku_id TEXT NOT NULL,
        week_start_date TEXT NOT NULL, -- ISO8601 Date (Monday)
        run_id TEXT, 
        model_used TEXT, 
        confidence_score REAL, 
        base_demand INTEGER, 
        elasticity_coeff REAL, 
        elasticity_vector JSON, 
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(sku_id) REFERENCES universal_objects(obj_id)
    )
    """,
    
    # B. The Forecast Audit Log (Rich Path)
    """
    CREATE TABLE IF NOT EXISTS forecast_audit_log (
        audit_id TEXT PRIMARY KEY,
        sku_id TEXT NOT NULL,
        run_id TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        data_quality_score REAL,
        cleansing_log JSON,
        feature_manifest JSON,
        tournament_results JSON,
        driver_importance JSON,
        FOREIGN KEY(sku_id) REFERENCES universal_objects(obj_id)
    )
    """,

    # C. Forecast Snapshots (The Time Capsule - NEW)
    # Used to verify "What did we predict 3 months ago for today?"
    """
    CREATE TABLE IF NOT EXISTS forecast_snapshots (
        snapshot_id TEXT PRIMARY KEY,
        sku_id TEXT NOT NULL,
        target_date TEXT NOT NULL, -- The date being predicted
        generated_at TEXT NOT NULL, -- When the prediction was made
        lag_weeks INTEGER, -- 4, 12, 24
        forecast_qty INTEGER,
        FOREIGN KEY(sku_id) REFERENCES universal_objects(obj_id)
    )
    """,

    # D. Accuracy Matrix (The Scorecard - NEW)
    # Stores pre-calculated WMAPE for the Dashboard Heatmap
    """
    CREATE TABLE IF NOT EXISTS accuracy_matrix (
        node_id TEXT, -- 'GLOBAL', 'DIV-FOOTWEAR', 'SKU-123'
        lag_bucket TEXT, -- '1_MONTH', '3_MONTH', '6_MONTH'
        wmape_score REAL, -- 0.0 to 1.0 (Lower is better, or inverted to accuracy %)
        bias_score REAL, -- +10% means over-forecasted
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (node_id, lag_bucket)
    )
    """,
    
    # =========================================================
    # 5. PLANNING PLANE (THE HUMAN LAYER)
    # =========================================================
    """
    CREATE TABLE IF NOT EXISTS plan_overrides (
        id TEXT PRIMARY KEY,
        entity_id TEXT NOT NULL,
        period_id TEXT NOT NULL,
        user_adjustment_pct REAL,
        user_adjustment_qty INTEGER,
        rationale TEXT,
        user_id TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
]

def init_db(db_path):
    """
    Initializes the SQLite database with the full Auctorian Schema.
    Idempotent: Can be run multiple times safely.
    """
    with sqlite3.connect(db_path) as conn:
        for stmt in INIT_STMTS:
            try:
                conn.execute(stmt)
            except sqlite3.OperationalError as e:
                print(f"⚠️ Schema Warning: {e}")
    print(f"✅ Database Schema Initialized at {db_path}")
