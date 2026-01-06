import pandas as pd
import sqlite3
from .domain_model import domain_mgr

class FeatureStore:
    """
    The Data Refinery (v9.7 - Batch Optimized).
    Transforms raw Graph Events into an ML-ready Matrix.
    """
    
    def __init__(self):
        self.db_path = domain_mgr.db_path

    def build_master_table(self) -> pd.DataFrame:
        """
        Pivots the Event Graph and aligns features for 'Future-Aware' training.
        """
        with sqlite3.connect(self.db_path) as conn:
            # 1. Fetch ALL relevant events
            query = """
                SELECT 
                    timestamp as date,
                    primary_target_id as node_id,
                    event_type,
                    value
                FROM universal_events 
                WHERE event_type IN ('SALES_QTY', 'PRICE', 'PROMO_FLAG')
                ORDER BY date ASC
            """
            df = pd.read_sql(query, conn)
        
        if df.empty:
            return pd.DataFrame()

        # 2. Pivot: Turn Event Types into Columns
        df_pivot = df.pivot_table(
            index=['date', 'node_id'], 
            columns='event_type', 
            values='value', 
            aggfunc='mean'
        ).reset_index()

        # 3. Fill Missing Data (Imputation)
        for col in ['SALES_QTY', 'PRICE', 'PROMO_FLAG']:
            if col not in df_pivot.columns:
                df_pivot[col] = 0 if col != 'PRICE' else 50.0

        df_pivot['SALES_QTY'] = df_pivot['SALES_QTY'].fillna(0)
        df_pivot['PRICE'] = df_pivot.groupby('node_id')['PRICE'].ffill().bfill()
        df_pivot['PROMO_FLAG'] = df_pivot['PROMO_FLAG'].fillna(0)

        # 4. Feature Engineering
        df_pivot['date_obj'] = pd.to_datetime(df_pivot['date'])
        df_pivot['day_of_week'] = df_pivot['date_obj'].dt.dayofweek
        df_pivot['is_weekend'] = (df_pivot['day_of_week'] >= 5).astype(int)
        
        df_pivot['lag_1'] = df_pivot.groupby('node_id')['SALES_QTY'].shift(1)
        df_pivot['ma_7'] = df_pivot.groupby('node_id')['SALES_QTY'].transform(
            lambda x: x.shift(1).rolling(7).mean()
        )

        # 5. Cleanup
        df_clean = df_pivot.dropna().drop(columns=['date_obj'])
        
        return df_clean

    def get_latest_features(self, node_id: str) -> pd.DataFrame:
        """Single node fetch (Legacy/Slow)"""
        df = self.build_master_table()
        if df.empty: return None
        node_data = df[df['node_id'] == node_id]
        if node_data.empty: return None
        return node_data.iloc[[-1]]

    def get_all_latest_features(self) -> pd.DataFrame:
        """
        Batch fetch (Fast). 
        Returns the most recent row for EVERY node in one DataFrame.
        """
        df = self.build_master_table()
        if df.empty: return pd.DataFrame()
        # Group by node_id and take the last row (latest date)
        return df.groupby('node_id').tail(1)

# Singleton Instance
feature_store = FeatureStore()
