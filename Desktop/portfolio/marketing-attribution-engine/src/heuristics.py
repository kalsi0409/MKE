import pandas as pd
import numpy as np

class HeuristicAttributionEngine:
    def __init__(self, clicks_path="data/raw/user_clicks.csv", convs_path="data/raw/conversions.csv"):
        self.clicks = pd.read_csv(clicks_path)
        self.convs = pd.read_csv(convs_path)
        
        self.clicks['timestamp'] = pd.to_datetime(self.clicks['timestamp'])
        self.convs['conversion_timestamp'] = pd.to_datetime(self.convs['conversion_timestamp'])
        
    def _prepare_converted_journeys(self):
        joined = self.clicks.merge(self.convs, on='user_id', how='inner')
        valid_touches = joined[joined['timestamp'] <= joined['conversion_timestamp']].copy()
        
        valid_touches['true_position'] = valid_touches.sort_values('timestamp').groupby('user_id').cumcount() + 1
        valid_touches['max_position'] = valid_touches.groupby('user_id')['true_position'].transform('max')
        
        # Calculate time delta in days between the touchpoint and conversion
        valid_touches['days_to_conversion'] = (valid_touches['conversion_timestamp'] - valid_touches['timestamp']).dt.total_seconds() / 86400.0
        
        return valid_touches

    def compute_attributions(self):
        df = self._prepare_converted_journeys()
        
        # 1. First-Touch
        df['is_first'] = df['true_position'] == 1
        df['first_revenue'] = df['revenue'] * df['is_first']
        
        # 2. Last-Touch
        df['is_last'] = df['true_position'] == df['max_position']
        df['last_revenue'] = df['revenue'] * df['is_last']
        
        # 3. Linear
        df['linear_weight'] = 1.0 / df['max_position']
        df['linear_revenue'] = df['revenue'] * df['linear_weight']
        
        # 4. Position-Based (U-Shaped)
        def calculate_u_shaped_weight(row):
            pos = row['true_position']
            max_pos = row['max_position']
            if max_pos == 1: return 1.0
            elif max_pos == 2: return 0.5 if (pos == 1 or pos == 2) else 0.0
            else:
                if pos == 1 or pos == max_pos: return 0.40
                else: return 0.20 / (max_pos - 2)
                    
        df['u_shaped_weight'] = df.apply(calculate_u_shaped_weight, axis=1)
        df['position_based_revenue'] = df['revenue'] * df['u_shaped_weight']
        
        # 5. ADVANCED ADDITION: Exponential Time-Decay Model (7-Day Half-Life)
        half_life = 7.0
        df['decay_weight'] = 2 ** (-df['days_to_conversion'] / half_life)
        
        # Normalize weights per user profile so total revenue distribution adds up to exactly 100%
        df['total_decay_weight'] = df.groupby('user_id')['decay_weight'].transform('sum')
        df['normalized_decay_weight'] = df['decay_weight'] / df['total_decay_weight']
        df['time_decay_revenue'] = df['revenue'] * df['normalized_decay_weight']
        
        # Aggregate summaries
        summary = df.groupby('channel').agg({
            'first_revenue': 'sum',
            'last_revenue': 'sum',
            'linear_revenue': 'sum',
            'position_based_revenue': 'sum',
            'time_decay_revenue': 'sum'
        }).round(2).reset_index()
        
        return summary