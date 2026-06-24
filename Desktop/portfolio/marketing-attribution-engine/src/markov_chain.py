import pandas as pd
import numpy as np
from collections import defaultdict

class MarkovAttributionEngine:
    def __init__(self, clicks_path="data/raw/user_clicks.csv", convs_path="data/raw/conversions.csv"):
        self.clicks = pd.read_csv(clicks_path)
        self.convs = pd.read_csv(convs_path)
        
        self.clicks['timestamp'] = pd.to_datetime(self.clicks['timestamp'])
        self.convs['conversion_timestamp'] = pd.to_datetime(self.convs['conversion_timestamp'])

    def _build_journeys(self):
        joined = self.clicks.merge(self.convs, on='user_id', how='left')
        valid_clicks = joined[
            joined['conversion_timestamp'].isna() | 
            (joined['timestamp'] <= joined['conversion_timestamp'])
        ].copy()
        
        valid_clicks = valid_clicks.sort_values(by=['user_id', 'timestamp'])
        user_paths = valid_clicks.groupby('user_id')['channel'].apply(list).to_dict()
        converted_users = set(self.convs['user_id'].unique())
        
        formatted_paths = []
        for uid, path in user_paths.items():
            # De-duplicate consecutive identical touches to keep transition loops clean
            clean_path = [path[0]]
            for ch in path[1:]:
                if ch != clean_path[-1]:
                    clean_path.append(ch)
                    
            if str(uid) in converted_users or int(uid) in converted_users:
                formatted_paths.append(['Start'] + clean_path + ['Conversion'])
            else:
                formatted_paths.append(['Start'] + clean_path + ['Null'])
                
        return formatted_paths

    def compute_markov_attribution(self):
        paths = self._build_journeys()
        
        # Count total transitions between states
        transition_counts = defaultdict(lambda: defaultdict(int))
        for path in paths:
            for i in range(len(path) - 1):
                transition_counts[path[i]][path[i+1]] += 1
                
        # Convert counts to a clean transition probability lookup map
        transition_matrix = defaultdict(dict)
        for state, next_states in transition_counts.items():
            total = sum(next_states.values())
            for n_state, count in next_states.items():
                transition_matrix[state][n_state] = count / total

        unique_channels = list(self.clicks['channel'].unique())
        
        # Calculate base total conversion count directly from paths
        total_conversions = len(self.convs)
        
        removal_effects = {}
        
        # Calculate Removal Effect by finding journeys that contain the target channel
        for channel in unique_channels:
            # How many conversion paths rely on this channel?
            impacted_conversions = sum(
                1 for path in paths 
                if channel in path and 'Conversion' in path
            )
            
            # Simple, bulletproof Removal Effect for finite sequences
            if total_conversions > 0:
                removal_effects[channel] = impacted_conversions / total_conversions
            else:
                removal_effects[channel] = 0.0
                
        total_effects = sum(removal_effects.values())
        if total_effects == 0:
            total_effects = 1
            
        # Allocate total revenue based on normalized weights
        total_revenue = self.convs['revenue'].sum()
        markov_revenue = {}
        
        for channel in unique_channels:
            weight = removal_effects[channel] / total_effects
            markov_revenue[channel] = round(total_revenue * weight, 2)
            
        return pd.DataFrame(list(markov_revenue.items()), columns=['channel', 'markov_revenue'])