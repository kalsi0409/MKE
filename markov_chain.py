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
        """Builds chronological linear paths per user, tracking conversions or drops."""
        # Join clicks and conversions to flag paths
        joined = self.clicks.merge(self.convs, on='user_id', how='left')
        
        # Filter out clicks that happened after a conversion timestamp
        valid_clicks = joined[
            joined['conversion_timestamp'].isna() | 
            (joined['timestamp'] <= joined['conversion_timestamp'])
        ].copy()
        
        # Sort chronologically to get authentic paths
        valid_clicks = valid_clicks.sort_values(by=['user_id', 'timestamp'])
        
        # Group channels into a list per user
        user_paths = valid_clicks.groupby('user_id')['channel'].apply(list).to_dict()
        converted_users = set(self.convs['user_id'].unique())
        
        formatted_paths = []
        for uid, path in user_paths.items():
            if str(uid) in converted_users or int(uid) in converted_users:
                # Path successfully reached conversion state
                formatted_paths.append(['Start'] + path + ['Conversion'])
            else:
                # Path abandoned / failed to convert
                formatted_paths.append(['Start'] + path + ['Null'])
                
        return formatted_paths

    def _compute_transition_matrix(self, paths):
        """Calculates structural state-to-state transition probabilities."""
        transitions = defaultdict(lambda: defaultdict(int))
        
        for path in paths:
            for i in range(len(path) - 1):
                transitions[path[i]][path[i+1]] += 1
                
        matrix = defaultdict(dict)
        for state, next_states in transitions.items():
            total_transitions = sum(next_states.values())
            for n_state, count in next_states.items():
                matrix[state][n_state] = count / total_transitions
                
        return matrix

    def _calculate_base_conversion_probability(self, matrix):
        """Simulates conversion rate through matrix algebra or iterative paths."""
        # Simplified structural path tracker for base probability
        return self._simulate_conversion_rate(matrix, omitted_channel=None)

    def _simulate_conversion_rate(self, matrix, omitted_channel=None, iterations=10000):
        """Simulates path traversals to find expected conversion rate."""
        conversions = 0
        for _ in range(iterations):
            state = 'Start'
            while state not in ['Conversion', 'Null']:
                next_states = matrix[state]
                if omitted_channel and omitted_channel in next_states:
                    # Dynamically remove omitted channel and re-normalize probabilities
                    filtered_states = {k: v for k, v in next_states.items() if k != omitted_channel}
                    if not filtered_states:
                        state = 'Null'
                        break
                    total = sum(filtered_states.values())
                    probs = {k: v/total for k, v in filtered_states.items()}
                else:
                    probs = next_states
                
                state = np.random.choice(list(probs.keys()), p=list(probs.values()))
                if state == 'Conversion':
                    conversions += 1
                    
        return conversions / iterations

    def compute_markov_attribution(self):
        """Calculates Removal Effect index and distributes total revenue accordingly."""
        paths = self._build_journeys()
        matrix = self._compute_transition_matrix(paths)
        
        unique_channels = list(self.clicks['channel'].unique())
        base_prob = self._calculate_base_conversion_probability(matrix)
        
        removal_effects = {}
        total_removal_effect = 0
        
        # Calculate individual impact of removing each channel
        for channel in unique_channels:
            prob_without_channel = self._simulate_conversion_rate(matrix, omitted_channel=channel)
            # Removal Effect formula: 1 - (Prob without channel / Base Prob)
            removal_effect = max(0, 1.0 - (prob_without_channel / base_prob))
            removal_effects[channel] = removal_effect
            total_removal_effect += removal_effect
            
        # Distribute actual total revenue proportionally to removal effect weights
        total_revenue = self.convs['revenue'].sum()
        markov_revenue = {}
        
        for channel in unique_channels:
            weight = removal_effects[channel] / total_removal_effect
            markov_revenue[channel] = round(total_revenue * weight, 2)
            
        df_markov = pd.DataFrame(list(markov_revenue.items()), columns=['channel', 'markov_revenue'])
        return df_markov