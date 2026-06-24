import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class MarketingDataFactory:
    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(self.seed)
        self.channels = ['Paid Social', 'Paid Search', 'Email Marketing', 'Organic Search', 'Affiliate']
        
        # Base conversion probabilities per channel touch
        self.base_conv_probs = {
            'Paid Social': 0.015,
            'Paid Search': 0.040,
            'Email Marketing': 0.075,
            'Organic Search': 0.010,
            'Affiliate': 0.025
        }
        
    def generate_clickstream(self, num_users=10000):
        """
        Generates realistic multi-channel click logs and subsequent 
        conversions with intentional funnel positioning bias.
        """
        click_logs = []
        conversion_logs = []
        start_date = datetime(2026, 1, 1)
        
        print(f"STATUS: Synthesizing marketing journeys for {num_users} users...")
        
        for uid in range(100000, 100000 + num_users):
            # 1. Determine path length (1 to 6 touchpoints before converting or dropping)
            path_length = np.random.choice([1, 2, 3, 4, 5, 6], p=[0.35, 0.25, 0.20, 0.10, 0.06, 0.04])
            
            user_time = start_date + timedelta(days=np.random.randint(0, 45), 
                                               seconds=np.random.randint(0, 86400))
            has_converted = False
            
            for position in range(path_length):
                # Apply structural funnel probabilities based on path location
                if position == 0:
                    probs = [0.40, 0.15, 0.10, 0.25, 0.10] # Top of funnel (Social/Organic heavy)
                elif position == path_length - 1:
                    probs = [0.10, 0.40, 0.35, 0.05, 0.10] # Bottom of funnel (Search/Email heavy)
                else:
                    probs = [0.20, 0.25, 0.25, 0.15, 0.15] # Mid-funnel distribution
                    
                channel = np.random.choice(self.channels, p=probs)
                user_time += timedelta(hours=np.random.randint(1, 72))
                
                click_logs.append({
                    'user_id': str(uid),
                    'timestamp': user_time,
                    'channel': channel,
                    'interaction_order': position + 1
                })
                
                # 2. Dynamic conversion chance compounding with interaction experience
                conversion_chance = self.base_conv_probs[channel] * (1.15 ** position)
                
                if np.random.rand() < conversion_chance and not has_converted:
                    has_converted = True
                    conversion_time = user_time + timedelta(minutes=np.random.randint(5, 45))
                    revenue = round(float(np.random.exponential(scale=65.0) + 15.0), 2)
                    
                    conversion_logs.append({
                        'user_id': str(uid),
                        'conversion_timestamp': conversion_time,
                        'revenue': revenue,
                        'order_id': f"ORD_{uid}_{np.random.randint(1000, 9999)}"
                    })
                    break # User converts; journey successfully ends
                    
        return pd.DataFrame(click_logs), pd.DataFrame(conversion_logs)

    def save_data(self, clicks, conversions, output_dir="data/raw"):
        """Safely saves dataframes to disk directory."""
        os.makedirs(output_dir, exist_ok=True)
        clicks.to_csv(f"{output_dir}/user_clicks.csv", index=False)
        conversions.to_csv(f"{output_dir}/conversions.csv", index=False)
        print(f"SUCCESS: Successfully exported datasets to '{output_dir}/'")