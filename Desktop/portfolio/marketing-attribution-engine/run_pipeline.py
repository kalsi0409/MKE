from src.data_factory import MarketingDataFactory
from src.heuristics import HeuristicAttributionEngine
from src.markov_chain import MarkovAttributionEngine
from src.reporter import ExecutiveReporter
import pandas as pd

def main():
    print("STARTING PIPELINE: Marketing Analytics Attribution Pipeline...")
    
    # --- PHASE 1: DATA SYNTHESIS ---
    factory = MarketingDataFactory(seed=42)
    clicks_df, conversions_df = factory.generate_clickstream(num_users=25000)
    factory.save_data(clicks_df, conversions_df)
    
    # --- PHASE 2: CALCULATING HEURISTIC MODELS ---
    print("\nPHASE 2: CALCULATING HEURISTIC MODELS (First, Last, Linear, U-Shaped)...")
    h_engine = HeuristicAttributionEngine()
    heuristic_results = h_engine.compute_attributions()
    
    # --- PHASE 3: CALCULATING MARKOV ALGORITHMIC MODEL ---
    print("\nPHASE 3: RUNNING MARKOV CHAIN ALGORITHMIC ENGINE...")
    m_engine = MarkovAttributionEngine()
    markov_results = m_engine.compute_markov_attribution()
    
    # --- PHASE 4: FINAL INTEGRATED LOOK ---
    final_comparison = heuristic_results.merge(markov_results, on='channel')
    
    print("\n==========================================================================")
    print("FINAL INTEGRATED MARKETING ANALYTICS PORTFOLIO VIEW:")
    print("==========================================================================")
    print(final_comparison.to_string(index=False))
    print("==========================================================================")
    
    # --- PHASE 5: ENTERPRISE AUTOMATED REPORTING EXPORT ---
    print("\nPHASE 5: INITIALIZING AUTOMATED REPORTING PIPELINE EXPORT...")
    ExecutiveReporter.generate_markdown_report(final_comparison)

if __name__ == "__main__":
    main()