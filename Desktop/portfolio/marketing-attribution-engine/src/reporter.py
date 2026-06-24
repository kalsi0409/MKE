import os
import pandas as pd

class ExecutiveReporter:
    @staticmethod
    def generate_markdown_report(final_df, output_path="data/Executive_Marketing_Report.txt"):
        """Generates a clean text/markdown corporate report for executive stakeholders."""
        total_rev = final_df['linear_revenue'].sum()
        
        # Isolate the channel with the highest variance between Last-Touch and Markov
        final_df['variance'] = final_df['markov_revenue'] - final_df['last_revenue']
        most_undervalued = final_df.sort_values(by='variance', ascending=False).iloc[0]
        most_overvalued = final_df.sort_values(by='variance', ascending=True).iloc[0]
        
        report_content = f"""==========================================================================
EXECUTIVE MARKETING PERFORMANCE & BUDGET OPTIMIZATION REPORT
Generated Framework: Algorithmic Attribution Engine (Markov Model)
==========================================================================

[1] SYSTEM OVERVIEW METRICS:
--------------------------------------------------------------------------
* Gross Portfolio Attributed Revenue  : ${total_rev:,.2f}
* Monitored Programmatic Channels     : {len(final_df)} Active Niche Networks
* Core Mathematical Optimization Basis: Markov Chain Absorption Path Auditing

[2] INTEGRATED MULTI-MODEL COMPARISON MATRIX:
--------------------------------------------------------------------------
{final_df[['channel', 'first_revenue', 'last_revenue', 'position_based_revenue', 'markov_revenue']].to_string(index=False)}

[3] CORE STRATEGIC BUDGET REALLOCATION INSIGHTS:
--------------------------------------------------------------------------
* STRATEGIC OPPORTUNITY (SCALE SPEND):
  The '{most_undervalued['channel']}' network is structurally undervalued by ${abs(most_undervalued['variance']):,.2f} 
  when evaluating through legacy Last-Touch constraints. Markov Chain path evaluation proves its high value 
  introducing user paths. Recommendation: INCREASE budget allocation immediately.

* AD WASTE MITIGATION (REDUCE SPEND):
  The '{most_overvalued['channel']}' network is over-credited by ${abs(most_overvalued['variance']):,.2f} 
  using traditional Last-Touch models. It routinely captures late transactional conversions that would have 
  occurred organically or via alternative assists. Recommendation: LOWER budget exposure by 15-20%.

==========================================================================
END OF REPORT -- AUTOMATED ANALYTICS PIPELINE ENGINE REPORTING LAYER
==========================================================================
"""
        with open(output_path, "w") as f:
            f.write(report_content)
        print(f"REPORT GENERATION SUCCESS: Corporate executive asset exported cleanly to '{output_path}'")