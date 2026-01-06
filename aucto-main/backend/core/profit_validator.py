from typing import Dict, Any

class ProfitValidator:
    """
    The Economic Guardian.
    Rejects any decision that destroys Net Present Value (NPV) or violates Brand DNA.
    """
    def __init__(self, dna_config: Dict[str, Any]):
        self.dna = dna_config
        self.wacc = self.dna['physics'].get('wacc_daily', 0.0003)
        self.hurdle = self.dna['physics'].get('hurdle_rate_roi', 0.15)

    def validate(self, action_type: str, revenue_impact: float, 
                 cost_impact: float, duration_days: int) -> Dict[str, Any]:
        """
        Calculates Economic Profit (EVA) and ROI to validate decisions.
        """
        # 1. Gross Profit
        gross_profit = revenue_impact - cost_impact
        
        # 2. Opportunity Cost (The "Hidden" Tax)
        # Cost of tying up cash in inventory instead of investing it.
        # Formula: Capital * Daily_Rate * Days_Held
        capital_cost = cost_impact * (self.wacc * duration_days)
        
        # 3. Economic Value Added (EVA) - The True Profit
        economic_profit = gross_profit - capital_cost
        
        # 4. Return on Investment (ROI)
        roi = (economic_profit / cost_impact) if cost_impact > 0 else 0.0
            
        # 5. The Verdict
        if economic_profit < 0:
            return {
                "approved": False, 
                "reason": f"Destroys Value. Economic Loss: ${abs(economic_profit):.2f} (CapCost: ${capital_cost:.2f})"
            }
            
        if roi < self.hurdle:
            return {
                "approved": False, 
                "reason": f"Inefficient Capital. ROI {roi*100:.1f}% < Hurdle {self.hurdle*100:.0f}%"
            }
            
        return {
            "approved": True, 
            "reason": f"Accretive. ROI: {roi*100:.1f}%. Adds ${economic_profit:.2f} Value."
        }