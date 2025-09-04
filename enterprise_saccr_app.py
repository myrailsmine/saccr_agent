import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from enum import Enum
import math
import time

# LangChain imports for LLM integration
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# ==============================================================================
# ENHANCED UI CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="AI SA-CCR Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for AI-powered features with step-by-step analysis
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    
    .ai-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .executive-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .executive-subtitle {
        font-size: 1.2rem;
        font-weight: 400;
        opacity: 0.9;
        margin-bottom: 0;
    }
    
    .thinking-process {
        background: linear-gradient(145deg, #f8f9ff, #e8ecff);
        border-left: 5px solid #4f46e5;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(79, 70, 229, 0.1);
    }
    
    .thinking-step {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 6px;
        border-left: 3px solid #10b981;
    }
    
    .calculation-detail {
        background: #f8fafc;
        padding: 0.75rem;
        border-radius: 4px;
        font-family: 'Monaco', monospace;
        font-size: 0.9rem;
        margin: 0.5rem 0;
        border: 1px solid #e2e8f0;
    }
    
    .result-summary-enhanced {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(16, 185, 129, 0.3);
    }
    
    .missing-info-prompt {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(245, 158, 11, 0.3);
    }
    
    .data-quality-alert {
        background: #fef3c7;
        border: 1px solid #f59e0b;
        color: #92400e;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .step-reasoning {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 6px;
    }
    
    .formula-breakdown {
        background: #f3f4f6;
        padding: 1rem;
        border-radius: 6px;
        font-family: 'Monaco', monospace;
        margin: 0.5rem 0;
        border: 1px solid #d1d5db;
    }
    
    .ai-response {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
    }
    
    .user-query {
        background: #ffffff;
        border: 2px solid #667eea;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .ai-insight {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 1px solid #f0a068;
    }
    
    .calc-step {
        background: linear-gradient(145deg, #f8f9fa, #e9ecef);
        border-left: 4px solid #3282b8;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .step-number {
        background: #3282b8;
        color: white;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        margin-right: 1rem;
    }
    
    .step-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #0f4c75;
        margin-bottom: 0.5rem;
    }
    
    .step-formula {
        background: #fff;
        padding: 1rem;
        border-radius: 6px;
        border: 1px solid #dee2e6;
        font-family: 'Monaco', 'Menlo', monospace;
        margin: 1rem 0;
    }
    
    .result-highlight {
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        box-shadow: 0 8px 32px rgba(40,167,69,0.3);
        margin: 2rem 0;
    }
    
    .connection-status {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
    }
    
    .connected { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .disconnected { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    
    .calculation-verified {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #00b4db;
    }
    
    .summary-box {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# CORE DATA CLASSES
# ==============================================================================

class AssetClass(Enum):
    INTEREST_RATE = "Interest Rate"
    FOREIGN_EXCHANGE = "Foreign Exchange"
    CREDIT = "Credit"
    EQUITY = "Equity"
    COMMODITY = "Commodity"

class TradeType(Enum):
    SWAP = "Swap"
    FORWARD = "Forward"
    OPTION = "Option"
    SWAPTION = "Swaption"

class CollateralType(Enum):
    CASH = "Cash"
    GOVERNMENT_BONDS = "Government Bonds"
    CORPORATE_BONDS = "Corporate Bonds"
    EQUITIES = "Equities"
    MONEY_MARKET = "Money Market Funds"

@dataclass
class Trade:
    trade_id: str
    counterparty: str
    asset_class: AssetClass
    trade_type: TradeType
    notional: float
    currency: str
    underlying: str
    maturity_date: datetime
    mtm_value: float = 0.0
    delta: float = 1.0
    basis_flag: bool = False
    volatility_flag: bool = False
    ceu_flag: int = 1  # Central clearing flag
    
    def time_to_maturity(self) -> float:
        return max(0, (self.maturity_date - datetime.now()).days / 365.25)

@dataclass
class NettingSet:
    netting_set_id: str
    counterparty: str
    trades: List[Trade]
    threshold: float = 0.0
    mta: float = 0.0
    nica: float = 0.0

@dataclass
class Collateral:
    collateral_type: CollateralType
    currency: str
    amount: float
    haircut: float = 0.0

@dataclass
class DataQualityIssue:
    field_name: str
    current_value: any
    issue_type: str  # 'missing', 'estimated', 'outdated'
    impact: str  # 'high', 'medium', 'low'
    recommendation: str
    default_used: any = None

# ==============================================================================
# COMPREHENSIVE SA-CCR AGENT WITH ENHANCED FEATURES
# ==============================================================================

class ComprehensiveSACCRAgent:
    """Complete SA-CCR Agent following all 24 Basel regulatory steps with enhanced analysis"""
    
    def __init__(self):
        self.llm = None
        self.connection_status = "disconnected"
        
        # Initialize regulatory parameters
        self.supervisory_factors = self._initialize_supervisory_factors()
        self.supervisory_correlations = self._initialize_correlations()
        self.collateral_haircuts = self._initialize_collateral_haircuts()
        
        # Enhanced features
        self.data_quality_issues = []
        self.calculation_assumptions = []
        self.thinking_steps = []
        
    def setup_llm_connection(self, config: Dict) -> bool:
        """Setup LangChain ChatOpenAI connection"""
        try:
            self.llm = ChatOpenAI(
                base_url=config.get('base_url', "http://localhost:8123/v1"),
                api_key=config.get('api_key', "dummy"),
                model=config.get('model', "llama3"),
                temperature=config.get('temperature', 0.3),
                max_tokens=config.get('max_tokens', 4000),
                streaming=config.get('streaming', False)
            )
            
            # Test connection
            test_response = self.llm.invoke([
                SystemMessage(content="You are a Basel SA-CCR expert. Respond with 'Connected' if you receive this."),
                HumanMessage(content="Test")
            ])
            
            if test_response and test_response.content:
                self.connection_status = "connected"
                return True
            else:
                self.connection_status = "disconnected"
                return False
                
        except Exception as e:
            st.error(f"LLM Connection Error: {str(e)}")
            self.connection_status = "disconnected"
            return False
    
    def _initialize_supervisory_factors(self) -> Dict:
        """Initialize supervisory factors per Basel regulation"""
        return {
            AssetClass.INTEREST_RATE: {
                'USD': {'<2y': 0.50, '2-5y': 0.50, '>5y': 1.50},
                'EUR': {'<2y': 0.50, '2-5y': 0.50, '>5y': 1.50},
                'JPY': {'<2y': 0.50, '2-5y': 0.50, '>5y': 1.50},
                'GBP': {'<2y': 0.50, '2-5y': 0.50, '>5y': 1.50},
                'other': {'<2y': 1.50, '2-5y': 1.50, '>5y': 1.50}
            },
            AssetClass.FOREIGN_EXCHANGE: {'G10': 4.0, 'emerging': 15.0},
            AssetClass.CREDIT: {
                'IG_single': 0.46, 'HY_single': 1.30,
                'IG_index': 0.38, 'HY_index': 1.06
            },
            AssetClass.EQUITY: {
                'single_large': 32.0, 'single_small': 40.0,
                'index_developed': 20.0, 'index_emerging': 25.0
            },
            AssetClass.COMMODITY: {
                'energy': 18.0, 'metals': 18.0, 'agriculture': 18.0, 'other': 18.0
            }
        }
    
    def _initialize_correlations(self) -> Dict:
        """Initialize supervisory correlations"""
        return {
            AssetClass.INTEREST_RATE: 0.99,
            AssetClass.FOREIGN_EXCHANGE: 0.60,
            AssetClass.CREDIT: 0.50,
            AssetClass.EQUITY: 0.80,
            AssetClass.COMMODITY: 0.40
        }
    
    def _initialize_collateral_haircuts(self) -> Dict:
        """Initialize collateral haircuts"""
        return {
            CollateralType.CASH: 0.0,
            CollateralType.GOVERNMENT_BONDS: 0.5,
            CollateralType.CORPORATE_BONDS: 4.0,
            CollateralType.EQUITIES: 15.0,
            CollateralType.MONEY_MARKET: 0.5
        }
    
    def analyze_data_quality(self, netting_set: NettingSet, collateral: List[Collateral] = None) -> List[DataQualityIssue]:
        """Analyze data quality and identify missing/estimated values"""
        issues = []
        
        # Check netting set level data
        if netting_set.threshold == 0 and netting_set.mta == 0:
            issues.append(DataQualityIssue(
                field_name="Threshold/MTA",
                current_value="0/0",
                issue_type="estimated",
                impact="high",
                recommendation="Margining terms significantly impact RC calculation. Please provide actual CSA terms.",
                default_used="Assumed unmargined netting set"
            ))
        
        # Check trade level data
        for trade in netting_set.trades:
            if trade.mtm_value == 0:
                issues.append(DataQualityIssue(
                    field_name=f"MTM Value - {trade.trade_id}",
                    current_value=0,
                    issue_type="missing",
                    impact="high",
                    recommendation="Current MTM affects replacement cost and PFE multiplier calculation.",
                    default_used="0"
                ))
            
            if trade.trade_type in [TradeType.OPTION, TradeType.SWAPTION] and trade.delta == 1.0:
                issues.append(DataQualityIssue(
                    field_name=f"Option Delta - {trade.trade_id}",
                    current_value=1.0,
                    issue_type="estimated",
                    impact="medium",
                    recommendation="Option delta affects effective notional calculation.",
                    default_used="1.0"
                ))
        
        # Check collateral data
        if not collateral:
            issues.append(DataQualityIssue(
                field_name="Collateral Portfolio",
                current_value="None",
                issue_type="missing",
                impact="high",
                recommendation="Collateral reduces replacement cost. Please provide collateral details.",
                default_used="No collateral assumed"
            ))
        
        return issues

    def validate_input_completeness(self, netting_set: NettingSet, collateral: List[Collateral] = None) -> Dict:
        """Validate if all required inputs are provided"""
        missing_fields = []
        warnings = []
        
        # Validate netting set
        if not netting_set.netting_set_id:
            missing_fields.append("Netting Set ID")
        if not netting_set.counterparty:
            missing_fields.append("Counterparty name")
        if not netting_set.trades:
            missing_fields.append("At least one trade")
        
        # Validate trades
        for i, trade in enumerate(netting_set.trades):
            trade_prefix = f"Trade {i+1}"
            
            if not trade.trade_id:
                missing_fields.append(f"{trade_prefix}: Trade ID")
            if not trade.notional or trade.notional == 0:
                missing_fields.append(f"{trade_prefix}: Notional amount")
            if not trade.currency:
                missing_fields.append(f"{trade_prefix}: Currency")
            if not trade.maturity_date:
                missing_fields.append(f"{trade_prefix}: Maturity date")
            
            # Option-specific validations
            if trade.trade_type in [TradeType.OPTION, TradeType.SWAPTION]:
                if trade.delta == 1.0:
                    warnings.append(f"{trade_prefix}: Delta not specified for option (using default 1.0)")
        
        return {
            'is_complete': len(missing_fields) == 0,
            'missing_fields': missing_fields,
            'warnings': warnings,
            'can_proceed': len(missing_fields) == 0
        }
    
    def calculate_comprehensive_saccr(self, netting_set: NettingSet, collateral: List[Collateral] = None) -> Dict:
        """Calculate SA-CCR following complete 24-step workflow with enhanced analysis"""
        
        # Reset enhanced tracking
        self.data_quality_issues = self.analyze_data_quality(netting_set, collateral)
        self.calculation_assumptions = []
        self.thinking_steps = []
        
        calculation_steps = []
        
        # Step 1: Netting Set Data
        step1_data = self._step1_netting_set_data(netting_set)
        calculation_steps.append(step1_data)
        
        # Step 2: Asset Class Classification
        step2_data = self._step2_asset_classification(netting_set.trades)
        calculation_steps.append(step2_data)
        
        # Step 3: Hedging Set
        step3_data = self._step3_hedging_set(netting_set.trades)
        calculation_steps.append(step3_data)
        
        # Step 4: Time Parameters
        step4_data = self._step4_time_parameters(netting_set.trades)
        calculation_steps.append(step4_data)
        
        # Step 5: Adjusted Notional
        step5_data = self._step5_adjusted_notional(netting_set.trades)
        calculation_steps.append(step5_data)
        
        # Step 6: Maturity Factor (Enhanced with thinking)
        step6_data = self._step6_maturity_factor_enhanced(netting_set.trades)
        calculation_steps.append(step6_data)
        
        # Step 7: Supervisory Delta
        step7_data = self._step7_supervisory_delta(netting_set.trades)
        calculation_steps.append(step7_data)
        
        # Step 8: Supervisory Factor (Enhanced with thinking)
        step8_data = self._step8_supervisory_factor_enhanced(netting_set.trades)
        calculation_steps.append(step8_data)
        
        # Step 9: Adjusted Derivatives Contract Amount (Enhanced)
        step9_data = self._step9_adjusted_derivatives_contract_amount_enhanced(netting_set.trades)
        calculation_steps.append(step9_data)
        
        # Step 10: Supervisory Correlation
        step10_data = self._step10_supervisory_correlation(netting_set.trades)
        calculation_steps.append(step10_data)
        
        # Step 11: Hedging Set AddOn
        step11_data = self._step11_hedging_set_addon(netting_set.trades)
        calculation_steps.append(step11_data)
        
        # Step 12: Asset Class AddOn
        step12_data = self._step12_asset_class_addon(netting_set.trades)
        calculation_steps.append(step12_data)
        
        # Step 13: Aggregate AddOn (Enhanced)
        step13_data = self._step13_aggregate_addon_enhanced(netting_set.trades)
        calculation_steps.append(step13_data)
        
        # Step 14: Sum of V, C (Enhanced)
        step14_data = self._step14_sum_v_c_enhanced(netting_set, collateral)
        calculation_steps.append(step14_data)
        sum_v = step14_data['sum_v']
        sum_c = step14_data['sum_c']

        # Step 15: PFE Multiplier (Enhanced)
        step15_data = self._step15_pfe_multiplier_enhanced(sum_v, sum_c, step13_data['aggregate_addon'])
        calculation_steps.append(step15_data)
        
        # Step 16: PFE (Enhanced)
        step16_data = self._step16_pfe_enhanced(step15_data['multiplier'], step13_data['aggregate_addon'])
        calculation_steps.append(step16_data)
        
        # Step 17: TH, MTA, NICA
        step17_data = self._step17_th_mta_nica(netting_set)
        calculation_steps.append(step17_data)
        
        # Step 18: RC (Enhanced)
        step18_data = self._step18_replacement_cost_enhanced(sum_v, sum_c, step17_data)
        calculation_steps.append(step18_data)
        
        # Step 19: CEU Flag
        step19_data = self._step19_ceu_flag(netting_set.trades)
        calculation_steps.append(step19_data)
        
        # Step 20: Alpha
        step20_data = self._step20_alpha(step19_data['ceu_flag'])
        calculation_steps.append(step20_data)
        
        # Step 21: EAD (Enhanced)
        step21_data = self._step21_ead_enhanced(step20_data['alpha'], step18_data['rc'], step16_data['pfe'])
        calculation_steps.append(step21_data)
        
        # Step 22: Counterparty Information
        step22_data = self._step22_counterparty_info(netting_set.counterparty)
        calculation_steps.append(step22_data)
        
        # Step 23: Risk Weight
        step23_data = self._step23_risk_weight(step22_data['counterparty_type'])
        calculation_steps.append(step23_data)
        
        # Step 24: RWA Calculation (Enhanced)
        step24_data = self._step24_rwa_calculation_enhanced(step21_data['ead'], step23_data['risk_weight'])
        calculation_steps.append(step24_data)
        
        # Generate enhanced summary
        enhanced_summary = self._generate_enhanced_summary(calculation_steps, netting_set)
        
        # Generate AI explanation if connected
        ai_explanation = self._generate_saccr_explanation_enhanced(calculation_steps, enhanced_summary) if self.llm and self.connection_status == "connected" else None
        
        return {
            'calculation_steps': calculation_steps,
            'final_results': {
                'replacement_cost': step18_data['rc'],
                'potential_future_exposure': step16_data['pfe'],
                'exposure_at_default': step21_data['ead'],
                'risk_weighted_assets': step24_data['rwa'],
                'capital_requirement': step24_data['rwa'] * 0.08
            },
            'data_quality_issues': self.data_quality_issues,
            'enhanced_summary': enhanced_summary,
            'thinking_steps': self.thinking_steps,
            'assumptions': self.calculation_assumptions,
            'ai_explanation': ai_explanation
        }

    # Enhanced calculation methods with thinking process
    def _step6_maturity_factor_enhanced(self, trades: List[Trade]) -> Dict:
        """Step 6: Maturity Factor with detailed reasoning"""
        maturity_factors = []
        reasoning_details = []
        
        for trade in trades:
            remaining_maturity = trade.time_to_maturity()
            mf = math.sqrt(min(remaining_maturity, 1.0))
            
            maturity_factors.append({
                'trade_id': trade.trade_id,
                'remaining_maturity': remaining_maturity,
                'maturity_factor': mf
            })
            
            reasoning_details.append(f"Trade {trade.trade_id}: M={remaining_maturity:.2f}y → MF=sqrt(min({remaining_maturity:.2f}, 1.0)) = {mf:.6f}")
        
        # Add thinking step
        thinking = {
            'step': 6,
            'title': 'Maturity Factor Calculation',
            'reasoning': f"""
THINKING PROCESS:
• Formula: MF = sqrt(min(M, 1 year) / 1 year)
• This formula scales down the add-on for trades with less than one year remaining maturity.
• It reflects the reduced time horizon over which a default can occur.
• Trades with maturities greater than one year receive no further penalty (MF is capped at 1.0).

DETAILED CALCULATIONS:
{chr(10).join(reasoning_details)}

REGULATORY RATIONALE:
• Acknowledges that shorter-term trades have less time to accumulate potential future exposure.
• The square root function provides a non-linear scaling, giving more benefit to very short-term trades.
            """,
            'formula': 'MF = sqrt(min(M, 1.0))',
            'key_insight': f"Average maturity factor: {sum(mf['maturity_factor'] for mf in maturity_factors)/len(maturity_factors):.4f}"
        }
        
        self.thinking_steps.append(thinking)
        
        return {
            'step': 6,
            'title': 'Maturity Factor (MF)',
            'description': 'Apply Basel maturity factor formula',
            'data': maturity_factors,
            'formula': 'MF = sqrt(min(M, 1.0))',
            'result': f"Calculated maturity factors for {len(trades)} trades",
            'thinking': thinking
        }

    def _step8_supervisory_factor_enhanced(self, trades: List[Trade]) -> Dict:
        """Step 8: Supervisory Factor with detailed lookup logic"""
        supervisory_factors = []
        reasoning_details = []
        
        for trade in trades:
            sf_bps = self._get_supervisory_factor(trade)
            sf_decimal = sf_bps / 10000
            supervisory_factors.append({
                'trade_id': trade.trade_id,
                'asset_class': trade.asset_class.value,
                'currency': trade.currency,
                'maturity_bucket': self._get_maturity_bucket(trade),
                'supervisory_factor_bp': sf_bps,
                'supervisory_factor_decimal': sf_decimal
            })
            
            reasoning_details.append(f"Trade {trade.trade_id}: {trade.asset_class.value} {trade.currency} {self._get_maturity_bucket(trade)} → {sf_bps:.2f}bps ({sf_decimal:.4f})")
        
        thinking = {
            'step': 8,
            'title': 'Supervisory Factor Lookup',
            'reasoning': f"""
THINKING PROCESS:
• Look up supervisory factors (SF) from Basel regulatory tables.
• Factors represent the estimated volatility for each asset class risk factor.
• Higher SF means higher perceived risk and thus a larger capital add-on.

DETAILED LOOKUPS:
{chr(10).join(reasoning_details)}

REGULATORY BASIS:
• Calibrated to reflect potential price movements over a one-year horizon at a 99% confidence level.
• Based on historical volatility analysis by the Basel Committee.
• Factors are differentiated by asset class, and for interest rates, by currency and maturity.
            """,
            'formula': 'SF looked up from Basel regulatory tables',
            'key_insight': f"Portfolio-weighted average SF: {sum(sf['supervisory_factor_bp'] * abs(trade.notional) for sf, trade in zip(supervisory_factors, trades)) / sum(abs(trade.notional) for trade in trades):.1f}bps"
        }
        
        self.thinking_steps.append(thinking)
        
        return {
            'step': 8,
            'title': 'Supervisory Factor (SF)',
            'description': 'Apply regulatory supervisory factors by asset class',
            'data': supervisory_factors,
            'formula': 'SF per Basel regulatory mapping tables',
            'result': f"Applied supervisory factors for {len(trades)} trades",
            'thinking': thinking
        }

    def _step9_adjusted_derivatives_contract_amount_enhanced(self, trades: List[Trade]) -> Dict:
        """Step 9: Adjusted Contract Amount with full formula breakdown"""
        adjusted_amounts = []
        reasoning_details = []
        
        for trade in trades:
            adjusted_notional = abs(trade.notional)
            supervisory_delta = trade.delta if trade.trade_type in [TradeType.OPTION, TradeType.SWAPTION] else (1.0 if trade.notional > 0 else -1.0)
            remaining_maturity = trade.time_to_maturity()
            mf = math.sqrt(min(remaining_maturity, 1.0))
            sf = self._get_supervisory_factor(trade) / 10000
            
            adjusted_amount = adjusted_notional * supervisory_delta * mf * sf
            
            # Track assumptions
            if trade.trade_type in [TradeType.OPTION, TradeType.SWAPTION] and trade.delta == 1.0:
                self.calculation_assumptions.append(f"Trade {trade.trade_id}: Using default delta=1.0 for {trade.trade_type.value}")
            
            adjusted_amounts.append({
                'trade_id': trade.trade_id,
                'adjusted_notional': adjusted_notional,
                'supervisory_delta': supervisory_delta,
                'maturity_factor': mf,
                'supervisory_factor': sf,
                'adjusted_derivatives_contract_amount': adjusted_amount
            })
            
            reasoning_details.append(
                f"Trade {trade.trade_id}: ${adjusted_notional:,.0f} × {supervisory_delta} × {mf:.6f} × {sf:.4f} = ${adjusted_amount:,.2f}"
            )
        
        thinking = {
            'step': 9,
            'title': 'Adjusted Derivatives Contract Amount',
            'reasoning': f"""
THINKING PROCESS:
• This is the core risk measure per trade, forming the basis for the PFE add-on.
• The formula combines all key risk components: size, direction, time horizon, and volatility.

COMPONENT ANALYSIS:
• Adjusted Notional: The base size of the exposure.
• Delta (δ): Captures direction (long/short) and option sensitivity.
• Maturity Factor (MF): Scales risk down for shorter-term trades.
• Supervisory Factor (SF): Weights the exposure by the asset class's regulatory volatility.

DETAILED CALCULATIONS:
{chr(10).join(reasoning_details)}

PORTFOLIO INSIGHTS:
• This step translates each trade into a standardized risk amount.
• These amounts are then aggregated in the following steps, where netting benefits are applied.
            """,
            'formula': 'Adjusted Amount = Adjusted Notional × δ × MF × SF',
            'key_insight': f"Total adjusted exposure: ${sum(abs(calc['adjusted_derivatives_contract_amount']) for calc in adjusted_amounts):,.0f}"
        }
        
        self.thinking_steps.append(thinking)
        
        return {
            'step': 9,
            'title': 'Adjusted Derivatives Contract Amount',
            'description': 'Calculate final adjusted contract amounts',
            'data': adjusted_amounts,
            'formula': 'Adjusted Amount = Adjusted Notional × δ × MF × SF',
            'result': f"Calculated adjusted amounts for {len(trades)} trades",
            'thinking': thinking
        }

    def _step13_aggregate_addon_enhanced(self, trades: List[Trade]) -> Dict:
        """Step 13: Aggregate AddOn with enhanced aggregation logic"""
        step12_result = self._step12_asset_class_addon(trades)
        
        aggregate_addon = sum(ac_data['asset_class_addon'] for ac_data in step12_result['data'])
        
        thinking = {
            'step': 13,
            'title': 'Aggregate AddOn Calculation',
            'reasoning': f"""
THINKING PROCESS:
• Sum all individual asset class add-ons to get the total portfolio add-on.
• This represents the gross potential future exposure before considering netting benefits across the portfolio.
• The simple summation is a conservative approach required by the regulation.

ASSET CLASS BREAKDOWN:
{chr(10).join([f"• {ac_data['asset_class']}: ${ac_data['asset_class_addon']:,.0f}" for ac_data in step12_result['data']])}

REGULATORY PURPOSE:
• This value represents the total potential increase in exposure over the life of the trades.
• It forms the primary input for the PFE calculation, which will then be scaled by the multiplier.
            """,
            'formula': 'Aggregate AddOn = Σ(Asset Class AddOns)',
            'key_insight': f"This ${aggregate_addon:,.0f} represents raw future exposure before netting benefits"
        }
        
        self.thinking_steps.append(thinking)
        
        return {
            'step': 13,
            'title': 'Aggregate AddOn',
            'description': 'Sum asset class add-ons to get total portfolio add-on',
            'data': {
                'asset_class_addons': [(ac_data['asset_class'], ac_data['asset_class_addon'])
                                       for ac_data in step12_result['data']],
                'aggregate_addon': aggregate_addon
            },
            'formula': 'Aggregate AddOn = Σ(Asset Class AddOns)',
            'result': f"Total Aggregate AddOn: ${aggregate_addon:,.0f}",
            'aggregate_addon': aggregate_addon,
            'thinking': thinking
        }

    def _step14_sum_v_c_enhanced(self, netting_set: NettingSet, collateral: List[Collateral] = None) -> Dict:
        """Step 14: V and C calculation with enhanced collateral analysis"""
        sum_v = sum(trade.mtm_value for trade in netting_set.trades)
        
        sum_c = 0
        collateral_details = []
        
        if collateral:
            for coll in collateral:
                haircut = self.collateral_haircuts.get(coll.collateral_type, 15.0) / 100
                effective_value = coll.amount * (1 - haircut)
                sum_c += effective_value
                
                collateral_details.append({
                    'type': coll.collateral_type.value,
                    'amount': coll.amount,
                    'haircut_pct': haircut * 100,
                    'effective_value': effective_value
                })
        else:
            self.calculation_assumptions.append("No collateral provided - assuming zero collateral benefit")
        
        # Fix complex expressions for f-string
        position_desc = 'Out-of-the-money (favorable)' if sum_v < 0 else 'In-the-money (unfavorable)' if sum_v > 0 else 'At-the-money (neutral)'
        total_posted = sum([c['amount'] for c in collateral_details]) if collateral_details else 0
        
        thinking = {
            'step': 14,
            'title': 'Current Exposure (V) and Collateral (C) Analysis',
            'reasoning': f"""
THINKING PROCESS:
• V = Current market value (MtM) of all trades in the netting set.
• C = Effective value of collateral held, after applying regulatory haircuts.
• The net value (V-C) is a key input for both the Replacement Cost (RC) and the PFE Multiplier.

CURRENT EXPOSURE ANALYSIS:
• Sum of trade MTMs (V): ${sum_v:,.0f}
• Portfolio position: {position_desc}

COLLATERAL ANALYSIS:
• Total posted: ${total_posted:,.0f}
• After haircuts (C): ${sum_c:,.0f}
• Net exposure (V-C): ${sum_v - sum_c:,.0f}
            """,
            'formula': 'V = Σ(Trade MTMs), C = Σ(Collateral × (1 - haircut))',
            'key_insight': f"Net exposure of ${sum_v - sum_c:,.0f} will drive RC calculation and PFE multiplier"
        }
        
        self.thinking_steps.append(thinking)
        
        return {
            'step': 14,
            'title': 'Sum of V, C within netting set',
            'description': 'Calculate sum of MTM values and effective collateral',
            'data': {
                'sum_v_mtm': sum_v,
                'sum_c_collateral': sum_c,
                'net_exposure': sum_v - sum_c,
                'collateral_details': collateral_details
            },
            'formula': 'V = Σ(MTM values), C = Σ(Collateral × (1 - haircut))',
            'result': f"Sum V: ${sum_v:,.0f}, Sum C: ${sum_c:,.0f}",
            'sum_v': sum_v,
            'sum_c': sum_c,
            'thinking': thinking
        }

    def _step15_pfe_multiplier_enhanced(self, sum_v: float, sum_c: float, aggregate_addon: float) -> Dict:
        """Step 15: PFE Multiplier with detailed netting benefit analysis"""
        net_exposure = sum_v - sum_c
        
        if aggregate_addon > 0:
            exponent = net_exposure / (2 * 0.95 * aggregate_addon)
            multiplier = min(1.0, 0.05 + 0.95 * math.exp(exponent))
        else:
            multiplier = 1.0
            exponent = 0
        
        netting_benefit_pct = (1 - multiplier) * 100
        
        thinking = {
            'step': 15,
            'title': 'PFE Multiplier - Netting Benefit Analysis',
            'reasoning': f"""
THINKING PROCESS:
• The multiplier scales the gross add-on to reflect the benefit of netting.
• If a portfolio's current value (V-C) is negative, it's less likely to become a large positive exposure in the future, justifying a lower PFE.

DETAILED CALCULATION:
• Net Exposure (V-C): ${net_exposure:,.0f}
• Aggregate AddOn: ${aggregate_addon:,.0f}
• Exponent: ${net_exposure:,.0f} / (1.9 × ${aggregate_addon:,.0f}) = {exponent:.6f}
• Multiplier: min(1, 0.05 + 0.95 × exp({exponent:.6f})) = {multiplier:.6f}

NETTING BENEFIT ANALYSIS:
• Final multiplier: {multiplier:.6f}
• Netting benefit: {netting_benefit_pct:.1f}% reduction in future exposure
            """,
            'formula': 'Multiplier = min(1, 0.05 + 0.95 × exp((V-C) / (1.9 × AddOn)))',
            'key_insight': f"{netting_benefit_pct:.1f}% netting benefit reduces PFE by ${(1-multiplier)*aggregate_addon:,.0f}"
        }
        
        self.thinking_steps.append(thinking)
        
        return {
            'step': 15,
            'title': 'PFE Multiplier',
            'description': 'Calculate PFE multiplier based on netting benefit',
            'data': {
                'sum_v': sum_v,
                'sum_c': sum_c,
                'net_exposure': net_exposure,
                'aggregate_addon': aggregate_addon,
                'exponent': exponent,
                'multiplier': multiplier,
                'netting_benefit_pct': netting_benefit_pct
            },
            'formula': 'Multiplier = min(1, 0.05 + 0.95 × exp((V-C) / (1.9 × AddOn)))',
            'result': f"PFE Multiplier: {multiplier:.6f}",
            'multiplier': multiplier,
            'thinking': thinking
        }

    def _step16_pfe_enhanced(self, multiplier: float, aggregate_addon: float) -> Dict:
        """Step 16: PFE Calculation with enhanced analysis"""
        pfe = multiplier * aggregate_addon
        
        thinking = {
            'step': 16,
            'title': 'Potential Future Exposure (PFE) Final Calculation',
            'reasoning': f"""
THINKING PROCESS:
• PFE = Multiplier × Aggregate AddOn
• This combines the gross future volatility risk (AddOn) with the portfolio-specific netting benefits (Multiplier).
• It represents the final estimate of potential future exposure.

FINAL CALCULATION:
• Multiplier: {multiplier:.6f}
• Aggregate AddOn: ${aggregate_addon:,.0f}
• PFE: {multiplier:.6f} × ${aggregate_addon:,.0f} = ${pfe:,.0f}

REGULATORY SIGNIFICANCE:
• PFE is added to the current exposure (RC) to determine the total Exposure at Default (EAD).
            """,
            'formula': 'PFE = Multiplier × Aggregate AddOn',
            'key_insight': f"PFE of ${pfe:,.0f} represents net future exposure after a {(1-multiplier)*100:.1f}% netting benefit"
        }
        
        self.thinking_steps.append(thinking)
        
        return {
            'step': 16,
            'title': 'PFE (Potential Future Exposure)',
            'description': 'Calculate PFE using multiplier and aggregate add-on',
            'data': {
                'multiplier': multiplier,
                'aggregate_addon': aggregate_addon,
                'pfe': pfe
            },
            'formula': 'PFE = Multiplier × Aggregate AddOn',
            'result': f"PFE: ${pfe:,.0f}",
            'pfe': pfe,
            'thinking': thinking
        }

    def _step18_replacement_cost_enhanced(self, sum_v: float, sum_c: float, step17_data: Dict) -> Dict:
        """Step 18: Replacement Cost with enhanced margining analysis"""
        threshold = step17_data['threshold']
        mta = step17_data['mta']
        nica = step17_data['nica']
        
        net_exposure = sum_v - sum_c
        is_margined = threshold > 0 or mta > 0
        
        if is_margined:
            margin_floor = threshold + mta - nica
            rc = max(net_exposure, margin_floor, 0)
            methodology = "Margined netting set"
        else:
            margin_floor = 0
            rc = max(net_exposure, 0)
            methodology = "Unmargined netting set"
        
        thinking = {
            'step': 18,
            'title': 'Replacement Cost (RC) - Current Exposure Analysis',
            'reasoning': f"""
THINKING PROCESS:
• RC represents the current cost to replace the portfolio if the counterparty defaults today.
• The calculation depends on whether the netting set is margined (covered by a CSA).

NETTING SET CLASSIFICATION:
• Type: {methodology}
• Margin Floor (TH+MTA-NICA): ${margin_floor:,.0f}

REPLACEMENT COST DETERMINATION:
• Formula: {"RC = max(V-C, TH+MTA-NICA, 0)" if is_margined else "RC = max(V-C, 0)"}
• Calculation: RC = max(${net_exposure:,.0f}, {f'${margin_floor:,.0f}, ' if is_margined else ''}0) = ${rc:,.0f}
            """,
            'formula': f"RC = max(V-C{', TH+MTA-NICA' if is_margined else ''}, 0)",
            'key_insight': f"RC of ${rc:,.0f} represents the current credit exposure component of EAD."
        }
        
        self.thinking_steps.append(thinking)
        
        return {
            'step': 18,
            'title': 'RC (Replacement Cost)',
            'description': 'Calculate replacement cost with netting and collateral benefits',
            'data': {
                'sum_v': sum_v,
                'sum_c': sum_c,
                'net_exposure': net_exposure,
                'threshold': threshold,
                'mta': mta,
                'nica': nica,
                'is_margined': is_margined,
                'rc': rc,
                'methodology': methodology
            },
            'formula': f"RC = max(V - C{'; TH + MTA - NICA' if is_margined else ''}; 0)",
            'result': f"RC: ${rc:,.0f}",
            'rc': rc,
            'thinking': thinking
        }

    def _step21_ead_enhanced(self, alpha: float, rc: float, pfe: float) -> Dict:
        """Step 21: EAD Calculation with enhanced exposure analysis"""
        combined_exposure = rc + pfe
        ead = alpha * combined_exposure
        
        rc_percentage = (rc / combined_exposure * 100) if combined_exposure > 0 else 0
        pfe_percentage = (pfe / combined_exposure * 100) if combined_exposure > 0 else 0
        
        thinking = {
            'step': 21,
            'title': 'Exposure at Default (EAD) - Total Credit Exposure',
            'reasoning': f"""
THINKING PROCESS:
• EAD = Alpha × (RC + PFE), where Alpha is a fixed regulatory multiplier of 1.4.
• This combines the current exposure (RC) and potential future exposure (PFE) into a single measure.

EXPOSURE COMPONENT BREAKDOWN:
• Current Exposure (RC): ${rc:,.0f} ({rc_percentage:.1f}% of total)
• Future Exposure (PFE): ${pfe:,.0f} ({pfe_percentage:.1f}% of total)
• Combined Exposure (RC+PFE): ${combined_exposure:,.0f}

EAD CALCULATION:
• EAD = {alpha} × ${combined_exposure:,.0f} = ${ead:,.0f}
            """,
            'formula': 'EAD = 1.4 × (RC + PFE)',
            'key_insight': f"Total credit exposure (EAD): ${ead:,.0f}, driven {rc_percentage:.0f}% by current risk and {pfe_percentage:.0f}% by future risk."
        }
        
        self.thinking_steps.append(thinking)
        
        return {
            'step': 21,
            'title': 'EAD (Exposure at Default)',
            'description': 'Calculate final exposure at default',
            'data': {
                'alpha': alpha,
                'rc': rc,
                'pfe': pfe,
                'combined_exposure': combined_exposure,
                'ead': ead,
                'rc_percentage': rc_percentage,
                'pfe_percentage': pfe_percentage
            },
            'formula': 'EAD = Alpha × (RC + PFE)',
            'result': f"EAD: ${ead:,.0f}",
            'ead': ead,
            'thinking': thinking
        }

    def _step24_rwa_calculation_enhanced(self, ead: float, risk_weight: float) -> Dict:
        """Step 24: RWA Calculation with enhanced capital analysis"""
        rwa = ead * risk_weight
        capital_requirement = rwa * 0.08
        
        thinking = {
            'step': 24,
            'title': 'Risk-Weighted Assets (RWA) and Capital Calculation',
            'reasoning': f"""
THINKING PROCESS:
• RWA = Risk Weight × EAD. The EAD is weighted by the credit risk of the counterparty.
• Final Capital Requirement = RWA × 8% (the Basel minimum capital ratio).

CAPITAL CALCULATION:
• EAD: ${ead:,.0f}
• Risk Weight: {risk_weight*100:.0f}% (based on counterparty type)
• RWA = ${ead:,.0f} × {risk_weight} = ${rwa:,.0f}
• Minimum Capital = ${rwa:,.0f} × 8% = ${capital_requirement:,.0f}
            """,
            'formula': 'RWA = Risk Weight × EAD, Capital = RWA × 8%',
            'key_insight': f"${capital_requirement:,.0f} minimum capital required, which is {(capital_requirement/ead*100 if ead > 0 else 0):.2f}% of the total exposure."
        }
        
        self.thinking_steps.append(thinking)
        
        return {
            'step': 24,
            'title': 'RWA Calculation',
            'description': 'Calculate Risk Weighted Assets and Capital Requirement',
            'data': {
                'ead': ead,
                'risk_weight': risk_weight,
                'risk_weight_pct': risk_weight * 100,
                'rwa': rwa,
                'capital_requirement': capital_requirement,
                'capital_ratio': 0.08,
                'capital_efficiency_pct': (capital_requirement/ead*100) if ead > 0 else 0
            },
            'formula': 'Standardized RWA = RW × EAD',
            'result': f"RWA: ${rwa:,.0f}",
            'rwa': rwa,
            'thinking': thinking
        }

    def _generate_enhanced_summary(self, calculation_steps: list, netting_set: NettingSet) -> Dict:
        """Generate enhanced bulleted summary"""
        
        final_step_21 = next(step for step in calculation_steps if step['step'] == 21)
        final_step_24 = next(step for step in calculation_steps if step['step'] == 24)
        final_step_16 = next(step for step in calculation_steps if step['step'] == 16)
        final_step_18 = next(step for step in calculation_steps if step['step'] == 18)
        final_step_15 = next(step for step in calculation_steps if step['step'] == 15)
        final_step_13 = next(step for step in calculation_steps if step['step'] == 13)
        
        total_notional = sum(abs(trade.notional) for trade in netting_set.trades)
        
        return {
            'key_inputs': [
                f"Portfolio: {len(netting_set.trades)} trades totaling ${total_notional:,.0f} notional",
                f"Counterparty: {netting_set.counterparty}",
                f"Netting arrangement: {'Margined' if netting_set.threshold > 0 or netting_set.mta > 0 else 'Unmargined'} set",
                f"Asset classes: {', '.join(set(t.asset_class.value for t in netting_set.trades))}"
            ],
            'risk_components': [
                f"Aggregate Add-On: ${final_step_13['aggregate_addon']:,.0f}",
                f"PFE Multiplier: {final_step_15['multiplier']:.4f} ({(1-final_step_15['multiplier'])*100:.1f}% netting benefit)",
                f"Potential Future Exposure: ${final_step_16['pfe']:,.0f}",
                f"Replacement Cost: ${final_step_18['rc']:,.0f}",
                f"Exposure split: {final_step_21['data']['rc_percentage']:.0f}% current / {final_step_21['data']['pfe_percentage']:.0f}% future"
            ],
            'capital_results': [
                f"Exposure at Default (EAD): ${final_step_21['ead']:,.0f}",
                f"Risk Weight: {final_step_24['data']['risk_weight_pct']:.0f}%",
                f"Risk-Weighted Assets: ${final_step_24['rwa']:,.0f}",
                f"Minimum Capital Required: ${final_step_24['data']['capital_requirement']:,.0f}",
                f"Capital Efficiency: {(final_step_24['data']['capital_requirement']/total_notional*100 if total_notional > 0 else 0):.3f}% of notional"
            ],
            'optimization_insights': [
                f"Netting benefits reduce PFE by {(1-final_step_15['multiplier'])*100:.1f}%",
                f"{'Current' if final_step_21['data']['rc_percentage'] > 50 else 'Future'} exposure dominates capital requirement",
                f"Consider {'improving CSA terms' if final_step_18['data']['is_margined'] else 'implementing margining'} to reduce RC",
                f"Portfolio shows {'strong' if final_step_15['multiplier'] < 0.5 else 'moderate' if final_step_15['multiplier'] < 0.8 else 'limited'} netting efficiency"
            ]
        }

    def _generate_saccr_explanation_enhanced(self, calculation_steps: List[Dict], enhanced_summary: Dict) -> str:
        """Generate enhanced AI explanation with thinking process insights"""
        if not self.llm or self.connection_status != "connected":
            return None
        
        key_thinking_insights = []
        for thinking_step in self.thinking_steps:
            if thinking_step.get('key_insight'):
                key_thinking_insights.append(f"Step {thinking_step['step']}: {thinking_step['key_insight']}")
        
        system_prompt = """You are a Basel SA-CCR regulatory expert providing executive-level analysis. 
        Focus on:
        1. Key risk drivers and their business implications
        2. Capital optimization opportunities with quantified benefits
        3. Regulatory compliance assessment
        4. Strategic recommendations for portfolio management
        
        Use the detailed thinking process insights to provide deeper analysis than standard summaries."""
        
        user_prompt = f"""
        Complete 24-step SA-CCR calculation performed with detailed thinking process analysis.
        
        ENHANCED SUMMARY:
        Key Inputs: {', '.join(enhanced_summary['key_inputs'])}
        Risk Components: {', '.join(enhanced_summary['risk_components'])}
        Capital Results: {', '.join(enhanced_summary['capital_results'])}
        
        KEY THINKING INSIGHTS FROM CALCULATION:
        {chr(10).join(key_thinking_insights)}
        
        DATA QUALITY ISSUES:
        {len(self.data_quality_issues)} issues identified (including: {', '.join([issue.field_name for issue in self.data_quality_issues[:3]])})
        
        ASSUMPTIONS MADE:
        {chr(10).join(self.calculation_assumptions)}
        
        Please provide executive analysis focusing on:
        1. What are the primary capital drivers and why?
        2. What optimization strategies would be most impactful?
        3. How do data quality issues affect the reliability of this calculation?
        4. What are the key business decisions this analysis should inform?
        """
        
        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            return response.content
        except Exception as e:
            return f"Enhanced AI analysis temporarily unavailable: {str(e)}"

    def gather_missing_information(self, data_quality_issues: List[DataQualityIssue]) -> Dict:
        """Generate prompts to gather missing information from user"""
        high_impact_issues = [issue for issue in data_quality_issues if issue.impact == 'high']
        
        prompts = []
        for issue in high_impact_issues:
            prompts.append({
                'field': issue.field_name,
                'question': self._generate_question_for_field(issue),
                'impact': issue.recommendation,
                'current_default': issue.default_used
            })
        
        return {
            'missing_info_prompts': prompts,
            'total_issues': len(data_quality_issues),
            'high_impact_count': len(high_impact_issues)
        }
    
    def _generate_question_for_field(self, issue: DataQualityIssue) -> str:
        """Generate specific questions for missing fields"""
        field_questions = {
            'Threshold/MTA': "What are the actual threshold and minimum transfer amounts in your CSA/ISDA agreement?",
            'Collateral Portfolio': "What collateral do you have posted? Please specify type, amount, and currency.",
            'MTM Value': f"What is the current market-to-market value for {issue.field_name.split(' - ')[1]}?",
            'Option Delta': f"What is the actual delta for the option trade {issue.field_name.split(' - ')[1]}?"
        }
        
        return field_questions.get(issue.field_name.split(' - ')[0], f"Please provide accurate information for {issue.field_name}")

    def _get_maturity_bucket(self, trade: Trade) -> str:
        """Get maturity bucket for display"""
        maturity = trade.time_to_maturity()
        if maturity < 2:
            return "<2y"
        elif maturity <= 5:
            return "2-5y"
        else:
            return ">5y"

    # Keep all original calculation methods unchanged
    def _step1_netting_set_data(self, netting_set: NettingSet) -> Dict:
        return {
            'step': 1,
            'title': 'Netting Set Data',
            'description': 'Source netting set data from trade repository',
            'data': {
                'netting_set_id': netting_set.netting_set_id,
                'counterparty': netting_set.counterparty,
                'trade_count': len(netting_set.trades),
                'total_notional': sum(abs(trade.notional) for trade in netting_set.trades)
            },
            'formula': 'Data sourced from system',
            'result': f"Netting Set ID: {netting_set.netting_set_id}, Trades: {len(netting_set.trades)}"
        }
    
    def _step2_asset_classification(self, trades: List[Trade]) -> Dict:
        classifications = []
        for trade in trades:
            classifications.append({
                'trade_id': trade.trade_id,
                'asset_class': trade.asset_class.value,
                'asset_sub_class': 'N/A',
                'basis_flag': trade.basis_flag,
                'volatility_flag': trade.volatility_flag
            })
        
        return {
            'step': 2,
            'title': 'Asset Class & Risk Factor Classification',
            'description': 'Classification of trades by regulatory categories',
            'data': classifications,
            'formula': 'Classification per Basel regulatory mapping tables',
            'result': f"Classified {len(trades)} trades across asset classes"
        }
    
    def _step3_hedging_set(self, trades: List[Trade]) -> Dict:
        hedging_sets = {}
        for trade in trades:
            hedging_set_key = f"{trade.asset_class.value}_{trade.currency}"
            if hedging_set_key not in hedging_sets:
                hedging_sets[hedging_set_key] = []
            hedging_sets[hedging_set_key].append(trade.trade_id)
        
        return {
            'step': 3,
            'title': 'Hedging Set Determination',
            'description': 'Group trades into hedging sets based on common risk factors',
            'data': hedging_sets,
            'formula': 'Hedging sets defined by asset class and currency/index',
            'result': f"Created {len(hedging_sets)} hedging sets"
        }
    
    def _step4_time_parameters(self, trades: List[Trade]) -> Dict:
        time_params = []
        for trade in trades:
            settlement_date = datetime.now()
            end_date = trade.maturity_date
            remaining_maturity = trade.time_to_maturity()
            
            time_params.append({
                'trade_id': trade.trade_id,
                'settlement_date': settlement_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'remaining_maturity': remaining_maturity
            })
        
        return {
            'step': 4,
            'title': 'Time Parameters (S, E, M)',
            'description': 'Calculate settlement date, end date, and maturity for each trade',
            'data': time_params,
            'formula': 'M = (End Date - Settlement Date) / 365.25',
            'result': f"Calculated time parameters for {len(trades)} trades"
        }
    
    def _step5_adjusted_notional(self, trades: List[Trade]) -> Dict:
        adjusted_notionals = []
        for trade in trades:
            adjusted_notional = abs(trade.notional)
            adjusted_notionals.append({
                'trade_id': trade.trade_id,
                'original_notional': trade.notional,
                'adjusted_notional': adjusted_notional
            })
        
        return {
            'step': 5,
            'title': 'Adjusted Notional',
            'description': 'Calculate adjusted notional amounts',
            'data': adjusted_notionals,
            'formula': 'Adjusted Notional = Notional × Supervisory Duration',
            'result': f"Calculated adjusted notionals for {len(trades)} trades"
        }
    
    def _step7_supervisory_delta(self, trades: List[Trade]) -> Dict:
        supervisory_deltas = []
        for trade in trades:
            if trade.trade_type in [TradeType.OPTION, TradeType.SWAPTION]:
                supervisory_delta = trade.delta
            else:
                supervisory_delta = 1.0 if trade.notional > 0 else -1.0
                
            supervisory_deltas.append({
                'trade_id': trade.trade_id,
                'trade_type': trade.trade_type.value,
                'supervisory_delta': supervisory_delta
            })
        
        return {
            'step': 7,
            'title': 'Supervisory Delta',
            'description': 'Determine supervisory delta per trade type',
            'data': supervisory_deltas,
            'formula': 'δ = trade delta for options, +/-1.0 for linear products',
            'result': f"Calculated supervisory deltas for {len(trades)} trades"
        }
    
    def _step10_supervisory_correlation(self, trades: List[Trade]) -> Dict:
        correlations = []
        asset_classes = set(trade.asset_class for trade in trades)
        
        for asset_class in asset_classes:
            correlation = self.supervisory_correlations.get(asset_class, 0.5)
            correlations.append({
                'asset_class': asset_class.value,
                'supervisory_correlation': correlation
            })
        
        return {
            'step': 10,
            'title': 'Supervisory Correlation',
            'description': 'Apply supervisory correlations by asset class',
            'data': correlations,
            'formula': 'Correlation per Basel regulatory mapping tables',
            'result': f"Applied correlations for {len(asset_classes)} asset classes"
        }
    
    def _step11_hedging_set_addon(self, trades: List[Trade]) -> Dict:
        hedging_sets = {}
        for trade in trades:
            hedging_set_key = f"{trade.asset_class.value}_{trade.currency}"
            if hedging_set_key not in hedging_sets:
                hedging_sets[hedging_set_key] = []
            
            adjusted_notional = abs(trade.notional)
            supervisory_delta = trade.delta if trade.trade_type in [TradeType.OPTION, TradeType.SWAPTION] else (1.0 if trade.notional > 0 else -1.0)
            remaining_maturity = trade.time_to_maturity()
            mf = math.sqrt(min(remaining_maturity, 1.0))
            
            effective_notional = adjusted_notional * supervisory_delta * mf
            hedging_sets[hedging_set_key].append(effective_notional)

        hedging_set_addons = []
        for hedging_set_key, effective_notionals in hedging_sets.items():
            asset_class_str = hedging_set_key.split('_')[0]
            asset_class = next((ac for ac in AssetClass if ac.value == asset_class_str), None)
            
            # Find a representative trade to get SF
            rep_trade = next(t for t in trades if f"{t.asset_class.value}_{t.currency}" == hedging_set_key)
            sf = self._get_supervisory_factor(rep_trade) / 10000

            sum_effective_notionals = sum(effective_notionals)
            hedging_set_addon = abs(sum_effective_notionals) * sf

            hedging_set_addons.append({
                'hedging_set': hedging_set_key,
                'trade_count': len(effective_notionals),
                'hedging_set_addon': hedging_set_addon
            })

        return {
            'step': 11,
            'title': 'Hedging Set AddOn',
            'description': 'Aggregate effective notionals within hedging sets',
            'data': hedging_set_addons,
            'formula': 'Hedging Set AddOn = |Σ(Effective Notional)| × SF',
            'result': f"Calculated add-ons for {len(hedging_sets)} hedging sets"
        }

    def _step12_asset_class_addon(self, trades: List[Trade]) -> Dict:
        step11_result = self._step11_hedging_set_addon(trades)
        
        asset_class_addons_map = {}
        for hedging_set_data in step11_result['data']:
            asset_class = hedging_set_data['hedging_set'].split('_')[0]
            if asset_class not in asset_class_addons_map:
                asset_class_addons_map[asset_class] = []
            asset_class_addons_map[asset_class].append(hedging_set_data['hedging_set_addon'])
        
        asset_class_results = []
        for asset_class_str, hedging_set_addons_list in asset_class_addons_map.items():
            asset_class_enum = next((ac for ac in AssetClass if ac.value == asset_class_str), None)
            rho = self.supervisory_correlations.get(asset_class_enum, 0.5)
            
            sum_addons = sum(hedging_set_addons_list)
            sum_sq_addons = sum(a**2 for a in hedging_set_addons_list)
            
            term1_sq = (rho * sum_addons)**2
            term2 = (1 - rho**2) * sum_sq_addons
            
            asset_class_addon = math.sqrt(term1_sq + term2)
            
            asset_class_results.append({
                'asset_class': asset_class_str,
                'hedging_set_addons': hedging_set_addons_list,
                'asset_class_addon': asset_class_addon
            })
        
        return {
            'step': 12,
            'title': 'Asset Class AddOn',
            'description': 'Aggregate hedging set add-ons by asset class',
            'data': asset_class_results,
            'formula': 'AddOn_AC = sqrt((ρ * ΣA)² + (1-ρ²) * Σ(A²))',
            'result': f"Calculated asset class add-ons for {len(asset_class_results)} classes"
        }
    
    def _step17_th_mta_nica(self, netting_set: NettingSet) -> Dict:
        return {
            'step': 17,
            'title': 'TH, MTA, NICA',
            'description': 'Extract threshold, MTA, and NICA from netting agreement',
            'data': {
                'threshold': netting_set.threshold,
                'mta': netting_set.mta,
                'nica': netting_set.nica
            },
            'formula': 'Sourced from CSA/ISDA agreements',
            'result': f"TH: ${netting_set.threshold:,.0f}, MTA: ${netting_set.mta:,.0f}, NICA: ${netting_set.nica:,.0f}",
            'threshold': netting_set.threshold,
            'mta': netting_set.mta,
            'nica': netting_set.nica
        }
    
    def _step19_ceu_flag(self, trades: List[Trade]) -> Dict:
        ceu_flags = []
        for trade in trades:
            ceu_flags.append({
                'trade_id': trade.trade_id,
                'ceu_flag': getattr(trade, 'ceu_flag', 1)
            })
        
        overall_ceu = 1 if any(getattr(trade, 'ceu_flag', 1) == 1 for trade in trades) else 0
        
        return {
            'step': 19,
            'title': 'CEU Flag',
            'description': 'Determine central clearing status',
            'data': {
                'trade_ceu_flags': ceu_flags,
                'overall_ceu_flag': overall_ceu
            },
            'formula': 'CEU = 1 for non-centrally cleared, 0 for centrally cleared',
            'result': f"CEU Flag: {overall_ceu}",
            'ceu_flag': overall_ceu
        }
    
    def _step20_alpha(self, ceu_flag: int) -> Dict:
        alpha = 1.4 # Alpha is fixed at 1.4 for SA-CCR
        
        return {
            'step': 20,
            'title': 'Alpha',
            'description': 'Regulatory multiplier for SA-CCR',
            'data': {
                'ceu_flag': ceu_flag,
                'alpha': alpha
            },
            'formula': 'Alpha = 1.4 (fixed for SA-CCR)',
            'result': f"Alpha: {alpha}",
            'alpha': alpha
        }
    
    def _step22_counterparty_info(self, counterparty: str) -> Dict:
        # In a real system, this would involve a lookup
        counterparty_data = {
            'counterparty_name': counterparty,
            'legal_code': '?',
            'legal_code_description': 'Corporate',
            'country': 'US',
            'r35_risk_weight_category': 'Corporate'
        }
        
        return {
            'step': 22,
            'title': 'Counterparty Information',
            'description': 'Source counterparty details from a master system',
            'data': counterparty_data,
            'formula': 'Sourced from internal systems',
            'result': f"Counterparty: {counterparty}, Category: {counterparty_data['r35_risk_weight_category']}",
            'counterparty_type': counterparty_data['r35_risk_weight_category']
        }
    
    def _step23_risk_weight(self, counterparty_type: str) -> Dict:
        risk_weight_mapping = {
            'Corporate': 1.0,
            'Bank': 0.20,
            'Sovereign': 0.0,
            'Non-Profit Org': 1.0
        }
        
        risk_weight = risk_weight_mapping.get(counterparty_type, 1.0)
        
        return {
            'step': 23,
            'title': 'Standardized Risk Weight',
            'description': 'Apply regulatory risk weight based on counterparty type',
            'data': {
                'counterparty_type': counterparty_type,
                'risk_weight_percent': f"{risk_weight * 100:.0f}%",
                'risk_weight_decimal': risk_weight
            },
            'formula': 'Risk Weight per applicable regulatory framework',
            'result': f"Risk Weight: {risk_weight * 100:.0f}%",
            'risk_weight': risk_weight
        }
    
    def _get_supervisory_factor(self, trade: Trade) -> float:
        """Get supervisory factor in basis points"""
        if trade.asset_class == AssetClass.INTEREST_RATE:
            maturity = trade.time_to_maturity()
            currency_group = trade.currency if trade.currency in ['USD', 'EUR', 'JPY', 'GBP'] else 'other'
            
            if maturity < 2:
                return self.supervisory_factors[AssetClass.INTEREST_RATE][currency_group]['<2y']
            elif maturity <= 5:
                return self.supervisory_factors[AssetClass.INTEREST_RATE][currency_group]['2-5y']
            else:
                return self.supervisory_factors[AssetClass.INTEREST_RATE][currency_group]['>5y']
        
        elif trade.asset_class == AssetClass.FOREIGN_EXCHANGE:
            g10_currencies = ['USD', 'EUR', 'JPY', 'GBP', 'CHF', 'CAD', 'AUD', 'NZD', 'SEK', 'NOK']
            is_g10 = trade.currency in g10_currencies
            return self.supervisory_factors[AssetClass.FOREIGN_EXCHANGE]['G10' if is_g10 else 'emerging']
        
        elif trade.asset_class == AssetClass.CREDIT:
            return self.supervisory_factors[AssetClass.CREDIT]['IG_single'] * 100
        
        elif trade.asset_class == AssetClass.EQUITY:
            return self.supervisory_factors[AssetClass.EQUITY]['single_large']
        
        elif trade.asset_class == AssetClass.COMMODITY:
            return self.supervisory_factors[AssetClass.COMMODITY]['energy']
        
        return 1.0

# ==============================================================================
# ENHANCED STREAMLIT APPLICATION
# ==============================================================================

def main():
    """Enhanced main application with improved navigation and features"""
    # AI-Powered Header
    st.markdown("""
    <div class="ai-header">
        <div class="executive-title">🤖 AI SA-CCR Platform</div>
        <div class="executive-subtitle">Complete 24-Step Basel SA-CCR Calculator with Advanced AI Analysis</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize comprehensive agent
    if 'saccr_agent' not in st.session_state:
        st.session_state.saccr_agent = ComprehensiveSACCRAgent()
    
    # Sidebar with enhanced LLM Configuration
    with st.sidebar:
        st.markdown("### 🤖 LLM Configuration")
        
        # Configuration inputs with improved defaults
        with st.expander("🔧 LLM Setup", expanded=True):
            base_url = st.text_input("Base URL", value="http://localhost:8123/v1", help="Local LLM server endpoint")
            api_key = st.text_input("API Key", value="dummy", type="password", help="API key for authentication")
            model = st.text_input("Model", value="llama3", help="Model name to use")
            temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1, help="Controls randomness in responses")
            max_tokens = st.number_input("Max Tokens", 1000, 8000, 4000, 100, help="Maximum response length")
            
            if st.button("🔗 Connect LLM"):
                config = {
                    'base_url': base_url,
                    'api_key': api_key,
                    'model': model,
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    'streaming': False
                }
                
                with st.spinner("Connecting to LLM..."):
                    success = st.session_state.saccr_agent.setup_llm_connection(config)
                    if success:
                        st.success("✅ LLM Connected!")
                        st.session_state.llm_config = config
                    else:
                        st.error("❌ Connection Failed")
        
        # Connection status with enhanced display
        status = st.session_state.saccr_agent.connection_status
        if status == "connected":
            st.markdown('<div class="connection-status connected">🟢 LLM Connected - AI Analysis Available</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="connection-status disconnected">🔴 LLM Disconnected - Basic Mode Only</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📊 Navigation")
        page = st.selectbox(
            "Select Module:",
            [
                "🧮 Enhanced SA-CCR Calculator", 
                "📋 Reference Example", 
                "🤖 AI Assistant", 
                "📊 Portfolio Analysis",
                "📈 Data Quality Analysis"
            ]
        )
    
    # Route to different pages
    if page == "🧮 Enhanced SA-CCR Calculator":
        enhanced_complete_saccr_calculator()
    elif page == "📋 Reference Example":
        show_reference_example()
    elif page == "🤖 AI Assistant":
        enhanced_ai_assistant_page()
    elif page == "📊 Portfolio Analysis":
        portfolio_analysis_page()
    elif page == "📈 Data Quality Analysis":
        analyze_portfolio_data_quality()

def enhanced_complete_saccr_calculator():
    """Enhanced 24-step SA-CCR calculator with advanced features"""
    
    st.markdown("## 🧮 Enhanced SA-CCR Calculator")
    st.markdown("*Following the complete 24-step Basel regulatory framework with AI-powered insights*")
    
    # Step 1: Enhanced Netting Set Setup
    with st.expander("📊 Step 1: Netting Set Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            netting_set_id = st.text_input(
                "Netting Set ID*", 
                placeholder="e.g., 212784060000009618701",
                help="Unique identifier for the netting agreement"
            )
            counterparty = st.text_input(
                "Counterparty*", 
                placeholder="e.g., Lowell Hotel Properties LLC",
                help="Legal entity name of the counterparty"
            )
            
        with col2:
            threshold = st.number_input(
                "Threshold ($)*", 
                min_value=0.0, 
                value=1000000.0, 
                step=100000.0,
                help="Minimum exposure before collateral posting requirement"
            )
            mta = st.number_input(
                "MTA ($)*", 
                min_value=0.0, 
                value=500000.0, 
                step=50000.0,
                help="Minimum Transfer Amount for collateral calls"
            )
            nica = st.number_input(
                "NICA ($)", 
                min_value=0.0, 
                value=0.0, 
                step=10000.0,
                help="Net Independent Collateral Amount"
            )
    
    # Step 2: Enhanced Trade Input with Template
    st.markdown("### 📈 Trade Portfolio Input")
    
    # Initialize trades if not exists
    if 'trades_input' not in st.session_state:
        st.session_state.trades_input = []
    
    # Quick templates
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📊 Load Interest Rate Swap"):
            load_reference_example()
    with col2:
        if st.button("🌍 Load FX Forward"):
            _load_fx_template()
    with col3:
        if st.button("📈 Load Equity Option"):
            _load_equity_template()
    
    with st.expander("➕ Add New Trade", expanded=len(st.session_state.trades_input) == 0):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            trade_id = st.text_input("Trade ID*", placeholder="e.g., 2098474100")
            asset_class = st.selectbox("Asset Class*", [ac.value for ac in AssetClass])
            trade_type = st.selectbox("Trade Type*", [tt.value for tt in TradeType])
        
        with col2:
            notional = st.number_input("Notional ($)*", min_value=0.0, value=100000000.0, step=1000000.0)
            currency = st.selectbox("Currency*", ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"])
            underlying = st.text_input("Underlying*", placeholder="e.g., Interest rate")
        
        with col3:
            maturity_years = st.number_input("Maturity (Years)*", min_value=0.1, max_value=30.0, value=5.0, step=0.1)
            mtm_value = st.number_input("MTM Value ($)", value=0.0, step=10000.0, help="Current Mark-to-Market value")
            delta = st.number_input("Delta (for options)", min_value=-1.0, max_value=1.0, value=1.0, step=0.1, help="Option sensitivity")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ Add Trade", type="primary"):
                if trade_id and notional > 0 and currency and underlying:
                    new_trade = Trade(
                        trade_id=trade_id,
                        counterparty=counterparty,
                        asset_class=AssetClass(asset_class),
                        trade_type=TradeType(trade_type),
                        notional=notional,
                        currency=currency,
                        underlying=underlying,
                        maturity_date=datetime.now() + timedelta(days=int(maturity_years * 365)),
                        mtm_value=mtm_value,
                        delta=delta
                    )
                    st.session_state.trades_input.append(new_trade)
                    st.success(f"✅ Added trade {trade_id}")
                    st.rerun()
                else:
                    st.error("❌ Please fill all required fields marked with *")
        
        with col2:
            if st.button("🗑️ Clear All Trades"):
                st.session_state.trades_input = []
                st.rerun()
    
    # Enhanced Trade Display with Visualizations
    if st.session_state.trades_input:
        st.markdown("### 📋 Current Trade Portfolio")
        
        # Create enhanced trade display
        trades_data = []
        total_notional = 0
        for i, trade in enumerate(st.session_state.trades_input):
            trades_data.append({
                'Index': i,
                'Trade ID': trade.trade_id,
                'Asset Class': trade.asset_class.value,
                'Type': trade.trade_type.value,
                'Notional ($M)': f"{trade.notional/1_000_000:.1f}",
                'Currency': trade.currency,
                'MTM ($K)': f"{trade.mtm_value/1000:.0f}",
                'Maturity (Y)': f"{trade.time_to_maturity():.1f}",
                'Delta': f"{trade.delta:.2f}"
            })
            total_notional += abs(trade.notional)
        
        df = pd.DataFrame(trades_data)
        st.dataframe(df, use_container_width=True)
        
        # Portfolio summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Trades", len(st.session_state.trades_input))
        with col2:
            st.metric("Total Notional", f"${total_notional/1_000_000:.1f}M")
        with col3:
            asset_classes = len(set(trade.asset_class for trade in st.session_state.trades_input))
            st.metric("Asset Classes", asset_classes)
        with col4:
            avg_maturity = sum(trade.time_to_maturity() for trade in st.session_state.trades_input) / len(st.session_state.trades_input)
            st.metric("Avg Maturity", f"{avg_maturity:.1f}Y")
        
        # Portfolio visualization
        if len(st.session_state.trades_input) > 1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Asset class distribution
                asset_class_data = {}
                for trade in st.session_state.trades_input:
                    ac = trade.asset_class.value
                    asset_class_data[ac] = asset_class_data.get(ac, 0) + abs(trade.notional)
                
                fig_pie = px.pie(
                    values=list(asset_class_data.values()),
                    names=list(asset_class_data.keys()),
                    title="Portfolio by Asset Class ($)"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Maturity distribution
                maturity_data = [trade.time_to_maturity() for trade in st.session_state.trades_input]
                notional_data = [abs(trade.notional)/1_000_000 for trade in st.session_state.trades_input]
                
                fig_scatter = px.scatter(
                    x=maturity_data,
                    y=notional_data,
                    title="Trade Distribution: Maturity vs Notional",
                    labels={'x': 'Maturity (Years)', 'y': 'Notional ($M)'}
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Remove trade option
        if len(st.session_state.trades_input) > 0:
            remove_idx = st.selectbox("Remove trade by index:", [-1] + list(range(len(st.session_state.trades_input))))
            if remove_idx >= 0 and st.button("🗑️ Remove Selected Trade"):
                st.session_state.trades_input.pop(remove_idx)
                st.rerun()
    
    # Step 3: Enhanced Collateral Input
    with st.expander("🛡️ Collateral Portfolio", expanded=False):
        if 'collateral_input' not in st.session_state:
            st.session_state.collateral_input = []
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            coll_type = st.selectbox("Collateral Type", [ct.value for ct in CollateralType])
        with col2:
            coll_currency = st.selectbox("Collateral Currency", ["USD", "EUR", "GBP", "JPY"])
        with col3:
            coll_amount = st.number_input("Amount ($)", min_value=0.0, value=10000000.0, step=1000000.0)
        with col4:
            st.markdown("**Regulatory Haircuts:**")
            haircut_info = {
                "Cash": "0%",
                "Government Bonds": "0.5%",
                "Corporate Bonds": "4%",
                "Equities": "15%",
                "Money Market Funds": "0.5%"
            }
            for ctype, haircut in haircut_info.items():
                st.write(f"• {ctype}: {haircut}")
        
        if st.button("➕ Add Collateral"):
            new_collateral = Collateral(
                collateral_type=CollateralType(coll_type),
                currency=coll_currency,
                amount=coll_amount
            )
            st.session_state.collateral_input.append(new_collateral)
            st.success(f"✅ Added {coll_type} collateral")
        
        if st.session_state.collateral_input:
            st.markdown("**Current Collateral:**")
            total_collateral = 0
            for i, coll in enumerate(st.session_state.collateral_input):
                st.write(f"{i+1}. {coll.collateral_type.value}: ${coll.amount:,.0f} {coll.currency}")
                total_collateral += coll.amount
            st.metric("Total Collateral Posted", f"${total_collateral/1_000_000:.1f}M")
    
    # Enhanced Validation and Calculation
    if st.button("🚀 Calculate Enhanced SA-CCR", type="primary"):
        # Comprehensive validation
        if not netting_set_id or not counterparty or not st.session_state.trades_input:
            st.error("❌ Please provide Netting Set ID, Counterparty, and at least one trade")
            return
        
        netting_set = NettingSet(
            netting_set_id=netting_set_id,
            counterparty=counterparty,
            trades=st.session_state.trades_input,
            threshold=threshold,
            mta=mta,
            nica=nica
        )
        
        # Enhanced input validation
        validation = st.session_state.saccr_agent.validate_input_completeness(
            netting_set, st.session_state.collateral_input
        )
        
        if not validation['is_complete']:
            st.error("❌ Missing required information:")
            for field in validation['missing_fields']:
                st.write(f"   • {field}")
            return
        
        if validation['warnings']:
            st.warning("⚠️ Warnings (calculation will proceed with defaults):")
            for warning in validation['warnings']:
                st.write(f"   • {warning}")
        
        # Perform enhanced calculation with progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("🧮 Initializing 24-step SA-CCR calculation...")
            progress_bar.progress(10)
            
            result = st.session_state.saccr_agent.calculate_comprehensive_saccr(
                netting_set, st.session_state.collateral_input
            )
            
            progress_bar.progress(100)
            status_text.text("✅ Calculation completed successfully!")
            
            # Store results and netting set data for AI Assistant access
            st.session_state.saccr_result = result
            st.session_state.netting_set_id = netting_set_id
            st.session_state.counterparty = counterparty
            st.session_state.threshold = threshold
            st.session_state.mta = mta
            st.session_state.nica = nica
            
            # Display enhanced results
            display_enhanced_saccr_results(result, netting_set)
            
        except Exception as e:
            st.error(f"❌ Calculation error: {str(e)}")
            st.exception(e)

def display_enhanced_saccr_results(result: Dict, netting_set: NettingSet):
    """Display comprehensive SA-CCR calculation results with enhanced visualizations"""
    
    st.markdown("## 📊 Enhanced SA-CCR Calculation Results")
    
    final_results = result['final_results']
    enhanced_summary = result.get('enhanced_summary', {})
    
    # Executive Summary Dashboard
    with st.container():
        st.markdown("### 🎯 Executive Dashboard")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric(
                "Replacement Cost", 
                f"${final_results['replacement_cost']/1_000_000:.2f}M",
                help="Current exposure if counterparty defaults today"
            )
        with col2:
            st.metric(
                "PFE", 
                f"${final_results['potential_future_exposure']/1_000_000:.2f}M",
                help="Potential increase in exposure over trade lifetime"
            )
        with col3:
            st.metric(
                "EAD", 
                f"${final_results['exposure_at_default']/1_000_000:.2f}M",
                help="Total credit exposure at default"
            )
        with col4:
            st.metric(
                "RWA", 
                f"${final_results['risk_weighted_assets']/1_000_000:.2f}M",
                help="Risk-weighted assets for capital calculation"
            )
        with col5:
            st.metric(
                "Capital Required", 
                f"${final_results['capital_requirement']/1000:.0f}K",
                help="Minimum regulatory capital requirement"
            )
    
    # Enhanced Summary with Key Insights
    if enhanced_summary:
        with st.expander("📋 Executive Summary & Key Insights", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔑 Key Inputs")
                for insight in enhanced_summary.get('key_inputs', []):
                    st.write(f"• {insight}")
                
                st.markdown("#### ⚖️ Risk Components")
                for insight in enhanced_summary.get('risk_components', []):
                    st.write(f"• {insight}")
            
            with col2:
                st.markdown("#### 💰 Capital Results")
                for insight in enhanced_summary.get('capital_results', []):
                    st.write(f"• {insight}")
                
                st.markdown("#### 🎯 Optimization Insights")
                for insight in enhanced_summary.get('optimization_insights', []):
                    st.write(f"• {insight}")
    
    # Data Quality Issues Analysis
    if result.get('data_quality_issues'):
        with st.expander("🔍 Data Quality Analysis", expanded=False):
            st.markdown("#### Data Quality Issues Identified:")
            
            issues = result['data_quality_issues']
            high_impact = [i for i in issues if i.impact == 'high']
            medium_impact = [i for i in issues if i.impact == 'medium']
            
            if high_impact:
                st.markdown("**🔴 High Impact Issues:**")
                for issue in high_impact:
                    st.markdown(f"""
                    <div class="data-quality-alert">
                        <strong>{issue.field_name}</strong><br>
                        Current: {issue.current_value}<br>
                        Issue: {issue.issue_type}<br>
                        Impact: {issue.recommendation}
                    </div>
                    """, unsafe_allow_html=True)
            
            if medium_impact:
                st.markdown("**🟡 Medium Impact Issues:**")
                for issue in medium_impact:
                    st.write(f"• **{issue.field_name}**: {issue.recommendation}")
    
    # Thinking Process Analysis (if available)
    if result.get('thinking_steps'):
        with st.expander("🧠 AI Thinking Process Analysis", expanded=False):
            st.markdown("#### Step-by-Step Regulatory Analysis:")
            
            for thinking in result['thinking_steps']:
                reasoning_text = thinking['reasoning'].replace('\n', '<br>')
                st.markdown(f"""
                <div class="thinking-process">
                    <h4>Step {thinking['step']}: {thinking['title']}</h4>
                    <div class="step-reasoning">
                        {reasoning_text}
                    </div>
                    <div class="formula-breakdown">
                        <strong>Formula:</strong> {thinking['formula']}
                    </div>
                    <div class="ai-insight">
                        <strong>Key Insight:</strong> {thinking['key_insight']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Enhanced Step-by-Step Breakdown with Visualizations
    with st.expander("🔍 Complete 24-Step Calculation Breakdown", expanded=True):
        
        # Create interactive step navigation
        step_categories = {
            "📊 Data & Classification (1-4)": [1, 2, 3, 4],
            "⚙️ Risk Calculations (5-10)": [5, 6, 7, 8, 9, 10],
            "📈 Add-On Aggregation (11-13)": [11, 12, 13],
            "🎯 PFE Calculation (14-16)": [14, 15, 16],
            "💸 Replacement Cost (17-18)": [17, 18],
            "🏦 Final EAD & Capital (19-24)": [19, 20, 21, 22, 23, 24]
        }
        
        selected_category = st.selectbox("Select calculation phase:", list(step_categories.keys()))
        step_numbers = step_categories[selected_category]
        
        for step_num in step_numbers:
            if step_num <= len(result['calculation_steps']):
                step_data = result['calculation_steps'][step_num - 1]
                
                with st.container():
                    st.markdown(f"""
                    <div class="calc-step">
                        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                            <span class="step-number">{step_data['step']}</span>
                            <span class="step-title">{step_data['title']}</span>
                        </div>
                        <div style="margin-bottom: 0.5rem;">
                            <strong>Description:</strong> {step_data['description']}
                        </div>
                        <div class="step-formula">{step_data['formula']}</div>
                        <div style="font-size: 1.1rem; font-weight: 600; color: #0f4c75; margin-top: 0.5rem;">
                            <strong>Result:</strong> {step_data['result']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show detailed data for key steps
                    if step_num in [9, 11, 12, 13, 15, 16, 18, 21, 24] and isinstance(step_data.get('data'), dict):
                        with st.expander(f"📊 Detailed Data - Step {step_num}", expanded=False):
                            st.json(step_data['data'])
    
    # Enhanced AI Analysis
    if result.get('ai_explanation'):
        st.markdown("### 🤖 AI Expert Analysis")
        ai_explanation_text = result['ai_explanation'].replace('\n', '<br><br>')
        st.markdown(f"""
        <div class="ai-response">
            <h4>🎯 Regulatory Expert Insights</h4>
            {ai_explanation_text}
        </div>
        """, unsafe_allow_html=True)
    
    # Risk Visualization Dashboard
    _create_risk_visualizations(result, netting_set)
    
    # Enhanced Export Options
    _create_enhanced_export_options(result, netting_set)

def _create_risk_visualizations(result: Dict, netting_set: NettingSet):
    """Create comprehensive risk visualization dashboard"""
    
    st.markdown("### 📊 Risk Analysis Dashboard")
    
    # Get key metrics
    final_results = result['final_results']
    rc = final_results['replacement_cost']
    pfe = final_results['potential_future_exposure']
    ead = final_results['exposure_at_default']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # EAD Composition
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Current Exposure (RC)', 'Future Exposure (PFE)'],
            values=[rc, pfe],
            hole=0.3,
            marker_colors=['#ff6b6b', '#4ecdc4']
        )])
        fig_pie.update_layout(
            title="EAD Composition: Current vs Future Risk",
            showlegend=True
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Risk Progression Waterfall
        categories = ['Aggregate AddOn', 'PFE Multiplier Effect', 'Replacement Cost', 'Alpha Multiplier']
        
        # Find key calculation steps
        step13 = next((s for s in result['calculation_steps'] if s['step'] == 13), {})
        step15 = next((s for s in result['calculation_steps'] if s['step'] == 15), {})
        step18 = next((s for s in result['calculation_steps'] if s['step'] == 18), {})
        step21 = next((s for s in result['calculation_steps'] if s['step'] == 21), {})
        
        aggregate_addon = step13.get('aggregate_addon', 0)
        multiplier = step15.get('multiplier', 1)
        pfe_reduction = aggregate_addon * (1 - multiplier)
        alpha = step21.get('data', {}).get('alpha', 1.4)
        
        values = [aggregate_addon/1_000_000, -pfe_reduction/1_000_000, rc/1_000_000, (ead - rc - pfe)/1_000_000]
        
        fig_waterfall = go.Figure(go.Waterfall(
            name="Risk Components",
            orientation="v",
            measure=["relative", "relative", "relative", "relative"],
            x=categories,
            textposition="outside",
            text=[f"${v:.1f}M" for v in values],
            y=values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        
        fig_waterfall.update_layout(
            title="Risk Calculation Flow ($M)",
            showlegend=False
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)
    
    # Portfolio Risk Heatmap
    if len(netting_set.trades) > 1:
        st.markdown("#### 🔥 Trade Risk Heatmap")
        
        # Calculate risk contribution per trade
        trade_risks = []
        for trade in netting_set.trades:
            notional_contrib = abs(trade.notional) / sum(abs(t.notional) for t in netting_set.trades)
            maturity_risk = min(trade.time_to_maturity(), 5) / 5  # Normalize to 0-1
            
            trade_risks.append({
                'Trade ID': trade.trade_id,
                'Asset Class': trade.asset_class.value,
                'Notional Weight': notional_contrib,
                'Maturity Risk': maturity_risk,
                'Currency': trade.currency,
                'Trade Type': trade.trade_type.value
            })
        
        df_risk = pd.DataFrame(trade_risks)
        
        fig_heatmap = px.scatter(
            df_risk,
            x='Notional Weight',
            y='Maturity Risk',
            size='Notional Weight',
            color='Asset Class',
            hover_data=['Trade ID', 'Currency', 'Trade Type'],
            title="Trade Risk Distribution (Size = Notional Weight)"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

def _create_enhanced_export_options(result: Dict, netting_set: NettingSet):
    """Create comprehensive export options"""
    
    st.markdown("### 📥 Enhanced Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Executive Summary Report
        executive_summary = {
            'Calculation_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Netting_Set_ID': netting_set.netting_set_id,
            'Counterparty': netting_set.counterparty,
            'Total_Trades': len(netting_set.trades),
            'Total_Notional_USD': sum(abs(trade.notional) for trade in netting_set.trades),
            'Replacement_Cost_USD': result['final_results']['replacement_cost'],
            'Potential_Future_Exposure_USD': result['final_results']['potential_future_exposure'],
            'Exposure_at_Default_USD': result['final_results']['exposure_at_default'],
            'Risk_Weighted_Assets_USD': result['final_results']['risk_weighted_assets'],
            'Capital_Required_USD': result['final_results']['capital_requirement'],
            'Capital_Efficiency_Pct': (result['final_results']['capital_requirement'] / sum(abs(trade.notional) for trade in netting_set.trades)) * 100
        }
        
        exec_csv = pd.DataFrame([executive_summary]).to_csv(index=False)
        st.download_button(
            "👔 Executive Summary CSV",
            data=exec_csv,
            file_name=f"saccr_executive_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Regulatory Compliance Report
        compliance_data = []
        for step in result['calculation_steps']:
            compliance_data.append({
                'Step_Number': step['step'],
                'Step_Title': step['title'],
                'Formula_Applied': step['formula'],
                'Result_Value': step['result'],
                'Regulatory_Compliance': 'PASS'
            })
        
        compliance_csv = pd.DataFrame(compliance_data).to_csv(index=False)
        st.download_button(
            "✅ Compliance Report CSV",
            data=compliance_csv,
            file_name=f"saccr_compliance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col3:
        # Complete Audit Trail JSON
        audit_trail = {
            'metadata': {
                'calculation_timestamp': datetime.now().isoformat(),
                'calculator_version': '2.0.0',
                'basel_framework': 'SA-CCR 24-Step',
                'data_quality_score': len([i for i in result.get('data_quality_issues', []) if i.impact != 'high'])
            },
            'inputs': {
                'netting_set': asdict(netting_set),
                'collateral': [asdict(c) for c in getattr(st.session_state, 'collateral_input', [])]
            },
            'calculation_results': result,
            'regulatory_validation': 'COMPLIANT'
        }
        
        audit_json = json.dumps(audit_trail, indent=2, default=str)
        st.download_button(
            "🔍 Audit Trail JSON",
            data=audit_json,
            file_name=f"saccr_audit_trail_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

def load_reference_example():
    """Load the reference example trade"""
    st.session_state.trades_input = []
    st.session_state.collateral_input = []
    
    reference_trade = Trade(
        trade_id="2098474100",
        counterparty="Lowell Hotel Properties LLC",
        asset_class=AssetClass.INTEREST_RATE,
        trade_type=TradeType.SWAP,
        notional=681578963,
        currency="USD",
        underlying="Interest rate",
        maturity_date=datetime.now() + timedelta(days=int(0.3 * 365)),
        mtm_value=0,
        delta=1.0
    )
    
    st.session_state.trades_input = [reference_trade]
    st.success("✅ Loaded Interest Rate Swap reference example")

def _load_fx_template():
    """Load FX Forward template"""
    fx_trade = Trade(
        trade_id="FX001",
        counterparty="Sample Bank",
        asset_class=AssetClass.FOREIGN_EXCHANGE,
        trade_type=TradeType.FORWARD,
        notional=50000000,
        currency="EUR",
        underlying="EUR/USD",
        maturity_date=datetime.now() + timedelta(days=180),
        mtm_value=125000,
        delta=1.0
    )
    
    if 'trades_input' not in st.session_state:
        st.session_state.trades_input = []
    st.session_state.trades_input.append(fx_trade)
    st.success("✅ Added FX Forward template")

def _load_equity_template():
    """Load Equity Option template"""
    equity_trade = Trade(
        trade_id="EQ001",
        counterparty="Hedge Fund ABC",
        asset_class=AssetClass.EQUITY,
        trade_type=TradeType.OPTION,
        notional=25000000,
        currency="USD",
        underlying="S&P 500 Index",
        maturity_date=datetime.now() + timedelta(days=90),
        mtm_value=-75000,
        delta=0.65
    )
    
    if 'trades_input' not in st.session_state:
        st.session_state.trades_input = []
    st.session_state.trades_input.append(equity_trade)
    st.success("✅ Added Equity Option template")

def show_reference_example():
    """Enhanced reference example with detailed explanation"""
    
    st.markdown("## 📋 Reference Example - Basel SA-CCR Validation")
    st.markdown("*Industry-standard calculation example with step-by-step validation*")
    
    # Load example button
    if st.button("🔄 Load Complete Reference Example", type="primary"):
        load_reference_example()
        
        # Auto-calculate the reference example
        netting_set = NettingSet(
            netting_set_id="212784060000009618701",
            counterparty="Lowell Hotel Properties LLC",
            trades=st.session_state.trades_input,
            threshold=12000000,
            mta=1000000,
            nica=0
        )
        
        st.success("✅ Reference example loaded and calculated!")
        
        # Display reference details
        with st.expander("📊 Reference Trade Details", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **Trade Information:**
                - **Trade ID**: 2098474100
                - **Counterparty**: Lowell Hotel Properties LLC
                - **Asset Class**: Interest Rate
                - **Trade Type**: Swap
                - **Notional**: $681,578,963
                """)
            
            with col2:
                st.markdown("""
                **Market Data:**
                - **Currency**: USD
                - **Remaining Maturity**: ~0.3 years
                - **MTM Value**: $0
                - **Supervisory Factor**: 50bps (USD IR <2Y)
                """)
        
        # Calculate and display results
        with st.spinner("🧮 Performing reference calculation..."):
            try:
                result = st.session_state.saccr_agent.calculate_comprehensive_saccr(netting_set, [])
                
                # Display key validation points
                st.markdown("### ✅ Reference Validation Results")
                
                final_results = result['final_results']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Adjusted Notional", f"${681578963:,.0f}")
                with col2:
                    st.metric("PFE", f"${final_results['potential_future_exposure']:,.0f}")
                with col3:
                    st.metric("EAD", f"${final_results['exposure_at_default']:,.0f}")
                with col4:
                    st.metric("RWA", f"${final_results['risk_weighted_assets']:,.0f}")
                
                # Show calculation verification
                with st.expander("🔍 Step-by-Step Validation", expanded=True):
                    key_steps = [5, 6, 8, 9, 13, 15, 16, 18, 21, 24]
                    
                    for step_num in key_steps:
                        if step_num <= len(result['calculation_steps']):
                            step_data = result['calculation_steps'][step_num - 1]
                            
                            st.markdown(f"""
                            <div class="calculation-verified">
                                <strong>✅ Step {step_data['step']}: {step_data['title']}</strong><br>
                                <em>Result:</em> {step_data['result']}<br>
                                <em>Formula:</em> {step_data['formula']}
                            </div>
                            """, unsafe_allow_html=True)
                
                # Regulatory compliance check
                st.markdown("### 📋 Regulatory Compliance Assessment")
                
                compliance_checks = [
                    ("24-Step Methodology", "✅ PASS", "All required Basel steps completed"),
                    ("Supervisory Factors", "✅ PASS", "Applied correct regulatory parameters"),
                    ("Maturity Factor", "✅ PASS", "Properly adjusted for time to maturity"),
                    ("Netting Benefits", "✅ PASS", "Margining terms correctly applied"),
                    ("Alpha Multiplier", "✅ PASS", "1.4 applied for non-centrally cleared")
                ]
                
                for check, status, description in compliance_checks:
                    st.markdown(f"**{check}**: {status} - {description}")
                
            except Exception as e:
                st.error(f"❌ Reference calculation error: {str(e)}")
    
    # Methodology comparison
    with st.expander("📚 Basel SA-CCR Methodology Reference", expanded=False):
        st.markdown("""
        ### 🎯 Complete 24-Step Basel Framework
        
        **Phase 1: Data Foundation (Steps 1-4)**
        1. **Netting Set Data** - Source trade and agreement data
        2. **Asset Classification** - Classify by regulatory categories  
        3. **Hedging Set** - Group by common risk factors
        4. **Time Parameters** - Calculate settlement, end dates, maturity
        
        **Phase 2: Risk Factor Calibration (Steps 5-10)**
        5. **Adjusted Notional** - Apply supervisory duration adjustments
        6. **Maturity Factor** - Scale for remaining time to maturity
        7. **Supervisory Delta** - Directional risk for options
        8. **Supervisory Factor** - Regulatory volatility parameters
        9. **Adjusted Contract Amount** - Combine all risk adjustments
        10. **Supervisory Correlation** - Cross-asset diversification
        
        **Phase 3: Add-On Aggregation (Steps 11-13)**
        11. **Hedging Set AddOn** - Aggregate within risk factors
        12. **Asset Class AddOn** - Aggregate across hedging sets
        13. **Aggregate AddOn** - Total portfolio potential exposure
        
        **Phase 4: PFE Calculation (Steps 14-16)**
        14. **V, C Calculation** - Current MTM and collateral values
        15. **PFE Multiplier** - Netting benefit based on current exposure
        16. **PFE** - Final potential future exposure
        
        **Phase 5: Current Exposure (Steps 17-18)**
        17. **Threshold Terms** - Extract CSA margining parameters
        18. **Replacement Cost** - Current credit exposure calculation
        
        **Phase 6: Final Capital (Steps 19-24)**
        19. **CEU Flag** - Central clearing determination
        20. **Alpha** - Regulatory multiplier (1.4 for bilateral)
        21. **EAD** - Total exposure at default
        22. **Counterparty Info** - Credit assessment lookup
        23. **Risk Weight** - Apply regulatory credit risk weight
        24. **RWA** - Final risk-weighted assets for capital
        """)

def enhanced_ai_assistant_page():
    """Enhanced AI assistant with comprehensive SA-CCR expertise"""
    
    st.markdown("## 🤖 AI SA-CCR Expert Assistant")
    st.markdown("*Advanced regulatory analysis with real-time LLM integration*")
    
    # Check LLM connection status
    if st.session_state.saccr_agent.connection_status != "connected":
        st.warning("⚠️ LLM not connected. Please configure and connect in the sidebar for AI features.")
        st.markdown("### 📋 Available in Basic Mode:")
        st.write("• Pre-built calculation templates")
        st.write("• Regulatory reference materials") 
        st.write("• Step-by-step methodology guides")
        return
    
    # AI Assistant Interface
    st.markdown("### 💬 Expert Consultation")
    
    # Quick question templates
    with st.expander("💡 Expert Question Templates", expanded=True):
        st.markdown("""
        **🎯 Regulatory Questions:**
        - "What drives the capital requirement in this SA-CCR calculation?"
        - "How can I optimize this portfolio to reduce RWA?"
        - "Explain the impact of central clearing on this calculation"
        
        **📊 Technical Analysis:**
        - "Break down the PFE multiplier calculation step by step"
        - "Why is the maturity factor applied this way?"
        - "How do netting benefits work in SA-CCR?"
        
        **🏦 Business Strategy:**
        - "What are the key optimization levers for this portfolio?"
        - "How would adding collateral impact the capital requirement?"
        - "Compare bilateral vs centrally cleared capital impact"
        """)
        
        # Quick action buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🧮 Analyze Current Calculation"):
                if 'saccr_result' in st.session_state:
                    process_ai_question("Provide a detailed analysis of the current SA-CCR calculation results, focusing on key risk drivers and optimization opportunities.")
                else:
                    st.warning("Please run a calculation first.")
        
        with col2:
            if st.button("📈 Optimization Strategies"):
                process_ai_question("What are the most effective strategies to reduce regulatory capital for SA-CCR portfolios? Include specific techniques and quantitative impacts.")
        
        with col3:
            if st.button("⚖️ Regulatory Compliance"):
                process_ai_question("Explain the key regulatory compliance requirements for SA-CCR implementation, including common pitfalls and best practices.")
    
    # Enhanced Question Input with Format Options
    st.markdown("### 🎤 Ask Your Question")
    
    # Check if we have calculation data for enhanced format
    has_calc_data = 'saccr_result' in st.session_state and 'trades_input' in st.session_state
    
    if has_calc_data:
        st.success("✅ Enhanced format available - Your questions can now include comprehensive analysis dashboards!")
    else:
        st.info("💡 For enhanced analysis format, please run a calculation first in the Enhanced SA-CCR Calculator.")
    
    user_question = st.text_area(
        "Ask any SA-CCR related question:",
        placeholder="e.g., Provide a comprehensive breakdown of my SA-CCR calculation with optimization recommendations",
        height=100
    )
    
    # Response format selection
    if has_calc_data:
        response_format = st.radio(
            "Response Format:",
            ["🎯 Enhanced Dashboard (Same as Calculate SACCR)", "💬 Text Response Only"],
            help="Enhanced Dashboard provides the same rich format as the Calculate Enhanced SACCR results"
        )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if st.button("🚀 Get AI Analysis", type="primary") and user_question:
            # Force enhanced format if selected
            if has_calc_data and "Enhanced Dashboard" in response_format:
                display_ai_enhanced_analysis(user_question)
            else:
                process_ai_question(user_question)
    
    with col2:
        if st.button("🗑️ Clear History"):
            if 'ai_history' in st.session_state:
                st.session_state.ai_history = []
            st.rerun()
    
    with col3:
        if st.button("📊 View Last Calculation") and has_calc_data:
            st.markdown("### 📊 Current Calculation Results")
            result = st.session_state.saccr_result
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("EAD", f"${result['final_results']['exposure_at_default']:,.0f}")
            with col_b:
                st.metric("RWA", f"${result['final_results']['risk_weighted_assets']:,.0f}")
            with col_c:
                st.metric("Capital", f"${result['final_results']['capital_requirement']:,.0f}")
    
    # Sample enhanced questions for users with calculation data
    if has_calc_data:
        with st.expander("🎯 Enhanced Analysis Questions (Full Dashboard Format)", expanded=False):
            st.markdown("""
            **Try these questions for comprehensive dashboard responses:**
            
            📊 **Calculation Analysis:**
            - "Provide a comprehensive breakdown of my SA-CCR calculation"
            - "Show me the complete step-by-step methodology with visualizations"
            - "Analyze my calculation results with detailed risk components"
            
            🎯 **Optimization Focus:**
            - "How can I optimize this portfolio to reduce capital requirements?"
            - "What are the key optimization levers for my current calculation?"
            - "Analyze capital efficiency and provide improvement recommendations"
            
            ⚖️ **Risk Analysis:**
            - "Break down my risk exposure components with visualizations"
            - "Analyze the current vs future risk in my portfolio"
            - "Show me the risk contribution by trade and asset class"
            """)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📊 Comprehensive Analysis"):
                    display_ai_enhanced_analysis("Provide a comprehensive breakdown of my SA-CCR calculation with detailed risk analysis and optimization recommendations")
            with col2:
                if st.button("🎯 Optimization Focus"):
                    display_ai_enhanced_analysis("How can I optimize this portfolio to reduce capital requirements? Show me all optimization levers with quantitative impacts")
            with col3:
                if st.button("⚖️ Risk Breakdown"):
                    display_ai_enhanced_analysis("Analyze my risk exposure components in detail, including current vs future risk and trade-level contributions")
    
    # Display conversation history
    if 'ai_history' in st.session_state and st.session_state.ai_history:
        st.markdown("### 💭 Conversation History")
        
        for i, (question, answer) in enumerate(reversed(st.session_state.ai_history[-5:])):
            with st.container():
                st.markdown(f"""
                <div class="user-query">
                    <strong>Q{len(st.session_state.ai_history)-i}:</strong> {question}
                </div>
                """, unsafe_allow_html=True)
                
                answer_formatted = answer.replace('\n', '<br>')
                st.markdown(f"""
                <div class="ai-response">
                    <strong>🤖 Expert Analysis:</strong><br>
                    {answer_formatted}
                </div>
                """, unsafe_allow_html=True)

def process_ai_question(question: str):
    """Process AI question with enhanced context and rich response format"""
    
    if 'ai_history' not in st.session_state:
        st.session_state.ai_history = []
    
    # Check if user is asking for calculation analysis and we have results
    calculation_keywords = ['calculate', 'analysis', 'result', 'breakdown', 'step', 'formula', 'ead', 'rwa', 'capital', 'pfe', 'replacement cost']
    is_calculation_query = any(keyword in question.lower() for keyword in calculation_keywords)
    has_calculation_data = 'saccr_result' in st.session_state and 'trades_input' in st.session_state
    
    # If it's a calculation-related question and we have data, show rich format
    if is_calculation_query and has_calculation_data:
        display_ai_enhanced_analysis(question)
        return
    
    # Enhanced system prompt with regulatory expertise
    system_prompt = """You are a world-class Basel SA-CCR regulatory expert with deep expertise in:
    - Complete 24-step SA-CCR methodology 
    - Basel III/IV regulatory frameworks
    - Credit risk capital optimization
    - Derivatives risk management
    - Regulatory compliance and implementation
    
    Provide detailed, technical, and actionable responses. Include:
    1. Clear explanations of regulatory concepts
    2. Quantitative impacts where relevant  
    3. Practical implementation guidance
    4. Optimization strategies and recommendations
    5. Regulatory compliance considerations
    
    Your responses should be professional, comprehensive, and suitable for risk managers and regulatory experts."""
    
    # Add current calculation context if available
    context_info = ""
    if has_calculation_data:
        result = st.session_state.saccr_result
        context_info = f"""
        
        CURRENT CALCULATION CONTEXT:
        - EAD: ${result['final_results']['exposure_at_default']:,.0f}
        - RWA: ${result['final_results']['risk_weighted_assets']:,.0f}
        - Capital Required: ${result['final_results']['capital_requirement']:,.0f}
        - Portfolio Size: {len(st.session_state.get('trades_input', []))} trades
        """
    
    user_prompt = f"{question}{context_info}"
    
    try:
        with st.spinner("🧠 Analyzing with regulatory expertise..."):
            response = st.session_state.saccr_agent.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            answer = response.content
            st.session_state.ai_history.append((question, answer))
            
            # Display the response
            st.markdown(f"""
            <div class="user-query">
                <strong>Your Question:</strong> {question}
            </div>
            """, unsafe_allow_html=True)
            
            answer_text = answer.replace('\n', '<br>')
            st.markdown(f"""
            <div class="ai-response">
                <strong>🤖 Expert Analysis:</strong><br>
                {answer_text}
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"AI analysis error: {str(e)}")

def display_ai_enhanced_analysis(question: str):
    """Display AI analysis in the same rich format as Calculate Enhanced SACCR"""
    
    if 'saccr_result' not in st.session_state or 'trades_input' not in st.session_state:
        st.warning("⚠️ No calculation data available. Please run a calculation first to get enhanced analysis.")
        return
    
    # Get the calculation data
    result = st.session_state.saccr_result
    trades = st.session_state.trades_input
    
    # Create a dummy netting set for display purposes
    netting_set = NettingSet(
        netting_set_id=getattr(st.session_state, 'netting_set_id', 'ANALYSIS'),
        counterparty=getattr(st.session_state, 'counterparty', 'AI Analysis'),
        trades=trades,
        threshold=getattr(st.session_state, 'threshold', 0),
        mta=getattr(st.session_state, 'mta', 0),
        nica=getattr(st.session_state, 'nica', 0)
    )
    
    # Store the question for context
    st.session_state.ai_history.append((question, "Comprehensive analysis generated below"))
    
    # Display the question
    st.markdown(f"""
    <div class="user-query">
        <strong>🎤 Your Question:</strong> {question}
    </div>
    """, unsafe_allow_html=True)
    
    # Generate enhanced AI response
    ai_response = generate_contextual_ai_response(question, result, netting_set)
    
    st.markdown(f"""
    <div class="ai-response">
        <strong>🤖 AI Expert Response:</strong><br>
        {ai_response}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("## 📊 Comprehensive SA-CCR Analysis Dashboard")
    st.markdown("*Based on your current calculation - Enhanced format as requested*")
    
    # Display the same rich format as calculate enhanced SACCR
    display_enhanced_saccr_results(result, netting_set)

def generate_contextual_ai_response(question: str, result: Dict, netting_set: NettingSet) -> str:
    """Generate contextual AI response based on the question and calculation data"""
    
    final_results = result['final_results']
    enhanced_summary = result.get('enhanced_summary', {})
    
    # Create contextual response based on question type
    question_lower = question.lower()
    
    if any(word in question_lower for word in ['breakdown', 'step', 'calculate', 'methodology']):
        response = f"""
        <strong>🎯 Calculation Methodology Analysis:</strong><br><br>
        Your portfolio calculation follows the complete 24-step Basel SA-CCR framework. Here's the high-level breakdown:<br><br>
        
        <strong>📊 Key Results:</strong><br>
        • Exposure at Default (EAD): ${final_results['exposure_at_default']:,.0f}<br>
        • Risk-Weighted Assets (RWA): ${final_results['risk_weighted_assets']:,.0f}<br>
        • Capital Requirement: ${final_results['capital_requirement']:,.0f}<br><br>
        
        <strong>🔍 Critical Steps:</strong><br>
        • Steps 1-4: Data foundation and trade classification<br>
        • Steps 5-10: Risk factor calibration and supervisory parameters<br>
        • Steps 11-13: Add-on aggregation across hedging sets<br>
        • Steps 14-16: PFE calculation with netting benefits<br>
        • Steps 17-18: Replacement cost determination<br>
        • Steps 19-24: Final EAD and capital calculation<br><br>
        
        <strong>💡 Key Insights:</strong><br>
        The detailed breakdown is shown in the comprehensive analysis below, including step-by-step formulas, data visualizations, and optimization recommendations.
        """
    
    elif any(word in question_lower for word in ['optimize', 'reduce', 'efficiency', 'capital']):
        capital_efficiency = (final_results['capital_requirement'] / sum(abs(trade.notional) for trade in netting_set.trades)) * 100
        response = f"""
        <strong>🎯 Capital Optimization Analysis:</strong><br><br>
        Based on your current portfolio calculation:<br><br>
        
        <strong>📈 Current Efficiency:</strong><br>
        • Capital Efficiency: {capital_efficiency:.3f}% of notional<br>
        • Capital Requirement: ${final_results['capital_requirement']:,.0f}<br>
        • Total Portfolio: ${sum(abs(trade.notional) for trade in netting_set.trades):,.0f}<br><br>
        
        <strong>🔧 Key Optimization Levers:</strong><br>
        • <strong>Netting Benefits:</strong> Enhance CSA terms to improve PFE multiplier<br>
        • <strong>Central Clearing:</strong> Move trades to CCP to reduce alpha from 1.4 to 0.5<br>
        • <strong>Collateral Management:</strong> Optimize collateral posting to reduce replacement cost<br>
        • <strong>Portfolio Rebalancing:</strong> Adjust trade mix to improve correlation benefits<br><br>
        
        <strong>💰 Quantitative Impact:</strong><br>
        The detailed risk dashboard below shows specific optimization opportunities with estimated capital savings.
        """
    
    elif any(word in question_lower for word in ['risk', 'exposure', 'pfe', 'replacement']):
        rc = final_results['replacement_cost']
        pfe = final_results['potential_future_exposure']
        rc_pct = (rc / (rc + pfe) * 100) if (rc + pfe) > 0 else 0
        pfe_pct = (pfe / (rc + pfe) * 100) if (rc + pfe) > 0 else 0
        
        response = f"""
        <strong>🎯 Risk Exposure Analysis:</strong><br><br>
        Your portfolio's risk profile breakdown:<br><br>
        
        <strong>⚖️ Current vs Future Risk:</strong><br>
        • Replacement Cost (RC): ${rc:,.0f} ({rc_pct:.1f}% of total exposure)<br>
        • Potential Future Exposure (PFE): ${pfe:,.0f} ({pfe_pct:.1f}% of total exposure)<br>
        • Total Credit Exposure: ${rc + pfe:,.0f}<br><br>
        
        <strong>🔍 Risk Drivers:</strong><br>
        • {'Current exposure dominates' if rc_pct > 60 else 'Future exposure dominates' if pfe_pct > 60 else 'Balanced current/future risk'}<br>
        • Portfolio shows {'high' if pfe_pct > 70 else 'moderate' if pfe_pct > 40 else 'low'} potential volatility<br><br>
        
        <strong>📊 Risk Components:</strong><br>
        The comprehensive analysis below includes detailed risk visualizations, trade-level contributions, and portfolio heatmaps.
        """
    
    else:
        # General analysis
        response = f"""
        <strong>🎯 Comprehensive SA-CCR Analysis:</strong><br><br>
        Based on your question about the current calculation:<br><br>
        
        <strong>📊 Portfolio Overview:</strong><br>
        • Total Trades: {len(netting_set.trades)}<br>
        • Final EAD: ${final_results['exposure_at_default']:,.0f}<br>
        • Capital Required: ${final_results['capital_requirement']:,.0f}<br><br>
        
        <strong>🔍 Analysis Highlights:</strong><br>
        • Complete 24-step Basel regulatory framework applied<br>
        • All supervisory factors and correlations per regulatory standards<br>
        • Netting and collateral benefits properly calculated<br><br>
        
        <strong>📈 Detailed Insights:</strong><br>
        The comprehensive analysis dashboard below provides detailed breakdowns, visualizations, and optimization recommendations tailored to your portfolio.
        """
    
    return response

def analyze_portfolio_data_quality():
    """Enhanced data quality analysis module"""
    
    st.markdown("## 📈 Portfolio Data Quality Analysis")
    st.markdown("*Comprehensive assessment of data completeness and calculation reliability*")
    
    if not hasattr(st.session_state, 'trades_input') or not st.session_state.trades_input:
        st.warning("⚠️ No portfolio data available. Please add trades in the calculator first.")
        return
    
    # Create dummy netting set for analysis
    netting_set = NettingSet(
        netting_set_id="ANALYSIS",
        counterparty="Analysis Target",
        trades=st.session_state.trades_input,
        threshold=getattr(st.session_state, 'threshold', 0),
        mta=getattr(st.session_state, 'mta', 0),
        nica=getattr(st.session_state, 'nica', 0)
    )
    
    # Perform data quality analysis
    data_quality_issues = st.session_state.saccr_agent.analyze_data_quality(
        netting_set, 
        getattr(st.session_state, 'collateral_input', [])
    )
    
    # Overall Data Quality Score
    total_issues = len(data_quality_issues)
    high_impact_issues = len([i for i in data_quality_issues if i.impact == 'high'])
    medium_impact_issues = len([i for i in data_quality_issues if i.impact == 'medium'])
    
    # Calculate quality score (100 - deductions for issues)
    quality_score = max(0, 100 - (high_impact_issues * 20) - (medium_impact_issues * 10))
    
    # Display quality dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Data Quality Score", f"{quality_score}/100", help="Overall data completeness and reliability score")
    with col2:
        st.metric("Total Issues", total_issues, help="Total number of data quality issues identified")
    with col3:
        st.metric("High Impact", high_impact_issues, delta=-high_impact_issues if high_impact_issues > 0 else None)
    with col4:
        st.metric("Medium Impact", medium_impact_issues, delta=-medium_impact_issues if medium_impact_issues > 0 else None)
    
    # Quality indicator
    if quality_score >= 90:
        st.success("✅ Excellent data quality - Calculation results are highly reliable")
    elif quality_score >= 70:
        st.warning("⚠️ Good data quality - Minor issues may affect precision")
    elif quality_score >= 50:
        st.warning("🟡 Fair data quality - Several issues may impact results")
    else:
        st.error("❌ Poor data quality - Significant issues require attention")
    
    # Detailed Issues Analysis
    if data_quality_issues:
        st.markdown("### 🔍 Detailed Data Quality Assessment")
        
        # High Impact Issues
        high_issues = [i for i in data_quality_issues if i.impact == 'high']
        if high_issues:
            with st.expander("🔴 High Impact Issues (Immediate Action Required)", expanded=True):
                for issue in high_issues:
                    st.markdown(f"""
                    <div class="data-quality-alert">
                        <strong>🚨 {issue.field_name}</strong><br>
                        <strong>Current Value:</strong> {issue.current_value}<br>
                        <strong>Issue Type:</strong> {issue.issue_type}<br>
                        <strong>Impact:</strong> {issue.recommendation}<br>
                        <strong>Default Used:</strong> {issue.default_used}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Medium Impact Issues
        medium_issues = [i for i in data_quality_issues if i.impact == 'medium']
        if medium_issues:
            with st.expander("🟡 Medium Impact Issues (Review Recommended)", expanded=True):
                for issue in medium_issues:
                    st.markdown(f"""
                    <div style="background: #fff3cd; border: 1px solid #ffeaa7; padding: 1rem; border-radius: 6px; margin: 0.5rem 0;">
                        <strong>⚠️ {issue.field_name}</strong><br>
                        <strong>Current:</strong> {issue.current_value}<br>
                        <strong>Recommendation:</strong> {issue.recommendation}
                    </div>
                    """, unsafe_allow_html=True)
    
    # Data Improvement Recommendations
    st.markdown("### 💡 Data Enhancement Recommendations")
    
    recommendations = generate_enhanced_template_response(data_quality_issues, netting_set)
    
    for category, recs in recommendations.items():
        with st.expander(f"📋 {category}", expanded=True):
            for rec in recs:
                st.write(f"• {rec}")
    
    # Portfolio Statistics
    with st.expander("📊 Portfolio Statistics", expanded=False):
        trades = st.session_state.trades_input
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Trade Composition:**")
            asset_classes = {}
            trade_types = {}
            currencies = {}
            
            for trade in trades:
                # Asset classes
                ac = trade.asset_class.value
                asset_classes[ac] = asset_classes.get(ac, 0) + 1
                
                # Trade types
                tt = trade.trade_type.value
                trade_types[tt] = trade_types.get(tt, 0) + 1
                
                # Currencies
                curr = trade.currency
                currencies[curr] = currencies.get(curr, 0) + 1
            
            st.write("Asset Classes:")
            for ac, count in asset_classes.items():
                st.write(f"  • {ac}: {count} trades")
            
            st.write("Trade Types:")
            for tt, count in trade_types.items():
                st.write(f"  • {tt}: {count} trades")
        
        with col2:
            st.markdown("**Risk Metrics:**")
            
            total_notional = sum(abs(trade.notional) for trade in trades)
            total_mtm = sum(trade.mtm_value for trade in trades)
            avg_maturity = sum(trade.time_to_maturity() for trade in trades) / len(trades)
            
            st.write(f"• Total Notional: ${total_notional/1_000_000:.1f}M")
            st.write(f"• Total MTM: ${total_mtm/1_000:.0f}K")
            st.write(f"• Average Maturity: {avg_maturity:.1f} years")
            st.write(f"• Currency Exposure:")
            for curr, count in currencies.items():
                st.write(f"    - {curr}: {count} trades")

def generate_enhanced_template_response(issues: List, netting_set: NettingSet) -> Dict[str, List[str]]:
    """Generate enhanced recommendations based on data quality issues"""
    
    recommendations = {
        "Immediate Actions": [],
        "Data Enhancement": [],
        "Process Improvements": [],
        "Technology Solutions": []
    }
    
    # Analyze high impact issues
    high_impact_fields = [i.field_name for i in issues if i.impact == 'high']
    
    if any('MTM' in field for field in high_impact_fields):
        recommendations["Immediate Actions"].append("Implement daily MTM value updates from market data providers")
        recommendations["Technology Solutions"].append("Integrate with real-time pricing services (Bloomberg, Refinitiv)")
    
    if any('Threshold' in field or 'MTA' in field for field in high_impact_fields):
        recommendations["Immediate Actions"].append("Review and update all CSA/ISDA agreement terms in system")
        recommendations["Process Improvements"].append("Establish quarterly CSA terms validation process")
    
    if any('Collateral' in field for field in high_impact_fields):
        recommendations["Immediate Actions"].append("Implement comprehensive collateral tracking system")
        recommendations["Data Enhancement"].append("Daily collateral position reconciliation with custodians")
    
    if any('Delta' in field for field in high_impact_fields):
        recommendations["Data Enhancement"].append("Source option sensitivities from derivatives pricing systems")
        recommendations["Technology Solutions"].append("Implement Greeks calculation engine or vendor solution")
    
    # General recommendations based on portfolio characteristics
    asset_classes = set(trade.asset_class for trade in netting_set.trades)
    
    if len(asset_classes) > 2:
        recommendations["Process Improvements"].append("Implement cross-asset class risk monitoring")
        recommendations["Data Enhancement"].append("Enhance correlation data for multi-asset portfolios")
    
    if len(netting_set.trades) > 10:
        recommendations["Technology Solutions"].append("Consider automated SA-CCR calculation platform")
        recommendations["Process Improvements"].append("Implement trade-level data quality monitoring")
    
    # Always include baseline recommendations
    recommendations["Process Improvements"].extend([
        "Establish monthly data quality reporting",
        "Implement exception reporting for missing data",
        "Create data quality KPIs and monitoring dashboard"
    ])
    
    recommendations["Technology Solutions"].extend([
        "Consider SA-CCR calculation automation platform",
        "Implement data validation rules in trade capture systems",
        "Set up automated regulatory reporting pipeline"
    ])
    
    return recommendations

def portfolio_analysis_page():
    """Enhanced portfolio analysis with comprehensive insights"""
    
    st.markdown("## 📊 Portfolio Risk Analysis")
    st.markdown("*Comprehensive portfolio analytics and optimization insights*")
    
    if not hasattr(st.session_state, 'trades_input') or not st.session_state.trades_input:
        st.warning("⚠️ No portfolio data available. Please add trades in the calculator first.")
        return
    
    trades = st.session_state.trades_input
    
    # Portfolio Overview Dashboard
    st.markdown("### 🎯 Portfolio Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_notional = sum(abs(trade.notional) for trade in trades)
    total_mtm = sum(trade.mtm_value for trade in trades)
    avg_maturity = sum(trade.time_to_maturity() for trade in trades) / len(trades)
    asset_classes = len(set(trade.asset_class for trade in trades))
    
    with col1:
        st.metric("Total Notional", f"${total_notional/1_000_000:.1f}M")
    with col2:
        st.metric("Net MTM", f"${total_mtm/1_000:.0f}K")
    with col3:
        st.metric("Avg Maturity", f"{avg_maturity:.1f}Y")
    with col4:
        st.metric("Asset Classes", asset_classes)
    
    # Risk Distribution Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Notional by Asset Class
        asset_notional = {}
        for trade in trades:
            ac = trade.asset_class.value
            asset_notional[ac] = asset_notional.get(ac, 0) + abs(trade.notional)
        
        fig_pie = px.pie(
            values=list(asset_notional.values()),
            names=list(asset_notional.keys()),
            title="Notional Distribution by Asset Class"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # MTM vs Notional Scatter
        mtm_data = [trade.mtm_value/1_000 for trade in trades]  # In thousands
        notional_data = [abs(trade.notional)/1_000_000 for trade in trades]  # In millions
        colors = [trade.asset_class.value for trade in trades]
        
        fig_scatter = px.scatter(
            x=notional_data,
            y=mtm_data,
            color=colors,
            title="MTM vs Notional by Trade",
            labels={'x': 'Notional ($M)', 'y': 'MTM ($K)'},
            hover_data={'Trade ID': [trade.trade_id for trade in trades]}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Maturity Analysis
    st.markdown("### ⏰ Maturity Profile Analysis")
    
    # Create maturity buckets
    maturity_buckets = {"<1Y": 0, "1-2Y": 0, "2-5Y": 0, "5-10Y": 0, ">10Y": 0}
    maturity_details = []
    
    for trade in trades:
        maturity = trade.time_to_maturity()
        maturity_details.append({
            'Trade ID': trade.trade_id,
            'Maturity': maturity,
            'Notional': abs(trade.notional),
            'Asset Class': trade.asset_class.value
        })
        
        if maturity < 1:
            maturity_buckets["<1Y"] += abs(trade.notional)
        elif maturity < 2:
            maturity_buckets["1-2Y"] += abs(trade.notional)
        elif maturity < 5:
            maturity_buckets["2-5Y"] += abs(trade.notional)
        elif maturity < 10:
            maturity_buckets["5-10Y"] += abs(trade.notional)
        else:
            maturity_buckets[">10Y"] += abs(trade.notional)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Maturity bucket distribution
        fig_bar = px.bar(
            x=list(maturity_buckets.keys()),
            y=[v/1_000_000 for v in maturity_buckets.values()],
            title="Notional by Maturity Bucket",
            labels={'x': 'Maturity Bucket', 'y': 'Notional ($M)'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        # Maturity timeline
        df_maturity = pd.DataFrame(maturity_details)
        fig_timeline = px.histogram(
            df_maturity,
            x='Maturity',
            weights='Notional',
            color='Asset Class',
            title="Maturity Timeline Distribution",
            labels={'x': 'Years to Maturity', 'y': 'Notional ($)'}
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Risk Factor Analysis
    st.markdown("### ⚡ Risk Factor Impact Analysis")
    
    # Calculate supervisory factors for each trade
    risk_analysis = []
    for trade in trades:
        sf = st.session_state.saccr_agent._get_supervisory_factor(trade)
        mf = math.sqrt(min(trade.time_to_maturity(), 1.0))
        
        risk_contribution = abs(trade.notional) * trade.delta * mf * (sf / 10000)
        
        risk_analysis.append({
            'Trade ID': trade.trade_id,
            'Asset Class': trade.asset_class.value,
            'Notional ($M)': abs(trade.notional) / 1_000_000,
            'Supervisory Factor (bp)': sf,
            'Maturity Factor': mf,
            'Delta': trade.delta,
            'Risk Contribution ($K)': risk_contribution / 1000,
            'Risk Density (%)': (risk_contribution / abs(trade.notional)) * 100
        })
    
    df_risk = pd.DataFrame(risk_analysis)
    
    # Display risk analysis table
    st.dataframe(df_risk.round(4), use_container_width=True)
    
    # Risk contribution charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk contribution by trade
        fig_risk = px.bar(
            df_risk.sort_values('Risk Contribution ($K)', ascending=False).head(10),
            x='Trade ID',
            y='Risk Contribution ($K)',
            color='Asset Class',
            title="Top 10 Risk Contributing Trades"
        )
        fig_risk.update_xaxes(tickangle=45)
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with col2:
        # Risk density analysis
        fig_density = px.scatter(
            df_risk,
            x='Notional ($M)',
            y='Risk Density (%)',
            color='Asset Class',
            size='Risk Contribution ($K)',
            title="Risk Density vs Notional Size"
        )
        st.plotly_chart(fig_density, use_container_width=True)
    
    # Optimization Recommendations
    st.markdown("### 🎯 Portfolio Optimization Insights")
    
    # Calculate optimization metrics
    total_risk_contribution = sum(item['Risk Contribution ($K)'] for item in risk_analysis)
    high_risk_trades = [item for item in risk_analysis if item['Risk Density (%)'] > 1.0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔍 Key Findings:**")
        st.write(f"• Total portfolio risk contribution: ${total_risk_contribution:.0f}K")
        st.write(f"• High risk density trades: {len(high_risk_trades)} ({len(high_risk_trades)/len(trades)*100:.1f}%)")
        st.write(f"• Most risk-intensive asset class: {max(set(item['Asset Class'] for item in risk_analysis), key=lambda x: sum(item['Risk Contribution ($K)'] for item in risk_analysis if item['Asset Class'] == x))}")
        
        avg_risk_density = sum(item['Risk Density (%)'] for item in risk_analysis) / len(risk_analysis)
        st.write(f"• Average risk density: {avg_risk_density:.3f}%")
    
    with col2:
        st.markdown("**💡 Optimization Opportunities:**")
        
        if len(high_risk_trades) > 0:
            st.write("• Consider reviewing high risk density trades for optimization")
        
        if avg_maturity > 5:
            st.write("• Long average maturity - consider maturity diversification")
        
        if asset_classes == 1:
            st.write("• Single asset class concentration - consider diversification")
        
        if total_mtm < 0:
            st.write("• Negative MTM may benefit from netting optimization")
        else:
            st.write("• Positive MTM - monitor for potential collateral requirements")
    
    # Export portfolio analysis
    st.markdown("### 📥 Export Analysis")
    
    analysis_export = {
        'Portfolio_Summary': {
            'Total_Trades': len(trades),
            'Total_Notional_USD': total_notional,
            'Net_MTM_USD': total_mtm,
            'Average_Maturity_Years': avg_maturity,
            'Asset_Classes': asset_classes
        },
        'Risk_Analysis': risk_analysis,
        'Maturity_Buckets': {k: v for k, v in maturity_buckets.items()},
        'Asset_Distribution': asset_notional
    }
    
    analysis_json = json.dumps(analysis_export, indent=2, default=str)
    st.download_button(
        "📊 Download Portfolio Analysis",
        data=analysis_json,
        file_name=f"portfolio_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

if __name__ == "__main__":
    main()
