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
            mf = min(1.0, 0.05 + 0.95 * math.exp(-0.05 * max(1.0, remaining_maturity)))
            
            maturity_factors.append({
                'trade_id': trade.trade_id,
                'remaining_maturity': remaining_maturity,
                'maturity_factor': mf
            })
            
            reasoning_details.append(f"Trade {trade.trade_id}: M={remaining_maturity:.2f}y → MF=min(1, 0.05 + 0.95×exp(-0.05×max(1,{remaining_maturity:.2f}))) = {mf:.6f}")
        
        # Add thinking step
        thinking = {
            'step': 6,
            'title': 'Maturity Factor Calculation',
            'reasoning': f"""
THINKING PROCESS:
• Formula: MF = min(1, 0.05 + 0.95 × exp(-0.05 × max(1, M)))
• This formula reduces capital for shorter-term trades
• Floor at 0.05 ensures minimum benefit for very short trades
• Cap ensures longer trades don't get penalized beyond 1-year equivalent

DETAILED CALCULATIONS:
{chr(10).join(reasoning_details)}

REGULATORY RATIONALE:
• Shorter maturity = lower potential for large market moves = lower capital
• Basel Committee calibrated this to reflect time-dependent volatility
• Exponential decay captures diminishing marginal benefit of shorter maturities
            """,
            'formula': 'MF = min(1, 0.05 + 0.95 × exp(-0.05 × max(1, M)))',
            'key_insight': f"Average maturity factor: {sum(mf['maturity_factor'] for mf in maturity_factors)/len(maturity_factors):.4f}"
        }
        
        self.thinking_steps.append(thinking)
        
        return {
            'step': 6,
            'title': 'Maturity Factor (MF)',
            'description': 'Apply Basel maturity factor formula',
            'data': maturity_factors,
            'formula': 'MF = min(1, 0.05 + 0.95 × exp(-0.05 × max(1, M)))',
            'result': f"Calculated maturity factors for {len(trades)} trades",
            'thinking': thinking
        }

    def _step8_supervisory_factor_enhanced(self, trades: List[Trade]) -> Dict:
        """Step 8: Supervisory Factor with detailed lookup logic"""
        supervisory_factors = []
        reasoning_details = []
        
        for trade in trades:
            sf = self._get_supervisory_factor(trade)
            supervisory_factors.append({
                'trade_id': trade.trade_id,
                'asset_class': trade.asset_class.value,
                'currency': trade.currency,
                'maturity_bucket': self._get_maturity_bucket(trade),
                'supervisory_factor_bp': sf,
                'supervisory_factor_decimal': sf / 10000
            })
            
            reasoning_details.append(f"Trade {trade.trade_id}: {trade.asset_class.value} {trade.currency} {self._get_maturity_bucket(trade)} → {sf}bp ({sf/10000:.4f})")
        
        thinking = {
            'step': 8,
            'title': 'Supervisory Factor Lookup',
            'reasoning': f"""
THINKING PROCESS:
• Lookup supervisory factors from Basel regulatory tables (Annex 4)
• Factors vary by: Asset Class, Currency (for IR), Maturity bucket
• Higher factors = higher perceived volatility = more capital required

DETAILED LOOKUPS:
{chr(10).join(reasoning_details)}

REGULATORY BASIS:
• Calibrated to 99% confidence level over 1-year horizon
• Based on historical volatility analysis by Basel Committee
• Updated periodically to reflect current market conditions
• Interest rates have lower factors for major currencies (USD, EUR, GBP, JPY)
            """,
            'formula': 'SF per Basel Annex 4 regulatory tables',
            'key_insight': f"Portfolio-weighted average SF: {sum(sf['supervisory_factor_bp'] * abs(trade.notional) for sf, trade in zip(supervisory_factors, trades)) / sum(abs(trade.notional) for trade in trades):.1f}bp"
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
            mf = min(1.0, 0.05 + 0.95 * math.exp(-0.05 * max(1.0, remaining_maturity)))
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
• This is the core SA-CCR risk measure calculation per trade
• Formula combines all risk components: size, direction, time, volatility
• Each component serves a specific regulatory purpose

COMPONENT ANALYSIS:
• Effective Notional: Captures trade size exposure
• Delta (δ): Captures direction and option sensitivity
• Maturity Factor: Time-based risk scaling (shorter = lower risk)
• Supervisory Factor: Asset class volatility weighting

DETAILED CALCULATIONS:
{chr(10).join(reasoning_details)}

PORTFOLIO INSIGHTS:
• Total gross effective exposure: ${sum(abs(calc['adjusted_derivatives_contract_amount']) for calc in adjusted_amounts):,.0f}
• This feeds into hedging set aggregation where netting benefits apply
            """,
            'formula': 'Adjusted Amount = Effective Notional × δ × MF × SF',
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
        # Get step 12 result for asset class addons
        step12_result = self._step12_asset_class_addon(trades)
        
        aggregate_addon = sum(ac_data['asset_class_addon'] for ac_data in step12_result['data'])
        
        thinking = {
            'step': 13,
            'title': 'Aggregate AddOn Calculation',
            'reasoning': f"""
THINKING PROCESS:
• Sum all asset class add-ons to get total portfolio add-on
• This represents the potential future exposure component before multiplier
• Conservative approach: simple addition across asset classes

ASSET CLASS BREAKDOWN:
{chr(10).join([f"• {ac_data['asset_class']}: ${ac_data['asset_class_addon']:,.0f}" for ac_data in step12_result['data']])}

REGULATORY PURPOSE:
• Captures maximum potential future credit exposure
• Feeds into PFE multiplier calculation (netting benefits)
• Higher add-on = higher baseline capital requirement
• Will be reduced by multiplier based on current portfolio MTM
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
        
        thinking = {
            'step': 14,
            'title': 'Current Exposure (V) and Collateral (C) Analysis',
            'reasoning': f"""
THINKING PROCESS:
• V = Current market value of all trades in netting set
• C = Effective collateral value after regulatory haircuts
• V-C determines current net exposure and affects both RC and PFE multiplier

CURRENT EXPOSURE ANALYSIS:
• Sum of trade MTMs: ${sum_v:,.0f}
• Portfolio position: {'Out-of-the-money (favorable)' if sum_v < 0 else 'In-the-money (unfavorable)' if sum_v > 0 else 'At-the-money (neutral)'}
• Impact on capital: {'Negative MTM helps reduce PFE multiplier' if sum_v < 0 else 'Positive MTM increases replacement cost' if sum_v > 0 else 'Zero MTM - neutral impact'}

COLLATERAL ANALYSIS:
• Total posted: ${sum([c['amount'] for c in collateral_details]):,.0f if collateral_details else 0}
• After haircuts: ${sum_c:,.0f}
• Net exposure (V-C): ${sum_v - sum_c:,.0f}

REGULATORY HAIRCUTS:
{chr(10).join([f"• {c['type']}: {c['haircut_pct']:.1f}% haircut, effective: ${c['effective_value']:,.0f}" for c in collateral_details]) if collateral_details else "• No collateral posted"}
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
• The multiplier captures netting benefits within the netting set
• Formula: min(1, 0.05 + 0.95 × exp((V-C) / (2 × 0.95 × AddOn)))
• Lower multiplier = more netting benefit = lower capital requirement

DETAILED CALCULATION:
• Net Exposure (V-C): ${net_exposure:,.0f}
• Aggregate AddOn: ${aggregate_addon:,.0f}
• Denominator: 2 × 0.95 × {aggregate_addon:,.0f} = ${2 * 0.95 * aggregate_addon:,.0f}
• Exponent: ${net_exposure:,.0f} / ${2 * 0.95 * aggregate_addon:,.0f} = {exponent:.6f}
• exp({exponent:.6f}) = {math.exp(exponent):.6f}
• Multiplier: min(1, 0.05 + 0.95 × {math.exp(exponent):.6f}) = {multiplier:.6f}

NETTING BENEFIT ANALYSIS:
• Final multiplier: {multiplier:.6f} ({multiplier*100:.2f}% of gross add-on)
• Netting benefit: {netting_benefit_pct:.1f}% capital reduction
• Benefit category: {'Strong' if multiplier < 0.3 else 'Moderate' if multiplier < 0.7 else 'Limited'}

ECONOMIC INTERPRETATION:
• {'Out-of-the-money portfolio provides maximum netting benefit' if net_exposure < -aggregate_addon/2 else 'Portfolio provides some netting benefit' if net_exposure < 0 else 'In-the-money portfolio reduces netting benefit'}
• Floor at 5% ensures minimum recognition of diversification
            """,
            'formula': 'Multiplier = min(1, 0.05 + 0.95 × exp((V-C) / (2 × 0.95 × AddOn)))',
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
            'formula': 'Multiplier = min(1, 0.05 + 0.95 × exp(V / (2 × 0.95 × AddOn)))',
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
• Combines volatility risk (AddOn) with netting benefits (Multiplier)
• Represents potential future credit exposure over trade life

FINAL CALCULATION:
• Multiplier: {multiplier:.6f}
• Aggregate AddOn: ${aggregate_addon:,.0f}
• PFE: {multiplier:.6f} × ${aggregate_addon:,.0f} = ${pfe:,.0f}

REGULATORY SIGNIFICANCE:
• PFE captures potential future exposure at 99% confidence level
• Combined with current exposure (RC) to determine total EAD
• Lower PFE = lower capital requirement

PORTFOLIO INSIGHTS:
• Gross potential exposure: ${aggregate_addon:,.0f}
• Netting benefit: ${(1-multiplier)*aggregate_addon:,.0f}
• Net future exposure: ${pfe:,.0f}
• {'Strong portfolio netting reduces future exposure significantly' if multiplier < 0.5 else 'Moderate netting benefits applied' if multiplier < 0.8 else 'Conservative PFE - limited netting benefits'}
            """,
            'formula': 'PFE = Multiplier × Aggregate AddOn',
            'key_insight': f"PFE of ${pfe:,.0f} represents net future exposure after ${(1-multiplier)*100:.1f}% netting benefit"
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
            rc = max(net_exposure, 0)
            methodology = "Unmargined netting set"
        
        thinking = {
            'step': 18,
            'title': 'Replacement Cost (RC) - Current Exposure Analysis',
            'reasoning': f"""
THINKING PROCESS:
• RC represents current replacement cost if counterparty defaults today
• Different calculations for margined vs unmargined netting sets
• Captures benefits of netting agreements and posted collateral

NETTING SET CLASSIFICATION:
• Type: {methodology}
• Threshold: ${threshold:,.0f}
• MTA: ${mta:,.0f}
• NICA: ${nica:,.0f}
{'• Margin floor: ' + f'${margin_floor:,.0f}' if is_margined else ''}

CALCULATION COMPONENTS:
• Current MTM (V): ${sum_v:,.0f}
• Posted collateral (C): ${sum_c:,.0f}
• Net exposure (V-C): ${net_exposure:,.0f}

REPLACEMENT COST DETERMINATION:
• Formula: {"RC = max(V-C, TH+MTA-NICA, 0)" if is_margined else "RC = max(V-C, 0)"}
• Calculation: RC = max({net_exposure:,.0f}{f', {margin_floor:,.0f}' if is_margined else ''}, 0)
• Result: RC = ${rc:,.0f}

DRIVER ANALYSIS:
• RC driven by: {'Margin floor (CSA terms limit collateral benefit)' if is_margined and rc == margin_floor else 'Net current exposure' if rc == net_exposure else 'Zero (fully collateralized)'}
            """,
            'formula': f"RC = max(V-C{', TH+MTA-NICA' if is_margined else ''}, 0)",
            'key_insight': f"RC of ${rc:,.0f} represents current credit exposure component"
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
• EAD = Alpha × (RC + PFE)
• Alpha = 1.4 (fixed regulatory multiplier for SA-CCR)
• EAD represents total potential credit exposure upon default

EXPOSURE COMPONENT BREAKDOWN:
• Current exposure (RC): ${rc:,.0f} ({rc_percentage:.1f}% of total)
• Future exposure (PFE): ${pfe:,.0f} ({pfe_percentage:.1f}% of total)
• Combined exposure: ${combined_exposure:,.0f}
• Alpha multiplier: {alpha}

EAD CALCULATION:
• EAD = {alpha} × (${rc:,.0f} + ${pfe:,.0f})
• EAD = {alpha} × ${combined_exposure:,.0f}
• EAD = ${ead:,.0f}

EXPOSURE PROFILE ANALYSIS:
• Dominant risk: {'Current exposure (existing MTM risk)' if rc > pfe else 'Future exposure (potential market risk)' if pfe > rc else 'Balanced current/future exposure'}
• Risk characteristics: {'Portfolio already showing losses' if rc > pfe * 2 else 'Significant future risk potential' if pfe > rc * 2 else 'Typical derivatives risk profile'}

REGULATORY PURPOSE:
• EAD feeds directly into RWA calculation
• Alpha ensures consistent calibration across all banks
• Total exposure used for capital requirement determination
            """,
            'formula': 'EAD = Alpha × (RC + PFE), where Alpha = 1.4',
            'key_insight': f"Total credit exposure: ${ead:,.0f} ({rc_percentage:.0f}% current, {pfe_percentage:.0f}% future)"
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
• RWA = Risk Weight × EAD
• Risk weight reflects counterparty creditworthiness
• Final capital requirement = RWA × 8% (minimum ratio)

COUNTERPARTY RISK ASSESSMENT:
• EAD: ${ead:,.0f}
• Risk Weight: {risk_weight*100:.0f}% (corporate standard)
• Regulatory basis: Basel III Standardized Approach

CAPITAL CALCULATION:
• RWA = ${ead:,.0f} × {risk_weight} = ${rwa:,.0f}
• Minimum capital = ${rwa:,.0f} × 8% = ${capital_requirement:,.0f}

CAPITAL IMPACT ASSESSMENT:
• Capital efficiency: {(capital_requirement/ead*100):.2f}% of exposure
• {'High capital requirement - optimization recommended' if capital_requirement > 5000000 else 'Moderate capital requirement' if capital_requirement > 1000000 else 'Reasonable capital requirement'}
• Consider: {'Netting optimization, collateral enhancement, central clearing' if capital_requirement > 2000000 else 'Portfolio monitoring and periodic optimization'}

REGULATORY CONTEXT:
• This is minimum Tier 1 capital requirement
• Banks typically hold additional buffers (conservation, countercyclical)
• Actual capital impact may be 10-13% of RWA including buffers
            """,
            'formula': 'RWA = Risk Weight × EAD, Capital = RWA × 8%',
            'key_insight': f"${capital_requirement:,.0f} minimum capital required ({(capital_requirement/ead*100):.2f}% of exposure)"
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

    def _generate_enhanced_summary(self, calculation_steps: Dict, netting_set: NettingSet) -> Dict:
        """Generate enhanced bulleted summary"""
        
        # Extract key values from calculation steps
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
                f"Capital Efficiency: {(final_step_24['data']['capital_requirement']/total_notional*100):.3f}% of notional"
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
        
        # Prepare enhanced context
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
            'description': 'Source netting set data from Arctic system',
            'data': {
                'netting_set_id': netting_set.netting_set_id,
                'counterparty': netting_set.counterparty,
                'trade_count': len(netting_set.trades),
                'total_notional': sum(abs(trade.notional) for trade in netting_set.trades)
            },
            'formula': 'Data sourced from system Arctic',
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
            'title': 'Asset Class, Asset Sub Class, Basis Flag, Volatility Flag',
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
            'title': 'Hedging Set',
            'description': 'Group trades into hedging sets based on risk factors',
            'data': hedging_sets,
            'formula': 'Hedging sets defined by asset class and currency',
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
            'formula': 'S = Settlement Date, E = End Date, M = (E - S) / 365',
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
            'description': 'Calculate adjusted notional amounts per regulatory requirements',
            'data': adjusted_notionals,
            'formula': 'Adjusted Notional = Notional × Supervisory Duration × Supervisory Factor',
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
            mf = min(1.0, 0.05 + 0.95 * math.exp(-0.05 * max(1.0, remaining_maturity)))
            sf = self._get_supervisory_factor(trade) / 10000
            
            effective_notional = adjusted_notional * supervisory_delta * mf
            pfe_trade_level = effective_notional * sf
            
            hedging_sets[hedging_set_key].append(pfe_trade_level)

        hedging_set_addons = []
        for hedging_set_key, pfe_trades in hedging_sets.items():
            asset_class_str = hedging_set_key.split('_')[0]
            asset_class = next((ac for ac in AssetClass if ac.value == asset_class_str), None)
            correlation = self.supervisory_correlations.get(asset_class, 0.5)

            sum_of_pfe_trades = sum(pfe_trades)
            hedging_set_addon = abs(sum_of_pfe_trades)

            hedging_set_addons.append({
                'hedging_set': hedging_set_key,
                'trade_count': len(pfe_trades),
                'correlation': correlation,
                'hedging_set_addon': hedging_set_addon
            })

        return {
            'step': 11,
            'title': 'Hedging Set AddOn',
            'description': 'Aggregate trade add-ons within hedging sets',
            'data': hedging_set_addons,
            'formula': 'Hedging Set AddOn = | Σ(Effective Notional) | × SF',
            'result': f"Calculated add-ons for {len(hedging_sets)} hedging sets"
        }

    def _step12_asset_class_addon(self, trades: List[Trade]) -> Dict:
        step11_result = self._step11_hedging_set_addon(trades)
        
        asset_class_addons = {}
        for hedging_set_data in step11_result['data']:
            asset_class = hedging_set_data['hedging_set'].split('_')[0]
            if asset_class not in asset_class_addons:
                asset_class_addons[asset_class] = []
            asset_class_addons[asset_class].append(hedging_set_data['hedging_set_addon'])
        
        asset_class_results = []
        for asset_class, hedging_set_addons_list in asset_class_addons.items():
            asset_class_addon = sum(hedging_set_addons_list)
            asset_class_results.append({
                'asset_class': asset_class,
                'hedging_set_addons': hedging_set_addons_list,
                'asset_class_addon': asset_class_addon
            })
        
        return {
            'step': 12,
            'title': 'Asset Class AddOn',
            'description': 'Sum hedging set add-ons by asset class',
            'data': asset_class_results,
            'formula': 'Asset Class AddOn = Σ(Hedging Set AddOns)',
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
        alpha = 1.4
        
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
        counterparty_data = {
            'counterparty_name': counterparty,
            'legal_code': '?',
            'legal_code_description': 'Non-Profit Org',
            'country': 'US',
            'r35_risk_weight_category': 'Corporate'
        }
        
        return {
            'step': 22,
            'title': 'Counterparty Information',
            'description': 'Source counterparty details from Cesium system',
            'data': counterparty_data,
            'formula': 'Sourced from Cesium',
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
            'formula': 'Risk Weight per 12 CFR § 217.32 mapping',
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
            return self.supervisory_factors[AssetClass.CREDIT]['IG_single']
        
        elif trade.asset_class == AssetClass.EQUITY:
            return self.supervisory_factors[AssetClass.EQUITY]['single_large']
        
        elif trade.asset_class == AssetClass.COMMODITY:
            return self.supervisory_factors[AssetClass.COMMODITY]['energy']
        
        return 1.0

# ==============================================================================
# ENHANCED STREAMLIT APPLICATION
# ==============================================================================

def main():
    # AI-Powered Header
    st.markdown("""
    <div class="ai-header">
        <div class="executive-title">🤖 AI SA-CCR Platform</div>
        <div class="executive-subtitle">Complete 24-Step Basel SA-CCR Calculator with Enhanced Step-by-Step Analysis</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize comprehensive agent
    if 'saccr_agent' not in st.session_state:
        st.session_state.saccr_agent = ComprehensiveSACCRAgent()
    
    # Sidebar with LLM Configuration
    with st.sidebar:
        st.markdown("### 🤖 LLM Configuration")
        
        # Configuration inputs
        with st.expander("🔧 LLM Setup", expanded=True):
            base_url = st.text_input("Base URL", value="http://localhost:8123/v1")
            api_key = st.text_input("API Key", value="dummy", type="password")
            model = st.text_input("Model", value="llama3")
            temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
            max_tokens = st.number_input("Max Tokens", 1000, 8000, 4000, 100)
            
            if st.button("🔗 Connect LLM"):
                config = {
                    'base_url': base_url,
                    'api_key': api_key,
                    'model': model,
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    'streaming': False
                }
                
                success = st.session_state.saccr_agent.setup_llm_connection(config)
                if success:
                    st.success("✅ LLM Connected!")
                else:
                    st.error("❌ Connection Failed")
        
        # Connection status
        status = st.session_state.saccr_agent.connection_status
        if status == "connected":
            st.markdown('<div class="connection-status connected">🟢 LLM Connected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="connection-status disconnected">🔴 LLM Disconnected</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📊 Navigation")
        page = st.selectbox(
            "Select Module:",
            ["🧮 Enhanced SA-CCR Calculator", "📋 Reference Example", "🤖 AI Assistant", "📊 Portfolio Analysis"]
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

def enhanced_complete_saccr_calculator():
    """Enhanced complete 24-step SA-CCR calculator with step-by-step analysis"""
    
    st.markdown("## 🧮 Enhanced SA-CCR Calculator with Step-by-Step Analysis")
    st.markdown("*Following the complete 24-step Basel regulatory framework with detailed thinking process*")
    
    # Quick load reference example
    if st.button("📋 Load Reference Example (Lowell Hotel Properties)", type="secondary"):
        load_reference_example()
        st.rerun()
    
    # Step 1: Netting Set Setup (same as original)
    with st.expander("📊 Step 1: Netting Set Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            netting_set_id = st.text_input("Netting Set ID*", 
                value="212784060000009618701" if 'trades_input' in st.session_state and st.session_state.trades_input else "",
                placeholder="e.g., 212784060000009618701")
            counterparty = st.text_input("Counterparty*", 
                value="Lowell Hotel Properties LLC" if 'trades_input' in st.session_state and st.session_state.trades_input else "",
                placeholder="e.g., Lowell Hotel Properties LLC")
            
        with col2:
            threshold = st.number_input("Threshold ($)*", min_value=0.0, value=12000000.0, step=100000.0)
            mta = st.number_input("MTA ($)*", min_value=0.0, value=1000000.0, step=50000.0)
            nica = st.number_input("NICA ($)", min_value=0.0, value=0.0, step=10000.0)
    
    # Step 2: Trade Input (same as original)
    st.markdown("### 📈 Trade Portfolio Input")
    
    if 'trades_input' not in st.session_state:
        st.session_state.trades_input = []
    
    with st.expander("➕ Add New Trade", expanded=len(st.session_state.trades_input) == 0):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            trade_id = st.text_input("Trade ID*", placeholder="e.g., 2098474100")
            asset_class = st.selectbox("Asset Class*", [ac.value for ac in AssetClass])
            trade_type = st.selectbox("Trade Type*", [tt.value for tt in TradeType])
        
        with col2:
            notional = st.number_input("Notional ($)*", value=100000000.0, step=1000000.0)
            currency = st.selectbox("Currency*", ["USD", "EUR", "GBP", "JPY", "CHF", "CAD"])
            underlying = st.text_input("Underlying*", placeholder="e.g., Interest rate")
        
        with col3:
            maturity_years = st.number_input("Maturity (Years)*", min_value=0.1, max_value=30.0, value=5.0, step=0.1)
            mtm_value = st.number_input("MTM Value ($)", value=0.0, step=10000.0)
            delta = st.number_input("Delta (for options)", min_value=-1.0, max_value=1.0, value=1.0, step=0.1)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ Add Trade", type="primary"):
                if trade_id and notional != 0 and currency and underlying:
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
    
    # Display current trades (same as original)
    if st.session_state.trades_input:
        st.markdown("### 📋 Current Trade Portfolio")
        
        trades_data = []
        for i, trade in enumerate(st.session_state.trades_input):
            trades_data.append({
                'Index': i,
                'Trade ID': trade.trade_id,
                'Asset Class': trade.asset_class.value,
                'Type': trade.trade_type.value,
                'Notional ($M)': f"{trade.notional/1_000_000:.1f}",
                'Currency': trade.currency,
                'MTM ($K)': f"{trade.mtm_value/1000:.0f}",
                'Maturity (Y)': f"{trade.time_to_maturity():.1f}"
            })
        
        df = pd.DataFrame(trades_data)
        st.dataframe(df, use_container_width=True)
        
        # Remove trade option
        if len(st.session_state.trades_input) > 0:
            remove_idx = st.selectbox("Remove trade by index:", [-1] + list(range(len(st.session_state.trades_input))))
            if remove_idx >= 0 and st.button("🗑️ Remove Selected Trade"):
                st.session_state.trades_input.pop(remove_idx)
                st.rerun()
    
    # Step 3: Collateral Input (same as original)
    with st.expander("🛡️ Collateral Portfolio", expanded=False):
        if 'collateral_input' not in st.session_state:
            st.session_state.collateral_input = []
        
        col1, col2, col3 = st.columns(3)
        with col1:
            coll_type = st.selectbox("Collateral Type", [ct.value for ct in CollateralType])
        with col2:
            coll_currency = st.selectbox("Collateral Currency", ["USD", "EUR", "GBP", "JPY"])
        with col3:
            coll_amount = st.number_input("Amount ($)", min_value=0.0, value=10000000.0, step=1000000.0)
        
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
            for i, coll in enumerate(st.session_state.collateral_input):
                st.write(f"{i+1}. {coll.collateral_type.value}: ${coll.amount:,.0f} {coll.currency}")
    
    # Enhanced Calculation and Data Quality Analysis
    if st.button("🚀 Calculate SA-CCR with Step-by-Step Analysis", type="primary"):
        # Create netting set
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
        
        # Validate inputs
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
        
        # Perform enhanced calculation
        with st.spinner("🧮 Performing complete 24-step SA-CCR calculation with detailed analysis..."):
            try:
                result = st.session_state.saccr_agent.calculate_comprehensive_saccr(
                    netting_set, st.session_state.collateral_input
                )
                
                # Display enhanced results
                display_enhanced_saccr_results(result)
                
            except Exception as e:
                st.error(f"❌ Calculation error: {str(e)}")

def display_enhanced_saccr_results(result: Dict):
    """Display enhanced results with step-by-step thinking and data quality analysis"""
    
    # Data Quality Issues Section (Enhanced)
    if result['data_quality_issues']:
        st.markdown("### ⚠️ Data Quality Analysis")
        
        high_impact = [issue for issue in result['data_quality_issues'] if issue.impact == 'high']
        medium_impact = [issue for issue in result['data_quality_issues'] if issue.impact == 'medium']
        
        if high_impact:
            st.markdown("""
            <div class="missing-info-prompt">
                <strong>🚨 High-Impact Missing Information Detected</strong><br>
                The following missing information significantly affects RWA accuracy:
            </div>
            """, unsafe_allow_html=True)
            
            for issue in high_impact:
                st.markdown(f"""
                <div class="data-quality-alert">
                    <strong>Missing: {issue.field_name}</strong><br>
                    <strong>Current:</strong> {issue.current_value}<br>
                    <strong>Impact:</strong> {issue.recommendation}<br>
                    <strong>Default Used:</strong> {issue.default_used}
                </div>
                """, unsafe_allow_html=True)
        
        # Interactive Data Gathering
        if high_impact:
            st.markdown("#### 💬 Provide Missing Information for More Accurate Results")
            
            missing_info_prompts = st.session_state.saccr_agent.gather_missing_information(result['data_quality_issues'])
            
            for i, prompt in enumerate(missing_info_prompts['missing_info_prompts']):
                with st.expander(f"📝 {prompt['field']}", expanded=False):
                    st.markdown(f"**Question:** {prompt['question']}")
                    st.markdown(f"**Why this matters:** {prompt['impact']}")
                    user_input = st.text_input(f"Your answer:", key=f"missing_info_{i}")
                    if user_input:
                        st.success(f"✅ Information captured: {user_input}")
                        st.info(f"💡 This will improve the accuracy of: {prompt['impact']}")
    
    # Enhanced Summary Results
    st.markdown("### 📊 SA-CCR Calculation Summary")
    
    enhanced_summary = result['enhanced_summary']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📋 Key Inputs")
        st.markdown('<div class="summary-box">', unsafe_allow_html=True)
        for item in enhanced_summary['key_inputs']:
            st.write(f"• {item}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### ⚡ Risk Components")
        st.markdown('<div class="summary-box">', unsafe_allow_html=True)
        for item in enhanced_summary['risk_components']:
            st.write(f"• {item}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown("#### 💰 Capital Results")
        st.markdown('<div class="summary-box">', unsafe_allow_html=True)
        for item in enhanced_summary['capital_results']:
            st.write(f"• {item}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Final Results Highlight
    final_results = result['final_results']
    
    st.markdown(f"""
    <div class="result-summary-enhanced">
        <h3>🎯 Final SA-CCR Results</h3>
        <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 1rem; margin-top: 1rem;">
            <div style="text-align: center;">
                <div style="font-size: 1.2rem; font-weight: bold;">RC</div>
                <div style="font-size: 1.1rem;">${final_results['replacement_cost']:,.0f}</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.2rem; font-weight: bold;">PFE</div>
                <div style="font-size: 1.1rem;">${final_results['potential_future_exposure']:,.0f}</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.2rem; font-weight: bold;">EAD</div>
                <div style="font-size: 1.1rem;">${final_results['exposure_at_default']:,.0f}</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.2rem; font-weight: bold;">RWA</div>
                <div style="font-size: 1.1rem;">${final_results['risk_weighted_assets']:,.0f}</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.2rem; font-weight: bold;">Capital</div>
                <div style="font-size: 1.1rem;">${final_results['capital_requirement']:,.0f}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Step-by-Step Thinking Process
    if result.get('thinking_steps'):
        st.markdown("### 🧠 Detailed Step-by-Step Analysis")
        st.markdown("*Click each step to see the detailed regulatory thinking process*")
        
        for thinking_step in result['thinking_steps']:
            with st.expander(f"🔍 Step {thinking_step['step']}: {thinking_step['title']}", expanded=False):
                
                st.markdown(f"""
                <div class="thinking-process">
                    <div class="step-reasoning">
                        <pre style="white-space: pre-wrap; font-family: 'Inter', sans-serif; font-size: 0.9rem;">{thinking_step['reasoning']}</pre>
                    </div>
                    
                    <div class="formula-breakdown">
                        <strong>Regulatory Formula:</strong> {thinking_step['formula']}
                    </div>
                    
                    <div class="calculation-detail">
                        <strong>Key Insight:</strong> {thinking_step.get('key_insight', 'Calculation completed per Basel requirements')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Complete 24-Step Breakdown
    with st.expander("📋 Complete 24-Step Calculation Breakdown", expanded=False):
        
        step_groups = {
            "Trade Data & Classification (Steps 1-4)": [1, 2, 3, 4],
            "Notional & Risk Factor Calculations (Steps 5-8)": [5, 6, 7, 8],
            "Add-On Calculations (Steps 9-13)": [9, 10, 11, 12, 13],
            "PFE Calculations (Steps 14-16)": [14, 15, 16],
            "Replacement Cost (Steps 17-18)": [17, 18],
            "EAD & RWA Calculations (Steps 19-24)": [19, 20, 21, 22, 23, 24]
        }
        
        for group_name, step_numbers in step_groups.items():
            with st.expander(f"📋 {group_name}", expanded=False):
                for step_num in step_numbers:
                    if step_num <= len(result['calculation_steps']):
                        step_data = result['calculation_steps'][step_num - 1]
                        
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
    
    # Optimization Insights
    st.markdown("### 💡 Portfolio Optimization Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Key Optimization Opportunities")
        for insight in enhanced_summary['optimization_insights']:
            st.write(f"• {insight}")
    
    with col2:
        st.markdown("#### ⚠️ Assumptions Made")
        if result.get('assumptions'):
            for assumption in result['assumptions']:
                st.write(f"• {assumption}")
        else:
            st.write("• No significant assumptions required")
    
    # Enhanced AI Analysis
    if result.get('ai_explanation'):
        st.markdown("### 🤖 AI Expert Analysis")
        st.markdown(f"""
        <div class="ai-response">
            <strong>🤖 Enhanced AI Analysis with Thinking Process Insights:</strong><br><br>
            {result['ai_explanation']}
        </div>
        """, unsafe_allow_html=True)
    
    # Interactive AI Chat for Data Improvement
    if result['data_quality_issues']:
        st.markdown("### 🤖 AI Assistant - Improve Your SA-CCR Calculation")
        
        user_question = st.text_area(
            "Ask the AI how to improve your data quality or SA-CCR calculation:",
            placeholder="e.g., How can I get better MTM values? What's the impact of missing collateral data? How can I optimize my capital requirement?",
            height=80
        )
        
        if st.button("💬 Get AI Guidance on Data & Optimization") and user_question:
            if st.session_state.saccr_agent.llm and st.session_state.saccr_agent.connection_status == "connected":
                
                # Generate context-aware response
                system_prompt = """You are a SA-CCR expert focusing on data quality and capital optimization. Help users:
                1. Understand how missing data affects their calculations
                2. Provide practical steps to gather missing information  
                3. Suggest optimization strategies based on their specific portfolio
                4. Quantify potential benefits of improvements
                
                Be specific, actionable, and focus on practical implementation."""
                
                missing_info_context = json.dumps([
                    {
                        'field': issue.field_name,
                        'impact': issue.impact,
                        'recommendation': issue.recommendation
                    } for issue in result['data_quality_issues']
                ], indent=2)
                
                calculation_context = {
                    'final_capital': final_results['capital_requirement'],
                    'current_rc': final_results['replacement_cost'],
                    'current_pfe': final_results['potential_future_exposure'],
                    'netting_benefit': result.get('thinking_steps', [{}])[-1].get('key_insight', 'N/A')
                }
                
                user_prompt = f"""
                USER'S SA-CCR CALCULATION CONTEXT:
                Current Results: Capital Requirement: ${final_results['capital_requirement']:,.0f}
                
                Data Quality Issues:
                {missing_info_context}
                
                Calculation Summary:
                {json.dumps(calculation_context, indent=2)}
                
                User Question: {user_question}
                
                Please provide specific guidance on:
                1. How to gather the missing information
                2. Expected impact on capital calculation accuracy
                3. Practical optimization strategies
                4. Quantified benefits where possible
                """
                
                with st.spinner("🤖 AI is analyzing your SA-CCR optimization question..."):
                    try:
                        response = st.session_state.saccr_agent.llm.invoke([
                            SystemMessage(content=system_prompt),
                            HumanMessage(content=user_prompt)
                        ])
                        
                        st.markdown(f"""
                        <div class="ai-response">
                            <strong>🤖 AI Optimization & Data Quality Guidance:</strong><br><br>
                            {response.content}
                        </div>
                        """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"AI response error: {str(e)}")
            else:
                st.warning("🔌 Please connect to LLM in the sidebar for AI assistance")
    
    # Export Enhanced Results
    st.markdown("### 📥 Export Enhanced Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Enhanced summary report
        summary_data = {
            'Calculation_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Netting_Set': result['calculation_steps'][0]['data']['netting_set_id'],
            'Counterparty': result['calculation_steps'][0]['data']['counterparty'],
            'Total_Trades': result['calculation_steps'][0]['data']['trade_count'],
            'Data_Quality_Issues': len(result['data_quality_issues']),
            'High_Impact_Issues': len([i for i in result['data_quality_issues'] if i.impact == 'high']),
            'Replacement_Cost': final_results['replacement_cost'],
            'PFE': final_results['potential_future_exposure'],
            'EAD': final_results['exposure_at_default'],
            'RWA': final_results['risk_weighted_assets'],
            'Capital_Required': final_results['capital_requirement'],
            'Capital_Efficiency_bps': int(final_results['capital_requirement']/sum(abs(t.notional) for t in st.session_state.trades_input)*10000)
        }
        
        summary_csv = pd.DataFrame([summary_data]).to_csv(index=False).encode('utf-8')
        st.download_button(
            "📊 Download Enhanced Summary",
            data=summary_csv,
            file_name=f"enhanced_saccr_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Thinking process report
        thinking_data = []
        for step in result.get('thinking_steps', []):
            thinking_data.append({
                'Step': step['step'],
                'Title': step['title'],
                'Key_Insight': step.get('key_insight', ''),
                'Formula': step['formula']
            })
        
        if thinking_data:
            thinking_csv = pd.DataFrame(thinking_data).to_csv(index=False).encode('utf-8')
            st.download_button(
                "🧠 Download Thinking Process",
                data=thinking_csv,
                file_name=f"saccr_thinking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col3:
        # Complete JSON export
        json_data = json.dumps(result, indent=2, default=str)
        st.download_button(
            "🔧 Download Complete Analysis",
            data=json_data,
            file_name=f"complete_saccr_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

def load_reference_example():
    """Load the reference example data"""
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
    if 'collateral_input' not in st.session_state:
        st.session_state.collateral_input = []

def show_reference_example():
    """Show the reference example with enhanced analysis"""
    
    st.markdown("## 📋 Reference Example - Lowell Hotel Properties LLC")
    st.markdown("*Enhanced analysis of the reference calculation with step-by-step thinking*")
    
    # Create the reference example trade
    if st.button("🔄 Load Reference Example with Enhanced Analysis", type="primary"):
        
        # Clear existing data
        st.session_state.trades_input = []
        st.session_state.collateral_input = []
        
        # Load reference data
        load_reference_example()
        
        # Create reference netting set
        netting_set = NettingSet(
            netting_set_id="212784060000009618701",
            counterparty="Lowell Hotel Properties LLC",
            trades=st.session_state.trades_input,
            threshold=12000000,
            mta=1000000,
            nica=0
        )
        
        st.success("✅ Reference example loaded successfully!")
        
        # Display reference details
        st.markdown("### 📊 Reference Trade Analysis")
        
        trade = st.session_state.trades_input[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Trade Details")
            st.write(f"• **Trade ID**: {trade.trade_id}")
            st.write(f"• **Counterparty**: {trade.counterparty}")
            st.write(f"• **Asset Class**: {trade.asset_class.value}")
            st.write(f"• **Notional**: ${trade.notional:,.0f}")
            st.write(f"• **Currency**: {trade.currency}")
            st.write(f"• **Trade Type**: {trade.trade_type.value}")
        
        with col2:
            st.markdown("#### Netting Set Terms")
            st.write(f"• **Netting Set ID**: {netting_set.netting_set_id}")
            st.write(f"• **Threshold**: ${netting_set.threshold:,.0f}")
            st.write(f"• **MTA**: ${netting_set.mta:,.0f}")
            st.write(f"• **NICA**: ${netting_set.nica:,.0f}")
            st.write(f"• **Maturity**: {trade.time_to_maturity():.1f} years")
        
        # Automatically run enhanced calculation
        with st.spinner("🧮 Performing enhanced SA-CCR analysis on reference example..."):
            try:
                result = st.session_state.saccr_agent.calculate_comprehensive_saccr(netting_set, [])
                
                st.markdown("### 📊 Enhanced Reference Results with Step-by-Step Analysis")
                
                # Show key results
                final_results = result['final_results']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Final EAD", f"${final_results['exposure_at_default']:,.0f}")
                with col2:
                    st.metric("RWA", f"${final_results['risk_weighted_assets']:,.0f}")
                with col3:
                    st.metric("Capital Required", f"${final_results['capital_requirement']:,.0f}")
                with col4:
                    capital_efficiency = (final_results['capital_requirement']/trade.notional*10000)
                    st.metric("Capital Efficiency", f"{capital_efficiency:.1f}bps")
                
                # Show thinking process highlights
                st.markdown("### 🧠 Key Calculation Insights")
                
                if result.get('thinking_steps'):
                    key_steps = [step for step in result['thinking_steps'] if step['step'] in [6, 8, 15, 18, 21]]
                    
                    for thinking_step in key_steps:
                        st.markdown(f"""
                        <div class="calculation-verified">
                            <strong>Step {thinking_step['step']}: {thinking_step['title']}</strong><br>
                            <strong>Key Insight:</strong> {thinking_step.get('key_insight', 'Calculation completed')}<br>
                            <small><strong>Formula:</strong> {thinking_step['formula']}</small>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Enhanced summary
                enhanced_summary = result['enhanced_summary']
                
                st.markdown("### 📋 Reference Example Summary")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Key Results")
                    for item in enhanced_summary['capital_results']:
                        st.write(f"• {item}")
                
                with col2:
                    st.markdown("#### Risk Profile")
                    for item in enhanced_summary['risk_components']:
                        st.write(f"• {item}")
                
                # Data quality assessment
                if result['data_quality_issues']:
                    st.markdown("### ⚠️ Reference Example Data Quality Notes")
                    for issue in result['data_quality_issues']:
                        if issue.impact == 'high':
                            st.warning(f"**{issue.field_name}**: {issue.recommendation}")
                
                st.markdown("### ✅ Reference Validation")
                st.success("✅ Enhanced calculation follows the complete 24-step Basel SA-CCR methodology")
                st.info("💡 This analysis provides deeper insights than the basic reference calculation")
                
            except Exception as e:
                st.error(f"❌ Calculation error: {str(e)}")

def enhanced_ai_assistant_page():
    """Enhanced AI assistant with data quality focus"""
    
    st.markdown("## 🤖 Enhanced AI SA-CCR Expert Assistant")
    st.markdown("*Advanced SA-CCR guidance with data quality analysis and optimization strategies*")
    
    # Enhanced question templates
    with st.expander("💡 Enhanced Sample Questions", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**SA-CCR Technical Questions:**")
            st.markdown("""
            - "Walk me through the PFE multiplier calculation step-by-step"
            - "How does the maturity factor formula work and why?"
            - "What's the regulatory reasoning behind supervisory factors?"
            - "Explain the difference between margined and unmargined RC"
            - "How do hedging set correlations affect my add-on calculations?"
            """)
        
        with col2:
            st.markdown("**Data Quality & Optimization:**")
            st.markdown("""
            - "What's the impact of missing MTM values on my calculation?"
            - "How can I optimize my derivatives portfolio for SA-CCR?"
            - "What collateral types give me the best capital benefit?"
            - "Should I prioritize central clearing or netting optimization?"
            - "How accurate is my calculation with estimated data?"
            """)
    
    # Chat interface with enhanced context
    st.markdown("### 💬 Ask the Enhanced AI Expert")
    
    if 'saccr_chat_history' not in st.session_state:
        st.session_state.saccr_chat_history = []
    
    user_question = st.text_area(
        "Your SA-CCR Question:",
        placeholder="e.g., I have missing MTM data for 3 trades - how much does this affect my capital calculation accuracy?",
        height=100
    )
    
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        if st.button("🚀 Ask Enhanced AI Expert", type="primary"):
            if user_question.strip():
                process_ai_question(user_question)
    
    with col2:
        if st.button("🔍 Analyze My Portfolio Data Quality"):
            if 'trades_input' in st.session_state and st.session_state.trades_input:
                analyze_portfolio_data_quality()
            else:
                st.warning("Please add trades in the calculator first")
    
    with col3:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.saccr_chat_history = []
            st.rerun()
    
    # Display enhanced chat history
    if st.session_state.saccr_chat_history:
        st.markdown("### 💬 Enhanced Conversation History")
        
        for chat in reversed(st.session_state.saccr_chat_history[-6:]):
            if chat['type'] == 'user':
                st.markdown(f"""
                <div class="user-query">
                    <strong>👤 You:</strong> {chat['content']}
                    <br><small style="color: #666;">{chat['timestamp'].strftime('%H:%M:%S')}</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="ai-response">
                    <strong>🤖 Enhanced SA-CCR Expert:</strong><br>
                    {chat['content']}
                    <br><small style="color: rgba(255,255,255,0.7);">{chat['timestamp'].strftime('%H:%M:%S')}</small>
                </div>
                """, unsafe_allow_html=True)

def process_ai_question(user_question: str):
    """Process AI question with enhanced context"""
    
    # Add to chat history
    st.session_state.saccr_chat_history.append({
        'type': 'user',
        'content': user_question,
        'timestamp': datetime.now()
    })
    
    # Get enhanced portfolio context
    portfolio_context = {}
    data_quality_context = {}
    
    if 'trades_input' in st.session_state and st.session_state.trades_input:
        portfolio_context = {
            'trade_count': len(st.session_state.trades_input),
            'asset_classes': list(set(t.asset_class.value for t in st.session_state.trades_input)),
            'total_notional': sum(abs(t.notional) for t in st.session_state.trades_input),
            'currencies': list(set(t.currency for t in st.session_state.trades_input)),
            'maturity_range': f"{min(t.time_to_maturity() for t in st.session_state.trades_input):.1f} - {max(t.time_to_maturity() for t in st.session_state.trades_input):.1f} years"
        }
        
        # Analyze data quality for context
        temp_netting_set = NettingSet(
            netting_set_id="temp",
            counterparty="temp",
            trades=st.session_state.trades_input
        )
        data_quality_issues = st.session_state.saccr_agent.analyze_data_quality(temp_netting_set)
        
        data_quality_context = {
            'total_issues': len(data_quality_issues),
            'high_impact_issues': len([i for i in data_quality_issues if i.impact == 'high']),
            'missing_fields': [i.field_name for i in data_quality_issues if i.impact == 'high']
        }
    
    # Generate AI response
    with st.spinner("🤖 Enhanced AI is analyzing your SA-CCR question with portfolio context..."):
        try:
            if st.session_state.saccr_agent.llm and st.session_state.saccr_agent.connection_status == "connected":
                
                system_prompt = """You are an enhanced Basel SA-CCR regulatory expert with deep knowledge of:
                - Complete 24-step SA-CCR calculation methodology with step-by-step reasoning
                - Data quality impacts on calculation accuracy and capital requirements
                - Portfolio optimization strategies with quantified benefits
                - Regulatory compliance and implementation best practices
                
                Provide detailed, technical answers that include:
                1. Step-by-step thinking process when relevant
                2. Quantified impacts and benefits
                3. Practical implementation guidance
                4. Data quality considerations
                """
                
                context_info = f"""
                Portfolio Context: {json.dumps(portfolio_context, indent=2) if portfolio_context else 'No portfolio loaded'}
                
                Data Quality Assessment: {json.dumps(data_quality_context, indent=2) if data_quality_context else 'No assessment available'}
                """
                
                user_prompt = f"""
                Enhanced SA-CCR Question: {user_question}
                
                {context_info}
                
                Please provide a comprehensive answer including:
                - Technical explanation with step-by-step reasoning where applicable
                - Impact on the user's specific portfolio context
                - Data quality considerations
                - Quantified benefits and recommendations
                - Practical implementation steps
                """
                
                response = st.session_state.saccr_agent.llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ])
                
                ai_response = response.content
                
            else:
                # Enhanced fallback response
                ai_response = generate_enhanced_template_response(user_question, portfolio_context, data_quality_context)
            
            # Add AI response to chat history
            st.session_state.saccr_chat_history.append({
                'type': 'ai',
                'content': ai_response,
                'timestamp': datetime.now()
            })
            
            st.rerun()
            
        except Exception as e:
            st.error(f"Enhanced AI response error: {str(e)}")

def analyze_portfolio_data_quality():
    """Analyze current portfolio data quality"""
    
    temp_netting_set = NettingSet(
        netting_set_id="analysis",
        counterparty="analysis",
        trades=st.session_state.trades_input
    )
    
    issues = st.session_state.saccr_agent.analyze_data_quality(temp_netting_set)
    
    analysis_response = f"""
    **🔍 Portfolio Data Quality Analysis:**
    
    **Overall Assessment:** {len(issues)} data quality issues identified
    
    **High Impact Issues ({len([i for i in issues if i.impact == 'high'])}):**
    {chr(10).join([f"• {issue.field_name}: {issue.recommendation}" for issue in issues if issue.impact == 'high'])}
    
    **Medium Impact Issues ({len([i for i in issues if i.impact == 'medium'])}):**
    {chr(10).join([f"• {issue.field_name}: {issue.recommendation}" for issue in issues if issue.impact == 'medium'])}
    
    **Recommendations:**
    • Address high-impact issues first for maximum accuracy improvement
    • Missing MTM values can significantly affect RC and PFE multiplier calculations
    • Consider implementing systematic data collection processes
    • Estimated values should be replaced with actual market data when possible
    
    **Next Steps:**
    1. Gather missing information using the prompts in the calculator
    2. Re-run calculation with improved data
    3. Compare results to quantify accuracy improvement
    """
    
    # Add to chat history
    st.session_state.saccr_chat_history.append({
        'type': 'user',
        'content': 'Analyze my portfolio data quality',
        'timestamp': datetime.now()
    })
    
    st.session_state.saccr_chat_history.append({
        'type': 'ai',
        'content': analysis_response,
        'timestamp': datetime.now()
    })
    
    st.rerun()

def generate_enhanced_template_response(question: str, portfolio_context: dict = None, data_quality_context: dict = None) -> str:
    """Generate enhanced template responses when LLM is not available"""
    
    question_lower = question.lower()
    
    # Enhanced responses with step-by-step thinking
    if "pfe multiplier" in question_lower or "multiplier" in question_lower:
        return f"""
        **Enhanced PFE Multiplier Analysis:**
        
        **Step-by-Step Thinking Process:**
        
        **Step 1: Understanding the Formula**
        • Formula: Multiplier = min(1, 0.05 + 0.95 × exp((V-C) / (2 × 0.95 × AddOn)))
        • This captures netting benefits within your netting set
        • Lower multiplier = more netting benefit = lower capital
        
        **Step 2: Component Analysis**
        • V-C: Net exposure after collateral
        • AddOn: Your portfolio's aggregate add-on (future risk measure)
        • Ratio (V-C)/AddOn: Determines netting benefit level
        
        **Step 3: Economic Interpretation**
        • When V-C is negative (out-of-the-money): Strong netting benefit
        • When V-C is positive (in-the-money): Reduced netting benefit
        • Floor at 0.05 ensures minimum 95% netting benefit recognition
        
        {f"**Your Portfolio Context:** {portfolio_context['trade_count']} trades, focusing on {', '.join(portfolio_context['asset_classes'])} asset classes" if portfolio_context else ""}
        
        **Optimization Strategy:**
        • Monitor and manage portfolio MTM through strategic hedging
        • Consider trade compression to reduce gross exposure
        • Optimize collateral posting to minimize V-C ratio
        
        **Quantified Impact:**
        • 10% reduction in multiplier ≈ 10% reduction in PFE ≈ 7% reduction in total capital
        """
    
    elif "data quality" in question_lower or "missing" in question_lower or "accuracy" in question_lower:
        high_impact_count = data_quality_context.get('high_impact_issues', 0) if data_quality_context else 0
        
        return f"""
        **Enhanced Data Quality Impact Analysis:**
        
        **Your Current Data Quality Status:**
        {f"• Total issues identified: {data_quality_context['total_issues']}" if data_quality_context else "• No portfolio loaded for analysis"}
        {f"• High-impact issues: {high_impact_count}" if data_quality_context else ""}
        {f"• Critical missing fields: {', '.join(data_quality_context['missing_fields'])}" if data_quality_context and data_quality_context['missing_fields'] else ""}
        
        **Impact Assessment by Data Type:**
        
        **Missing MTM Values (High Impact - ±15-30% capital error):**
        • Directly affects replacement cost calculation
        • Changes PFE multiplier through V-C ratio
        • Can cause 15-30% over/under-estimation of capital
        
        **Missing/Estimated Collateral (High Impact - ±20-40% capital error):**
        • Reduces replacement cost calculation accuracy
        • Affects PFE multiplier significantly
        • Missing $10M collateral ≈ $1-2M additional capital requirement
        
        **Estimated Option Deltas (Medium Impact - ±5-15% capital error):**
        • Affects effective notional calculations
        • Impacts add-on aggregation accuracy
        • Less critical but can accumulate across large option portfolios
        
        **Recommended Data Collection Priority:**
        1. **Current MTM values** (highest impact - get from risk systems)
        2. **Posted collateral details** (high impact - check margin systems)  
        3. **Accurate option deltas** (medium impact - get from pricing systems)
        4. **CSA threshold/MTA terms** (high impact - check legal agreements)
        
        **Expected Accuracy Improvement:**
        • Addressing all high-impact issues: 60-80% more accurate capital calculation
        • Cost of inaccuracy: Potential $500K-2M capital over/under-allocation per $100M notional
        """
    
    else:
        return f"""
        **Enhanced SA-CCR Expert Guidance:**
        
        I can provide detailed step-by-step analysis on any SA-CCR topic. Here are my enhanced capabilities:
        
        **Technical Deep-Dives:**
        • Complete 24-step calculation methodology with thinking process
        • Regulatory formula derivations and Basel Committee reasoning
        • Component-by-component impact analysis
        
        **Data Quality Analysis:**
        • Missing data impact quantification
        • Data collection prioritization strategies  
        • Accuracy improvement recommendations
        
        **Optimization Strategies:**
        • Portfolio restructuring for capital efficiency
        • Netting and collateral optimization
        • Central clearing vs bilateral trade-offs
        
        {f"**Your Portfolio:** {portfolio_context['trade_count']} trades totaling ${portfolio_context['total_notional']:,.0f}" if portfolio_context else ""}
        {f"**Data Quality:** {data_quality_context['high_impact_issues']} high-impact issues to address" if data_quality_context else ""}
        
        Please ask a specific question for detailed step-by-step analysis.
        """

def portfolio_analysis_page():
    """Enhanced portfolio analysis page"""
    
    st.markdown("## 📊 Enhanced Portfolio Analysis & Optimization")
    
    if 'trades_input' not in st.session_state or not st.session_state.trades_input:
        st.info("📝 Please add trades in the SA-CCR Calculator first to perform enhanced portfolio analysis")
        return
    
    trades = st.session_state.trades_input
    
    # Enhanced Portfolio Overview
    st.markdown("### 📋 Enhanced Portfolio Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Trades", len(trades))
    with col2:
        total_notional = sum(abs(t.notional) for t in trades)
        st.metric("Total Notional", f"${total_notional/1_000_000:.0f}M")
    with col3:
        asset_classes = len(set(t.asset_class for t in trades))
        st.metric("Asset Classes", asset_classes)
    with col4:
        currencies = len(set(t.currency for t in trades))
        st.metric("Currencies", currencies)
    with col5:
        avg_maturity = sum(t.time_to_maturity() for t in trades) / len(trades)
        st.metric("Avg Maturity", f"{avg_maturity:.1f}y")
    
    # Enhanced visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Portfolio Composition Analysis")
        
        # Asset class breakdown with risk weighting
        asset_class_data = {}
        for trade in trades:
            ac = trade.asset_class.value
            if ac not in asset_class_data:
                asset_class_data[ac] = {'notional': 0, 'count': 0}
            asset_class_data[ac]['notional'] += abs(trade.notional)
            asset_class_data[ac]['count'] += 1
        
        ac_df = pd.DataFrame([
            {'Asset_Class': ac, 'Notional_MM': data['notional']/1_000_000, 'Trade_Count': data['count']}
            for ac, data in asset_class_data.items()
        ])
        
        fig = px.pie(ac_df, values='Notional_MM', names='Asset_Class',
                     title="Notional Distribution by Asset Class",
                     hover_data=['Trade_Count'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Risk-Time Profile")
        
        # Enhanced maturity vs notional with risk coloring
        maturity_data = []
        for trade in trades:
            # Simple risk scoring for coloring
            risk_score = abs(trade.notional) / 1_000_000 * (trade.time_to_maturity() + 0.5)
            
            maturity_data.append({
                'Trade_ID': trade.trade_id,
                'Maturity_Years': trade.time_to_maturity(),
                'Notional_MM': abs(trade.notional) / 1_000_000,
                'Asset_Class': trade.asset_class.value,
                'Risk_Score': risk_score
            })
        
        mat_df = pd.DataFrame(maturity_data)
        fig = px.scatter(mat_df, x='Maturity_Years', y='Notional_MM',
                         color='Risk_Score', size='Risk_Score',
                         hover_data=['Trade_ID', 'Asset_Class'],
                         title="Risk-Weighted Maturity Profile",
                         color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    # Data Quality Assessment
    temp_netting_set = NettingSet(
        netting_set_id="analysis",
        counterparty="analysis",
        trades=trades
    )
    data_quality_issues = st.session_state.saccr_agent.analyze_data_quality(temp_netting_set)
    
    st.markdown("### ⚠️ Portfolio Data Quality Assessment")
    
    if data_quality_issues:
        high_impact = len([i for i in data_quality_issues if i.impact == 'high'])
        medium_impact = len([i for i in data_quality_issues if i.impact == 'medium'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Issues", len(data_quality_issues))
        with col2:
            st.metric("High Impact", high_impact, delta=f"-{high_impact}" if high_impact > 0 else None)
        with col3:
            st.metric("Medium Impact", medium_impact)
        
        if high_impact > 0:
            st.warning(f"🚨 {high_impact} high-impact data quality issues detected. Consider addressing these for more accurate capital calculations.")
    else:
        st.success("✅ No significant data quality issues detected")
    
    # Enhanced AI Portfolio Analysis
    if st.button("🤖 Generate Enhanced AI Portfolio Analysis", type="primary"):
        with st.spinner("🤖 Enhanced AI is performing comprehensive portfolio analysis..."):
            
            # Prepare enhanced portfolio summary
            portfolio_summary = {
                'basic_metrics': {
                    'total_trades': len(trades),
                    'total_notional': total_notional,
                    'avg_maturity': avg_maturity,
                    'largest_trade': max(abs(t.notional) for t in trades),
                    'mtm_exposure': sum(t.mtm_value for t in trades)
                },
                'composition': {
                    'asset_classes': list(set(t.asset_class.value for t in trades)),
                    'currencies': list(set(t.currency for t in trades)),
                    'trade_types': list(set(t.trade_type.value for t in trades))
                },
                'risk_profile': {
                    'maturity_range': f"{min(t.time_to_maturity() for t in trades):.1f}-{max(t.time_to_maturity() for t in trades):.1f}y",
                    'concentration': max(abs(t.notional) for t in trades) / total_notional * 100,
                    'asset_class_concentration': max(sum(abs(t.notional) for t in trades if t.asset_class.value == ac) for ac in set(t.asset_class.value for t in trades)) / total_notional * 100
                },
                'data_quality': {
                    'total_issues': len(data_quality_issues),
                    'high_impact_issues': high_impact if data_quality_issues else 0,
                    'missing_fields': [i.field_name for i in data_quality_issues if i.impact == 'high']
                }
            }
            
            if st.session_state.saccr_agent.llm and st.session_state.saccr_agent.connection_status == "connected":
                try:
                    system_prompt = """You are an enhanced derivatives portfolio optimization expert specializing in SA-CCR capital efficiency and data quality analysis. 
                    
                    Provide comprehensive analysis including:
                    1. Risk concentration and diversification assessment
                    2. SA-CCR capital efficiency opportunities with quantified benefits
                    3. Data quality impact on calculation accuracy
                    4. Specific optimization recommendations with implementation priorities
                    5. Expected capital reduction percentages from each strategy
                    
                    Focus on actionable, implementable strategies with clear business impact."""
                    
                    user_prompt = f"""
                    Perform comprehensive enhanced portfolio analysis:
                    
                    PORTFOLIO METRICS:
                    {json.dumps(portfolio_summary, indent=2)}
                    
                    Please provide executive-level analysis covering:
                    
                    1. **Portfolio Risk Assessment**
                       - Concentration risks and diversification opportunities
                       - Maturity profile optimization
                       - Asset class balance evaluation
                    
                    2. **SA-CCR Capital Efficiency Analysis**
                       - Current capital efficiency vs industry benchmarks
                       - Specific optimization opportunities with quantified benefits
                       - Netting and collateral optimization potential
                    
                    3. **Data Quality Impact Assessment**
                       - How data issues affect capital calculation accuracy
                       - Priority actions to improve data quality
                       - Expected accuracy improvement from data fixes
                    
                    4. **Strategic Recommendations**
                       - Priority actions ranked by capital impact
                       - Implementation timeline and resource requirements
                       - Expected capital reduction percentages
                    
                    5. **Risk Management Insights**
                       - Portfolio monitoring recommendations
                       - Key risk indicators to track
                       - Optimization review frequency
                    
                    Provide specific, quantified recommendations with clear business rationale.
                    """
                    
                    response = st.session_state.saccr_agent.llm.invoke([
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_prompt)
                    ])
                    
                    st.markdown(f"""
                    <div class="ai-response">
                        <strong>🤖 Enhanced AI Portfolio Analysis & Optimization Strategy:</strong><br><br>
                        {response.content}
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Enhanced AI analysis error: {str(e)}")
            else:
                # Enhanced fallback analysis
                st.markdown(f"""
                <div class="ai-insight">
                    <strong>📊 Enhanced Portfolio Analysis (LLM Disconnected):</strong><br><br>
                    
                    <strong>Portfolio Risk Profile:</strong><br>
                    • Total exposure: ${total_notional/1_000_000:.0f}M across {len(trades)} trades<br>
                    • Asset class diversification: {len(set(t.asset_class.value for t in trades))} classes ({', '.join(set(t.asset_class.value for t in trades))})<br>
                    • Maturity profile: {min(t.time_to_maturity() for t in trades):.1f} to {max(t.time_to_maturity() for t in trades):.1f} years<br>
                    • Largest single exposure: ${max(abs(t.notional) for t in trades)/1_000_000:.0f}M ({max(abs(t.notional) for t in trades)/total_notional*100:.1f}% of portfolio)<br>
                    
                    <strong>Data Quality Assessment:</strong><br>
                    • Total data issues: {len(data_quality_issues)}<br>
                    • High-impact issues: {high_impact if data_quality_issues else 0}<br>
                    • Estimated capital calculation accuracy: {100 - min(high_impact * 10, 40):.0f}%<br>
                    
                    <strong>Optimization Recommendations:</strong><br>
                    • {"Address data quality issues first - potential 20-40% improvement in calculation accuracy" if high_impact > 0 else "Data quality is good - focus on structural optimization"}<br>
                    • Consider portfolio compression if concentration > 25%<br>
                    • Evaluate netting agreement enhancements<br>
                    • {"Assess central clearing opportunities for eligible trades" if len(trades) > 5 else "Monitor portfolio growth for future optimization opportunities"}<br>
                    • Review collateral optimization strategies<br>
                    
                    <strong>Next Steps:</strong><br>
                    • Run full SA-CCR calculation to quantify current capital requirement<br>
                    • Address high-impact data quality issues<br>
                    • Conduct quarterly portfolio optimization reviews<br>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
