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
    page_title="Enhanced AI SA-CCR Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with step-by-step thinking styles
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
    
    .result-summary {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 1.5rem;
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
    
    .connection-status {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
    }
    
    .connected { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .disconnected { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# CORE DATA CLASSES (same as before)
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
# ENHANCED SA-CCR AGENT WITH STEP-BY-STEP ANALYSIS
# ==============================================================================

class EnhancedSACCRAgent:
    """Enhanced SA-CCR Agent with detailed step-by-step analysis and data gathering"""
    
    def __init__(self):
        self.llm = None
        self.connection_status = "disconnected"
        
        # Initialize regulatory parameters
        self.supervisory_factors = self._initialize_supervisory_factors()
        self.supervisory_correlations = self._initialize_correlations()
        self.collateral_haircuts = self._initialize_collateral_haircuts()
        
        # Data quality tracking
        self.data_quality_issues = []
        self.calculation_assumptions = []
    
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
    
    def calculate_with_thinking_process(self, netting_set: NettingSet, collateral: List[Collateral] = None) -> Dict:
        """Calculate SA-CCR with detailed step-by-step thinking process"""
        
        # Reset tracking
        self.data_quality_issues = []
        self.calculation_assumptions = []
        
        # Analyze data quality first
        quality_issues = self.analyze_data_quality(netting_set, collateral)
        self.data_quality_issues = quality_issues
        
        thinking_steps = []
        calculation_results = {}
        
        # Step-by-step calculation with reasoning
        step_1_thinking, step_1_result = self._step_1_with_thinking(netting_set)
        thinking_steps.append(step_1_thinking)
        calculation_results['step_1'] = step_1_result
        
        step_6_thinking, step_6_result = self._step_6_with_thinking(netting_set.trades)
        thinking_steps.append(step_6_thinking)
        calculation_results['step_6'] = step_6_result
        
        step_8_thinking, step_8_result = self._step_8_with_thinking(netting_set.trades)
        thinking_steps.append(step_8_thinking)
        calculation_results['step_8'] = step_8_result
        
        step_9_thinking, step_9_result = self._step_9_with_thinking(netting_set.trades, step_6_result, step_8_result)
        thinking_steps.append(step_9_thinking)
        calculation_results['step_9'] = step_9_result
        
        step_13_thinking, step_13_result = self._step_13_with_thinking(step_9_result)
        thinking_steps.append(step_13_thinking)
        calculation_results['step_13'] = step_13_result
        
        step_14_thinking, step_14_result = self._step_14_with_thinking(netting_set, collateral)
        thinking_steps.append(step_14_thinking)
        calculation_results['step_14'] = step_14_result
        
        step_15_thinking, step_15_result = self._step_15_with_thinking(step_14_result, step_13_result)
        thinking_steps.append(step_15_thinking)
        calculation_results['step_15'] = step_15_result
        
        step_16_thinking, step_16_result = self._step_16_with_thinking(step_15_result, step_13_result)
        thinking_steps.append(step_16_thinking)
        calculation_results['step_16'] = step_16_result
        
        step_18_thinking, step_18_result = self._step_18_with_thinking(netting_set, step_14_result)
        thinking_steps.append(step_18_thinking)
        calculation_results['step_18'] = step_18_result
        
        step_21_thinking, step_21_result = self._step_21_with_thinking(step_18_result, step_16_result)
        thinking_steps.append(step_21_thinking)
        calculation_results['step_21'] = step_21_result
        
        step_24_thinking, step_24_result = self._step_24_with_thinking(netting_set.counterparty, step_21_result)
        thinking_steps.append(step_24_thinking)
        calculation_results['step_24'] = step_24_result
        
        # Generate summary
        summary = self._generate_calculation_summary(calculation_results)
        
        return {
            'thinking_steps': thinking_steps,
            'calculation_results': calculation_results,
            'data_quality_issues': quality_issues,
            'summary': summary,
            'assumptions': self.calculation_assumptions
        }
    
    def _step_1_with_thinking(self, netting_set: NettingSet) -> Tuple[Dict, Dict]:
        """Step 1: Netting Set Data with thinking process"""
        
        thinking = {
            'step': 1,
            'title': 'Portfolio Data Collection & Validation',
            'reasoning': f"""
            THINKING PROCESS:
            • Analyzing netting set: {netting_set.netting_set_id}
            • Counterparty: {netting_set.counterparty}
            • Trade count: {len(netting_set.trades)}
            • Total gross notional: ${sum(abs(t.notional) for t in netting_set.trades):,.0f}
            
            KEY OBSERVATIONS:
            • This appears to be a {"margined" if netting_set.threshold > 0 or netting_set.mta > 0 else "unmargined"} netting set
            • Threshold: ${netting_set.threshold:,.0f}, MTA: ${netting_set.mta:,.0f}
            • NICA: ${netting_set.nica:,.0f}
            """,
            'assumptions': [],
            'data_sources': 'System Arctic - Trade Repository'
        }
        
        result = {
            'netting_set_id': netting_set.netting_set_id,
            'counterparty': netting_set.counterparty,
            'trade_count': len(netting_set.trades),
            'total_notional': sum(abs(trade.notional) for trade in netting_set.trades),
            'is_margined': netting_set.threshold > 0 or netting_set.mta > 0
        }
        
        return thinking, result
    
    def _step_6_with_thinking(self, trades: List[Trade]) -> Tuple[Dict, Dict]:
        """Step 6: Maturity Factor with detailed reasoning"""
        
        maturity_calcs = []
        reasoning_details = []
        
        for trade in trades:
            maturity = trade.time_to_maturity()
            mf = math.sqrt(min(maturity, 1.0) / 1.0)
            
            maturity_calcs.append({
                'trade_id': trade.trade_id,
                'maturity_years': maturity,
                'maturity_factor': mf
            })
            
            reasoning_details.append(f"Trade {trade.trade_id}: M={maturity:.2f}y → MF=√(min({maturity:.2f},1)/1) = {mf:.6f}")
        
        thinking = {
            'step': 6,
            'title': 'Maturity Factor Calculation',
            'reasoning': f"""
            THINKING PROCESS:
            • Formula: MF = √(min(M, 1 year) / 1 year)
            • This formula reduces capital for short-term trades
            • Cap at 1 year maturity means longer trades don't get additional penalty
            
            DETAILED CALCULATIONS:
            {chr(10).join(reasoning_details)}
            
            REGULATORY RATIONALE:
            • Shorter maturity = lower potential for large market moves
            • Basel Committee recognized that 1+ year trades have similar volatility
            """,
            'formula': 'MF = √(min(M, 1 year) / 1 year)',
            'assumptions': ['Using calendar days for maturity calculation']
        }
        
        result = {
            'trade_calculations': maturity_calcs,
            'average_maturity_factor': sum(calc['maturity_factor'] for calc in maturity_calcs) / len(maturity_calcs)
        }
        
        return thinking, result
    
    def _step_8_with_thinking(self, trades: List[Trade]) -> Tuple[Dict, Dict]:
        """Step 8: Supervisory Factor with detailed lookup logic"""
        
        sf_calcs = []
        reasoning_details = []
        
        for trade in trades:
            sf_bp = self._get_supervisory_factor_with_reasoning(trade)
            sf_decimal = sf_bp / 10000
            
            sf_calcs.append({
                'trade_id': trade.trade_id,
                'asset_class': trade.asset_class.value,
                'currency': trade.currency,
                'maturity': trade.time_to_maturity(),
                'supervisory_factor_bp': sf_bp,
                'supervisory_factor_decimal': sf_decimal
            })
            
            reasoning_details.append(f"Trade {trade.trade_id}: {trade.asset_class.value} {trade.currency} {trade.time_to_maturity():.1f}y → {sf_bp}bp ({sf_decimal:.4f})")
        
        thinking = {
            'step': 8,
            'title': 'Supervisory Factor Lookup',
            'reasoning': f"""
            THINKING PROCESS:
            • Lookup supervisory factors from Basel regulatory tables
            • Factors vary by: Asset Class, Currency, Maturity bucket
            • Higher factors = higher perceived volatility = more capital
            
            DETAILED LOOKUPS:
            {chr(10).join(reasoning_details)}
            
            REGULATORY BASIS:
            • Calibrated to 99% confidence level over 1-year horizon
            • Based on historical volatility analysis by Basel Committee
            • Regular reviews ensure factors remain appropriate
            """,
            'formula': 'SF per Basel Annex 4 regulatory tables',
            'assumptions': ['Using current Basel III supervisory factors']
        }
        
        result = {'trade_calculations': sf_calcs}
        
        return thinking, result
    
    def _step_9_with_thinking(self, trades: List[Trade], maturity_result: Dict, sf_result: Dict) -> Tuple[Dict, Dict]:
        """Step 9: Adjusted Contract Amount with full formula breakdown"""
        
        adjusted_calcs = []
        reasoning_details = []
        
        for i, trade in enumerate(trades):
            adjusted_notional = abs(trade.notional)
            
            # Get pre-calculated values
            mf = maturity_result['trade_calculations'][i]['maturity_factor']
            sf = sf_result['trade_calculations'][i]['supervisory_factor_decimal']
            
            # Supervisory delta
            if trade.trade_type in [TradeType.OPTION, TradeType.SWAPTION]:
                delta = trade.delta
                if delta == 1.0:
                    self.calculation_assumptions.append(f"Trade {trade.trade_id}: Using default delta=1.0 for {trade.trade_type.value}")
            else:
                delta = 1.0 if trade.notional > 0 else -1.0
            
            # Final calculation
            adjusted_amount = adjusted_notional * delta * mf * sf
            
            adjusted_calcs.append({
                'trade_id': trade.trade_id,
                'adjusted_notional': adjusted_notional,
                'supervisory_delta': delta,
                'maturity_factor': mf,
                'supervisory_factor': sf,
                'adjusted_amount': adjusted_amount
            })
            
            reasoning_details.append(
                f"Trade {trade.trade_id}: ${adjusted_notional:,.0f} × {delta} × {mf:.6f} × {sf:.4f} = ${adjusted_amount:,.2f}"
            )
        
        thinking = {
            'step': 9,
            'title': 'Adjusted Derivatives Contract Amount',
            'reasoning': f"""
            THINKING PROCESS:
            • This is the core SA-CCR risk measure calculation
            • Formula: Effective Notional × δ × MF × SF
            • Each component serves a specific regulatory purpose
            
            COMPONENT ANALYSIS:
            • Effective Notional: Trade size exposure
            • Delta (δ): Direction and option sensitivity
            • Maturity Factor: Time-based risk scaling
            • Supervisory Factor: Asset class volatility weighting
            
            DETAILED CALCULATIONS:
            {chr(10).join(reasoning_details)}
            
            TOTAL PORTFOLIO IMPACT:
            • Sum of adjusted amounts feeds into hedging set aggregation
            • Sign of amounts allows for netting within hedging sets
            """,
            'formula': 'Adjusted Amount = Effective Notional × δ × MF × SF',
            'assumptions': self.calculation_assumptions.copy()
        }
        
        result = {
            'trade_calculations': adjusted_calcs,
            'total_adjusted_amount': sum(abs(calc['adjusted_amount']) for calc in adjusted_calcs)
        }
        
        return thinking, result
    
    def _step_13_with_thinking(self, step_9_result: Dict) -> Tuple[Dict, Dict]:
        """Step 13: Aggregate AddOn with hedging set logic"""
        
        # Simplified aggregation for demo - real SA-CCR has complex correlation formulas
        aggregate_addon = step_9_result['total_adjusted_amount']
        
        thinking = {
            'step': 13,
            'title': 'Aggregate AddOn Calculation',
            'reasoning': f"""
            THINKING PROCESS:
            • Aggregate all hedging set add-ons into portfolio total
            • This represents the potential future exposure component
            • Simplified calculation: sum of absolute adjusted amounts
            
            HEDGING SET ANALYSIS:
            • Each asset class + currency combination forms a hedging set
            • Within hedging sets: correlation-based netting applies
            • Across hedging sets: simple addition (conservative approach)
            
            CALCULATION:
            • Total Adjusted Amounts: ${step_9_result['total_adjusted_amount']:,.2f}
            • This becomes the baseline for PFE calculation
            
            REGULATORY PURPOSE:
            • Captures potential future credit exposure
            • Feeds into PFE multiplier calculation
            • Higher add-on = higher capital requirement
            """,
            'formula': 'Aggregate AddOn = Σ(Hedging Set AddOns)',
            'assumptions': ['Simplified aggregation - no cross-asset correlation adjustments applied']
        }
        
        result = {
            'aggregate_addon': aggregate_addon,
            'hedging_set_count': 1,  # Simplified
            'methodology': 'Conservative sum of absolute amounts'
        }
        
        return thinking, result
    
    def _step_14_with_thinking(self, netting_set: NettingSet, collateral: List[Collateral] = None) -> Tuple[Dict, Dict]:
        """Step 14: V and C calculation with collateral analysis"""
        
        # Calculate V (sum of MTMs)
        sum_v = sum(trade.mtm_value for trade in netting_set.trades)
        
        # Calculate C (effective collateral after haircuts)
        sum_c = 0
        collateral_details = []
        
        if collateral:
            for coll in collateral:
                haircut_pct = self.collateral_haircuts.get(coll.collateral_type, 15.0)
                effective_value = coll.amount * (1 - haircut_pct / 100)
                sum_c += effective_value
                
                collateral_details.append({
                    'type': coll.collateral_type.value,
                    'amount': coll.amount,
                    'haircut_pct': haircut_pct,
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
            • These values directly impact both RC and PFE multiplier
            
            CURRENT EXPOSURE ANALYSIS:
            • Sum of trade MTMs: ${sum_v:,.0f}
            • {'Favorable' if sum_v < 0 else 'Unfavorable' if sum_v > 0 else 'Neutral'} portfolio position
            • {"Out-of-the-money portfolio helps with PFE multiplier" if sum_v < 0 else "In-the-money portfolio increases replacement cost" if sum_v > 0 else "Zero MTM - neutral impact"}
            
            COLLATERAL ANALYSIS:
            • Total posted collateral: ${sum([c['amount'] for c in collateral_details]):,.0f if collateral_details else 0}
            • After haircuts: ${sum_c:,.0f}
            • Net exposure (V-C): ${sum_v - sum_c:,.0f}
            
            REGULATORY HAIRCUTS APPLIED:
            {chr(10).join([f"• {c['type']}: {c['haircut_pct']:.1f}% haircut" for c in collateral_details]) if collateral_details else "• No collateral - no haircuts applied"}
            """,
            'formula': 'V = Σ(Trade MTMs), C = Σ(Collateral × (1 - haircut))',
            'assumptions': ['MTM values are current and accurate', 'Collateral haircuts per Basel standards']
        }
        
        result = {
            'sum_v': sum_v,
            'sum_c': sum_c,
            'net_exposure': sum_v - sum_c,
            'collateral_details': collateral_details
        }
        
        return thinking, result
    
    def _step_15_with_thinking(self, step_14_result: Dict, step_13_result: Dict) -> Tuple[Dict, Dict]:
        """Step 15: PFE Multiplier with detailed netting benefit analysis"""
        
        sum_v = step_14_result['sum_v']
        sum_c = step_14_result['sum_c']
        aggregate_addon = step_13_result['aggregate_addon']
        
        net_exposure = sum_v - sum_c
        
        if aggregate_addon > 0:
            exponent = net_exposure / (2 * 0.95 * aggregate_addon)
            multiplier = min(1.0, 0.05 + 0.95 * math.exp(exponent))
        else:
            multiplier = 1.0
            exponent = 0
        
        # Analyze netting benefit
        netting_benefit_pct = (1 - multiplier) * 100
        
        thinking = {
            'step': 15,
            'title': 'PFE Multiplier - Netting Benefit Analysis',
            'reasoning': f"""
            THINKING PROCESS:
            • The multiplier captures netting benefits within the netting set
            • Formula: min(1, 0.05 + 0.95 × exp((V-C) / (2 × 0.95 × AddOn)))
            • Lower multiplier = more netting benefit = lower capital
            
            DETAILED CALCULATION:
            • Net Exposure (V-C): ${net_exposure:,.0f}
            • Aggregate AddOn: ${aggregate_addon:,.0f}
            • Exponent: {net_exposure:,.0f} / (2 × 0.95 × {aggregate_addon:,.0f}) = {exponent:.6f}
            • exp({exponent:.6f}) = {math.exp(exponent):.6f}
            • Multiplier: min(1, 0.05 + 0.95 × {math.exp(exponent):.6f}) = {multiplier:.6f}
            
            NETTING BENEFIT ANALYSIS:
            • Multiplier: {multiplier:.6f} ({multiplier*100:.2f}%)
            • Netting benefit: {netting_benefit_pct:.1f}%
            • {"Strong netting benefit due to negative net exposure" if multiplier < 0.3 else "Moderate netting benefit" if multiplier < 0.7 else "Limited netting benefit"}
            
            ECONOMIC INTERPRETATION:
            • {"Portfolio is out-of-the-money - favorable for capital" if net_exposure < 0 else "Portfolio is in-the-money - less favorable for capital" if net_exposure > 0 else "Portfolio is at-the-money - neutral impact"}
            • Multiplier floors at 5% (regulatory minimum netting benefit)
            """,
            'formula': 'Multiplier = min(1, 0.05 + 0.95 × exp((V-C) / (2 × 0.95 × AddOn)))',
            'assumptions': ['Current MTM values accurately reflect market conditions']
        }
        
        result = {
            'sum_v': sum_v,
            'sum_c': sum_c,
            'net_exposure': net_exposure,
            'aggregate_addon': aggregate_addon,
            'exponent': exponent,
            'multiplier': multiplier,
            'netting_benefit_pct': netting_benefit_pct
        }
        
        return thinking, result
    
    def _step_16_with_thinking(self, step_15_result: Dict, step_13_result: Dict) -> Tuple[Dict, Dict]:
        """Step 16: PFE Calculation"""
        
        multiplier = step_15_result['multiplier']
        aggregate_addon = step_13_result['aggregate_addon']
        pfe = multiplier * aggregate_addon
        
        thinking = {
            'step': 16,
            'title': 'Potential Future Exposure (PFE) Final Calculation',
            'reasoning': f"""
            THINKING PROCESS:
            • PFE = Multiplier × Aggregate AddOn
            • This represents the potential future credit exposure
            • Incorporates both volatility (AddOn) and netting benefits (Multiplier)
            
            CALCULATION:
            • Multiplier: {multiplier:.6f}
            • Aggregate AddOn: ${aggregate_addon:,.2f}
            • PFE: {multiplier:.6f} × ${aggregate_addon:,.2f} = ${pfe:,.2f}
            
            REGULATORY SIGNIFICANCE:
            • PFE captures potential future exposure over the life of trades
            • Combined with current exposure (RC) to get total EAD
            • Lower PFE = lower capital requirement
            
            PORTFOLIO INSIGHTS:
            • {"Strong netting benefits reduce PFE significantly" if multiplier < 0.5 else "Moderate netting benefits applied" if multiplier < 0.8 else "Limited netting - conservative PFE calculation"}
            """,
            'formula': 'PFE = Multiplier × Aggregate AddOn',
            'assumptions': []
        }
        
        result = {
            'multiplier': multiplier,
            'aggregate_addon': aggregate_addon,
            'pfe': pfe
        }
        
        return thinking, result
    
    def _step_18_with_thinking(self, netting_set: NettingSet, step_14_result: Dict) -> Tuple[Dict, Dict]:
        """Step 18: Replacement Cost with margining analysis"""
        
        sum_v = step_14_result['sum_v']
        sum_c = step_14_result['sum_c']
        threshold = netting_set.threshold
        mta = netting_set.mta
        nica = netting_set.nica
        
        # Determine if margined
        is_margined = threshold > 0 or mta > 0
        
        if is_margined:
            rc = max(sum_v - sum_c, threshold + mta - nica, 0)
            methodology = "Margined netting set formula"
        else:
            rc = max(sum_v - sum_c, 0)
            methodology = "Unmargined netting set formula"
        
        thinking = {
            'step': 18,
            'title': 'Replacement Cost (RC) - Current Exposure Analysis',
            'reasoning': f"""
            THINKING PROCESS:
            • RC represents current replacement cost if counterparty defaults
            • Different formulas for margined vs unmargined netting sets
            • Captures benefits of netting and collateral
            
            NETTING SET ANALYSIS:
            • Type: {"Margined" if is_margined else "Unmargined"}
            • Threshold: ${threshold:,.0f}
            • MTA: ${mta:,.0f}
            • NICA: ${nica:,.0f}
            
            CALCULATION METHOD:
            • Formula: {"RC = max(V - C, TH + MTA - NICA, 0)" if is_margined else "RC = max(V - C, 0)"}
            • Current MTM (V): ${sum_v:,.0f}
            • Effective Collateral (C): ${sum_c:,.0f}
            • Net Exposure (V-C): ${sum_v - sum_c:,.0f}
            {f"• Margin Floor (TH + MTA - NICA): ${threshold + mta - nica:,.0f}" if is_margined else ""}
            
            RESULT ANALYSIS:
            • RC = max({sum_v - sum_c:,.0f}{f", {threshold + mta - nica:,.0f}" if is_margined else ""}, 0) = ${rc:,.0f}
            • {"RC driven by margin floor - collateral terms limit benefit" if is_margined and rc == threshold + mta - nica else "RC driven by net exposure" if rc == sum_v - sum_c else "RC floored at zero"}
            """,
            'formula': methodology,
            'assumptions': ['CSA terms accurately reflect legal agreements']
        }
        
        result = {
            'sum_v': sum_v,
            'sum_c': sum_c,
            'net_exposure': sum_v - sum_c,
            'threshold': threshold,
            'mta': mta,
            'nica': nica,
            'is_margined': is_margined,
            'rc': rc,
            'methodology': methodology
        }
        
        return thinking, result
    
    def _step_21_with_thinking(self, step_18_result: Dict, step_16_result: Dict) -> Tuple[Dict, Dict]:
        """Step 21: EAD Calculation"""
        
        alpha = 1.4  # Fixed for SA-CCR
        rc = step_18_result['rc']
        pfe = step_16_result['pfe']
        ead = alpha * (rc + pfe)
        
        thinking = {
            'step': 21,
            'title': 'Exposure at Default (EAD) - Total Credit Exposure',
            'reasoning': f"""
            THINKING PROCESS:
            • EAD = Alpha × (RC + PFE)
            • Alpha = 1.4 (fixed regulatory multiplier for SA-CCR)
            • EAD represents total potential credit exposure
            
            COMPONENT BREAKDOWN:
            • Current Exposure (RC): ${rc:,.0f}
            • Future Exposure (PFE): ${pfe:,.0f}
            • Combined Exposure: ${rc + pfe:,.0f}
            • Alpha Multiplier: {alpha}
            
            CALCULATION:
            • EAD = {alpha} × (${rc:,.0f} + ${pfe:,.0f})
            • EAD = {alpha} × ${rc + pfe:,.0f}
            • EAD = ${ead:,.0f}
            
            EXPOSURE ANALYSIS:
            • Current vs Future: RC={rc/(rc+pfe)*100:.1f}%, PFE={pfe/(rc+pfe)*100:.1f}%
            • {"Current exposure dominates" if rc > pfe else "Future exposure dominates" if pfe > rc else "Balanced current/future exposure"}
            
            REGULATORY PURPOSE:
            • EAD feeds directly into RWA calculation
            • Alpha ensures consistent calibration across banks
            • Higher EAD = higher capital requirement
            """,
            'formula': 'EAD = Alpha × (RC + PFE), where Alpha = 1.4',
            'assumptions': ['Alpha = 1.4 per Basel SA-CCR regulation']
        }
        
        result = {
            'alpha': alpha,
            'rc': rc,
            'pfe': pfe,
            'combined_exposure': rc + pfe,
            'ead': ead,
            'rc_percentage': rc / (rc + pfe) * 100 if (rc + pfe) > 0 else 0,
            'pfe_percentage': pfe / (rc + pfe) * 100 if (rc + pfe) > 0 else 0
        }
        
        return thinking, result
    
    def _step_24_with_thinking(self, counterparty: str, step_21_result: Dict) -> Tuple[Dict, Dict]:
        """Step 24: RWA Calculation with counterparty analysis"""
        
        ead = step_21_result['ead']
        
        # Simplified counterparty risk weight determination
        # In practice, this would involve detailed counterparty analysis
        risk_weight = self._determine_risk_weight(counterparty)
        rwa = ead * risk_weight
        capital_requirement = rwa * 0.08  # 8% capital ratio
        
        thinking = {
            'step': 24,
            'title': 'Risk-Weighted Assets (RWA) and Capital Calculation',
            'reasoning': f"""
            THINKING PROCESS:
            • RWA = Risk Weight × EAD
            • Risk weight determined by counterparty creditworthiness
            • Final capital requirement = RWA × 8%
            
            COUNTERPARTY ANALYSIS:
            • Counterparty: {counterparty}
            • Risk Category: Corporate (assumed)
            • Risk Weight: {risk_weight*100:.0f}%
            • Regulatory Basis: Basel III Standardized Approach
            
            FINAL CALCULATION:
            • EAD: ${ead:,.0f}
            • Risk Weight: {risk_weight*100:.0f}%
            • RWA = ${ead:,.0f} × {risk_weight} = ${rwa:,.0f}
            • Capital Requirement = ${rwa:,.0f} × 8% = ${capital_requirement:,.0f}
            
            CAPITAL IMPACT:
            • This represents the minimum regulatory capital required
            • Actual bank capital requirements may include additional buffers
            • {"High capital requirement - consider optimization strategies" if capital_requirement > 1000000 else "Moderate capital requirement" if capital_requirement > 100000 else "Low capital requirement"}
            """,
            'formula': 'RWA = Risk Weight × EAD, Capital = RWA × 8%',
            'assumptions': [f'Risk weight of {risk_weight*100:.0f}% assumed for corporate counterparty']
        }
        
        result = {
            'counterparty': counterparty,
            'ead': ead,
            'risk_weight': risk_weight,
            'risk_weight_pct': risk_weight * 100,
            'rwa': rwa,
            'capital_requirement': capital_requirement,
            'capital_ratio': 0.08
        }
        
        return thinking, result
    
    def _generate_calculation_summary(self, calculation_results: Dict) -> Dict:
        """Generate bulleted summary of key results"""
        
        summary = {
            'key_inputs': [
                f"Portfolio: {calculation_results['step_1']['trade_count']} trades, ${calculation_results['step_1']['total_notional']:,.0f} notional",
                f"Counterparty: {calculation_results['step_1']['counterparty']}",
                f"Netting: {'Margined' if calculation_results['step_1']['is_margined'] else 'Unmargined'} set"
            ],
            'risk_components': [
                f"Aggregate Add-On: ${calculation_results['step_13']['aggregate_addon']:,.0f}",
                f"PFE Multiplier: {calculation_results['step_15']['multiplier']:.4f} ({calculation_results['step_15']['netting_benefit_pct']:.1f}% netting benefit)",
                f"Potential Future Exposure: ${calculation_results['step_16']['pfe']:,.0f}",
                f"Replacement Cost: ${calculation_results['step_18']['rc']:,.0f}",
                f"Current vs Future Exposure: {calculation_results['step_21']['rc_percentage']:.1f}% / {calculation_results['step_21']['pfe_percentage']:.1f}%"
            ],
            'capital_results': [
                f"Exposure at Default (EAD): ${calculation_results['step_21']['ead']:,.0f}",
                f"Risk Weight: {calculation_results['step_24']['risk_weight_pct']:.0f}%",
                f"Risk-Weighted Assets: ${calculation_results['step_24']['rwa']:,.0f}",
                f"Minimum Capital Required: ${calculation_results['step_24']['capital_requirement']:,.0f}",
                f"Capital Efficiency: {(calculation_results['step_24']['capital_requirement']/calculation_results['step_1']['total_notional']*100):.3f}% of notional"
            ],
            'optimization_insights': [
                "Netting benefits reduce PFE by " + f"{calculation_results['step_15']['netting_benefit_pct']:.1f}%",
                "Current exposure " + ("dominates" if calculation_results['step_21']['rc'] > calculation_results['step_21']['pfe'] else "is secondary to") + " future exposure",
                "Consider " + ("improving collateral terms" if calculation_results['step_14']['sum_c'] == 0 else "optimizing collateral mix") + " for RC reduction"
            ]
        }
        
        return summary
    
    def _get_supervisory_factor_with_reasoning(self, trade: Trade) -> float:
        """Get supervisory factor with detailed reasoning"""
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
    
    def _determine_risk_weight(self, counterparty: str) -> float:
        """Determine risk weight - simplified logic"""
        # In practice, this would involve credit rating lookup
        return 1.0  # 100% for corporate
    
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

# ==============================================================================
# ENHANCED STREAMLIT APPLICATION
# ==============================================================================

def main():
    # AI-Powered Header
    st.markdown("""
    <div class="ai-header">
        <div class="executive-title">🤖 Enhanced AI SA-CCR Platform</div>
        <div class="executive-subtitle">Step-by-Step Analysis with Interactive Data Gathering</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize enhanced agent
    if 'enhanced_saccr_agent' not in st.session_state:
        st.session_state.enhanced_saccr_agent = EnhancedSACCRAgent()
    
    # Sidebar configuration (same as before)
    with st.sidebar:
        st.markdown("### 🤖 LLM Configuration")
        
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
                
                success = st.session_state.enhanced_saccr_agent.setup_llm_connection(config)
                if success:
                    st.success("✅ LLM Connected!")
                else:
                    st.error("❌ Connection Failed")
        
        # Connection status
        status = st.session_state.enhanced_saccr_agent.connection_status
        if status == "connected":
            st.markdown('<div class="connection-status connected">🟢 LLM Connected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="connection-status disconnected">🔴 LLM Disconnected</div>', unsafe_allow_html=True)
    
    # Main application
    enhanced_calculator()

def enhanced_calculator():
    """Enhanced calculator with step-by-step analysis"""
    
    st.markdown("## 🧮 Enhanced SA-CCR Calculator with Step-by-Step Analysis")
    
    # Load reference example button
    if st.button("📋 Load Reference Example", type="secondary"):
        load_reference_example()
        st.rerun()
    
    # Input section (simplified for demo)
    with st.expander("📊 Portfolio Input", expanded=True):
        if 'trades_input' not in st.session_state:
            st.session_state.trades_input = []
        
        # Show current trades
        if st.session_state.trades_input:
            st.markdown("**Current Portfolio:**")
            for i, trade in enumerate(st.session_state.trades_input):
                st.write(f"{i+1}. {trade.trade_id}: ${trade.notional:,.0f} {trade.currency} {trade.asset_class.value} {trade.trade_type.value}")
    
    # Data quality analysis button
    if st.session_state.trades_input:
        if st.button("🔍 Analyze Data Quality & Calculate SA-CCR", type="primary"):
            
            # Create netting set from current trades
            netting_set = NettingSet(
                netting_set_id="212784060000009618701",
                counterparty="Lowell Hotel Properties LLC",
                trades=st.session_state.trades_input,
                threshold=12000000,
                mta=1000000,
                nica=0
            )
            
            # Perform enhanced calculation
            with st.spinner("🧮 Performing step-by-step SA-CCR analysis..."):
                result = st.session_state.enhanced_saccr_agent.calculate_with_thinking_process(netting_set, [])
                
                # Display results
                display_enhanced_results(result)

def load_reference_example():
    """Load the reference example"""
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

def display_enhanced_results(result: Dict):
    """Display enhanced results with step-by-step thinking"""
    
    # Data Quality Issues First
    if result['data_quality_issues']:
        st.markdown("### ⚠️ Data Quality Analysis")
        
        high_impact = [issue for issue in result['data_quality_issues'] if issue.impact == 'high']
        
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
                    Current: {issue.current_value}<br>
                    Impact: {issue.recommendation}<br>
                    Default Used: {issue.default_used}
                </div>
                """, unsafe_allow_html=True)
        
        # Interactive prompting
        st.markdown("#### 💬 Provide Missing Information")
        
        missing_info_prompts = st.session_state.enhanced_saccr_agent.gather_missing_information(result['data_quality_issues'])
        
        for prompt in missing_info_prompts['missing_info_prompts']:
            st.markdown(f"**{prompt['field']}:** {prompt['question']}")
            user_input = st.text_input(f"Enter {prompt['field']}", key=f"input_{prompt['field']}")
            if user_input:
                st.info(f"Impact: {prompt['impact']}")
    
    # Step-by-Step Thinking Process
    st.markdown("### 🧠 Step-by-Step Calculation Analysis")
    
    for thinking_step in result['thinking_steps']:
        with st.expander(f"🔍 Step {thinking_step['step']}: {thinking_step['title']}", expanded=False):
            
            st.markdown(f"""
            <div class="thinking-process">
                <div class="step-reasoning">
                    <pre>{thinking_step['reasoning']}</pre>
                </div>
                
                <div class="formula-breakdown">
                    <strong>Formula:</strong> {thinking_step['formula']}
                </div>
                
                {f"<div class='calculation-detail'><strong>Assumptions:</strong><br>• " + "<br>• ".join(thinking_step.get('assumptions', [])) + "</div>" if thinking_step.get('assumptions') else ""}
            </div>
            """, unsafe_allow_html=True)
    
    # Results Summary
    st.markdown("### 📊 Calculation Results Summary")
    
    summary = result['summary']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📋 Key Inputs")
        for item in summary['key_inputs']:
            st.write(f"• {item}")
    
    with col2:
        st.markdown("#### ⚡ Risk Components")
        for item in summary['risk_components']:
            st.write(f"• {item}")
    
    with col3:
        st.markdown("#### 💰 Capital Results")
        for item in summary['capital_results']:
            st.write(f"• {item}")
    
    # Final Results
    final_ead = result['calculation_results']['step_21']['ead']
    final_rwa = result['calculation_results']['step_24']['rwa']
    final_capital = result['calculation_results']['step_24']['capital_requirement']
    
    st.markdown(f"""
    <div class="result-summary">
        <h3>🎯 Final SA-CCR Results</h3>
        <div style="display: flex; justify-content: space-around; margin-top: 1rem;">
            <div style="text-align: center;">
                <div style="font-size: 1.5rem; font-weight: bold;">EAD</div>
                <div style="font-size: 1.2rem;">${final_ead:,.0f}</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.5rem; font-weight: bold;">RWA</div>
                <div style="font-size: 1.2rem;">${final_rwa:,.0f}</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.5rem; font-weight: bold;">Capital Required</div>
                <div style="font-size: 1.2rem;">${final_capital:,.0f}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Optimization Insights
    st.markdown("### 💡 Optimization Insights")
    for insight in summary['optimization_insights']:
        st.write(f"• {insight}")
    
    # Interactive AI Chat for Missing Info
    if result['data_quality_issues']:
        st.markdown("### 🤖 AI Assistant - Improve Data Quality")
        
        user_question = st.text_area(
            "Ask the AI how to improve your data for more accurate RWA calculation:",
            placeholder="e.g., How can I get better MTM values for my trades? What collateral information do you need?",
            height=80
        )
        
        if st.button("💬 Get AI Guidance") and user_question:
            if st.session_state.enhanced_saccr_agent.llm and st.session_state.enhanced_saccr_agent.connection_status == "connected":
                
                # Generate context-aware response
                system_prompt = """You are a SA-CCR data quality expert. Help users understand how to gather missing information to improve their RWA calculation accuracy. Focus on practical, actionable guidance."""
                
                missing_info_context = json.dumps([
                    {
                        'field': issue.field_name,
                        'impact': issue.impact,
                        'recommendation': issue.recommendation
                    } for issue in result['data_quality_issues']
                ], indent=2)
                
                user_prompt = f"""
                Data Quality Issues Identified:
                {missing_info_context}
                
                User Question: {user_question}
                
                Please provide specific, actionable guidance on:
                1. Where to find this information
                2. How it impacts the SA-CCR calculation
                3. What level of accuracy is needed
                4. Practical steps to gather the data
                """
                
                with st.spinner("🤖 AI is analyzing your data quality question..."):
                    try:
                        response = st.session_state.enhanced_saccr_agent.llm.invoke([
                            SystemMessage(content=system_prompt),
                            HumanMessage(content=user_prompt)
                        ])
                        
                        st.markdown(f"""
                        <div class="ai-response">
                            <strong>🤖 AI Data Quality Guidance:</strong><br><br>
                            {response.content}
                        </div>
                        """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"AI response error: {str(e)}")

if __name__ == "__main__":
    main()
