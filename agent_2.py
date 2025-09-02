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
try:
    from langchain_openai import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    st.warning("LangChain not available. AI features will be disabled.")

# ==============================================================================
# ENTERPRISE UI CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="AI SA-CCR Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for AI-powered features
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
    
    .error-message {
        background: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
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
        return max(0.0, (self.maturity_date - datetime.now()).days / 365.25)

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

# ==============================================================================
# COMPREHENSIVE SA-CCR AGENT
# ==============================================================================

class ComprehensiveSACCRAgent:
    """Complete SA-CCR Agent following all 24 Basel regulatory steps"""
    
    def __init__(self):
        self.llm = None
        self.connection_status = "disconnected"
        
        # Initialize regulatory parameters
        self.supervisory_factors = self._initialize_supervisory_factors()
        self.supervisory_correlations = self._initialize_correlations()
        self.collateral_haircuts = self._initialize_collateral_haircuts()
        
    def setup_llm_connection(self, config: Dict) -> bool:
        """Setup LangChain ChatOpenAI connection"""
        if not LANGCHAIN_AVAILABLE:
            st.error("LangChain is not available. Please install: pip install langchain-openai")
            self.connection_status = "unavailable"
            return False
            
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
                'USD': {'<2y': 50, '2-5y': 50, '>5y': 150},  # Fixed: converted to basis points
                'EUR': {'<2y': 50, '2-5y': 50, '>5y': 150},
                'JPY': {'<2y': 50, '2-5y': 50, '>5y': 150},
                'GBP': {'<2y': 50, '2-5y': 50, '>5y': 150},
                'other': {'<2y': 150, '2-5y': 150, '>5y': 150}
            },
            AssetClass.FOREIGN_EXCHANGE: {'G10': 400, 'emerging': 1500},
            AssetClass.CREDIT: {
                'IG_single': 46, 'HY_single': 130, 
                'IG_index': 38, 'HY_index': 106
            },
            AssetClass.EQUITY: {
                'single_large': 3200, 'single_small': 4000,
                'index_developed': 2000, 'index_emerging': 2500
            },
            AssetClass.COMMODITY: {
                'energy': 1800, 'metals': 1800, 'agriculture': 1800, 'other': 1800
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
        """Calculate SA-CCR following complete 24-step workflow"""
        
        try:
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
            
            # Step 6: Maturity Factor
            step6_data = self._step6_maturity_factor(netting_set.trades)
            calculation_steps.append(step6_data)
            
            # Step 7: Supervisory Delta
            step7_data = self._step7_supervisory_delta(netting_set.trades)
            calculation_steps.append(step7_data)
            
            # Step 8: Supervisory Factor
            step8_data = self._step8_supervisory_factor(netting_set.trades)
            calculation_steps.append(step8_data)
            
            # Step 9: Adjusted Derivatives Contract Amount
            step9_data = self._step9_adjusted_derivatives_contract_amount(netting_set.trades)
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
            
            # Step 13: Aggregate AddOn
            step13_data = self._step13_aggregate_addon(netting_set.trades)
            calculation_steps.append(step13_data)
            
            # Step 14: Sum of V, C
            step14_data = self._step14_sum_v_c(netting_set, collateral)
            calculation_steps.append(step14_data)
            
            # Step 15: PFE Multiplier
            step15_data = self._step15_pfe_multiplier(netting_set, step13_data['aggregate_addon'])
            calculation_steps.append(step15_data)
            
            # Step 16: PFE
            step16_data = self._step16_pfe(step15_data['multiplier'], step13_data['aggregate_addon'])
            calculation_steps.append(step16_data)
            
            # Step 17: TH, MTA, NICA
            step17_data = self._step17_th_mta_nica(netting_set)
            calculation_steps.append(step17_data)
            
            # Step 18: RC
            step18_data = self._step18_replacement_cost(netting_set, collateral, step17_data)
            calculation_steps.append(step18_data)
            
            # Step 19: CEU Flag
            step19_data = self._step19_ceu_flag(netting_set.trades)
            calculation_steps.append(step19_data)
            
            # Step 20: Alpha
            step20_data = self._step20_alpha(step19_data['ceu_flag'])
            calculation_steps.append(step20_data)
            
            # Step 21: EAD
            step21_data = self._step21_ead(step20_data['alpha'], step18_data['rc'], step16_data['pfe'])
            calculation_steps.append(step21_data)
            
            # Step 22: Counterparty Information
            step22_data = self._step22_counterparty_info(netting_set.counterparty)
            calculation_steps.append(step22_data)
            
            # Step 23: Risk Weight
            step23_data = self._step23_risk_weight(step22_data['counterparty_type'])
            calculation_steps.append(step23_data)
            
            # Step 24: RWA Calculation
            step24_data = self._step24_rwa_calculation(step21_data['ead'], step23_data['risk_weight'])
            calculation_steps.append(step24_data)
            
            # Generate AI explanation if connected
            ai_explanation = None
            if self.llm and self.connection_status == "connected":
                try:
                    ai_explanation = self._generate_saccr_explanation(calculation_steps)
                except Exception as e:
                    st.warning(f"AI explanation failed: {str(e)}")
            
            return {
                'calculation_steps': calculation_steps,
                'final_results': {
                    'replacement_cost': step18_data['rc'],
                    'potential_future_exposure': step16_data['pfe'],
                    'exposure_at_default': step21_data['ead'],
                    'risk_weighted_assets': step24_data['rwa'],
                    'capital_requirement': step24_data['rwa'] * 0.08
                },
                'ai_explanation': ai_explanation
            }
            
        except Exception as e:
            raise Exception(f"SA-CCR calculation error: {str(e)}")
    
    # Implementation of all 24 SA-CCR calculation steps
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
            'formula': 'Adjusted Notional = |Notional|',
            'result': f"Calculated adjusted notionals for {len(trades)} trades"
        }
    
    def _step6_maturity_factor(self, trades: List[Trade]) -> Dict:
        maturity_factors = []
        for trade in trades:
            remaining_maturity = trade.time_to_maturity()
            mf = min(1.0, 0.05 + 0.95 * math.exp(-0.05 * max(1.0, remaining_maturity)))
            
            maturity_factors.append({
                'trade_id': trade.trade_id,
                'remaining_maturity': remaining_maturity,
                'maturity_factor': mf
            })
        
        return {
            'step': 6,
            'title': 'Maturity Factor (MF)',
            'description': 'Apply Basel maturity factor formula',
            'data': maturity_factors,
            'formula': 'MF = min(1, 0.05 + 0.95 × exp(-0.05 × max(1, M)))',
            'result': f"Calculated maturity factors for {len(trades)} trades"
        }
    
    def _step7_supervisory_delta(self, trades: List[Trade]) -> Dict:
        supervisory_deltas = []
        for trade in trades:
            if trade.trade_type in [TradeType.OPTION, TradeType.SWAPTION]:
                supervisory_delta = trade.delta
            else:
                supervisory_delta = 1.0
                
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
            'formula': 'δ = trade delta for options, 1.0 for linear products',
            'result': f"Calculated supervisory deltas for {len(trades)} trades"
        }
    
    def _step8_supervisory_factor(self, trades: List[Trade]) -> Dict:
        supervisory_factors = []
        for trade in trades:
            sf = self._get_supervisory_factor(trade)
            supervisory_factors.append({
                'trade_id': trade.trade_id,
                'asset_class': trade.asset_class.value,
                'supervisory_factor_bp': sf,
                'supervisory_factor_decimal': sf / 10000  # Fixed: convert basis points to decimal
            })
        
        return {
            'step': 8,
            'title': 'Supervisory Factor (SF)', 
            'description': 'Apply regulatory supervisory factors by asset class',
            'data': supervisory_factors,
            'formula': 'SF per Basel regulatory mapping tables',
            'result': f"Applied supervisory factors for {len(trades)} trades"
        }
    
    def _step9_adjusted_derivatives_contract_amount(self, trades: List[Trade]) -> Dict:
        adjusted_amounts = []
        for trade in trades:
            adjusted_notional = abs(trade.notional)
            supervisory_delta = trade.delta if trade.trade_type in [TradeType.OPTION, TradeType.SWAPTION] else 1.0
            remaining_maturity = trade.time_to_maturity()
            mf = min(1.0, 0.05 + 0.95 * math.exp(-0.05 * max(1.0, remaining_maturity)))
            sf = self._get_supervisory_factor(trade) / 10000  # Fixed: convert basis points to decimal
            
            adjusted_amount = adjusted_notional * abs(supervisory_delta) * mf * sf
            
            adjusted_amounts.append({
                'trade_id': trade.trade_id,
                'adjusted_notional': adjusted_notional,
                'supervisory_delta': supervisory_delta,
                'maturity_factor': mf,
                'supervisory_factor': sf,
                'adjusted_derivatives_contract_amount': adjusted_amount
            })
        
        return {
            'step': 9,
            'title': 'Adjusted Derivatives Contract Amount',
            'description': 'Calculate final adjusted contract amounts',
            'data': adjusted_amounts,
            'formula': 'Adjusted Amount = Adjusted Notional × δ × MF × SF',
            'result': f"Calculated adjusted amounts for {len(trades)} trades"
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
