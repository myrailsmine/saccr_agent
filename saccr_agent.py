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
        ai_explanation = self._generate_saccr_explanation(calculation_steps) if self.llm and self.connection_status == "connected" else None
        
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
            'formula': 'Adjusted Notional = Notional × Supervisory Duration × Supervisory Factor',
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
                'supervisory_factor_decimal': sf / 100
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
            sf = self._get_supervisory_factor(trade) / 100
            
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
    
    def _step11_hedging_set_addon(self, trades: List[Trade]) -> Dict:
        hedging_sets = {}
        for trade in trades:
            hedging_set_key = f"{trade.asset_class.value}_{trade.currency}"
            if hedging_set_key not in hedging_sets:
                hedging_sets[hedging_set_key] = []
            hedging_sets[hedging_set_key].append(trade)
        
        hedging_set_addons = []
        for hedging_set_key, set_trades in hedging_sets.items():
            trade_addons = []
            for trade in set_trades:
                adjusted_notional = abs(trade.notional)
                supervisory_delta = trade.delta if trade.trade_type in [TradeType.OPTION, TradeType.SWAPTION] else 1.0
                remaining_maturity = trade.time_to_maturity()
                mf = min(1.0, 0.05 + 0.95 * math.exp(-0.05 * max(1.0, remaining_maturity)))
                sf = self._get_supervisory_factor(trade) / 100
                
                trade_addon = adjusted_notional * abs(supervisory_delta) * mf * sf
                trade_addons.append(trade_addon)
            
            correlation = self.supervisory_correlations.get(set_trades[0].asset_class, 0.5)
            hedging_set_addon = sum(trade_addons) * math.sqrt(correlation)
            
            hedging_set_addons.append({
                'hedging_set': hedging_set_key,
                'trade_count': len(set_trades),
                'individual_addons': trade_addons,
                'correlation': correlation,
                'hedging_set_addon': hedging_set_addon
            })
        
        return {
            'step': 11,
            'title': 'Hedging Set AddOn',
            'description': 'Aggregate trade add-ons within hedging sets',
            'data': hedging_set_addons,
            'formula': 'Hedging Set AddOn = Σ(Trade AddOns) × √ρ',
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
        for asset_class, hedging_set_addons in asset_class_addons.items():
            asset_class_addon = sum(hedging_set_addons)
            asset_class_results.append({
                'asset_class': asset_class,
                'hedging_set_addons': hedging_set_addons,
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
    
    def _step13_aggregate_addon(self, trades: List[Trade]) -> Dict:
        step12_result = self._step12_asset_class_addon(trades)
        
        aggregate_addon = sum(ac_data['asset_class_addon'] for ac_data in step12_result['data'])
        
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
            'aggregate_addon': aggregate_addon
        }
    
    def _step14_sum_v_c(self, netting_set: NettingSet, collateral: List[Collateral] = None) -> Dict:
        sum_v = sum(trade.mtm_value for trade in netting_set.trades)
        
        sum_c = 0
        if collateral:
            for coll in collateral:
                haircut = self.collateral_haircuts.get(coll.collateral_type, 15.0) / 100
                effective_value = coll.amount * (1 - haircut)
                sum_c += effective_value
        
        return {
            'step': 14,
            'title': 'Sum of V, C within netting set',
            'description': 'Calculate sum of MTM values and effective collateral',
            'data': {
                'sum_v_mtm': sum_v,
                'sum_c_collateral': sum_c,
                'collateral_details': [(coll.collateral_type.value, coll.amount, 
                                      self.collateral_haircuts.get(coll.collateral_type, 15.0))
                                     for coll in (collateral or [])]
            },
            'formula': 'V = Σ(MTM values), C = Σ(Collateral × (1 - haircut))',
            'result': f"Sum V: ${sum_v:,.0f}, Sum C: ${sum_c:,.0f}",
            'sum_v': sum_v,
            'sum_c': sum_c
        }
    
    def _step15_pfe_multiplier(self, netting_set: NettingSet, aggregate_addon: float) -> Dict:
        sum_v = sum(trade.mtm_value for trade in netting_set.trades)
        
        if aggregate_addon > 0:
            multiplier = min(1.0, 0.05 + 0.95 * math.exp(-0.05 * max(0, sum_v) / aggregate_addon))
        else:
            multiplier = 1.0
        
        return {
            'step': 15,
            'title': 'PFE Multiplier',
            'description': 'Calculate PFE multiplier based on netting benefit',
            'data': {
                'sum_v': sum_v,
                'aggregate_addon': aggregate_addon,
                'multiplier': multiplier
            },
            'formula': 'Multiplier = min(1, 0.05 + 0.95 × exp(-0.05 × max(0, V) / AddOn))',
            'result': f"PFE Multiplier: {multiplier:.6f}",
            'multiplier': multiplier
        }
    
    def _step16_pfe(self, multiplier: float, aggregate_addon: float) -> Dict:
        pfe = multiplier * aggregate_addon
        
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
            'pfe': pfe
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
    
    def _step18_replacement_cost(self, netting_set: NettingSet, collateral: List[Collateral], step17_data: Dict) -> Dict:
        sum_v = sum(trade.mtm_value for trade in netting_set.trades)
        
        sum_c = 0
        if collateral:
            for coll in collateral:
                haircut = self.collateral_haircuts.get(coll.collateral_type, 15.0) / 100
                effective_value = coll.amount * (1 - haircut)
                sum_c += effective_value
        
        threshold = step17_data['threshold']
        mta = step17_data['mta']
        nica = step17_data['nica']
        
        rc_margined = max(sum_v - sum_c, threshold + mta - nica, 0)
        rc_unmargined = max(sum_v - sum_c, 0)
        
        is_margined = threshold > 0 or mta > 0
        rc = rc_margined if is_margined else rc_unmargined
        
        return {
            'step': 18,
            'title': 'RC (Replacement Cost)',
            'description': 'Calculate replacement cost with netting and collateral benefits',
            'data': {
                'sum_v': sum_v,
                'sum_c': sum_c,
                'threshold': threshold,
                'mta': mta,
                'nica': nica,
                'rc_margined': rc_margined,
                'rc_unmargined': rc_unmargined,
                'is_margined': is_margined,
                'rc_final': rc
            },
            'formula': 'RC = max(V - C; TH + MTA - NICA; 0) for margined sets',
            'result': f"RC: ${rc:,.0f}",
            'rc': rc
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
        alpha = 1.4 if ceu_flag == 1 else 0.5
        
        return {
            'step': 20,
            'title': 'Alpha',
            'description': 'Regulatory multiplier based on central clearing status',
            'data': {
                'ceu_flag': ceu_flag,
                'alpha': alpha
            },
            'formula': 'Alpha = 1.4 if CEU=1, 0.5 if CEU=0',
            'result': f"Alpha: {alpha}",
            'alpha': alpha
        }
    
    def _step21_ead(self, alpha: float, rc: float, pfe: float) -> Dict:
        ead = alpha * (rc + pfe)
        
        return {
            'step': 21,
            'title': 'EAD (Exposure at Default)',
            'description': 'Calculate final exposure at default',
            'data': {
                'alpha': alpha,
                'rc': rc,
                'pfe': pfe,
                'ead': ead
            },
            'formula': 'EAD = Alpha × (RC + PFE)',
            'result': f"EAD: ${ead:,.0f}",
            'ead': ead
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
    
    def _step24_rwa_calculation(self, ead: float, risk_weight: float) -> Dict:
        rwa = ead * risk_weight
        
        return {
            'step': 24,
            'title': 'RWA Calculation',
            'description': 'Calculate Risk Weighted Assets',
            'data': {
                'ead': ead,
                'risk_weight': risk_weight,
                'rwa': rwa
            },
            'formula': 'Standardized RWA = RW × EAD',
            'result': f"RWA: ${rwa:,.0f}",
            'rwa': rwa
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
    
    def _generate_saccr_explanation(self, calculation_steps: List[Dict]) -> str:
        """Generate AI explanation of the SA-CCR calculation"""
        final_results = {
            step['title']: step.get('result', 'N/A') 
            for step in calculation_steps[-5:]
        }
        
        system_prompt = """You are a Basel SA-CCR regulatory expert. Analyze the complete 24-step SA-CCR calculation and provide:
        1. Executive summary of key results
        2. Main risk drivers identified
        3. Regulatory compliance assessment  
        4. Optimization recommendations
        
        Focus on practical insights for risk managers."""
        
        user_prompt = f"""
        Complete SA-CCR calculation has been performed following all 24 Basel regulatory steps.
        
        Key Final Results:
        {json.dumps(final_results, indent=2)}
        
        Please provide executive-level analysis focusing on:
        - What drives the capital requirement
        - Portfolio risk characteristics
        - Potential optimization opportunities
        - Regulatory compliance status
        """
        
        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            return response.content
        except Exception as e:
            return f"AI analysis temporarily unavailable: {str(e)}"

# ==============================================================================
# STREAMLIT APPLICATION
# ==============================================================================

def main():
    # AI-Powered Header
    st.markdown("""
    <div class="ai-header">
        <div class="executive-title">🤖 AI SA-CCR Platform</div>
        <div class="executive-subtitle">Complete 24-Step Basel SA-CCR Calculator with LLM Integration</div>
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
            ["🧮 Complete SA-CCR Calculator", "📋 Reference Example", "🤖 AI Assistant", "📊 Portfolio Analysis"]
        )
    
    # Route to different pages
    if page == "🧮 Complete SA-CCR Calculator":
        complete_saccr_calculator()
    elif page == "📋 Reference Example":
        show_reference_example()
    elif page == "🤖 AI Assistant":
        ai_assistant_page()
    elif page == "📊 Portfolio Analysis":
        portfolio_analysis_page()

def complete_saccr_calculator():
    """Complete 24-step SA-CCR calculator with input validation"""
    
    st.markdown("## 🧮 Complete SA-CCR Calculator")
    st.markdown("*Following the complete 24-step Basel regulatory framework*")
    
    # Step 1: Netting Set Setup
    with st.expander("📊 Step 1: Netting Set Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            netting_set_id = st.text_input("Netting Set ID*", placeholder="e.g., 212784060000009618701")
            counterparty = st.text_input("Counterparty*", placeholder="e.g., Lowell Hotel Properties LLC")
            
        with col2:
            threshold = st.number_input("Threshold ($)*", min_value=0.0, value=1000000.0, step=100000.0)
            mta = st.number_input("MTA ($)*", min_value=0.0, value=500000.0, step=50000.0)
            nica = st.number_input("NICA ($)", min_value=0.0, value=0.0, step=10000.0)
    
    # Step 2: Trade Input
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
            notional = st.number_input("Notional ($)*", min_value=0.0, value=100000000.0, step=1000000.0)
            currency = st.selectbox("Currency*", ["USD", "EUR", "GBP", "JPY", "CHF", "CAD"])
            underlying = st.text_input("Underlying*", placeholder="e.g., Interest rate")
        
        with col3:
            maturity_years = st.number_input("Maturity (Years)*", min_value=0.1, max_value=30.0, value=5.0, step=0.1)
            mtm_value = st.number_input("MTM Value ($)", value=0.0, step=10000.0)
            delta = st.number_input("Delta (for options)", min_value=-1.0, max_value=1.0, value=1.0, step=0.1)
        
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
    
    # Display current trades
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
    
    # Step 3: Collateral Input
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
    
    # Validation and Calculation
    if st.button("🚀 Calculate Complete SA-CCR", type="primary"):
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
            
            st.markdown("### 📝 Please Provide Missing Information")
            st.markdown("The system has identified missing required fields above. Please fill them in and try again.")
            return
        
        if validation['warnings']:
            st.warning("⚠️ Warnings (calculation will proceed with defaults):")
            for warning in validation['warnings']:
                st.write(f"   • {warning}")
        
        # Perform calculation
        with st.spinner("🧮 Performing complete 24-step SA-CCR calculation..."):
            try:
                result = st.session_state.saccr_agent.calculate_comprehensive_saccr(
                    netting_set, st.session_state.collateral_input
                )
                
                # Display results
                display_saccr_results(result)
                
            except Exception as e:
                st.error(f"❌ Calculation error: {str(e)}")
