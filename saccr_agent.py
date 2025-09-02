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

# ==============================================================================
# ENTERPRISE UI CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="Basel Risk Capital Platform",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enterprise-grade CSS styling
st.markdown("""
<style>
    /* Import professional fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .main {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Executive dashboard header */
    .executive-header {
        background: linear-gradient(135deg, #0f4c75 0%, #3282b8 100%);
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
    
    /* KPI Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .kpi-card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #e1e8ed;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.12);
    }
    
    .kpi-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0f4c75;
        margin-bottom: 0.5rem;
    }
    
    .kpi-label {
        font-size: 1rem;
        color: #657786;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .kpi-change {
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    .positive { color: #28a745; }
    .negative { color: #dc3545; }
    .neutral { color: #6c757d; }
    
    /* Calculation Steps */
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
    
    /* Results highlighting */
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
    
    /* Warning alerts */
    .risk-alert {
        background: linear-gradient(135deg, #dc3545, #fd7e14);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    /* Professional tables */
    .dataframe {
        border: none !important;
    }
    
    .dataframe th {
        background: #0f4c75 !important;
        color: white !important;
        font-weight: 600 !important;
        text-align: center !important;
    }
    
    .dataframe td {
        text-align: center !important;
        padding: 12px !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Metric containers */
    .metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 0;
        border-bottom: 1px solid #e1e8ed;
    }
    
    .metric-label {
        font-weight: 500;
        color: #495057;
    }
    
    .metric-value {
        font-weight: 700;
        color: #0f4c75;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# CORE BASEL RISK CLASSES (Enhanced)
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
    
    def time_to_maturity(self) -> float:
        return max(0, (self.maturity_date - datetime.now()).days / 365.25)

@dataclass
class CreditFacility:
    facility_id: str
    counterparty: str
    facility_type: str
    approved_limit: float
    funded_amount: float
    currency: str
    term_years: float
    ccf: float = 0.0
    risk_weight: float = 1.0
    collateral_value: float = 0.0

@dataclass
class Collateral:
    collateral_type: CollateralType
    currency: str
    amount: float
    haircut: float = 0.0
    
    def effective_amount(self) -> float:
        return self.amount * (1 - self.haircut / 100)

class EnterpriseBaselAgent:
    """Enterprise-grade Basel Risk Capital Calculator"""
    
    def __init__(self):
        self.supervisory_factors = {
            AssetClass.INTEREST_RATE: {
                'USD': {'<2y': 0.5, '2-5y': 0.5, '>5y': 1.5},
                'EUR': {'<2y': 0.5, '2-5y': 0.5, '>5y': 1.5},
                'other': {'<2y': 1.5, '2-5y': 1.5, '>5y': 1.5}
            },
            AssetClass.FOREIGN_EXCHANGE: {'G10': 4.0, 'emerging': 15.0},
            AssetClass.CREDIT: {'IG': 0.46, 'HY': 1.30, 'IG_index': 0.38},
            AssetClass.EQUITY: {'single': 32.0, 'index': 20.0},
            AssetClass.COMMODITY: {'energy': 18.0, 'metals': 18.0, 'agriculture': 18.0}
        }
        
        self.collateral_haircuts = {
            CollateralType.CASH: 0.0,
            CollateralType.GOVERNMENT_BONDS: 0.5,  # 1Y UST
            CollateralType.CORPORATE_BONDS: 4.0,   # IG Corporate
            CollateralType.EQUITIES: 15.0,         # Listed equities
            CollateralType.MONEY_MARKET: 0.5       # Money market funds
        }
    
    def calculate_credit_rwa_detailed(self, facility: CreditFacility, collateral: List[Collateral] = None) -> Dict:
        """Calculate credit RWA with detailed steps"""
        steps = []
        
        # Step 1: Calculate undrawn exposure
        undrawn_amount = facility.approved_limit - facility.funded_amount
        undrawn_exposure = undrawn_amount * facility.ccf
        
        steps.append({
            'step': 1,
            'title': 'Calculate Undrawn Exposure',
            'formula': 'Undrawn Exposure = Undrawn Amount × CCF',
            'calculation': f'${undrawn_amount:,.0f} × {facility.ccf:.0%} = ${undrawn_exposure:,.0f}',
            'result': undrawn_exposure
        })
        
        # Step 2: Calculate gross exposure
        gross_exposure = facility.funded_amount + undrawn_exposure
        
        steps.append({
            'step': 2,
            'title': 'Calculate Gross Exposure',
            'formula': 'Gross Exposure = Funded Amount + Undrawn Exposure',
            'calculation': f'${facility.funded_amount:,.0f} + ${undrawn_exposure:,.0f} = ${gross_exposure:,.0f}',
            'result': gross_exposure
        })
        
        # Step 3: Apply collateral
        total_effective_collateral = 0
        collateral_details = []
        
        if collateral:
            for coll in collateral:
                haircut = self.collateral_haircuts.get(coll.collateral_type, 15.0)
                effective_value = coll.amount * (1 - haircut / 100)
                total_effective_collateral += effective_value
                
                collateral_details.append({
                    'type': coll.collateral_type.value,
                    'market_value': coll.amount,
                    'haircut': haircut,
                    'effective_value': effective_value
                })
        
        net_exposure = max(0, gross_exposure - total_effective_collateral)
        
        steps.append({
            'step': 3,
            'title': 'Apply Collateral and Calculate Net Exposure',
            'formula': 'Net Exposure = max(0, Gross Exposure - Effective Collateral)',
            'calculation': f'max(0, ${gross_exposure:,.0f} - ${total_effective_collateral:,.0f}) = ${net_exposure:,.0f}',
            'result': net_exposure,
            'collateral_details': collateral_details
        })
        
        # Step 4: Calculate RWA
        rwa = net_exposure * facility.risk_weight
        
        steps.append({
            'step': 4,
            'title': 'Calculate Risk Weighted Assets',
            'formula': 'RWA = Net Exposure × Risk Weight',
            'calculation': f'${net_exposure:,.0f} × {facility.risk_weight:.0%} = ${rwa:,.0f}',
            'result': rwa
        })
        
        # Step 5: Calculate capital requirement
        capital_requirement = rwa * 0.08
        
        steps.append({
            'step': 5,
            'title': 'Calculate Capital Requirement',
            'formula': 'Capital Requirement = RWA × 8%',
            'calculation': f'${rwa:,.0f} × 8% = ${capital_requirement:,.0f}',
            'result': capital_requirement
        })
        
        return {
            'steps': steps,
            'final_metrics': {
                'gross_exposure': gross_exposure,
                'net_exposure': net_exposure,
                'total_collateral': total_effective_collateral,
                'rwa': rwa,
                'capital_requirement': capital_requirement,
                'capital_ratio': capital_requirement / gross_exposure if gross_exposure > 0 else 0
            },
            'collateral_breakdown': collateral_details
        }
    
    def calculate_saccr_detailed(self, trades: List[Trade], collateral: List[Collateral] = None, 
                               threshold: float = 0, mta: float = 0) -> Dict:
        """Calculate SA-CCR with detailed step-by-step breakdown"""
        steps = []
        
        # Step 1: Calculate individual trade add-ons
        trade_addons = []
        total_gross_addon = 0
        
        for trade in trades:
            # Get supervisory factor
            if trade.asset_class == AssetClass.INTEREST_RATE:
                maturity = trade.time_to_maturity()
                if maturity < 2:
                    sf = 0.5
                elif maturity <= 5:
                    sf = 0.5
                else:
                    sf = 1.5
            elif trade.asset_class == AssetClass.FOREIGN_EXCHANGE:
                sf = 4.0  # Assume G10
            elif trade.asset_class == AssetClass.EQUITY:
                sf = 20.0 if 'index' in trade.underlying.lower() else 32.0
            else:
                sf = 18.0
            
            sf_decimal = sf / 100  # Convert to decimal
            
            # Calculate maturity factor
            remaining_maturity = trade.time_to_maturity()
            maturity_factor = min(1.0, 0.05 + 0.95 * math.exp(-0.05 * max(1.0, remaining_maturity)))
            
            # Calculate add-on
            if trade.trade_type in [TradeType.OPTION, TradeType.SWAPTION]:
                addon = sf_decimal * abs(trade.notional) * maturity_factor * abs(trade.delta)
            else:
                addon = sf_decimal * abs(trade.notional) * maturity_factor
            
            total_gross_addon += addon
            trade_addons.append({
                'trade_id': trade.trade_id,
                'asset_class': trade.asset_class.value,
                'supervisory_factor': sf,
                'maturity_factor': maturity_factor,
                'addon': addon
            })
        
        steps.append({
            'step': 1,
            'title': 'Calculate Individual Trade Add-ons',
            'formula': 'Add-on = Supervisory Factor × Notional × Maturity Factor × Delta (for options)',
            'result': total_gross_addon,
            'trade_details': trade_addons
        })
        
        # Step 2: Aggregate into hedging sets
        aggregated_addon = total_gross_addon * 0.95  # Simplified correlation benefit
        
        steps.append({
            'step': 2,
            'title': 'Aggregate into Hedging Sets',
            'formula': 'Hedging Set Add-on = Σ(Individual Add-ons) × Correlation Factor',
            'calculation': f'${total_gross_addon:,.0f} × 95% = ${aggregated_addon:,.0f}',
            'result': aggregated_addon
        })
        
        # Step 3: Calculate multiplier
        net_mtm = sum(trade.mtm_value for trade in trades)
        if total_gross_addon > 0:
            multiplier = min(1.0, 0.05 + 0.95 * math.exp(-0.05 * max(0, net_mtm) / total_gross_addon))
        else:
            multiplier = 1.0
        
        pfe = multiplier * aggregated_addon
        
        steps.append({
            'step': 3,
            'title': 'Calculate Potential Future Exposure (PFE)',
            'formula': 'PFE = Multiplier × Aggregated Add-on',
            'calculation': f'{multiplier:.3f} × ${aggregated_addon:,.0f} = ${pfe:,.0f}',
            'result': pfe,
            'multiplier': multiplier,
            'net_mtm': net_mtm
        })
        
        # Step 4: Calculate Replacement Cost
        total_mtm = sum(trade.mtm_value for trade in trades)
        total_effective_collateral = 0
        
        if collateral:
            total_effective_collateral = sum(coll.effective_amount() for coll in collateral)
        
        rc = max(total_mtm - total_effective_collateral, threshold + mta, 0)
        
        steps.append({
            'step': 4,
            'title': 'Calculate Replacement Cost (RC)',
            'formula': 'RC = max(MTM - Collateral, Threshold + MTA, 0)',
            'calculation': f'max(${total_mtm:,.0f} - ${total_effective_collateral:,.0f}, ${threshold + mta:,.0f}, 0) = ${rc:,.0f}',
            'result': rc
        })
        
        # Step 5: Calculate EAD
        alpha = 1.4
        ead = alpha * (rc + pfe)
        
        steps.append({
            'step': 5,
            'title': 'Calculate Exposure at Default (EAD)',
            'formula': 'EAD = α × (RC + PFE), where α = 1.4',
            'calculation': f'1.4 × (${rc:,.0f} + ${pfe:,.0f}) = ${ead:,.0f}',
            'result': ead
        })
        
        # Step 6: Calculate RWA and Capital
        risk_weight = 0.20  # 20% for derivatives
        rwa = ead * risk_weight
        capital_requirement = rwa * 0.08
        
        steps.append({
            'step': 6,
            'title': 'Calculate Capital Requirement',
            'formula': 'RWA = EAD × Risk Weight; Capital = RWA × 8%',
            'calculation': f'${ead:,.0f} × 20% = ${rwa:,.0f}; ${rwa:,.0f} × 8% = ${capital_requirement:,.0f}',
            'result': capital_requirement
        })
        
        return {
            'steps': steps,
            'final_metrics': {
                'replacement_cost': rc,
                'potential_future_exposure': pfe,
                'exposure_at_default': ead,
                'risk_weighted_assets': rwa,
                'capital_requirement': capital_requirement,
                'multiplier': multiplier
            }
        }

# ==============================================================================
# STREAMLIT APPLICATION
# ==============================================================================

def main():
    # Executive Dashboard Header
    st.markdown("""
    <div class="executive-header">
        <div class="executive-title">🏛️ Basel Risk Capital Platform</div>
        <div class="executive-subtitle">Enterprise-Grade SA-CCR & Credit Risk Capital Calculator</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'agent' not in st.session_state:
        st.session_state.agent = EnterpriseBaselAgent()
    
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("### 📊 Navigation")
        page = st.selectbox(
            "Select Analysis Type:",
            ["🏠 Executive Dashboard", "💳 Credit Risk Calculator", "📈 SA-CCR Calculator", "📋 GWIM Examples", "🔬 Stress Testing"]
        )
        
        st.markdown("---")
        st.markdown("### ⚙️ Configuration")
        st.number_input("Capital Ratio (%)", min_value=6.0, max_value=15.0, value=8.0, step=0.1, key="capital_ratio")
        st.selectbox("Base Currency", ["USD", "EUR", "GBP", "JPY"], key="base_currency")
        
        st.markdown("---")
        st.markdown("### 📈 Portfolio Summary")
        if 'trades' in st.session_state:
            st.metric("Total Trades", len(st.session_state.get('trades', [])))
        if 'facilities' in st.session_state:
            st.metric("Credit Facilities", len(st.session_state.get('facilities', [])))
    
    # Main content based on page selection
    if page == "🏠 Executive Dashboard":
        executive_dashboard()
    elif page == "💳 Credit Risk Calculator":
        credit_risk_calculator()
    elif page == "📈 SA-CCR Calculator":
        saccr_calculator()
    elif page == "📋 GWIM Examples":
        gwim_examples()
    elif page == "🔬 Stress Testing":
        stress_testing()

def executive_dashboard():
    """Executive-level dashboard with key metrics and insights"""
    
    st.markdown("## 📊 Executive Risk Dashboard")
    
    # Create sample portfolio metrics for demonstration
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-value">$847M</div>
            <div class="kpi-label">Portfolio Notional</div>
            <div class="kpi-change positive">+12.3% vs Q3</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-value">$23.4M</div>
            <div class="kpi-label">Total EAD</div>
            <div class="kpi-change negative">+5.7% vs Q3</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-value">$4.7M</div>
            <div class="kpi-label">Risk Weighted Assets</div>
            <div class="kpi-change positive">-2.1% vs Q3</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-value">$375K</div>
            <div class="kpi-label">Capital Requirement</div>
            <div class="kpi-change positive">-2.1% vs Q3</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-value">12.8%</div>
            <div class="kpi-label">Capital Ratio</div>
            <div class="kpi-change positive">+0.3% vs Q3</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Risk exposure breakdown chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Risk Exposure by Asset Class")
        
        # Sample data for executive view
        asset_data = pd.DataFrame({
            'Asset Class': ['Interest Rate', 'Foreign Exchange', 'Credit', 'Equity', 'Commodity'],
            'Exposure (EAD)': [8.5, 6.2, 4.1, 3.8, 0.8],
            'Capital Requirement': [136, 99, 66, 61, 13]
        })
        
        fig = px.sunburst(
            asset_data, 
            names='Asset Class', 
            values='Exposure (EAD)',
            title="Exposure at Default Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(font_family="Inter", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Capital Efficiency Trends")
        
        # Sample trend data
        trend_data = pd.DataFrame({
            'Quarter': ['Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2024'],
            'Capital Requirement': [425, 398, 384, 375],
            'Portfolio Notional': [720, 756, 812, 847],
            'Capital Efficiency': [0.059, 0.053, 0.047, 0.044]
        })
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=trend_data['Quarter'], y=trend_data['Capital Requirement'],
                      name="Capital Requirement ($K)", line=dict(color="#3282b8", width=3)),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(x=trend_data['Quarter'], y=trend_data['Capital Efficiency']*100,
                      name="Capital Efficiency (%)", line=dict(color="#28a745", width=3, dash="dot")),
            secondary_y=True,
        )
        
        fig.update_layout(title="Capital Efficiency Trend", font_family="Inter")
        fig.update_xaxes(title_text="Quarter")
        fig.update_yaxes(title_text="Capital Requirement ($K)", secondary_y=False)
        fig.update_yaxes(title_text="Capital Efficiency (%)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Executive summary table
    st.markdown("### 📋 Risk Summary by Counterparty")
    
    summary_data = pd.DataFrame({
        'Counterparty': ['Goldman Sachs', 'JP Morgan', 'Morgan Stanley', 'Citi', 'Deutsche Bank'],
        'Gross Exposure ($M)': [245.7, 189.3, 156.8, 134.2, 87.5],
        'Net Exposure ($M)': [12.3, 8.7, 6.4, 5.1, 2.9],
        'RWA ($M)': [2.46, 1.74, 1.28, 1.02, 0.58],
        'Capital Req ($K)': [197, 139, 102, 82, 46],
        'Utilization (%)': [67.8, 45.3, 78.9, 23.4, 89.1]
    })
    
    # Style the dataframe for executive presentation
    def style_summary_table(df):
        return df.style.format({
            'Gross Exposure ($M)': '${:.1f}',
            'Net Exposure ($M)': '${:.1f}',
            'RWA ($M)': '${:.2f}',
            'Capital Req ($K)': '${:.0f}',
            'Utilization (%)': '{:.1f}%'
        }).background_gradient(subset=['Capital Req ($K)'], cmap='RdYlGn_r')
    
    st.dataframe(style_summary_table(summary_data), use_container_width=True)

def gwim_examples():
    """GWIM Examples with step-by-step calculations"""
    
    st.markdown("## 📋 GWIM Examples - Step-by-Step Analysis")
    
    tab1, tab2 = st.tabs(["💼 Commercial Lending", "🏦 Securities Based Lending"])
    
    with tab1:
        st.markdown("### GWIM Example 1 - Custom Lending (Commercial)")
        
        with st.expander("📊 Input Parameters", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **Facility Details:**
                - Approved Commitment: $1,000,000
                - Term: 3 years
                - Credit Conversion Factor: 50%
                - Currently Funded: $400,000
                """)
            with col2:
                st.markdown("""
                **Risk Parameters:**
                - Risk Weight: 100% (Commercial)
                - Capital Ratio: 8%
                - Collateral: None
                """)
        
        # Create facility
        facility = CreditFacility(
            facility_id="GWIM_EX1",
            counterparty="US_Client_Commercial",
            facility_type="Commercial",
            approved_limit=1_000_000,
            funded_amount=400_000,
            currency="USD",
            term_years=3,
            ccf=0.50,
            risk_weight=1.0
        )
        
        # Calculate with detailed steps
        result = st.session_state.agent.calculate_credit_rwa_detailed(facility)
        
        st.markdown("### 🔍 Step-by-Step Calculation")
        
        for step in result['steps']:
            with st.container():
                st.markdown(f"""
                <div class="calc-step">
                    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                        <span class="step-number">{step['step']}</span>
                        <span class="step-title">{step['title']}</span>
                    </div>
                    <div class="step-formula">{step['formula']}</div>
                    <div style="font-size: 1.1rem; font-weight: 600; color: #0f4c75;">
                        {step['calculation']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Final result
        final_rwa = result['final_metrics']['rwa']
        st.markdown(f"""
        <div class="result-highlight">
            ✅ FINAL RESULT: Initial RWA = ${final_rwa:,.0f}
        </div>
        """, unsafe_allow_html=True)
        
        # Key metrics visualization
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Gross Exposure", f"${result['final_metrics']['gross_exposure']:,.0f}")
        with col2:
            st.metric("Net Exposure", f"${result['final_metrics']['net_exposure']:,.0f}")
        with col3:
            st.metric("RWA", f"${result['final_metrics']['rwa']:,.0f}")
        with col4:
            st.metric("Capital Required", f"${result['final_metrics']['capital_requirement']:,.0f}")
    
    with tab2:
        st.markdown("### GWIM Example 2 - Securities Based Lending (SBL)")
        
        with st.expander("📊 Input Parameters", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **Loan Details:**
                - Loan Amount: $1,000,000
                - Term: Demand (no term UCC)
                - Purpose: Securities Based Lending
                """)
            with col2:
                st.markdown("""
                **Risk Parameters:**
                - Risk Weight: 50% (Secured)
                - Capital Ratio: 8%
                """)
        
        st.markdown("**Collateral Portfolio:**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Cash Deposit", "$150,000", "0% haircut")
        with col2:
            st.metric("1Y UST", "$450,000", "0.5% haircut")
        with col3:
            st.metric("Money Market", "$300,000", "0.5% haircut")
        with col4:
            st.metric("Equity Funds", "$1,500,000", "12% haircut")
        
        # Create facility with collateral
        facility2 = CreditFacility(
            facility_id="GWIM_EX2",
            counterparty="US_Client_SBL",
            facility_type="SBL",
            approved_limit=1_000_000,
            funded_amount=1_000_000,
            currency="USD",
            term_years=0,
            ccf=1.0,
            risk_weight=0.50
        )
        
        collateral = [
            Collateral(CollateralType.CASH, "USD", 150_000, 0.0),
            Collateral(CollateralType.GOVERNMENT_BONDS, "USD", 450_000, 0.5),
            Collateral(CollateralType.MONEY_MARKET, "USD", 300_000, 0.5),
            Collateral(CollateralType.EQUITIES, "USD", 1_500_000, 12.0)
        ]
        
        result2 = st.session_state.agent.calculate_credit_rwa_detailed(facility2, collateral)
        
        st.markdown("### 🔍 Step-by-Step Calculation")
        
        for step in result2['steps']:
            with st.container():
                st.markdown(f"""
                <div class="calc-step">
                    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                        <span class="step-number">{step['step']}</span>
                        <span class="step-title">{step['title']}</span>
                    </div>
                    <div class="step-formula">{step['formula']}</div>
                    <div style="font-size: 1.1rem; font-weight: 600; color: #0f4c75;">
                        {step['calculation']}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show collateral details for step 3
                if step['step'] == 3 and 'collateral_details' in step:
                    st.markdown("**Collateral Breakdown:**")
                    coll_df = pd.DataFrame(step['collateral_details'])
                    coll_df['Market Value'] = coll_df['market_value'].apply(lambda x: f"${x:,.0f}")
                    coll_df['Haircut'] = coll_df['haircut'].apply(lambda x: f"{x:.1f}%")
                    coll_df['Effective Value'] = coll_df['effective_value'].apply(lambda x: f"${x:,.0f}")
                    st.dataframe(coll_df[['type', 'Market Value', 'Haircut', 'Effective Value']], 
                               use_container_width=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        # Final result
        final_rwa2 = result2['final_metrics']['rwa']
        st.markdown(f"""
        <div class="result-highlight">
            ✅ FINAL RESULT: RWA = ${final_rwa2:,.0f} (Loan is over-collateralized!)
        </div>
        """, unsafe_allow_html=True)
        
        # Collateral analysis
        st.markdown("### 📊 Collateral Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            # Collateral composition pie chart
            coll_data = pd.DataFrame(result2['collateral_breakdown'])
            fig = px.pie(coll_data, values='effective_value', names='type',
                        title='Effective Collateral Composition',
                        color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_layout(font_family="Inter")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Haircut impact analysis
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                name='Market Value',
                x=coll_data['type'],
                y=coll_data['market_value'],
                marker_color='lightblue'
            ))
            fig2.add_trace(go.Bar(
                name='Effective Value',
                x=coll_data['type'], 
                y=coll_data['effective_value'],
                marker_color='darkblue'
            ))
            fig2.update_layout(
                title='Collateral: Market vs Effective Value',
                xaxis_title='Collateral Type',
                yaxis_title='Value ($)',
                barmode='group',
                font_family="Inter"
            )
            st.plotly_chart(fig2, use_container_width=True)

def credit_risk_calculator():
    """Interactive Credit Risk Calculator"""
    
    st.markdown("## 💳 Credit Risk Calculator")
    
    with st.expander("🎯 Quick Setup", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            facility_type = st.selectbox("Facility Type", ["Commercial Lending", "Retail Mortgage", "Securities Based Lending", "Credit Card"])
            counterparty = st.text_input("Counterparty", value="Sample Client")
            
        with col2:
            approved_limit = st.number_input("Approved Limit ($)", min_value=0, value=1000000, step=50000)
            funded_amount = st.number_input("Funded Amount ($)", min_value=0, value=400000, step=10000)
            
        with col3:
            ccf = st.slider("Credit Conversion Factor (%)", 0, 100, 50) / 100
            risk_weight = st.slider("Risk Weight (%)", 0, 150, 100) / 100
    
    # Collateral section
    st.markdown("### 🛡️ Collateral Portfolio")
    
    col1, col2 = st.columns(2)
    with col1:
        add_collateral = st.checkbox("Add Collateral")
        
    collateral_list = []
    if add_collateral:
        with col2:
            num_collateral = st.number_input("Number of Collateral Items", 1, 5, 1)
        
        for i in range(num_collateral):
            st.markdown(f"**Collateral Item {i+1}**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                coll_type = st.selectbox(f"Type {i+1}", 
                                       [ct.value for ct in CollateralType], 
                                       key=f"coll_type_{i}")
            with col2:
                coll_currency = st.selectbox(f"Currency {i+1}", 
                                           ["USD", "EUR", "GBP", "JPY"], 
                                           key=f"coll_curr_{i}")
            with col3:
                coll_amount = st.number_input(f"Amount {i+1} ($)", 
                                            min_value=0, value=100000, 
                                            key=f"coll_amt_{i}")
            with col4:
                coll_haircut = st.number_input(f"Haircut {i+1} (%)", 
                                             min_value=0.0, max_value=50.0, 
                                             value=2.0, step=0.5, 
                                             key=f"coll_hair_{i}")
            
            collateral_list.append(Collateral(
                CollateralType(coll_type), coll_currency, coll_amount, coll_haircut
            ))
    
    # Calculate button
    if st.button("🚀 Calculate Risk Capital", type="primary"):
        # Create facility
        facility = CreditFacility(
            facility_id="USER_FACILITY",
            counterparty=counterparty,
            facility_type=facility_type,
            approved_limit=approved_limit,
            funded_amount=funded_amount,
            currency="USD",
            term_years=1,
            ccf=ccf,
            risk_weight=risk_weight
        )
        
        # Calculate with steps
        result = st.session_state.agent.calculate_credit_rwa_detailed(facility, collateral_list if add_collateral else None)
        
        # Display results
        st.markdown("### 📊 Calculation Results")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Gross Exposure", f"${result['final_metrics']['gross_exposure']:,.0f}")
        with col2:
            st.metric("Net Exposure", f"${result['final_metrics']['net_exposure']:,.0f}")
        with col3:
            st.metric("Risk Weighted Assets", f"${result['final_metrics']['rwa']:,.0f}")
        with col4:
            st.metric("Capital Requirement", f"${result['final_metrics']['capital_requirement']:,.0f}")
        
        # Detailed steps
        with st.expander("🔍 Detailed Calculation Steps", expanded=True):
            for step in result['steps']:
                st.markdown(f"""
                <div class="calc-step">
                    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                        <span class="step-number">{step['step']}</span>
                        <span class="step-title">{step['title']}</span>
                    </div>
                    <div class="step-formula">{step['formula']}</div>
                    <div style="font-size: 1.1rem; font-weight: 600; color: #0f4c75;">
                        {step['calculation']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

def saccr_calculator():
    """Interactive SA-CCR Calculator"""
    
    st.markdown("## 📈 SA-CCR Calculator")
    st.markdown("*Standardized Approach for Counterparty Credit Risk*")
    
    # Trade input section
    with st.expander("🎯 Trade Portfolio Setup", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Add New Trade**")
            trade_id = st.text_input("Trade ID", value="TRADE_001")
            counterparty = st.text_input("Counterparty", value="Goldman Sachs")
            asset_class = st.selectbox("Asset Class", [ac.value for ac in AssetClass])
            trade_type = st.selectbox("Trade Type", [tt.value for tt in TradeType])
            
        with col2:
            notional = st.number_input("Notional ($)", min_value=0, value=100000000, step=1000000)
            currency = st.selectbox("Currency", ["USD", "EUR", "GBP", "JPY", "CHF"])
            underlying = st.text_input("Underlying", value="USD 3M SOFR")
            maturity_years = st.number_input("Maturity (Years)", min_value=0.1, max_value=30.0, value=5.0, step=0.1)
            
        col3, col4 = st.columns(2)
        with col3:
            mtm_value = st.number_input("MTM Value ($)", value=0, step=10000)
            
        with col4:
            delta = st.number_input("Delta (for options)", min_value=-1.0, max_value=1.0, value=1.0, step=0.1)
        
        if st.button("➕ Add Trade"):
            if 'trades' not in st.session_state:
                st.session_state.trades = []
            
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
            
            st.session_state.trades.append(new_trade)
            st.success(f"✅ Added trade {trade_id}")
    
    # Display current trades
    if 'trades' in st.session_state and st.session_state.trades:
        st.markdown("### 📋 Current Trade Portfolio")
        
        trades_data = []
        for i, trade in enumerate(st.session_state.trades):
            trades_data.append({
                'ID': trade.trade_id,
                'Counterparty': trade.counterparty,
                'Asset Class': trade.asset_class.value,
                'Type': trade.trade_type.value,
                'Notional ($M)': f"{trade.notional/1_000_000:.1f}",
                'Currency': trade.currency,
                'MTM ($K)': f"{trade.mtm_value/1000:.0f}",
                'Maturity': f"{trade.time_to_maturity():.1f}Y"
            })
        
        trades_df = pd.DataFrame(trades_data)
        st.dataframe(trades_df, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear All Trades"):
                st.session_state.trades = []
                st.rerun()
        
        # Collateral setup
        st.markdown("### 🛡️ Collateral Setup")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            threshold = st.number_input("Threshold ($)", min_value=0, value=1000000, step=100000)
        with col2:
            mta = st.number_input("MTA ($)", min_value=0, value=500000, step=50000)
        with col3:
            add_saccr_collateral = st.checkbox("Add Collateral")
        
        saccr_collateral = []
        if add_saccr_collateral:
            st.markdown("**Collateral Items**")
            num_coll = st.number_input("Number of Items", 1, 3, 1)
            
            for i in range(num_coll):
                col1, col2, col3 = st.columns(3)
                with col1:
                    coll_type = st.selectbox(f"Collateral Type {i+1}", 
                                           [ct.value for ct in CollateralType], 
                                           key=f"saccr_coll_type_{i}")
                with col2:
                    coll_curr = st.selectbox(f"Currency {i+1}", 
                                           ["USD", "EUR", "GBP"], 
                                           key=f"saccr_coll_curr_{i}")
                with col3:
                    coll_amt = st.number_input(f"Amount {i+1} ($M)", 
                                             min_value=0.0, value=10.0, 
                                             key=f"saccr_coll_amt_{i}")
                
                saccr_collateral.append(Collateral(
                    CollateralType(coll_type), coll_curr, coll_amt * 1_000_000
                ))
        
        # Calculate SA-CCR
        if st.button("🚀 Calculate SA-CCR", type="primary"):
            result = st.session_state.agent.calculate_saccr_detailed(
                st.session_state.trades, 
                saccr_collateral if add_saccr_collateral else None,
                threshold, 
                mta
            )
            
            # Display results
            st.markdown("### 📊 SA-CCR Results")
            
            # Key metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Replacement Cost", f"${result['final_metrics']['replacement_cost']/1_000_000:.1f}M")
            with col2:
                st.metric("PFE", f"${result['final_metrics']['potential_future_exposure']/1_000_000:.1f}M")
            with col3:
                st.metric("EAD", f"${result['final_metrics']['exposure_at_default']/1_000_000:.1f}M")
            with col4:
                st.metric("RWA", f"${result['final_metrics']['risk_weighted_assets']/1_000_000:.1f}M")
            with col5:
                st.metric("Capital Required", f"${result['final_metrics']['capital_requirement']/1000:.0f}K")
            
            # Detailed calculation steps
            with st.expander("🔍 Detailed SA-CCR Steps", expanded=True):
                for step in result['steps']:
                    st.markdown(f"""
                    <div class="calc-step">
                        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                            <span class="step-number">{step['step']}</span>
                            <span class="step-title">{step['title']}</span>
                        </div>
                        <div class="step-formula">{step['formula']}</div>
                        <div style="font-size: 1.1rem; font-weight: 600; color: #0f4c75;">
                            {step['calculation']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show trade details for first step
                    if step['step'] == 1 and 'trade_details' in step:
                        trade_details_df = pd.DataFrame(step['trade_details'])
                        trade_details_df['Supervisory Factor'] = trade_details_df['supervisory_factor'].apply(lambda x: f"{x:.1f} bp")
                        trade_details_df['Maturity Factor'] = trade_details_df['maturity_factor'].apply(lambda x: f"{x:.3f}")
                        trade_details_df['Add-on'] = trade_details_df['addon'].apply(lambda x: f"${x:,.0f}")
                        st.dataframe(trade_details_df[['trade_id', 'asset_class', 'Supervisory Factor', 'Maturity Factor', 'Add-on']], 
                                   use_container_width=True)

def stress_testing():
    """Stress Testing Module"""
    
    st.markdown("## 🔬 Stress Testing & Scenario Analysis")
    
    st.markdown("""
    Perform comprehensive stress testing on your portfolio to understand potential 
    capital impacts under adverse market conditions.
    """)
    
    # Stress scenario selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Predefined Scenarios")
        
        scenarios = {
            "Market Shock": {"description": "+200bp rates, +20% volatility", "mtm_shock": 2.0, "addon_shock": 1.2},
            "Severe Stress": {"description": "+500bp rates, +50% volatility", "mtm_shock": 3.0, "addon_shock": 1.5},
            "Credit Crisis": {"description": "Credit spreads widen +300bp", "mtm_shock": 1.5, "addon_shock": 1.3},
            "Liquidity Crisis": {"description": "Collateral haircuts increase 50%", "mtm_shock": 1.2, "addon_shock": 1.1}
        }
        
        selected_scenario = st.selectbox("Select Stress Scenario", list(scenarios.keys()))
        
        st.info(f"📝 **{selected_scenario}**: {scenarios[selected_scenario]['description']}")
        
    with col2:
        st.markdown("### ⚙️ Custom Scenario")
        
        custom_mtm_shock = st.slider("MTM Shock Multiplier", 1.0, 5.0, 1.5, 0.1)
        custom_addon_shock = st.slider("Add-on Shock Multiplier", 1.0, 2.0, 1.2, 0.1)
        custom_collateral_shock = st.slider("Collateral Haircut Increase (%)", 0, 100, 25)
    
    # Portfolio summary for stress testing
    if 'trades' in st.session_state and st.session_state.trades:
        st.markdown("### 📋 Portfolio Summary")
        
        total_notional = sum(abs(trade.notional) for trade in st.session_state.trades)
        total_mtm = sum(trade.mtm_value for trade in st.session_state.trades)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Trades", len(st.session_state.trades))
        with col2:
            st.metric("Total Notional", f"${total_notional/1_000_000:.0f}M")
        with col3:
            st.metric("Net MTM", f"${total_mtm/1_000_000:+.1f}M")
        
        if st.button("🎯 Run Stress Test", type="primary"):
            # Simulate stress test results
            base_capital = 2_500_000  # Base capital requirement
            
            scenario = scenarios[selected_scenario]
            stressed_capital = base_capital * scenario['mtm_shock'] * scenario['addon_shock']
            
            impact = stressed_capital - base_capital
            impact_pct = (impact / base_capital) * 100
            
            # Results
            st.markdown("### 📊 Stress Test Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Base Capital", f"${base_capital/1_000_000:.1f}M")
            with col2:
                st.metric("Stressed Capital", f"${stressed_capital/1_000_000:.1f}M", 
                         f"+${impact/1_000_000:.1f}M")
            with col3:
                st.metric("Impact", f"+{impact_pct:.1f}%", 
                         delta_color="inverse")
            
            # Stress test visualization
            stress_data = pd.DataFrame({
                'Scenario': ['Base Case', selected_scenario],
                'Capital Requirement': [base_capital/1_000_000, stressed_capital/1_000_000],
                'Impact': [0, impact_pct]
            })
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Capital Requirement ($M)',
                x=stress_data['Scenario'],
                y=stress_data['Capital Requirement'],
                marker_color=['#3282b8', '#dc3545']
            ))
            
            fig.update_layout(
                title=f'Stress Test Results: {selected_scenario}',
                yaxis_title='Capital Requirement ($M)',
                font_family="Inter",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            if impact_pct > 50:
                st.markdown("""
                <div class="risk-alert">
                    ⚠️ <strong>High Risk Alert:</strong> Capital requirement increases by more than 50% under stress.
                    Consider portfolio rebalancing or additional hedging strategies.
                </div>
                """, unsafe_allow_html=True)
            elif impact_pct > 25:
                st.markdown("""
                <div class="warning-box">
                    ⚡ <strong>Moderate Risk:</strong> Significant capital impact under stress.
                    Monitor portfolio concentration and consider diversification.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-highlight">
                    ✅ <strong>Well Managed Risk:</strong> Portfolio shows resilience under stress conditions.
                </div>
                """, unsafe_allow_html=True)
    
    else:
        st.info("🔔 Please add trades in the SA-CCR Calculator to run stress tests.")

if __name__ == "__main__":
    main()
