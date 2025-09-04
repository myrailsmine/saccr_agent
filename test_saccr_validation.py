#!/usr/bin/env python3
"""
Test script to validate SA-CCR calculations against the provided images
"""

import sys
import os
from datetime import datetime, timedelta

# Add the current directory to Python path
sys.path.append('/app')

# Import the SA-CCR classes
from enterprise_saccr_app import (
    ComprehensiveSACCRAgent, Trade, NettingSet, AssetClass, TradeType, Collateral
)

def test_reference_example():
    """Test the reference example from the images"""
    
    print("=" * 80)
    print("SA-CCR CALCULATION VALIDATION - Testing Against Provided Images")
    print("=" * 80)
    
    # Create the exact trade from the images
    reference_trade = Trade(
        trade_id="2098474100",
        counterparty="Lowell Hotel Properties LLC", 
        asset_class=AssetClass.INTEREST_RATE,
        trade_type=TradeType.SWAP,
        notional=681578963,  # From images
        currency="USD",
        underlying="Interest rate",
        maturity_date=datetime.now() + timedelta(days=int(0.3 * 365)),
        mtm_value=8382419,  # From Step 14 in images
        delta=1.0
    )
    
    # Create netting set
    netting_set = NettingSet(
        netting_set_id="212784050000389187901",  # From images
        counterparty="Lowell Hotel Properties LLC",
        trades=[reference_trade],
        threshold=12000000,  # $12M
        mta=1000000,         # $1M
        nica=0               # $0
    )
    
    # Initialize SA-CCR agent
    agent = ComprehensiveSACCRAgent()
    
    # Calculate SA-CCR
    print("Calculating SA-CCR...")
    result = agent.calculate_comprehensive_saccr(netting_set)
    
    print("\nKEY RESULTS VALIDATION:")
    print("-" * 50)
    
    # Extract key results
    final_results = result['final_results']
    
    print(f"Trade ID: {reference_trade.trade_id}")
    print(f"Netting Set ID: {netting_set.netting_set_id}")
    print(f"Notional: ${reference_trade.notional:,.0f}")
    print(f"MTM Value: ${reference_trade.mtm_value:,.0f}")
    print()
    
    # Find specific steps for validation
    steps = result['calculation_steps']
    
    # Step 6: Maturity Factor
    step6 = next((s for s in steps if s['step'] == 6), None)
    if step6:
        mf = step6['data'][0]['maturity_factor']
        print(f"Step 6 - Maturity Factor: {mf:.6f}")
    
    # Step 8: Supervisory Factor
    step8 = next((s for s in steps if s['step'] == 8), None)
    if step8:
        sf_bp = step8['data'][0]['supervisory_factor_bp']
        sf_decimal = step8['data'][0]['supervisory_factor_decimal']
        print(f"Step 8 - Supervisory Factor: {sf_bp:.2f}bps ({sf_decimal:.4f} or {sf_decimal*100:.3f}%)")
        
        # Validate this matches the expected 0.5%
        expected_sf = 0.005  # 0.5%
        if abs(sf_decimal - expected_sf) < 0.0001:
            print("✅ Supervisory Factor CORRECT: 0.5%")
        else:
            print(f"❌ Supervisory Factor ERROR: Expected 0.5% ({expected_sf:.4f}), got {sf_decimal*100:.3f}% ({sf_decimal:.4f})")
    
    # Step 9: Adjusted Derivatives Contract Amount
    step9 = next((s for s in steps if s['step'] == 9), None)
    if step9:
        adjusted_amount = step9['data'][0]['adjusted_derivatives_contract_amount']
        print(f"Step 9 - Adjusted Contract Amount: ${adjusted_amount:,.2f}")
    
    # Step 13: Aggregate AddOn
    step13 = next((s for s in steps if s['step'] == 13), None)
    if step13:
        aggregate_addon = step13['aggregate_addon']
        print(f"Step 13 - Aggregate AddOn: ${aggregate_addon:,.0f}")
    
    # Step 16: PFE
    step16 = next((s for s in steps if s['step'] == 16), None)
    if step16:
        pfe = step16['pfe']
        print(f"Step 16 - PFE: ${pfe:,.0f}")
    
    # Step 18: RC
    step18 = next((s for s in steps if s['step'] == 18), None)
    if step18:
        rc_margined = step18['rc_margined']
        rc_unmargined = step18['rc_unmargined'] 
        print(f"Step 18 - RC Margined: ${rc_margined:,.0f}")
        print(f"Step 18 - RC Unmargined: ${rc_unmargined:,.0f}")
    
    # Step 21: EAD
    step21 = next((s for s in steps if s['step'] == 21), None)
    if step21:
        ead = step21['ead']
        print(f"Step 21 - EAD: ${ead:,.0f}")
    
    # Step 24: RWA
    step24 = next((s for s in steps if s['step'] == 24), None)
    if step24:
        rwa = step24['rwa']
        print(f"Step 24 - RWA: ${rwa:,.0f}")
        
    print()
    print("FINAL RESULTS SUMMARY:")
    print("-" * 30)
    print(f"Replacement Cost: ${final_results['replacement_cost']:,.0f}")
    print(f"Potential Future Exposure: ${final_results['potential_future_exposure']:,.0f}")
    print(f"Exposure at Default: ${final_results['exposure_at_default']:,.0f}")
    print(f"Risk Weighted Assets: ${final_results['risk_weighted_assets']:,.0f}")
    print(f"Capital Requirement: ${final_results['capital_requirement']:,.0f}")
    
    return result

if __name__ == "__main__":
    try:
        result = test_reference_example()
        print("\n✅ Test completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
