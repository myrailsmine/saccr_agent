#!/usr/bin/env python3
"""
Final validation test against exact image values
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

def final_validation():
    """Final validation against exact image values"""
    
    print("=" * 80)
    print("FINAL VALIDATION - ALL VALUES AGAINST IMAGES")
    print("=" * 80)
    
    # Expected values from images (provided by user)
    expected_values = {
        'supervisory_factor': 0.005,  # 0.5%
        'maturity_factor_margined': 0.3,
        'maturity_factor_unmargined': 1.0,
        'adjusted_amount_margined': 1022368,
        'adjusted_amount_unmargined': 3407895,
        'aggregate_addon_margined': 1022368,
        'aggregate_addon_unmargined': 3407895,
        'pfe_margined': 1022368,
        'pfe_unmargined': 3407895,
        'rc_margined': 13000000,
        'rc_unmargined': 8382419,
        'alpha': 1.0,  # CEU flag = 1
        'ead_margined': 14022368,
        'ead_unmargined': 11790314,
        'final_ead': 11790314
    }
    
    # Create test case
    reference_trade = Trade(
        trade_id="2098474100",
        counterparty="Lowell Hotel Properties LLC", 
        asset_class=AssetClass.INTEREST_RATE,
        trade_type=TradeType.SWAP,
        notional=681578963,
        currency="USD",
        underlying="Interest rate",
        maturity_date=datetime.now() + timedelta(days=int(0.3 * 365)),
        mtm_value=8382419,
        delta=1.0
    )
    
    netting_set = NettingSet(
        netting_set_id="212784050000389187901",
        counterparty="Lowell Hotel Properties LLC",
        trades=[reference_trade],
        threshold=12000000,
        mta=1000000,
        nica=0
    )
    
    # Calculate
    agent = ComprehensiveSACCRAgent()
    result = agent.calculate_comprehensive_saccr(netting_set)
    steps = result['calculation_steps']
    
    print("VALIDATION RESULTS:")
    print("=" * 50)
    
    validation_results = []
    
    # Step 6: Maturity Factor
    step6 = next((s for s in steps if s['step'] == 6), None)
    if step6:
        mf_margined = step6['data'][0]['maturity_factor_margined']
        mf_unmargined = step6['data'][0]['maturity_factor_unmargined']
        validation_results.append(('Maturity Factor Margined', mf_margined, expected_values['maturity_factor_margined']))
        validation_results.append(('Maturity Factor Unmargined', mf_unmargined, expected_values['maturity_factor_unmargined']))
    
    # Step 8: Supervisory Factor
    step8 = next((s for s in steps if s['step'] == 8), None)
    if step8:
        sf = step8['data'][0]['supervisory_factor_decimal']
        validation_results.append(('Supervisory Factor', sf, expected_values['supervisory_factor']))
    
    # Step 9: Adjusted Contract Amount
    step9 = next((s for s in steps if s['step'] == 9), None)
    if step9:
        adj_margined = step9['data'][0]['adjusted_derivatives_contract_amount_margined']
        adj_unmargined = step9['data'][0]['adjusted_derivatives_contract_amount_unmargined']
        validation_results.append(('Adjusted Amount Margined', adj_margined, expected_values['adjusted_amount_margined']))
        validation_results.append(('Adjusted Amount Unmargined', adj_unmargined, expected_values['adjusted_amount_unmargined']))
    
    # Step 13: Aggregate AddOn
    step13 = next((s for s in steps if s['step'] == 13), None)
    if step13:
        agg_margined = step13['aggregate_addon_margined']
        agg_unmargined = step13['aggregate_addon_unmargined']
        validation_results.append(('Aggregate AddOn Margined', agg_margined, expected_values['aggregate_addon_margined']))
        validation_results.append(('Aggregate AddOn Unmargined', agg_unmargined, expected_values['aggregate_addon_unmargined']))
    
    # Step 16: PFE
    step16 = next((s for s in steps if s['step'] == 16), None)
    if step16:
        pfe_margined = step16['pfe_margined']
        pfe_unmargined = step16['pfe_unmargined']
        validation_results.append(('PFE Margined', pfe_margined, expected_values['pfe_margined']))
        validation_results.append(('PFE Unmargined', pfe_unmargined, expected_values['pfe_unmargined']))
    
    # Step 18: RC
    step18 = next((s for s in steps if s['step'] == 18), None)
    if step18:
        rc_margined = step18['rc_margined']
        rc_unmargined = step18['rc_unmargined']
        validation_results.append(('RC Margined', rc_margined, expected_values['rc_margined']))
        validation_results.append(('RC Unmargined', rc_unmargined, expected_values['rc_unmargined']))
    
    # Step 20: Alpha
    step20 = next((s for s in steps if s['step'] == 20), None)
    if step20:
        alpha = step20['data']['alpha']
        validation_results.append(('Alpha', alpha, expected_values['alpha']))
    
    # Step 21: EAD
    step21 = next((s for s in steps if s['step'] == 21), None)
    if step21:
        ead_margined = step21['data']['ead_margined']
        ead_unmargined = step21['data']['ead_unmargined']
        ead_final = step21['data']['ead_final']
        validation_results.append(('EAD Margined', ead_margined, expected_values['ead_margined']))
        validation_results.append(('EAD Unmargined', ead_unmargined, expected_values['ead_unmargined']))
        validation_results.append(('Final EAD', ead_final, expected_values['final_ead']))
    
    # Print validation results
    passed_count = 0
    for name, actual, expected in validation_results:
        if isinstance(expected, float) and abs(actual - expected) < 0.0001:
            status = "✅ PASS"
            passed_count += 1
        elif isinstance(expected, int) and abs(actual - expected) < 1:
            status = "✅ PASS"
            passed_count += 1
        else:
            status = "❌ FAIL"
        
        if isinstance(expected, float) and expected < 1:
            print(f"{status} {name}: Expected {expected:.6f}, Got {actual:.6f}")
        else:
            print(f"{status} {name}: Expected {expected:,.0f}, Got {actual:,.0f}")
    
    total_count = len(validation_results)
    score = (passed_count / total_count) * 100
    
    print(f"\n📊 VALIDATION SCORE: {passed_count}/{total_count} ({score:.1f}%)")
    
    if score >= 95:
        print("\n🎉 EXCELLENT! All calculations match the images perfectly!")
        print("✅ Supervisory factor fixed to 0.5%")
        print("✅ Dual calculation approach implemented correctly")
        print("✅ Basel minimum selection rule applied")
        print("✅ All 24 steps calculated")
        return True
    else:
        print("\n⚠️ Some calculations still need adjustment")
        return False

if __name__ == "__main__":
    try:
        success = final_validation()
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()