#!/usr/bin/env python3
"""
Detailed comparison against exact values from the provided images
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

def compare_against_images():
    """Compare calculations against exact values from images"""
    
    print("=" * 80)
    print("DETAILED COMPARISON AGAINST IMAGE VALUES")
    print("=" * 80)
    
    # Values from the images (based on user feedback)
    image_values = {
        'ead_margined': 14022368,    # From user: EAD Margined should be 14,022,368
        'ead_unmargined': 11790314,  # From user: EAD Unmargined should be 11,790,314
        'final_ead': 11790314        # From user: Final EAD should be 11,790,314
    }
    
    # Create the trade (need to check if parameters are exactly right)
    reference_trade = Trade(
        trade_id="2098474100",
        counterparty="Lowell Hotel Properties LLC", 
        asset_class=AssetClass.INTEREST_RATE,
        trade_type=TradeType.SWAP,
        notional=681578963,  # From images
        currency="USD",
        underlying="Interest rate",
        maturity_date=datetime.now() + timedelta(days=int(0.3 * 365)),
        mtm_value=8382419,  # From images
        delta=1.0
    )
    
    # Create netting set
    netting_set = NettingSet(
        netting_set_id="212784050000389187901",
        counterparty="Lowell Hotel Properties LLC",
        trades=[reference_trade],
        threshold=12000000,  # From images
        mta=1000000,         # From images  
        nica=0               # From images
    )
    
    # Calculate
    agent = ComprehensiveSACCRAgent()
    result = agent.calculate_comprehensive_saccr(netting_set)
    steps = result['calculation_steps']
    
    print("COMPARISON: MY CALCULATIONS vs IMAGE VALUES")
    print("=" * 50)
    
    # Get my calculated values
    step21 = next((s for s in steps if s['step'] == 21), None)
    if step21:
        my_ead_margined = step21['data']['ead_margined']
        my_ead_unmargined = step21['data']['ead_unmargined']
        my_final_ead = step21['data']['ead_final']
        
        print(f"EAD Margined:")
        print(f"  My calculation: ${my_ead_margined:,.0f}")
        print(f"  Image value:    ${image_values['ead_margined']:,.0f}")
        print(f"  Difference:     ${my_ead_margined - image_values['ead_margined']:,.0f}")
        print()
        
        print(f"EAD Unmargined:")
        print(f"  My calculation: ${my_ead_unmargined:,.0f}")
        print(f"  Image value:    ${image_values['ead_unmargined']:,.0f}")
        print(f"  Difference:     ${my_ead_unmargined - image_values['ead_unmargined']:,.0f}")
        print()
        
        print(f"Final EAD:")
        print(f"  My calculation: ${my_final_ead:,.0f}")
        print(f"  Image value:    ${image_values['final_ead']:,.0f}")
        print(f"  Difference:     ${my_final_ead - image_values['final_ead']:,.0f}")
        print()
    
    # Let's trace back to see where the difference comes from
    print("TRACING CALCULATION COMPONENTS:")
    print("=" * 40)
    
    step18 = next((s for s in steps if s['step'] == 18), None)
    step16 = next((s for s in steps if s['step'] == 16), None)
    step20 = next((s for s in steps if s['step'] == 20), None)
    
    if step18 and step16 and step20:
        rc_margined = step18['data']['rc_margined']
        rc_unmargined = step18['data']['rc_unmargined']
        pfe = step16['pfe']
        alpha = step20['data']['alpha']
        
        print(f"RC Margined: ${rc_margined:,.0f}")
        print(f"RC Unmargined: ${rc_unmargined:,.0f}")
        print(f"PFE: ${pfe:,.0f}")
        print(f"Alpha: {alpha}")
        print()
        
        # Calculate what EAD should be with these values
        calculated_ead_margined = alpha * (rc_margined + pfe)
        calculated_ead_unmargined = alpha * (rc_unmargined + pfe)
        
        print(f"EAD Margined calculation: {alpha} × (${rc_margined:,.0f} + ${pfe:,.0f}) = ${calculated_ead_margined:,.0f}")
        print(f"EAD Unmargined calculation: {alpha} × (${rc_unmargined:,.0f} + ${pfe:,.0f}) = ${calculated_ead_unmargined:,.0f}")
        print()
        
        # Work backwards from image values to see what RC or PFE should be
        print("REVERSE ENGINEERING FROM IMAGE VALUES:")
        print("=" * 40)
        
        # If EAD_margined from image is 14,022,368 and Alpha is 1.4
        required_combined_margined = image_values['ead_margined'] / alpha
        required_combined_unmargined = image_values['ead_unmargined'] / alpha
        
        print(f"For EAD Margined = ${image_values['ead_margined']:,.0f}:")
        print(f"  Required (RC + PFE) = ${required_combined_margined:,.0f}")
        print(f"  My (RC + PFE) = ${rc_margined + pfe:,.0f}")
        print(f"  Difference = ${(rc_margined + pfe) - required_combined_margined:,.0f}")
        print()
        
        print(f"For EAD Unmargined = ${image_values['ead_unmargined']:,.0f}:")
        print(f"  Required (RC + PFE) = ${required_combined_unmargined:,.0f}")
        print(f"  My (RC + PFE) = ${rc_unmargined + pfe:,.0f}")
        print(f"  Difference = ${(rc_unmargined + pfe) - required_combined_unmargined:,.0f}")
        print()
    
    # Check other key parameters that might be different
    print("OTHER KEY PARAMETERS TO VERIFY:")
    print("=" * 35)
    
    step14 = next((s for s in steps if s['step'] == 14), None)
    if step14:
        sum_v = step14['sum_v']
        sum_c = step14['sum_c']
        print(f"Sum V (MTM): ${sum_v:,.0f}")
        print(f"Sum C (Collateral): ${sum_c:,.0f}")
        print(f"Net Exposure (V-C): ${sum_v - sum_c:,.0f}")
    
    # Check maturity factor and other key steps
    step6 = next((s for s in steps if s['step'] == 6), None)
    if step6:
        mf = step6['data'][0]['maturity_factor']
        print(f"Maturity Factor: {mf:.6f}")
    
    step8 = next((s for s in steps if s['step'] == 8), None)
    if step8:
        sf = step8['data'][0]['supervisory_factor_decimal']
        print(f"Supervisory Factor: {sf:.6f} ({sf*100:.3f}%)")
    
    step9 = next((s for s in steps if s['step'] == 9), None)
    if step9:
        adj_amount = step9['data'][0]['adjusted_derivatives_contract_amount']
        print(f"Adjusted Contract Amount: ${adj_amount:,.2f}")
    
    print("\n" + "="*80)
    print("CONCLUSION: Need to identify which parameters differ from images")
    print("="*80)

if __name__ == "__main__":
    try:
        compare_against_images()
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()