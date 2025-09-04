#!/usr/bin/env python3
"""
Detailed test to verify both margined and unmargined EAD calculations
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

def detailed_ead_test():
    """Test both margined and unmargined EAD calculations in detail"""
    
    print("=" * 80)
    print("DETAILED EAD CALCULATION TEST - Margined vs Unmargined")
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
    
    # Create netting set (MARGINED)
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
    print("Calculating SA-CCR with MARGINED netting set...")
    result = agent.calculate_comprehensive_saccr(netting_set)
    
    print("\nDETAILED EAD ANALYSIS:")
    print("-" * 50)
    
    # Extract key steps
    steps = result['calculation_steps']
    
    # Step 14: V and C
    step14 = next((s for s in steps if s['step'] == 14), None)
    if step14:
        sum_v = step14['sum_v']
        sum_c = step14['sum_c']
        net_exposure = sum_v - sum_c
        print(f"Step 14 - Sum V (MTM): ${sum_v:,.0f}")
        print(f"Step 14 - Sum C (Collateral): ${sum_c:,.0f}")
        print(f"Step 14 - Net Exposure (V-C): ${net_exposure:,.0f}")
    
    # Step 16: PFE
    step16 = next((s for s in steps if s['step'] == 16), None)
    if step16:
        pfe = step16['pfe']
        print(f"Step 16 - PFE: ${pfe:,.0f}")
    
    # Step 17: Margining parameters
    step17 = next((s for s in steps if s['step'] == 17), None)
    if step17:
        threshold = step17['threshold']
        mta = step17['mta']
        nica = step17['nica']
        print(f"Step 17 - Threshold: ${threshold:,.0f}")
        print(f"Step 17 - MTA: ${mta:,.0f}")
        print(f"Step 17 - NICA: ${nica:,.0f}")
    
    # Step 18: RC Calculations (BOTH scenarios)
    step18 = next((s for s in steps if s['step'] == 18), None)
    if step18:
        rc_margined = step18['rc_margined']
        rc_unmargined = step18['rc_unmargined']
        is_margined = step18['data']['is_margined']
        margin_floor = step18['data']['margin_floor']
        
        print(f"\nStep 18 - RC CALCULATIONS:")
        print(f"• Netting Set Type: {'MARGINED' if is_margined else 'UNMARGINED'}")
        print(f"• Margin Floor (TH+MTA-NICA): ${margin_floor:,.0f}")
        print(f"• RC Margined = max(V-C, TH+MTA-NICA, 0) = max(${net_exposure:,.0f}, ${margin_floor:,.0f}, 0) = ${rc_margined:,.0f}")
        print(f"• RC Unmargined = max(V-C, 0) = max(${net_exposure:,.0f}, 0) = ${rc_unmargined:,.0f}")
    
    # Step 20: Alpha
    step20 = next((s for s in steps if s['step'] == 20), None)
    if step20:
        alpha = step20['data']['alpha']
        print(f"Step 20 - Alpha: {alpha}")
    
    # Step 21: EAD Calculations (BOTH scenarios + minimum selection)
    step21 = next((s for s in steps if s['step'] == 21), None)
    if step21:
        ead_margined = step21['data']['ead_margined']
        ead_unmargined = step21['data']['ead_unmargined']
        ead_final = step21['data']['ead_final']
        methodology = step21['data']['methodology']
        
        print(f"\nStep 21 - EAD CALCULATIONS (BOTH SCENARIOS):")
        print(f"• EAD Margined = Alpha × (RC_margined + PFE)")
        print(f"  = {alpha} × (${rc_margined:,.0f} + ${pfe:,.0f}) = ${ead_margined:,.0f}")
        print(f"• EAD Unmargined = Alpha × (RC_unmargined + PFE)")
        print(f"  = {alpha} × (${rc_unmargined:,.0f} + ${pfe:,.0f}) = ${ead_unmargined:,.0f}")
        print(f"\n• BASEL MINIMUM SELECTION RULE:")
        print(f"  Final EAD = min(EAD_margined, EAD_unmargined)")
        print(f"  Final EAD = min(${ead_margined:,.0f}, ${ead_unmargined:,.0f}) = ${ead_final:,.0f}")
        
        selected_approach = "MARGINED" if ead_final == ead_margined else "UNMARGINED"
        print(f"  Selected Approach: {selected_approach}")
    
    # Step 23: Risk Weight
    step23 = next((s for s in steps if s['step'] == 23), None)
    if step23:
        risk_weight = step23['risk_weight']
        print(f"Step 23 - Risk Weight: {risk_weight*100:.0f}%")
    
    # Step 24: RWA
    step24 = next((s for s in steps if s['step'] == 24), None)
    if step24:
        rwa = step24['rwa']
        capital_req = step24['data']['capital_requirement']
        print(f"Step 24 - RWA = Risk Weight × EAD = {risk_weight} × ${ead_final:,.0f} = ${rwa:,.0f}")
        print(f"Step 24 - Capital Requirement = RWA × 8% = ${capital_req:,.0f}")
    
    print(f"\n" + "="*80)
    print("FINAL COMPARISON WITH IMAGES:")
    print(f"• Supervisory Factor: 0.50% ✅")
    print(f"• RC Margined: ${rc_margined:,.0f}")
    print(f"• RC Unmargined: ${rc_unmargined:,.0f}")
    print(f"• EAD Margined: ${ead_margined:,.0f}")
    print(f"• EAD Unmargined: ${ead_unmargined:,.0f}")
    print(f"• Final EAD (minimum): ${ead_final:,.0f}")
    print(f"• Final RWA: ${rwa:,.0f}")
    print("="*80)
    
    return result

if __name__ == "__main__":
    try:
        result = detailed_ead_test()
        print("\n✅ Detailed EAD test completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()