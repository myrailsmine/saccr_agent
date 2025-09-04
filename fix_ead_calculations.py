#!/usr/bin/env python3
"""
Fix EAD calculations to match exact image values
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

def analyze_required_adjustments():
    """Analyze what adjustments are needed to match image values"""
    
    print("=" * 80)
    print("ANALYZING REQUIRED ADJUSTMENTS TO MATCH IMAGE VALUES")
    print("=" * 80)
    
    # Target values from images (as provided by user)
    target_ead_margined = 14022368
    target_ead_unmargined = 11790314
    target_final_ead = 11790314
    
    # Current calculation
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
    
    agent = ComprehensiveSACCRAgent()
    result = agent.calculate_comprehensive_saccr(netting_set)
    steps = result['calculation_steps']
    
    # Extract current values
    step18 = next((s for s in steps if s['step'] == 18), None)
    step16 = next((s for s in steps if s['step'] == 16), None)
    step20 = next((s for s in steps if s['step'] == 20), None)
    
    current_rc_margined = step18['data']['rc_margined']
    current_rc_unmargined = step18['data']['rc_unmargined']
    current_pfe = step16['pfe']
    alpha = step20['data']['alpha']
    
    print("CURRENT VALUES:")
    print(f"RC Margined: ${current_rc_margined:,.0f}")
    print(f"RC Unmargined: ${current_rc_unmargined:,.0f}")
    print(f"PFE: ${current_pfe:,.0f}")
    print(f"Alpha: {alpha}")
    print()
    
    # Calculate what the combined (RC + PFE) should be for target EAD values
    required_combined_margined = target_ead_margined / alpha
    required_combined_unmargined = target_ead_unmargined / alpha
    
    print("REQUIRED VALUES TO MATCH IMAGES:")
    print(f"Target EAD Margined: ${target_ead_margined:,.0f}")
    print(f"Required (RC_margined + PFE): ${required_combined_margined:,.0f}")
    print(f"Current (RC_margined + PFE): ${current_rc_margined + current_pfe:,.0f}")
    print(f"Adjustment needed: ${required_combined_margined - (current_rc_margined + current_pfe):,.0f}")
    print()
    
    print(f"Target EAD Unmargined: ${target_ead_unmargined:,.0f}")
    print(f"Required (RC_unmargined + PFE): ${required_combined_unmargined:,.0f}")
    print(f"Current (RC_unmargined + PFE): ${current_rc_unmargined + current_pfe:,.0f}")
    print(f"Adjustment needed: ${required_combined_unmargined - (current_rc_unmargined + current_pfe):,.0f}")
    print()
    
    # Possible scenarios for adjustments
    print("POSSIBLE ADJUSTMENT SCENARIOS:")
    print("=" * 35)
    
    print("Scenario 1: Adjust PFE only")
    required_pfe_margined = required_combined_margined - current_rc_margined
    required_pfe_unmargined = required_combined_unmargined - current_rc_unmargined
    print(f"  For margined: PFE should be ${required_pfe_margined:,.0f} (current: ${current_pfe:,.0f})")
    print(f"  For unmargined: PFE should be ${required_pfe_unmargined:,.0f} (current: ${current_pfe:,.0f})")
    print(f"  Problem: PFE should be the same for both scenarios")
    print()
    
    print("Scenario 2: Adjust RC values")
    required_rc_margined = required_combined_margined - current_pfe
    required_rc_unmargined = required_combined_unmargined - current_pfe
    print(f"  RC Margined should be: ${required_rc_margined:,.0f} (current: ${current_rc_margined:,.0f})")
    print(f"  RC Unmargined should be: ${required_rc_unmargined:,.0f} (current: ${current_rc_unmargined:,.0f})")
    print()
    
    print("Scenario 3: Adjust margining parameters")
    # Work backwards from required RC values
    sum_v = 8382419  # Current MTM
    sum_c = 0        # No collateral
    net_exposure = sum_v - sum_c
    
    # For unmargined: RC = max(V-C, 0) = max(8382419, 0) = 8382419
    # This matches current calculation, so unmargined RC seems correct
    
    # For margined: RC = max(V-C, TH+MTA-NICA, 0)
    # Current: max(8382419, 13000000, 0) = 13000000
    # Required: ${required_rc_margined:,.0f}
    
    required_margin_floor = required_rc_margined  # Since V-C < required RC
    print(f"  Required margin floor (TH+MTA-NICA): ${required_margin_floor:,.0f}")
    print(f"  Current margin floor: ${current_rc_margined:,.0f}")
    print()
    
    # Try different parameter combinations
    print("SUGGESTED PARAMETER ADJUSTMENTS:")
    print("=" * 33)
    
    # If we need margin floor of required_rc_margined
    if required_margin_floor != current_rc_margined:
        adjustment = required_margin_floor - 13000000  # Current is TH(12M) + MTA(1M) - NICA(0) = 13M
        print(f"Option 1: Adjust threshold by ${adjustment:,.0f}")
        print(f"  New Threshold: ${12000000 + adjustment:,.0f}")
        print()
        
        print(f"Option 2: Adjust MTA by ${adjustment:,.0f}")
        print(f"  New MTA: ${1000000 + adjustment:,.0f}")
        print()
    
    # The most likely scenario is that some input parameters are different
    print("RECOMMENDATION:")
    print("Please check the exact values from the images for:")
    print("1. Threshold (TH) amount")
    print("2. MTA amount") 
    print("3. NICA amount")
    print("4. MTM value (Sum V)")
    print("5. Collateral amount (Sum C)")
    print("6. PFE calculation components")
    
    return {
        'required_rc_margined': required_rc_margined,
        'required_rc_unmargined': required_rc_unmargined,
        'current_pfe': current_pfe
    }

if __name__ == "__main__":
    try:
        results = analyze_required_adjustments()
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()