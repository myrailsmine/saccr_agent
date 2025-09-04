#!/usr/bin/env python3
"""
Simple test to verify the dual calculation implementation
"""

import sys
import os
from datetime import datetime, timedelta

# Add the current directory to Python path
sys.path.append('/app')

try:
    # Import the SA-CCR classes
    from enterprise_saccr_app import (
        ComprehensiveSACCRAgent, Trade, NettingSet, AssetClass, TradeType, Collateral
    )
    
    print("Imports successful")
    
    # Create simple test trade
    trade = Trade(
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
    
    print("Trade created successfully")
    
    # Create netting set
    netting_set = NettingSet(
        netting_set_id="212784050000389187901",
        counterparty="Lowell Hotel Properties LLC",
        trades=[trade],
        threshold=12000000,
        mta=1000000,
        nica=0
    )
    
    print("Netting set created successfully")
    
    # Test agent creation
    agent = ComprehensiveSACCRAgent()
    print("Agent created successfully")
    
    # Test calculation
    result = agent.calculate_comprehensive_saccr(netting_set)
    print("Calculation completed successfully")
    
    # Extract key results
    steps = result['calculation_steps']
    final_results = result['final_results']
    
    print(f"\nNumber of steps: {len(steps)}")
    print(f"Final EAD: ${final_results['exposure_at_default']:,.0f}")
    
    # Check for Step 21
    step21 = next((s for s in steps if s['step'] == 21), None)
    if step21:
        print(f"Step 21 EAD: ${step21['ead']:,.0f}")
        if 'data' in step21 and 'ead_margined' in step21['data']:
            print(f"EAD Margined: ${step21['data']['ead_margined']:,.0f}")
            print(f"EAD Unmargined: ${step21['data']['ead_unmargined']:,.0f}")
    
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()