#!/usr/bin/env python3
"""Quick test of unified production script with reduced parameters."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Override parameters for quick test
import scripts.production.generate_unified_production as gen

gen.DIMENSIONS = [8, 16]
gen.N_TRIALS = 10
gen.N_RHO_POINTS = 5
gen.TAU_VALUES = [0.99]

gen.main()
