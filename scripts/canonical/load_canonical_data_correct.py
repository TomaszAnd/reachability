#!/usr/bin/env python3
"""
CORRECT data loader for Canonical Ensemble based on DATA_PROVENANCE.md

This uses the EXACT access patterns documented and verified to work for v3 figures.
"""

import pickle
import numpy as np
from pathlib import Path

def load_canonical_data_correctly():
    """
    Load all canonical ensemble data using correct access patterns from DATA_PROVENANCE.md.
    
    Returns dict: {criterion: {d: {'K': array, 'P': array, 'err': array}}}
    """
    data = {
        'moment': {},
        'spectral': {},
        'krylov': {}
    }
    
    # =========================================================================
    # MOMENT: OLD comprehensive + NEW extension
    # =========================================================================
    print("Loading MOMENT data...")
    
    # OLD comprehensive (has transition region)
    try:
        with open('../data/raw_logs/comprehensive_reachability_20251209_153938.pkl', 'rb') as f:
            old_data = pickle.load(f)
        
        # ✅ CORRECT ACCESS: results['moment'][d] NOT results[d]['moment']!
        if 'results' in old_data and 'moment' in old_data['results']:
            for d in old_data['results']['moment'].keys():
                mom = old_data['results']['moment'][d]
                K = np.array(mom.get('K', []))
                P = np.array(mom.get('P', []))
                sem = np.array(mom.get('sem', mom.get('err', [])))
                
                if len(K) > 0:
                    data['moment'][d] = {
                        'K': K,
                        'P': P,
                        'err': sem
                    }
                    print(f"  Moment d={d}: {len(K)} points from OLD comprehensive")
    except Exception as e:
        print(f"  Warning: Could not load OLD comprehensive: {e}")
    
    # NEW moment extension (high ρ)
    try:
        with open('../data/raw_logs/moment_extension_all_dims_20251215_160333.pkl', 'rb') as f:
            ext_data = pickle.load(f)
        
        # ✅ CORRECT ACCESS: results[d] (NO nested 'moment' key!)
        if 'results' in ext_data:
            for d in ext_data['results'].keys():
                if isinstance(d, (int, np.integer)):
                    mom_ext = ext_data['results'][d]
                    K = np.array(mom_ext.get('K', []))
                    P = np.array(mom_ext.get('P', []))
                    sem = np.array(mom_ext.get('sem', mom_ext.get('err', [])))
                    
                    if len(K) > 0:
                        # Append to existing data
                        if d in data['moment']:
                            K_all = np.concatenate([data['moment'][d]['K'], K])
                            P_all = np.concatenate([data['moment'][d]['P'], P])
                            err_all = np.concatenate([data['moment'][d]['err'], sem])
                            
                            # Sort and deduplicate
                            idx = np.argsort(K_all)
                            K_all, P_all, err_all = K_all[idx], P_all[idx], err_all[idx]
                            _, unique_idx = np.unique(K_all, return_index=True)
                            
                            data['moment'][d] = {
                                'K': K_all[unique_idx],
                                'P': P_all[unique_idx],
                                'err': err_all[unique_idx]
                            }
                            print(f"  Moment d={d}: +{len(K)} points from NEW extension → {len(K_all[unique_idx])} total")
                        else:
                            data['moment'][d] = {'K': K, 'P': P, 'err': sem}
                            print(f"  Moment d={d}: {len(K)} points from NEW extension")
    except Exception as e:
        print(f"  Warning: Could not load moment extension: {e}")
    
    # =========================================================================
    # SPECTRAL: Use MERGED file (⭐ RECOMMENDED)
    # =========================================================================
    print("\nLoading SPECTRAL data...")
    
    try:
        with open('../data/raw_logs/spectral_complete_merged_20251216_153002.pkl', 'rb') as f:
            spectral_data = pickle.load(f)
        
        # ✅ CORRECT ACCESS: spectral[d]
        if 'spectral' in spectral_data:
            for d in spectral_data['spectral'].keys():
                if isinstance(d, (int, np.integer)):
                    spec = spectral_data['spectral'][d]
                    K = np.array(spec.get('K', []))
                    P = np.array(spec.get('P', []))
                    sem = np.array(spec.get('sem', spec.get('err', [])))
                    
                    if len(K) > 0:
                        data['spectral'][d] = {
                            'K': K,
                            'P': P,
                            'err': sem
                        }
                        print(f"  Spectral d={d}: {len(K)} points from MERGED")
    except Exception as e:
        print(f"  Warning: Could not load spectral merged: {e}")
    
    # =========================================================================
    # KRYLOV: FIXED + DENSE
    # =========================================================================
    print("\nLoading KRYLOV data...")
    
    # FIXED Krylov
    try:
        with open('../data/raw_logs/krylov_spectral_canonical_20251215_154634.pkl', 'rb') as f:
            fixed_data = pickle.load(f)
        
        # ✅ CORRECT ACCESS: results[d]['krylov']
        if 'results' in fixed_data:
            for d in fixed_data['results'].keys():
                if isinstance(d, (int, np.integer)) and 'krylov' in fixed_data['results'][d]:
                    K = np.array(fixed_data['results'][d].get('K', []))
                    kryl = fixed_data['results'][d]['krylov']
                    P = np.array(kryl.get('P', []))
                    sem = np.array(kryl.get('sem', kryl.get('err', [])))
                    
                    if len(K) > 0:
                        data['krylov'][d] = {
                            'K': K,
                            'P': P,
                            'err': sem
                        }
                        print(f"  Krylov d={d}: {len(K)} points from FIXED")
    except Exception as e:
        print(f"  Warning: Could not load krylov fixed: {e}")
    
    # DENSE Krylov (transition detail)
    try:
        with open('../data/raw_logs/krylov_dense_20251216_112335.pkl', 'rb') as f:
            dense_data = pickle.load(f)
        
        # ✅ CORRECT ACCESS: krylov[d]
        if 'krylov' in dense_data:
            for d in dense_data['krylov'].keys():
                if isinstance(d, (int, np.integer)):
                    kryl_dense = dense_data['krylov'][d]
                    K = np.array(kryl_dense.get('K', []))
                    P = np.array(kryl_dense.get('P', []))
                    sem = np.array(kryl_dense.get('sem', kryl_dense.get('err', [])))
                    
                    if len(K) > 0:
                        # Merge with FIXED data
                        if d in data['krylov']:
                            K_all = np.concatenate([data['krylov'][d]['K'], K])
                            P_all = np.concatenate([data['krylov'][d]['P'], P])
                            err_all = np.concatenate([data['krylov'][d]['err'], sem])
                            
                            # Sort and deduplicate
                            idx = np.argsort(K_all)
                            K_all, P_all, err_all = K_all[idx], P_all[idx], err_all[idx]
                            _, unique_idx = np.unique(K_all, return_index=True)
                            
                            data['krylov'][d] = {
                                'K': K_all[unique_idx],
                                'P': P_all[unique_idx],
                                'err': err_all[unique_idx]
                            }
                            print(f"  Krylov d={d}: +{len(K)} points from DENSE → {len(K_all[unique_idx])} total")
                        else:
                            data['krylov'][d] = {'K': K, 'P': P, 'err': sem}
                            print(f"  Krylov d={d}: {len(K)} points from DENSE")
    except Exception as e:
        print(f"  Warning: Could not load krylov dense: {e}")
    
    return data

if __name__ == '__main__':
    data = load_canonical_data_correctly()
    
    print("\n" + "="*70)
    print("FINAL DATA SUMMARY")
    print("="*70)
    for criterion in ['moment', 'spectral', 'krylov']:
        print(f"\n{criterion.upper()}:")
        for d in sorted(data[criterion].keys()):
            K = data[criterion][d]['K']
            P = data[criterion][d]['P']
            n_trans = np.sum((P > 0.05) & (P < 0.95))
            print(f"  d={d}: {len(K)} points, {n_trans} in transition, "
                  f"ρ∈[{K.min()/d**2:.4f}, {K.max()/d**2:.4f}]")
