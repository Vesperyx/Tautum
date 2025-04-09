"""
Large-Scale Prime Phase Identity Validator

This script tests the Prime-Phase Identity function:
φ'(P) = (P·2π·√τ_R) mod 2π

It analyzes the distribution of phase values for all primes up to a large limit
to determine if primes have a non-uniform distribution in phase space.

The code is optimized for consumer PC hardware, using efficient prime generation
and statistical analysis techniques.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import time
from scipy import stats
import multiprocessing as mp
from math import gcd
from functools import reduce

# Define the Rosetta constant and its derivatives
TAU_R = 2.203e-15  # Rosetta constant in seconds
SQRT_TAU_R = np.sqrt(TAU_R)  # Square root
PERIOD = 1 / SQRT_TAU_R  # Period in P for a full 2π cycle

# Print constants for reference
print(f"Rosetta constant (τ_R): {TAU_R:.6e} s")
print(f"Square root of τ_R: {SQRT_TAU_R:.6e} s^0.5")
print(f"Period (1/√τ_R): {PERIOD:.6e}")

def phi_prime(P):
    """
    Calculate φ' = (P * 2π * sqrt(τ_R)) modulo 2π.
    
    Parameters:
      P : scalar or numpy array
          The input number(s).
      
    Returns:
      phi_p: Values in the range [0, 2π)
    """
    value = P * 2 * np.pi * SQRT_TAU_R
    phi_p = np.mod(value, 2 * np.pi)
    return phi_p

def generate_primes_sieve(limit):
    """
    Generate all primes up to limit using the Sieve of Eratosthenes.
    
    Parameters:
        limit: Upper bound for prime generation
        
    Returns:
        numpy array of primes
    """
    print(f"Generating primes up to {limit:,} using Sieve of Eratosthenes...")
    start_time = time.time()
    
    # Initialize the sieve
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0:2] = False
    
    # Mark multiples of each prime as non-prime
    for i in range(2, int(np.sqrt(limit)) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    
    # Extract the primes
    primes = np.where(sieve)[0]
    
    elapsed = time.time() - start_time
    print(f"Generated {len(primes):,} primes in {elapsed:.2f} seconds")
    
    return primes

def analyze_prime_phase_distribution(primes, num_bins=1000):
    """
    Analyze the distribution of phase values for the given primes.
    
    Parameters:
        primes: numpy array of prime numbers
        num_bins: number of bins for histogram (higher = more detailed)
        
    Returns:
        dict with analysis results
    """
    print(f"Analyzing phase distribution for {len(primes):,} primes using {num_bins} bins...")
    start_time = time.time()
    
    # Calculate phase values for all primes
    phases = phi_prime(primes)
    
    # Create histogram of phases
    hist, bin_edges = np.histogram(phases, bins=num_bins, range=(0, 2*np.pi))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Expected count for uniform distribution
    expected_count = len(primes) / num_bins
    
    # Calculate chi-square statistic to test for uniformity
    chi2_stat = np.sum((hist - expected_count)**2 / expected_count)
    chi2_pvalue = 1 - stats.chi2.cdf(chi2_stat, num_bins - 1)
    
    # Calculate Kolmogorov-Smirnov test
    # First, create empirical CDF
    sorted_phases = np.sort(phases)
    empirical_cdf = np.arange(1, len(phases) + 1) / len(phases)
    
    # Theoretical CDF for uniform distribution
    theoretical_cdf = sorted_phases / (2 * np.pi)
    
    # KS test
    ks_stat, ks_pvalue = stats.kstest(empirical_cdf, theoretical_cdf)
    
    elapsed = time.time() - start_time
    print(f"Analysis completed in {elapsed:.2f} seconds")
    
    return {
        'phases': phases,
        'histogram': hist,
        'bin_centers': bin_centers,
        'chi2_stat': chi2_stat,
        'chi2_pvalue': chi2_pvalue,
        'ks_stat': ks_stat,
        'ks_pvalue': ks_pvalue,
        'expected_count': expected_count
    }

def phase_difference_analysis(primes):
    """
    Analyze the differences between consecutive prime phases.
    
    Parameters:
        primes: numpy array of prime numbers
        
    Returns:
        dict with analysis results
    """
    print(f"Analyzing phase differences between consecutive primes...")
    start_time = time.time()
    
    # Calculate phases
    phases = phi_prime(primes)
    
    # Calculate phase differences (handling the circular nature of phases)
    phase_diffs = np.diff(phases)
    
    # Adjust differences that cross the 0/2π boundary
    phase_diffs = np.where(phase_diffs > np.pi, phase_diffs - 2*np.pi, phase_diffs)
    phase_diffs = np.where(phase_diffs < -np.pi, phase_diffs + 2*np.pi, phase_diffs)
    
    # Calculate statistics
    mean_diff = np.mean(phase_diffs)
    std_diff = np.std(phase_diffs)
    
    elapsed = time.time() - start_time
    print(f"Phase difference analysis completed in {elapsed:.2f} seconds")
    
    return {
        'phase_diffs': phase_diffs,
        'mean_diff': mean_diff,
        'std_diff': std_diff
    }

def modular_analysis(primes, periods_to_test=None):
    """
    Analyze if primes have a pattern within specific periods in P-space.
    
    Parameters:
        primes: numpy array of prime numbers
        periods_to_test: list of period values to test (if None, uses default set)
        
    Returns:
        dict with analysis results
    """
    if periods_to_test is None:
        # Test the main period and some common fractions
        periods_to_test = [
            PERIOD,
            PERIOD/2,
            PERIOD/3,
            PERIOD/4,
            PERIOD/5,
            PERIOD/6,
            PERIOD/10
        ]
    
    print(f"Performing modular analysis with {len(periods_to_test)} different periods...")
    start_time = time.time()
    
    results = {}
    
    for period in periods_to_test:
        # Calculate the position within the period for each prime
        position = np.mod(primes, period) / period
        
        # Create histogram
        hist, bin_edges = np.histogram(position, bins=100, range=(0, 1))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Chi-square test for uniformity
        expected_count = len(primes) / 100
        chi2_stat = np.sum((hist - expected_count)**2 / expected_count)
        chi2_pvalue = 1 - stats.chi2.cdf(chi2_stat, 99)
        
        results[period] = {
            'histogram': hist,
            'bin_centers': bin_centers,
            'chi2_stat': chi2_stat,
            'chi2_pvalue': chi2_pvalue
        }
    
    elapsed = time.time() - start_time
    print(f"Modular analysis completed in {elapsed:.2f} seconds")
    
    return results

def visualize_results(phase_results, diff_results, mod_results, save_figs=True):
    """
    Visualize the analysis results with multiple plots.
    
    Parameters:
        phase_results: results from phase distribution analysis
        diff_results: results from phase difference analysis
        mod_results: results from modular analysis
        save_figs: whether to save figures to disk
    """
    print("Generating visualizations...")
    plt.figure(figsize=(18, 10))
    
    # 1. Phase distribution histogram
    plt.subplot(2, 3, 1)
    plt.bar(phase_results['bin_centers'], phase_results['histogram'], 
            width=2*np.pi/len(phase_results['bin_centers']), alpha=0.7)
    plt.axhline(phase_results['expected_count'], color='r', linestyle='--', 
               label=f'Expected (uniform)')
    plt.xlabel('Phase (radians)')
    plt.ylabel('Count')
    plt.title(f'Phase Distribution of Primes\nχ² p-value: {phase_results["chi2_pvalue"]:.6f}')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 2. Phase distribution on unit circle
    plt.subplot(2, 3, 2, polar=True)
    theta = phase_results['bin_centers']
    radii = phase_results['histogram']
    width = 2*np.pi/len(theta)
    plt.bar(theta, radii, width=width, alpha=0.7)
    plt.title('Phase Distribution (Polar)')
    
    # 3. Phase differences histogram
    plt.subplot(2, 3, 3)
    plt.hist(diff_results['phase_diffs'], bins=100, alpha=0.7)
    plt.xlabel('Phase Difference (radians)')
    plt.ylabel('Count')
    plt.title(f'Phase Differences Between Consecutive Primes\nMean: {diff_results["mean_diff"]:.6f}')
    plt.grid(alpha=0.3)
    
    # 4. Deviation from uniformity
    plt.subplot(2, 3, 4)
    deviation = (phase_results['histogram'] - phase_results['expected_count']) / np.sqrt(phase_results['expected_count'])
    plt.bar(phase_results['bin_centers'], deviation, width=2*np.pi/len(phase_results['bin_centers']), alpha=0.7)
    plt.axhline(0, color='r', linestyle='-')
    plt.axhline(2, color='g', linestyle='--', alpha=0.5, label='2σ')
    plt.axhline(-2, color='g', linestyle='--', alpha=0.5)
    plt.axhline(3, color='y', linestyle='--', alpha=0.5, label='3σ')
    plt.axhline(-3, color='y', linestyle='--', alpha=0.5)
    plt.xlabel('Phase (radians)')
    plt.ylabel('Deviation (sigma)')
    plt.title('Deviation from Uniform Distribution')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 5. Modular analysis (for the main period)
    plt.subplot(2, 3, 5)
    main_period_result = mod_results[PERIOD]
    plt.bar(main_period_result['bin_centers'], main_period_result['histogram'], alpha=0.7)
    expected_count = main_period_result['histogram'].mean()
    plt.axhline(expected_count, color='r', linestyle='--', label='Expected (uniform)')
    plt.xlabel('Position within Period')
    plt.ylabel('Count')
    plt.title(f'Distribution within Main Period\nχ² p-value: {main_period_result["chi2_pvalue"]:.6f}')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 6. Heatmap for modular analysis
    plt.subplot(2, 3, 6)
    periods = list(mod_results.keys())
    p_values = [mod_results[p]['chi2_pvalue'] for p in periods]
    
    # Convert periods to fractions of the main period for clearer labeling
    period_labels = []
    for p in periods:
        ratio = PERIOD / p
        if abs(ratio - round(ratio)) < 0.01:  # If it's very close to an integer
            period_labels.append(f'1/{int(ratio)}')
        else:
            period_labels.append(f'{p:.2e}')
    
    # Sort by p-value
    sorted_indices = np.argsort(p_values)
    periods_sorted = [period_labels[i] for i in sorted_indices]
    p_values_sorted = [p_values[i] for i in sorted_indices]
    
    plt.barh(periods_sorted, p_values_sorted, alpha=0.7)
    plt.axvline(0.05, color='r', linestyle='--', label='p=0.05')
    plt.xscale('log')
    plt.xlabel('p-value (log scale)')
    plt.ylabel('Period')
    plt.title('Chi-Square p-values for Different Periods')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_figs:
        plt.savefig('prime_phase_analysis.png', dpi=300, bbox_inches='tight')
    
    # Additional visualization: 2D heatmap of primes in phase space
    # This plots the primes by their value (x-axis) and phase (y-axis)
    plt.figure(figsize=(12, 8))
    
    # Take a sample of primes if there are too many
    max_primes_for_scatter = 10000
    if len(phase_results['phases']) > max_primes_for_scatter:
        # Take a random sample
        indices = np.random.choice(len(phase_results['phases']), max_primes_for_scatter, replace=False)
        scatter_primes = np.array([phase_results['phases'][i] for i in indices])
        scatter_values = np.array([phase_results['phases'][i] for i in indices])
    else:
        scatter_primes = phase_results['phases']
        scatter_values = phase_results['phases']
    
    # Calculate 2D histogram for density plot
    hist2d, xedges, yedges = np.histogram2d(
        phase_results['phases'], 
        np.mod(np.arange(len(phase_results['phases'])), PERIOD) / PERIOD,
        bins=[100, 100],
        range=[[0, 2*np.pi], [0, 1]]
    )
    
    # Create the plot
    plt.imshow(hist2d.T, aspect='auto', origin='lower', 
               extent=[0, 2*np.pi, 0, 1],
               interpolation='nearest', cmap='viridis',
               norm=LogNorm())
    plt.colorbar(label='Count (log scale)')
    plt.xlabel('Phase (radians)')
    plt.ylabel('Position within Period')
    plt.title('Prime Distribution in Phase Space')
    plt.grid(False)
    
    if save_figs:
        plt.savefig('prime_phase_heatmap.png', dpi=300, bbox_inches='tight')
    
    plt.show()

def batch_prime_analysis(start, end, batch_size=10**7):
    """
    Process and analyze primes in batches to handle very large numbers.
    
    Parameters:
        start: Starting number for prime search
        end: Ending number for prime search
        batch_size: Size of each batch
        
    Returns:
        Combined results from all batches
    """
    all_results = {
        'phases': np.array([]),
        'phase_diffs': np.array([])
    }
    
    current_start = start
    while current_start < end:
        current_end = min(current_start + batch_size, end)
        print(f"Processing batch from {current_start:,} to {current_end:,}...")
        
        # Generate primes in this range
        primes_segment = generate_primes_segment(current_start, current_end)
        
        if len(primes_segment) > 0:
            # Process this batch
            phases = phi_prime(primes_segment)
            
            # Store phases
            all_results['phases'] = np.append(all_results['phases'], phases)
            
            # Calculate phase differences if we have previous results
            if len(all_results['phase_diffs']) > 0 and len(all_results['phases']) > 1:
                # Calculate difference between last phase of previous batch and first phase of this batch
                prev_last = all_results['phases'][-len(phases)-1]
                curr_first = phases[0]
                boundary_diff = curr_first - prev_last
                # Adjust for circular boundary
                if boundary_diff > np.pi:
                    boundary_diff -= 2*np.pi
                elif boundary_diff < -np.pi:
                    boundary_diff += 2*np.pi
                
                # Calculate differences within this batch
                batch_diffs = np.diff(phases)
                batch_diffs = np.where(batch_diffs > np.pi, batch_diffs - 2*np.pi, batch_diffs)
                batch_diffs = np.where(batch_diffs < -np.pi, batch_diffs + 2*np.pi, batch_diffs)
                
                # Combine with boundary difference
                all_diffs = np.append(np.array([boundary_diff]), batch_diffs)
                all_results['phase_diffs'] = np.append(all_results['phase_diffs'], all_diffs)
            elif len(phases) > 1:
                # Only calculate differences within this first batch
                batch_diffs = np.diff(phases)
                batch_diffs = np.where(batch_diffs > np.pi, batch_diffs - 2*np.pi, batch_diffs)
                batch_diffs = np.where(batch_diffs < -np.pi, batch_diffs + 2*np.pi, batch_diffs)
                all_results['phase_diffs'] = batch_diffs
        
        current_start = current_end
    
    return all_results

def generate_primes_segment(start, end):
    """
    Generate primes in a specific range using a segmented sieve.
    More memory efficient for large numbers.
    
    Parameters:
        start: Lower bound
        end: Upper bound
        
    Returns:
        numpy array of primes in the range [start, end)
    """
    if start < 2:
        start = 2
    
    # First, generate small primes up to sqrt(end)
    sqrt_end = int(np.sqrt(end)) + 1
    small_primes = generate_primes_sieve(sqrt_end)
    
    # Now, use these small primes to sieve the segment
    segment_size = min(10**8, end - start)  # Limit memory usage
    
    all_primes = []
    
    # Process the range in segments
    for seg_start in range(start, end, segment_size):
        seg_end = min(seg_start + segment_size, end)
        
        # Create sieve for this segment
        sieve = np.ones(seg_end - seg_start, dtype=bool)
        
        # Mark multiples of each small prime
        for p in small_primes:
            # Find the first multiple of p that is >= seg_start
            first_multiple = ((seg_start + p - 1) // p) * p
            if first_multiple < seg_start:
                first_multiple += p
            
            # Mark all multiples of p in this segment
            for i in range(first_multiple, seg_end, p):
                sieve[i - seg_start] = False
        
        # If the segment includes 1, mark it as non-prime
        if seg_start <= 1 < seg_end:
            sieve[1 - seg_start] = False
        
        # Extract primes from this segment
        segment_primes = np.where(sieve)[0] + seg_start
        all_primes.append(segment_primes)
    
    # Combine all segments
    if all_primes:
        return np.concatenate(all_primes)
    else:
        return np.array([])

def run_prime_phase_analysis():
    """Main function to run the complete analysis."""
    # We'll run analysis with different upper limits to see if patterns emerge
    # or become stronger with more primes
    
    # The user can adjust these limits based on their hardware capabilities
    limits = [
        10**6,    # 1 million - Quick initial test
        10**7,    # 10 million - Medium-scale test
        10**8     # 100 million - Large-scale test (may take significant time)
    ]
    
    results = {}
    
    for limit in limits:
        print(f"\n{'='*80}")
        print(f"ANALYZING PRIMES UP TO {limit:,}")
        print(f"{'='*80}")
        
        # Generate primes
        primes = generate_primes_sieve(limit)
        
        # Basic analysis of phase distribution
        phase_results = analyze_prime_phase_distribution(primes)
        
        # Analysis of phase differences
        diff_results = phase_difference_analysis(primes)
        
        # Modular analysis
        mod_results = modular_analysis(primes)
        
        # Visualize results
        visualize_results(phase_results, diff_results, mod_results, save_figs=True)
        
        # Store results for comparison
        results[limit] = {
            'primes_count': len(primes),
            'phase_results': phase_results,
            'diff_results': diff_results,
            'mod_results': mod_results
        }
        
        # Check for significance
        if phase_results['chi2_pvalue'] < 0.01:
            print(f"\n⭐ SIGNIFICANT NON-UNIFORMITY DETECTED at limit {limit:,}")
            print(f"Chi-square p-value: {phase_results['chi2_pvalue']:.6e}")
        else:
            print(f"\nNo significant non-uniformity detected at limit {limit:,}")
            print(f"Chi-square p-value: {phase_results['chi2_pvalue']:.6f}")
    
    # Compare results across different limits
    print("\n\nFINAL COMPARISON ACROSS LIMITS:")
    print(f"{'Limit':<12} {'Primes Count':<15} {'Chi² p-value':<15} {'KS p-value':<15}")
    print(f"{'-'*60}")
    
    for limit, result in results.items():
        print(f"{limit:<12,} {result['primes_count']:<15,} "
              f"{result['phase_results']['chi2_pvalue']:<15.6e} "
              f"{result['phase_results']['ks_pvalue']:<15.6e}")
    
    return results

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run the analysis
    results = run_prime_phase_analysis()
    
    # Final conclusion
    min_pvalue = min(result['phase_results']['chi2_pvalue'] for result in results.values())
    
    print("\n\nFINAL CONCLUSION:")
    if min_pvalue < 0.01:
        print(f"✅ VALIDATED: Prime numbers show significant non-uniformity in phase space")
        print(f"Lowest p-value: {min_pvalue:.6e}")
    else:
        print(f"❌ INVALIDATED: No significant non-uniformity detected in prime phase distribution")
        print(f"Lowest p-value: {min_pvalue:.6f}")
