import zarr
import numpy as np

def is_valid_sample(sample, verbose=False):
    "Check if a sample is valid."
    # NOTE: These conditions are heuristic
    valid = (0 < np.max(sample) < 10000) and np.min(sample) >= 0
    if verbose and not valid:
        print(F"({0 < np.max(sample)} && {np.max(sample)< 10000}) && {np.min(sample) >= 0}")
    return valid

def analyze_zarr_array(zarr_path):
    """
    Analyze the Zarr array to understand data distribution and zero/NaN patterns.
    """
    # Open the Zarr store
    store = zarr.open(zarr_path, mode='r')
    minicubes_array = store['Minicubes']
    
    # Get metadata
    minicube_names = store['minicubes'][:]
    time_values = store['time'][:]
    variable_names = store['variable'][:]
    
    print("\n=== Zarr Array Analysis ===")
    print(f"Array shape: {minicubes_array.shape}")
    print(f"Data type: {minicubes_array.dtype}")
    
    # Sample random locations to analyze data distribution
    num_samples = 1000
    
    # Generate random indices
    m_indices = np.random.randint(0, minicubes_array.shape[0], size=num_samples)
    t_indices = np.random.randint(0, minicubes_array.shape[1], size=num_samples)
    v_indices = np.random.randint(0, minicubes_array.shape[2], size=num_samples)
    
    # Statistics containers
    all_zeros_count = 0
    mostly_zeros_count = 0
    nan_count = 0
    valid_count = 0
    bad_sample_count = 0
    values = []
    
    print(f"Analyzing {num_samples} random samples...")
    
    for i in range(num_samples):
        try:
            # Extract sample
            img = minicubes_array[m_indices[i], t_indices[i], v_indices[i], :, :]
            img_np = np.array(img)
            
            # Check for NaNs
            if np.isnan(img_np).any():
                nan_count += 1
                continue
                
            # Check for all zeros
            if np.allclose(img_np, 0, atol=1e-6):
                all_zeros_count += 1
                continue
                
            # Check for mostly zeros (>95%)
            if np.sum(np.abs(img_np) < 1e-6) / img_np.size > 0.95:
                mostly_zeros_count += 1

            if is_valid_sample(img_np):
                bad_sample_count += 1
                continue
            
            # Valid image
            valid_count += 1
            
            # Store some statistics about non-zero values
            non_zero = img_np[np.abs(img_np) > 1e-6]
            if len(non_zero) > 0:
                values.append({
                    'min': float(np.min(non_zero)),
                    'max': float(np.max(non_zero)),
                    'mean': float(np.mean(non_zero)),
                    # 'std': float(np.std(non_zero)), NOTE: This is typically undefined.
                    'minicube': minicube_names[m_indices[i]] if m_indices[i] < len(minicube_names) else f"idx_{m_indices[i]}",
                    'time': time_values[t_indices[i]] if t_indices[i] < len(time_values) else f"idx_{t_indices[i]}",
                    'variable': variable_names[v_indices[i]] if v_indices[i] < len(variable_names) else f"idx_{v_indices[i]}"
                })
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
    
    # Print analysis results
    print("\nAnalysis results:")
    print(f"- All zeros: {all_zeros_count}/{num_samples} ({all_zeros_count/num_samples*100:.1f}%)")
    print(f"- Mostly zeros (>95%): {mostly_zeros_count}/{num_samples} ({mostly_zeros_count/num_samples*100:.1f}%)")
    print(f"- Contains NaN: {nan_count}/{num_samples} ({nan_count/num_samples*100:.1f}%)")
    print(f"- Valid images: {valid_count}/{num_samples} ({valid_count/num_samples*100:.1f}%)")
    print(f"- Bad images: {bad_sample_count}/{num_samples} ({bad_sample_count/num_samples*100:.1f}%)")
    
    if values:
        # Calculate overall statistics
        all_mins = [v['min'] for v in values]
        all_maxs = [v['max'] for v in values]
        all_means = [v['mean'] for v in values]
        
        print("\nValue distribution for non-zero pixels:")
        print(f"- Min range: {np.min(all_mins)} to {np.max(all_mins)}")
        print(f"- Max range: {np.min(all_maxs)} to {np.max(all_maxs)}")
        print(f"- Mean range: {np.min(all_means)} to {np.max(all_means)}")
        
        # Analysis by variable
        var_stats = {}
        for v in values:
            var = v['variable']
            if var not in var_stats:
                var_stats[var] = {'mins': [], 'maxs': [], 'means': []}
            var_stats[var]['mins'].append(v['min'])
            var_stats[var]['maxs'].append(v['max'])
            var_stats[var]['means'].append(v['mean'])
        
        print("\nValue distribution by variable:")
        for var, stats in var_stats.items():
            if stats['mins']:
                print(f"- {var}:")
                print(f"  - Min range: {np.min(stats['mins'])} to {np.max(stats['mins'])}")
                print(f"  - Max range: {np.min(stats['maxs'])} to {np.max(stats['maxs'])}")
                print(f"  - Mean: {np.mean(stats['means']):.4f} ± {np.std(stats['means']):.4f}")
    
    # Analyze specific variables (like bands)
    band_vars = [v for v in variable_names if v.startswith('B')]
    other_vars = [v for v in variable_names if not v.startswith('B')]
    
    print("\nVariables:")
    print(f"- Band variables: {band_vars}")
    print(f"- Other variables: {other_vars}")
    
    # Return some useful information for further processing
    return {
        'all_zeros_percentage': all_zeros_count/num_samples*100,
        'nan_percentage': nan_count/num_samples*100,
        'valid_percentage': valid_count/num_samples*100,
        'variable_names': list(variable_names),
        'band_vars': band_vars,
        'other_vars': other_vars
    }

# Update main to use these new functions
if __name__ == "__main__":
    # Set the path to your Zarr store
    zarr_path = "../data"
    
    # Example 1: Analyze the Zarr array
    analysis_results = analyze_zarr_array(zarr_path)
    