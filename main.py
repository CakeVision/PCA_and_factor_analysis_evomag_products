import claude_pca
import claude_factor_analysis as factor_analysis
import pandas as pd


if __name__ == '__main__':
    df = pd.read_json('evomag_2024_11_12.json')
    claude_pca.run_pca_analysis(df)
    prepared_data = claude_pca.prepare_data(df)

    # 3. Run Factor Analysis
    print("\n=== Running Factor Analysis ===")
    fa_results = factor_analysis.perform_factor_analysis(prepared_data)
     # 4. Check Factor Analysis Results
    if 'error' in fa_results:
        print(f"Factor Analysis Error: {fa_results['error']}")
        print("Bartlett's test results:")
        print(f"Chi-square value: {fa_results['bartlett_test']['chi_square']:.2f}")
        print(f"p-value: {fa_results['bartlett_test']['p_value']:.4f}")
    else:
        # Print key results
        print("\nBartlett's test results:")
        print(f"Chi-square value: {fa_results['bartlett_test']['chi_square']:.2f}")
        print(f"p-value: {fa_results['bartlett_test']['p_value']:.4f}")

        print(f"\nOverall KMO score: {fa_results['kmo']['overall']:.3f}")

        print(f"\nNumber of factors extracted: {fa_results['n_factors']}")

        print("\nVariance explained by rotated factors:")
        print(fa_results['variance']['rotated'])

        # 5. Create visualizations
        print("\n=== Creating Visualizations ===")
        factor_analysis.plot_factor_analysis(fa_results)

        # 6. Save results if needed
        # Uncomment and modify these lines if you want to save results to files
        # fa_results['loadings']['rotated'].to_csv('rotated_loadings.csv')
        # fa_results['scores']['rotated'].to_csv('rotated_scores.csv')
        # fa_results['communalities'].to_csv('communalities.csv')