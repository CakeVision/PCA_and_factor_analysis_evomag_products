import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
import matplotlib.pyplot as plt
import seaborn as sns

def perform_factor_analysis(data):
    """
    Perform factor analysis on the dataset including Bartlett's test, KMO indices,
    and factor analysis with and without rotation.

    Parameters:
    data (pandas.DataFrame): Prepared dataset for factor analysis

    Returns:
    dict: Dictionary containing all factor analysis results
    """
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Perform Bartlett's test of sphericity
    chi_square_value, p_value = calculate_bartlett_sphericity(data_scaled)

    # Calculate KMO indices
    kmo_all, kmo_model = calculate_kmo(data_scaled)
    kmo_per_variable = pd.Series(kmo_all, index=data.columns)

    # Check if we should proceed (Bartlett's test)
    if p_value > 0.05:
        return {
            'bartlett_test': {'chi_square': chi_square_value, 'p_value': p_value},
            'error': 'Factor analysis not recommended: Failed Bartlett\'s test'
        }

    # Perform factor analysis without rotation
    fa_unrotated = FactorAnalyzer(rotation=None, n_factors=min(data.shape[1], data.shape[0]))
    fa_unrotated.fit(data_scaled)

    # Get eigenvalues and variance explained
    ev, v = fa_unrotated.get_eigenvalues()
    n_factors = sum(ev > 1)  # Kaiser criterion

    # Perform factor analysis with optimal number of factors
    fa_unrotated = FactorAnalyzer(rotation=None, n_factors=n_factors)
    fa_unrotated.fit(data_scaled)

    # Perform factor analysis with varimax rotation
    fa_rotated = FactorAnalyzer(rotation='varimax', n_factors=n_factors)
    fa_rotated.fit(data_scaled)

    # Get factor loadings
    loadings_unrotated = pd.DataFrame(
        fa_unrotated.loadings_,
        columns=[f'Factor{i+1}' for i in range(n_factors)],
        index=data.columns
    )

    loadings_rotated = pd.DataFrame(
        fa_rotated.loadings_,
        columns=[f'Factor{i+1}' for i in range(n_factors)],
        index=data.columns
    )

    # Calculate variance explained
    variance_unrotated = pd.DataFrame({
        'SS Loadings': fa_unrotated.get_factor_variance()[0],
        'Proportion Var': fa_unrotated.get_factor_variance()[1],
        'Cumulative Var': fa_unrotated.get_factor_variance()[2]
    }, index=[f'Factor{i+1}' for i in range(n_factors)])

    variance_rotated = pd.DataFrame({
        'SS Loadings': fa_rotated.get_factor_variance()[0],
        'Proportion Var': fa_rotated.get_factor_variance()[1],
        'Cumulative Var': fa_rotated.get_factor_variance()[2]
    }, index=[f'Factor{i+1}' for i in range(n_factors)])

    # Calculate communalities
    communalities = pd.DataFrame({
        'Initial': fa_unrotated.get_communalities(),
        'Extraction': fa_rotated.get_communalities()
    }, index=data.columns)

    # Calculate factor scores
    scores_unrotated = pd.DataFrame(
        fa_unrotated.transform(data_scaled),
        columns=[f'Factor{i+1}' for i in range(n_factors)]
    )

    scores_rotated = pd.DataFrame(
        fa_rotated.transform(data_scaled),
        columns=[f'Factor{i+1}' for i in range(n_factors)]
    )

    return {
        'bartlett_test': {'chi_square': chi_square_value, 'p_value': p_value},
        'kmo': {
            'overall': kmo_model,
            'per_variable': kmo_per_variable
        },
        'n_factors': n_factors,
        'loadings': {
            'unrotated': loadings_unrotated,
            'rotated': loadings_rotated
        },
        'variance': {
            'unrotated': variance_unrotated,
            'rotated': variance_rotated
        },
        'communalities': communalities,
        'scores': {
            'unrotated': scores_unrotated,
            'rotated': scores_rotated
        }
    }

def plot_factor_analysis(fa_results):
    """
    Create visualizations for factor analysis results.

    Parameters:
    fa_results (dict): Results from perform_factor_analysis function
    """
    # Plot KMO indices
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        pd.DataFrame(fa_results['kmo']['per_variable']).T,
        annot=True,
        cmap='YlOrRd',
        vmin=0,
        vmax=1
    )
    plt.title('KMO Indices per Variable')
    plt.tight_layout()
    plt.show()

    # Plot factor correlations (unrotated)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        fa_results['loadings']['unrotated'],
        annot=True,
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1
    )
    plt.title('Factor Correlations (Unrotated)')
    plt.tight_layout()
    plt.show()

    # Plot factor correlations (rotated)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        fa_results['loadings']['rotated'],
        annot=True,
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1
    )
    plt.title('Factor Correlations (Rotated)')
    plt.tight_layout()
    plt.show()

    # Plot communalities
    plt.figure(figsize=(10, 6))
    fa_results['communalities'].plot(kind='bar')
    plt.title('Communalities')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot factor scores (first two factors)
    if fa_results['n_factors'] >= 2:
        # Unrotated scores
        plt.figure(figsize=(10, 8))
        plt.scatter(
            fa_results['scores']['unrotated']['Factor1'],
            fa_results['scores']['unrotated']['Factor2']
        )
        plt.xlabel('Factor 1')
        plt.ylabel('Factor 2')
        plt.title('Factor Scores Plot (Unrotated)')
        plt.tight_layout()
        plt.show()

        # Rotated scores
        plt.figure(figsize=(10, 8))
        plt.scatter(
            fa_results['scores']['rotated']['Factor1'],
            fa_results['scores']['rotated']['Factor2']
        )
        plt.xlabel('Factor 1')
        plt.ylabel('Factor 2')
        plt.title('Factor Scores Plot (Rotated)')
        plt.tight_layout()
        plt.show()