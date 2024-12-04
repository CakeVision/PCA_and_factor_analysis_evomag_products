import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def clean_numeric_string(s):
    if isinstance(s, str):
        # Remove spaces and any other non-numeric characters (except decimal points)
        cleaned = ''.join(c for c in s if c.isdigit() or c == '.')
        return float(cleaned) if cleaned else 0
    return float(s) if s else 0

def prepare_data(df):
    # Extract numeric features for PCA
    numeric_features = ['price', 'rating', 'number_of_reviews']

    # Add processor features
    df['max_freq'] = df['specifications'].apply(lambda x: clean_numeric_string(x.get('Max Turbo Frequency (MHz)', 0)) if isinstance(x, dict) else 0)
    df['cache'] = df['specifications'].apply(lambda x: clean_numeric_string(x.get('Smart Cache (Kb)', 0)) if isinstance(x, dict) else 0)
    df['cores'] = df['specifications'].apply(lambda x: clean_numeric_string(x.get('Numar nuclee', 0)) if isinstance(x, dict) else 0)

    # Add memory and storage
    df['memory'] = df['specifications'].apply(lambda x: clean_numeric_string(x.get('Capacitate memorie', 0)) if isinstance(x, dict) else 0)
    df['storage'] = df['specifications'].apply(lambda x: clean_numeric_string(x.get('Capacitate SSD', '0').replace('GB', '')) if isinstance(x, dict) else 0)

    # Add display features
    df['screen_size'] = df['specifications'].apply(lambda x: clean_numeric_string(x.get('Diagonala', '0"').replace('"', '')) if isinstance(x, dict) else 0)
    df['refresh_rate'] = df['specifications'].apply(lambda x: clean_numeric_string(x.get('Rata Refresh', 0)) if isinstance(x, dict) else 0)

    features_for_pca = ['price', 'rating', 'number_of_reviews', 'max_freq', 'cache',
                       'cores', 'memory', 'storage', 'screen_size', 'refresh_rate']

    # Replace any remaining NaN values with 0
    return df[features_for_pca].fillna(0)

def perform_pca(data):
    # Standardize the features
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Perform PCA
    pca = PCA()
    pca_result = pca.fit_transform(data_scaled)

    # Calculate explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    # Create variance distribution table
    variance_df = pd.DataFrame({
        'Principal Component': [f'PC{i+1}' for i in range(len(explained_variance))],
        'Explained Variance Ratio': explained_variance,
        'Cumulative Variance Ratio': cumulative_variance
    })

    # Calculate component loadings (correlations)
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
        index=data.columns
    )

    # Calculate contributions
    contributions = loadings ** 2

    # Calculate communalities
    communalities = pd.DataFrame({
        'Feature': data.columns,
        'Communality': np.sum(loadings ** 2, axis=1)
    })

    return {
        'pca': pca,
        'scores': pca_result,
        'variance_distribution': variance_df,
        'loadings': loadings,
        'contributions': contributions,
        'communalities': communalities
    }

def plot_variance(variance_df):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(variance_df) + 1),
            variance_df['Cumulative Variance Ratio'],
            'bo-', linewidth=2)
    plt.axhline(y=0.8, color='r', linestyle='--', label='80% Threshold')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Scree Plot with Cumulative Explained Variance')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_correlation_circle(loadings, pc1=1, pc2=2):
    plt.figure(figsize=(10, 10))
    circle = plt.Circle((0, 0), radius=1, fill=False, color='gray', linestyle='--')
    plt.gca().add_patch(circle)

    for i, (x, y) in enumerate(zip(loadings[f'PC{pc1}'], loadings[f'PC{pc2}'])):
        plt.arrow(0, 0, x, y, head_width=0.05, head_length=0.05, fc='blue', ec='blue')
        plt.text(x*1.15, y*1.15, loadings.index[i], ha='center', va='center')

    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.xlabel(f'PC{pc1}')
    plt.ylabel(f'PC{pc2}')
    plt.title('Correlation Circle')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_scores(scores, variance_ratio, pc1=1, pc2=2):
    plt.figure(figsize=(10, 8))
    plt.scatter(scores[:, pc1-1], scores[:, pc2-1], alpha=0.5)
    plt.xlabel(f'PC{pc1} ({variance_ratio[pc1-1]:.1%} explained variance)')
    plt.ylabel(f'PC{pc2} ({variance_ratio[pc2-1]:.1%} explained variance)')
    plt.title('PCA Score Plot')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def run_pca_analysis(df):
    # Prepare data
    data = prepare_data(df)

    # Print the first few rows of prepared data to verify
    print("\nPrepared data sample:")
    print(data.head())
    print("\nData shape:", data.shape)

    # Perform PCA
    results = perform_pca(data)

    # Display results
    print("\nVariance Distribution:")
    print(results['variance_distribution'])

    print("\nComponent Loadings:")
    print(results['loadings'])

    print("\nFeature Contributions:")
    print(results['contributions'])

    print("\nCommunalities:")
    print(results['communalities'])

    # Create plots
    plot_variance(results['variance_distribution'])
    plot_correlation_circle(results['loadings'])
    plot_scores(results['scores'], results['pca'].explained_variance_ratio_)

    return results