import pandas as pd
import numpy as np
from typing import List, Dict
import json

def load_and_preprocess_data(filepath: str, sample_size: int = None) -> pd.DataFrame:
    """
    Load and preprocess the dataset
    
    Args:
        filepath: Path to CSV file
        sample_size: If specified, randomly sample this many rows
    
    Returns:
        Preprocessed DataFrame
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Handle different dataset formats
    if 'Hotel_Name' not in df.columns and 'name' in df.columns:
        df.rename(columns={'name': 'Hotel_Name'}, inplace=True)
    
    if 'Review' not in df.columns:
        if 'reviews.text' in df.columns:
            df.rename(columns={'reviews.text': 'Review'}, inplace=True)
        elif 'Positive_Review' in df.columns:
            df['Review'] = df['Positive_Review'] + " " + df.get('Negative_Review', '')
    
    # Clean missing values
    df = df.dropna(subset=['Hotel_Name'])
    
    # Sample if needed (for faster testing)
    if sample_size and len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
        print(f"Sampled {sample_size} rows")
    
    print(f"Loaded {len(df)} rows with {df['Hotel_Name'].nunique()} unique locations")
    
    return df

def export_results_to_json(results: pd.DataFrame, filepath: str = 'results.json'):
    """Export search results to JSON"""
    results_dict = results.to_dict('records')
    with open(filepath, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)
    print(f"Results exported to {filepath}")

def calculate_route_distance(locations: List[tuple]) -> float:
    """
    Calculate total distance for a route
    
    Args:
        locations: List of (lat, lng) tuples
    
    Returns:
        Total distance in km
    """
    from geopy.distance import geodesic
    
    total_distance = 0
    for i in range(len(locations) - 1):
        total_distance += geodesic(locations[i], locations[i+1]).km
    
    return total_distance

def generate_insights(results: pd.DataFrame) -> Dict:
    """
    Generate analytical insights from results
    """
    insights = {
        'total_locations': len(results),
        'avg_hidden_gem_score': results['hidden_gem_score'].mean(),
        'top_hidden_gem': results.nlargest(1, 'hidden_gem_score')['Hotel_Name'].values[0],
        'countries': results['Hotel_Address'].str.split().str[-1].unique().tolist(),
        'score_distribution': {
            'excellent': len(results[results['hidden_gem_score'] > 70]),
            'good': len(results[(results['hidden_gem_score'] > 50) & (results['hidden_gem_score'] <= 70)]),
            'average': len(results[results['hidden_gem_score'] <= 50])
        }
    }
    
    return insights