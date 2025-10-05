import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from geopy.distance import geodesic
import pickle
import os

class HiddenGemRecommender:
    def __init__(self):
        """Initialize the AI recommender system"""
        print("Loading AI models...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.locations_df = None
        self.embeddings = None
        self.bandit_counts = None
        self.bandit_rewards = None
        
    def load_data(self, filepath):
        """Load and process travel data"""
        print("📊 Loading dataset...")
        df = pd.read_csv(filepath)
        
        # This is the 515K Hotel Reviews dataset
        print("   Detected 515K Hotel Reviews format")
        
        # Combine positive and negative reviews
        df['Review'] = df['Positive_Review'].fillna('') + " " + df['Negative_Review'].fillna('')
        
        # Remove rows where both reviews are empty
        df = df[df['Review'].str.strip() != '']
        
        # Columns are already named 'lat' and 'lng' - perfect!
        
        # Data cleaning - drop rows with missing critical data
        df = df.dropna(subset=['Hotel_Name', 'lat', 'lng'])
        
        # Add Rating column (they use Reviewer_Score which is 0-10, convert to 0-5)
        df['Rating'] = df['Reviewer_Score'] / 2.0
        
        print(f"   Loaded {len(df)} reviews from {df['Hotel_Name'].nunique()} hotels")
        
        # Calculate hidden gem score
        print("💎 Calculating hidden gem scores...")
        df = self._add_hidden_gem_score(df)
        
        self.locations_df = df
        return df

    def _add_hidden_gem_score(self, df):
        """Calculate hidden gem score and add to dataframe"""
        # Group by hotel to get statistics
        hotel_stats = df.groupby('Hotel_Name').agg({
            'Rating': 'mean',
            'Review': 'count',
            'lat': 'first',
            'lng': 'first',
            'Hotel_Address': 'first'
        }).reset_index()
        
        hotel_stats.columns = ['Hotel_Name', 'avg_rating', 'review_count', 'lat', 'lng', 'Hotel_Address']
        
        # Hidden Gem Score Formula
        quality_score = hotel_stats['avg_rating'] / 5.0  # Normalize to 0-1
        popularity_penalty = 1 / (1 + np.log1p(hotel_stats['review_count']))
        
        hotel_stats['hidden_gem_score'] = quality_score * popularity_penalty * 100
        
        # Merge back to original dataframe
        df = df.merge(
            hotel_stats[['Hotel_Name', 'hidden_gem_score']], 
            on='Hotel_Name', 
            how='left'
        )
        
        return df
    
    def _calculate_hidden_gem_score(self, df):
        """
        Custom algorithm to find hidden gems
        Combines: quality, uniqueness, and under-tourism
        """
        # Group by location
        location_stats = df.groupby('Hotel_Name').agg({
            'Rating': 'mean',
            'Review': 'count'
        }).reset_index()
        
        location_stats.columns = ['Hotel_Name', 'avg_rating', 'review_count']
        
        # Scoring formula
        # Higher rating = better
        # More reviews = more popular (penalize)
        # Sweet spot: high quality, moderate popularity
        
        quality_score = location_stats['avg_rating'] / 5.0  # Normalize
        popularity_penalty = 1 / (1 + np.log1p(location_stats['review_count']))
        
        location_stats['hidden_gem_score'] = quality_score * popularity_penalty * 100
        
        # Merge back
        df = df.merge(location_stats[['Hotel_Name', 'hidden_gem_score']], on='Hotel_Name', how='left')
        
        return df['hidden_gem_score']
    
    def create_embeddings(self):
        """
        Create semantic embeddings for all locations
        This is the AI magic! 🧠
        """
        print("🧠 Creating semantic embeddings...")
        
        # Group by hotel and combine reviews
        location_data = self.locations_df.groupby('Hotel_Name').agg({
            'Review': lambda x: ' '.join(x[:10]),  # First 10 reviews per hotel
            'Hotel_Address': 'first',
            'lat': 'first',
            'lng': 'first',
            'Rating': 'first',
            'hidden_gem_score': 'first'
        }).reset_index()
        
        # Store this as our main dataframe (one row per hotel)
        self.locations_df = location_data
        
        # Create text for embeddings
        texts = (location_data['Hotel_Name'] + ". " + 
                location_data['Hotel_Address'] + ". " +
                location_data['Review']).tolist()
        
        print(f"   Creating embeddings for {len(texts)} unique hotels...")
        
        # Create embeddings
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        
        print(f"✅ Created {len(self.embeddings)} embeddings")
        
        # Initialize bandit for each location
        self.bandit_counts = np.ones(len(self.embeddings))
        self.bandit_rewards = np.zeros(len(self.embeddings))
        
        return self.embeddings
    
    def semantic_search(self, query, top_k=10, exploration_rate=0.2):
        """
        Search using natural language
        Uses multi-armed bandit for exploration-exploitation
        """
        print(f" Searching for: '{query}'")
        
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Calculate similarity
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Adjust top_k if we don't have enough locations
        available_locations = len(self.embeddings)
        actual_top_k = min(top_k, available_locations)
        
        if actual_top_k < top_k:
            print(f"   Only {available_locations} locations available (requested {top_k})")
        
        # Get top candidates (more than needed for bandit)
        candidate_multiplier = min(3, max(1, available_locations // actual_top_k))
        n_candidates = min(actual_top_k * candidate_multiplier, available_locations)
        top_indices = np.argsort(similarities)[::-1][:n_candidates]
        
        # Multi-armed bandit selection
        selected_indices = self._bandit_select(top_indices, actual_top_k, exploration_rate)
        
        results = self.locations_df.iloc[selected_indices].copy()
        results['similarity_score'] = similarities[selected_indices]
        
        # Sort by hidden gem score
        results = results.sort_values('hidden_gem_score', ascending=False)
        
        return results.head(actual_top_k)
    
    def _bandit_select(self, candidates, n_select, epsilon):
        """
        Multi-armed bandit: UCB algorithm
        Balances showing good places vs discovering new ones
        """
        selected = []
        candidates = candidates.copy()  # Don't modify original
        
        # Ensure we don't try to select more than available
        n_select = min(n_select, len(candidates))
        
        for _ in range(n_select):
            if len(candidates) == 0:
                break
                
            if np.random.random() < epsilon and len(candidates) > 1:
                # EXPLORE: Random selection
                idx_position = np.random.randint(0, len(candidates))
                idx = candidates[idx_position]
            else:
                # EXPLOIT: UCB score
                ucb_scores = (self.bandit_rewards[candidates] / self.bandit_counts[candidates]) + \
                            np.sqrt(2 * np.log(sum(self.bandit_counts)) / self.bandit_counts[candidates])
                idx = candidates[np.argmax(ucb_scores)]
            
            selected.append(idx)
            # Remove from candidates
            candidates = candidates[candidates != idx]
        
        return selected
    
    def update_bandit(self, location_idx, reward):
        """Update bandit based on user interaction"""
        self.bandit_counts[location_idx] += 1
        self.bandit_rewards[location_idx] += reward
    
    def create_itinerary(self, locations_df, n_days=3):
        """
        Create day-wise itinerary using geospatial clustering
        """
        print(f"Creating {n_days}-day itinerary...")
        
        # Extract coordinates
        coords = locations_df[['lat', 'lng']].values
        
        # Cluster into days using DBSCAN
        clustering = DBSCAN(eps=0.3, min_samples=1, metric='haversine')
        labels = clustering.fit_predict(np.radians(coords))
        
        locations_df['day'] = labels
        
        # Organize by day
        itinerary = {}
        for day in range(min(n_days, len(set(labels)))):
            day_locations = locations_df[locations_df['day'] == day]
            itinerary[f"Day {day + 1}"] = day_locations.sort_values('hidden_gem_score', ascending=False)
        
        return itinerary
    
    def save_model(self, path='model_cache.pkl'):
        """Save model for faster loading"""
        with open(path, 'wb') as f:
            pickle.dump({
                'embeddings': self.embeddings,
                'bandit_counts': self.bandit_counts,
                'bandit_rewards': self.bandit_rewards
            }, f)
        print(f"Model saved to {path}")
    
    def load_model(self, path='model_cache.pkl'):
        """Load saved model"""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self.embeddings = data['embeddings']
            self.bandit_counts = data['bandit_counts']
            self.bandit_rewards = data['bandit_rewards']
            print(f"Model loaded from {path}")
            return True
        return False