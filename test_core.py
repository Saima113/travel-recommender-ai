"""
Quick test to verify everything works
"""
from recommender import HiddenGemRecommender

print("=" * 60)
print("TESTING HIDDEN GEM TRAVEL AI")
print("=" * 60)

# Test 1: Initialize
print("\n✓ Test 1: Loading recommender...")
rec = HiddenGemRecommender()

# Test 2: Load data (use subset for speed)
print("\n✓ Test 2: Loading dataset...")
import pandas as pd

# Load only 5000 rows for faster testing
print("   Loading first 5000 reviews for testing...")
df = pd.read_csv('data/Hotel_Reviews.csv', nrows=5000)
df.to_csv('data/test_sample.csv', index=False)

rec.load_data('data/test_sample.csv')
print(f"    Loaded {len(rec.locations_df)} unique hotels")

# Test 3: Create embeddings
print("\n✓ Test 3: Creating embeddings...")
rec.create_embeddings()
print(f"    Created {len(rec.embeddings)} embeddings")

# Test 4: Semantic search
print("\n✓ Test 4: Testing semantic search...")
test_queries = [
    "peaceful mountain retreat with nature views",
    "romantic beach hotel with spa",
    "family friendly hotel near attractions"
]

for query in test_queries:
    print(f"\n   Query: '{query}'")
    results = rec.semantic_search(query, top_k=3)
    
    for idx, row in results.iterrows():
        print(f"      → {row['Hotel_Name']}")
        print(f"          Score: {row['hidden_gem_score']:.1f}/100 | "
              f" Match: {row['similarity_score']*100:.1f}%")

# Test 5: Itinerary
print("\n✓ Test 5: Creating itinerary...")
results = rec.semantic_search("luxury hotel with great location", top_k=5)
itinerary = rec.create_itinerary(results, n_days=3)
print(f"  Created {len(itinerary)}-day itinerary")
for day, locations in itinerary.items():
    print(f"   {day}: {len(locations)} locations")

# Test 6: Save model
print("\n✓ Test 6: Saving model...")
rec.save_model('test_model.pkl')
print("    Model saved")

print("\n" + "=" * 60)
print(" ALL TESTS PASSED! System is ready!")
print("=" * 60)
print("\n Next step: Run the Streamlit app")
print("   Command: streamlit run app.py")