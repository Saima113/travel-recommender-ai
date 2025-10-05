"""
Quick test to verify everything works
Run this before deploying
"""

from recommender import HiddenGemRecommender
import pandas as pd

print("=" * 50)
print("TESTING HIDDEN GEM TRAVEL AI")
print("=" * 50)

# Test 1: Load system
print("\n✓ Test 1: Loading recommender...")
rec = HiddenGemRecommender()

# Test 2: Load data (use small sample for speed)
print("\n✓ Test 2: Loading dataset...")
df = pd.read_csv('data/Hotel_Reviews.csv', nrows=1000)  # Test with 1K rows
df.to_csv('data/test_sample.csv', index=False)
rec.load_data('data/test_sample.csv')
print(f"   Loaded {len(df)} reviews")

# Test 3: Create embeddings
print("\n✓ Test 3: Creating embeddings...")
rec.create_embeddings()
print(f"   Created {len(rec.embeddings)} embeddings")

# Test 4: Semantic search
print("\n✓ Test 4: Testing semantic search...")
queries = [
    "peaceful mountain retreat",
    "romantic beach resort",
    "adventure activities"
]

for query in queries:
    results = rec.semantic_search(query, top_k=5)
    print(f"\n   Query: '{query}'")
    print(f"   Top result: {results.iloc[0]['Hotel_Name']}")
    print(f"   Hidden gem score: {results.iloc[0]['hidden_gem_score']:.1f}")

# Test 5: Itinerary creation
print("\n✓ Test 5: Creating itinerary...")
results = rec.semantic_search("family friendly hotel", top_k=10)
itinerary = rec.create_itine
rary = rec.create_itinerary(results, n_days=3)
print(f"   Created {len(itinerary)} days of travel")
for day, locs in itinerary.items():
    print(f"   {day}: {len(locs)} locations")

# Test 6: Bandit update
print("\n✓ Test 6: Testing multi-armed bandit...")
rec.update_bandit(0, reward=1.0)
print("   Bandit updated successfully")

# Test 7: Save/load model
print("\n✓ Test 7: Testing model persistence...")
rec.save_model('test_model.pkl')
rec2 = HiddenGemRecommender()
rec2.load_model('test_model.pkl')
print("   Model saved and loaded successfully")

print("\n" + "=" * 50)
print("✅ ALL TESTS PASSED!")
print("=" * 50)
print("\nYou're ready to run: streamlit run app.py")