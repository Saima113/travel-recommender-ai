import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
from recommender import HiddenGemRecommender
import time

# Page config
st.set_page_config(
    page_title="Hidden Gem Travel AI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'recommender' not in st.session_state:
    st.session_state.recommender = None
    st.session_state.search_results = None
    st.session_state.search_history = []

@st.cache_resource
def load_recommender_system(dataset_path):
    """Load and cache the recommender system"""
    rec = HiddenGemRecommender()
    
    # Try to load cached model first
    if not rec.load_model():
        df = rec.load_data(dataset_path)
        rec.create_embeddings()
        rec.save_model()
    else:
        df = rec.load_data(dataset_path)
    
    return rec

def create_map(locations_df):
    """Create interactive map with markers"""
    if locations_df.empty:
        return None
    
    # Center map on mean coordinates
    center_lat = locations_df['lat'].mean()
    center_lng = locations_df['lng'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=6,
        tiles='CartoDB positron'
    )
    
    # Add markers
    for idx, row in locations_df.iterrows():
        # Color based on hidden gem score
        if row['hidden_gem_score'] > 70:
            color = 'green'
            icon = 'star'
        elif row['hidden_gem_score'] > 50:
            color = 'blue'
            icon = 'heart'
        else:
            color = 'orange'
            icon = 'info-sign'
        
        popup_html = f"""
        <div style="font-family: Arial; width: 200px;">
            <h4 style="margin: 0; color: #667eea;">{row['Hotel_Name']}</h4>
            <hr style="margin: 5px 0;">
            <p style="margin: 5px 0;">
                <b>💎 Hidden Gem Score:</b> {row['hidden_gem_score']:.1f}/100
            </p>
            <p style="margin: 5px 0;">
                <b>⭐ Rating:</b> {row.get('Rating', 'N/A')}
            </p>
            <p style="margin: 5px 0;">
                <b>🎯 Match:</b> {row.get('similarity_score', 0)*100:.1f}%
            </p>
        </div>
        """
        
        folium.Marker(
            location=[row['lat'], row['lng']],
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=row['Hotel_Name'],
            icon=folium.Icon(color=color, icon=icon, prefix='glyphicon')
        ).add_to(m)
    
    return m

def create_score_distribution(df):
    """Create distribution chart for hidden gem scores"""
    fig = px.histogram(
        df,
        x='hidden_gem_score',
        nbins=30,
        title='Distribution of Hidden Gem Scores',
        labels={'hidden_gem_score': 'Hidden Gem Score', 'count': 'Number of Locations'},
        color_discrete_sequence=['#667eea']
    )
    fig.update_layout(
        template='plotly_white',
        showlegend=False,
        height=300
    )
    return fig

def create_similarity_chart(results):
    """Create bar chart for semantic similarity"""
    fig = go.Figure(data=[
        go.Bar(
            x=results['Hotel_Name'].head(10),
            y=results['similarity_score'].head(10) * 100,
            marker=dict(
                color=results['hidden_gem_score'].head(10),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Hidden Gem<br>Score")
            ),
            text=results['similarity_score'].head(10).apply(lambda x: f'{x*100:.1f}%'),
            textposition='outside',
        )
    ])
    fig.update_layout(
        title='Top 10 Semantic Matches',
        xaxis_title='Location',
        yaxis_title='Similarity Score (%)',
        template='plotly_white',
        height=400,
        xaxis_tickangle=-45
    )
    return fig

# Main App
def main():
    # Header
    st.markdown('<p class="main-header">✈️ Hidden Gem Travel AI 💎</p>', unsafe_allow_html=True)
    st.markdown("""
    <p style="text-align: center; font-size: 1.2rem; color: #666;">
        Discover Authentic Experiences Using AI-Powered Semantic Search & Multi-Armed Bandits
    </p>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/around-the-globe.png", width=100)
        st.title("⚙️ Configuration")
        
        # Dataset selection
        dataset_option = st.selectbox(
            "Choose Dataset",
            ["515K Hotel Reviews (Europe)", "TripAdvisor Reviews"]
        )
        
        if dataset_option == "515K Hotel Reviews (Europe)":
            dataset_path = "data/Hotel_Reviews.csv"
        else:
            dataset_path = "data/tripadvisor_hotel_reviews.csv"
        
        # Load system
        if st.button("🚀 Initialize AI System"):
            with st.spinner("Loading AI models... This may take 2-3 minutes..."):
                try:
                    st.session_state.recommender = load_recommender_system(dataset_path)
                    st.success("✅ AI System Ready!")
                except Exception as e:
                    st.error(f"Error loading dataset: {e}")
                    st.info("Please ensure the dataset is in the 'data/' folder")
        
        st.markdown("---")
        
        # Search parameters
        st.subheader("🎛️ Search Settings")
        top_k = st.slider("Number of Results", 5, 20, 10)
        exploration_rate = st.slider("Exploration Rate", 0.0, 0.5, 0.2, 0.05)
        
        st.info(f"""
        **Exploration Rate: {exploration_rate}**
        - Lower = Show proven gems
        - Higher = Discover new places
        """)
        
        st.markdown("---")
        
        # Stats
        if st.session_state.recommender:
            st.subheader("📊 System Stats")
            df = st.session_state.recommender.locations_df
            st.metric("Total Locations", f"{df['Hotel_Name'].nunique():,}")
            st.metric("Total Reviews", f"{len(df):,}")
            st.metric("Countries", f"{df['Hotel_Address'].str.split().str[-1].nunique()}")
    
    # Main content
    if st.session_state.recommender is None:
        st.info("👈 Please initialize the AI system from the sidebar")
        
        # Show system architecture
        st.subheader("🧠 System Architecture")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 1️⃣ Semantic Search
            - **BERT-based embeddings**
            - Understands meaning, not just keywords
            - Sentence Transformers (all-MiniLM-L6-v2)
            """)
        
        with col2:
            st.markdown("""
            ### 2️⃣ Multi-Armed Bandit
            - **UCB Algorithm**
            - Balances exploration vs exploitation
            - Online learning from user interactions
            """)
        
        with col3:
            st.markdown("""
            ### 3️⃣ Hidden Gem Score
            - **Custom algorithm**
            - Quality × Uniqueness ÷ Popularity
            - Geospatial density analysis
            """)
        
        return
    
    # Search interface
    st.subheader("🔍 Semantic Search")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "What kind of travel experience are you looking for?",
            placeholder="e.g., peaceful mountain retreat with authentic local cuisine and sunrise views",
            key="search_query"
        )
    
    with col2:
        search_button = st.button("🔎 Search", use_container_width=True)
    
    # Example queries
    st.caption("Try: *romantic coastal escape with historic charm* | *adventure activities in nature* | *quiet cultural immersion*")
    
    # Search execution
    if search_button and query:
        with st.spinner("🧠 AI is analyzing your preferences..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            results = st.session_state.recommender.semantic_search(
                query, 
                top_k=top_k,
                exploration_rate=exploration_rate
            )
            st.session_state.search_results = results
            st.session_state.search_history.append(query)
            
            progress_bar.empty()
    
    # Display results
    if st.session_state.search_results is not None:
        results = st.session_state.search_results
        
        st.success(f"Found {len(results)} hidden gems for you!")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Hidden Gem Score", f"{results['hidden_gem_score'].mean():.1f}/100")
        with col2:
            st.metric("Avg Semantic Match", f"{results['similarity_score'].mean()*100:.1f}%")
        with col3:
            st.metric("Top Score", f"{results['hidden_gem_score'].max():.1f}/100")
        with col4:
            st.metric("Countries", results['Hotel_Address'].str.split().str[-1].nunique())
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Map View", "📋 List View", "📊 Analytics", "🗓️ Itinerary"])
        
        with tab1:
            st.subheader("Interactive Location Map")
            map_obj = create_map(results)
            if map_obj:
                folium_static(map_obj, width=1200, height=600)
        
        with tab2:
            st.subheader("Detailed Results")
            
            for idx, row in results.iterrows():
                with st.expander(f"{'💚' if row['hidden_gem_score'] > 70 else '💙'} {row['Hotel_Name']} - Score: {row['hidden_gem_score']:.1f}/100"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**📍 Location:** {row['Hotel_Address']}")
                        st.markdown(f"**⭐ Rating:** {row.get('Rating', 'N/A')}")
                        st.markdown(f"**🎯 Semantic Match:** {row['similarity_score']*100:.1f}%")
                        st.markdown(f"**💎 Hidden Gem Score:** {row['hidden_gem_score']:.1f}/100")
                        
                        # Show a sample review
                        if 'Review' in row and pd.notna(row['Review']):
                            st.markdown(f"**📝 Sample Review:**")
                            st.info(row['Review'][:300] + "...")
                    
                    with col2:
                        st.markdown("**Coordinates:**")
                        st.code(f"Lat: {row['lat']:.4f}\nLng: {row['lng']:.4f}")
                        
                        # Like button (for bandit feedback)
                        if st.button(f"❤️ Like", key=f"like_{idx}"):
                            st.session_state.recommender.update_bandit(idx, reward=1.0)
                            st.success("Feedback recorded!")
        
        with tab3:
            st.subheader("Search Analytics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_sim = create_similarity_chart(results)
                st.plotly_chart(fig_sim, use_container_width=True)
            
            with col2:
                fig_dist = create_score_distribution(results)
                st.plotly_chart(fig_dist, use_container_width=True)
            
            # Scatter plot
            fig_scatter = px.scatter(
                results,
                x='similarity_score',
                y='hidden_gem_score',
                size='Rating' if 'Rating' in results.columns else None,
                hover_name='Hotel_Name',
                title='Semantic Match vs Hidden Gem Score',
                labels={
                    'similarity_score': 'Semantic Similarity',
                    'hidden_gem_score': 'Hidden Gem Score'
                },
                color='hidden_gem_score',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with tab4:
            st.subheader("🗓️ Smart Itinerary Builder")
            
            n_days = st.slider("Trip Duration (days)", 1, 7, 3)
            
            if st.button("Generate Itinerary"):
                with st.spinner("Creating optimized route..."):
                    itinerary = st.session_state.recommender.create_itinerary(
                        results.head(15), 
                        n_days=n_days
                    )
                    
                    for day, locations in itinerary.items():
                        st.markdown(f"### 📅 {day}")
                        
                        for idx, loc in locations.iterrows():
                            st.markdown(f"""
                            **{loc['Hotel_Name']}**  
                            📍 {loc['Hotel_Address']}  
                            💎 Hidden Gem Score: {loc['hidden_gem_score']:.1f}/100  
                            """)
                        
                        st.markdown("---")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem 0;">
        <p><b>Built by Saima</b> | Powered by Sentence-Transformers, Multi-Armed Bandits & Geospatial AI</p>
        <p>🔗 <a href="https://github.com/yourusername/travel-recommender-ai">GitHub</a> | 
           💼 <a href="https://linkedin.com/in/yourprofile">LinkedIn</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()