"""
Fantasy Cricket Team Predictor - Main Streamlit Application
ML-powered prediction system for Dream11 fantasy teams
"""

import streamlit as st
import pandas as pd
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.data_loader import load_dataset

# Page configuration
st.set_page_config(
    page_title="Fantasy Cricket Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Cricinfo Dashboard Style
def apply_custom_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700;900&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Roboto', 'Helvetica Neue', Arial, sans-serif;
    }
    
    /* Main Container - Clean White Background */
    .main {
        background: #f8f9fa;
        padding: 1rem 2rem;
    }
    
    /* Sidebar Styling - Cricinfo Green */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #09847A 0%, #076d64 100%);
        border-right: none;
    }
    
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    
    /* Main Header - Bold Cricinfo Style */
    .main-header {
        font-size: 3rem;
        font-weight: 900;
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
        color: #09847A;
        letter-spacing: -0.02em;
        text-transform: uppercase;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #5a5a5a;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Section Headers */
    h1, h2, h3 {
        color: #222222 !important;
        font-weight: 700 !important;
    }
    
    h2 {
        border-bottom: 3px solid #09847A;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem !important;
    }
    
    /* Cricinfo-style Cards */
    .cricinfo-card {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }
    
    .cricinfo-card:hover {
        box-shadow: 0 4px 12px rgba(9, 132, 122, 0.15);
        border-color: #09847A;
    }
    
    /* Match Card Style */
    .match-card {
        background: #ffffff;
        border-left: 4px solid #09847A;
        border-radius: 4px;
        padding: 1.25rem;
        margin: 1rem 0;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Stat Cards - Score Display Style */
    .stat-card {
        background: linear-gradient(135deg, #09847A 0%, #0a9b8f 100%);
        color: white;
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 3px 8px rgba(9, 132, 122, 0.3);
        transition: transform 0.2s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 16px rgba(9, 132, 122, 0.4);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 900;
        margin: 0;
        color: #ffffff;
    }
    
    .stat-label {
        font-size: 0.9rem;
        font-weight: 500;
        margin-top: 0.5rem;
        opacity: 0.95;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Buttons - Cricinfo Green */
    .stButton > button {
        background: #09847A;
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 700;
        border-radius: 4px;
        cursor: pointer;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        background: #076d64;
        box-shadow: 0 4px 12px rgba(9, 132, 122, 0.3);
        transform: translateY(-1px);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Input Fields - Clean Style */
    .stSelectbox > div > div,
    .stTextInput > div > div > input {
        background: #ffffff !important;
        border: 2px solid #e0e0e0 !important;
        border-radius: 4px !important;
        color: #222222 !important;
        font-weight: 500 !important;
    }
    
    .stSelectbox > div > div:hover,
    .stTextInput > div > div > input:hover {
        border-color: #09847A !important;
    }
    
    .stSelectbox > div > div:focus-within,
    .stTextInput > div > div > input:focus {
        border-color: #09847A !important;
        box-shadow: 0 0 0 2px rgba(9, 132, 122, 0.1) !important;
    }
    
    /* DataFrames - Table Style */
    .stDataFrame {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 4px;
    }
    
    /* Info/Success/Warning Boxes */
    .stAlert {
        background: #ffffff !important;
        border-radius: 4px !important;
        border-left: 4px solid #09847A !important;
        color: #222222 !important;
    }
    
    /* Radio Buttons */
    .stRadio > div {
        background: rgba(9, 132, 122, 0.05);
        border-radius: 4px;
        padding: 0.75rem;
    }
    
    /* Metrics - Bold Numbers */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 900 !important;
        color: #09847A !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #5a5a5a !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 0.85rem !important;
    }
    
    /* Dividers */
    hr {
        border-color: #e0e0e0 !important;
        margin: 2rem 0 !important;
    }
    
    /* Text Colors */
    p {
        color: #222222 !important;
        line-height: 1.6;
    }
    
    label {
        color: #5a5a5a !important;
        font-weight: 600 !important;
    }
    
    /* Live Score Style Badge */
    .live-badge {
        display: inline-block;
        background: #d32f2f;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-left: 0.5rem;
    }
    
    /* Captain/VC Badges - Sports Style */
    .captain-badge {
        display: inline-block;
        background: #ffa726;
        color: #000;
        padding: 0.3rem 0.75rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-left: 0.5rem;
        text-transform: uppercase;
    }
    
    .vc-badge {
        display: inline-block;
        background: #78909c;
        color: #fff;
        padding: 0.3rem 0.75rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-left: 0.5rem;
        text-transform: uppercase;
    }
    
    /* Score Display */
    .score-display {
        font-size: 3rem;
        font-weight: 900;
        color: #09847A;
        line-height: 1;
    }
    
    /* Step Timeline */
    .step-line {
        border-left: 3px solid #09847A;
        padding-left: 1.5rem;
        margin-left: 0.5rem;
    }
    
    /* Feature Box */
    .feature-box {
        background: #f5f5f5;
        border-left: 4px solid #09847A;
        padding: 1.25rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    
    .feature-box h4 {
        color: #09847A !important;
        margin-bottom: 0.75rem !important;
        font-weight: 700 !important;
    }
    
    .feature-box ul {
        margin: 0;
        padding-left: 1.25rem;
    }
    
    .feature-box li {
        color: #222222 !important;
        margin: 0.5rem 0;
    }
    
    /* Highlight Text */
    .highlight-text {
        color: #09847A !important;
        font-weight: 700;
    }
    
    /* Premium Content Banner */
    .premium-banner {
        background: linear-gradient(135deg, #09847A 0%, #0a9b8f 100%);
        color: white;
        padding: 2rem;
        border-radius: 8px;
        text-align: center;
        margin: 2rem 0;
    }
    
    /* Clean Table Headers */
    thead tr th {
        background-color: #09847A !important;
        color: white !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
    }
    </style>
    """, unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    # Dataset upload states
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'current_league' not in st.session_state:
        st.session_state.current_league = "BBL Women's"
    if 'model_info' not in st.session_state:
        st.session_state.model_info = None
    if 'loaded_model_name' not in st.session_state:
        st.session_state.loaded_model_name = None
    
    # Existing states
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'loader' not in st.session_state:
        st.session_state.loader = None
    if 'selected_team1' not in st.session_state:
        st.session_state.selected_team1 = None
    if 'selected_team2' not in st.session_state:
        st.session_state.selected_team2 = None
    if 'selected_venue' not in st.session_state:
        st.session_state.selected_venue = None
    if 'selected_players' not in st.session_state:
        st.session_state.selected_players = []
    if 'player_team_tags' not in st.session_state:
        st.session_state.player_team_tags = {}
    if 'predictions_ready' not in st.session_state:
        st.session_state.predictions_ready = False


def main():
    """Main application entry point."""
    apply_custom_css()
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🏏 Fantasy Cricket Team Predictor</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-header">ML-Powered Dream11 Team Predictions for {st.session_state.current_league}</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/cricket.png", width=80)
        st.title("Navigation")
        
        # Check if dataset is uploaded and model trained
        if not st.session_state.uploaded_data:
            page = "📤 Upload Dataset"
            st.info("📤 Upload a dataset to begin")
        elif not st.session_state.model_trained:
            page = "⚙️ Train Model"
            st.info("⚙️ Train model on uploaded data")
        else:
            page = st.radio(
                "Select Page",
                ["🏠 Home", "🎯 Team Selection", "🏟️ Ground Selection", 
                 "👥 Player Pool", "📊 Predictions"],
                label_visibility="collapsed"
            )
            
            # Model Library button
            st.markdown("---")
            if st.button("📚 Model Library", use_container_width=True):
                page = "📚 Model Library"
            
            # Add option to change dataset
            st.markdown("---")
            if st.button("🔄 Change Dataset", use_container_width=True):
                st.session_state.uploaded_data = None
                st.session_state.model_trained = False
                st.session_state.data_loaded = False
                st.rerun()
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown(f"""
        This app uses machine learning to predict fantasy cricket team performance
        for **{st.session_state.current_league}**.
        
        **Features:**
        - Upload any cricket dataset
        - On-demand model training
        - ML-powered predictions
        - Dream11 point system
        - Optimal team selection
        """)
        
        # Show model status
        if os.path.exists('models/fantasy_predictor.pkl'):
            st.success("✅ ML Model Loaded")
        else:
            st.warning("⚠️ ML Model Not Found")
            st.info("Run: `python scripts/train_model.py`")
    
    # Load data - from uploaded file or default
    if not st.session_state.data_loaded and st.session_state.uploaded_data is not None:
        with st.spinner("Loading uploaded dataset..."):
            try:
                from src.data.data_loader import DataLoader
                df = st.session_state.uploaded_data
                loader = DataLoader(df)
                st.session_state.df = df
                st.session_state.loader = loader
                st.session_state.data_loaded = True
            except Exception as e:
                st.error(f"Error loading dataset: {e}")
                return
    
    # Route to appropriate page
    if page == "📤 Upload Dataset":
        show_upload_page()
    elif page == "⚙️ Train Model":
        show_training_page()
    elif page == "📚 Model Library":
        show_model_library_page()
    elif page == "🏠 Home":
        show_home_page()
    elif page == "🎯 Team Selection":
        show_team_selection_page()
    elif page == "🏟️ Ground Selection":
       show_ground_selection_page()
    elif page == "👥 Player Pool":
        show_player_pool_page()
    elif page == "📊 Predictions":
        show_predictions_page()


def show_upload_page():
    """Dataset upload page."""
    st.markdown("## 📤 Upload Cricket Dataset")
    
    st.markdown("""
    <div class="cricinfo-card">
        <h3>Upload Your Cricket Dataset</h3>
        <p>Upload a ball-by-ball CSV file for any cricket league (IPL, BBL, CPL, PSL, etc.)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Required columns info
    st.markdown("### 📋 Required Columns")
    st.markdown("""
    <div class="feature-box">
        <p>Your CSV must contain these columns:</p>
        <ul>
            <li><code>match_id</code> - Unique match identifier</li>
            <li><code>batting_team</code> - Batting team name</li>
            <li><code>bowling_team</code> - Bowling team name</li>
            <li><code>striker</code> - Batsman on strike</li>
            <li><code>bowler</code> - Bowler name</li>
            <li><code>total_runs</code> - Runs scored on ball</li>
            <li><code>venue</code> - Match venue</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # League name input
    league_name = st.text_input(
        "League Name",
        value="Cricket League",
        placeholder="e.g., IPL 2024, BBL Women's, CPL 2023",
        help="Enter the name of your cricket league"
    )
    
    # File uploader
    uploaded_file =st.file_uploader(
        "Choose CSV file",
        type=['csv'],
        help="Upload ball-by-ball cricket data in CSV format"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully!")
            
            # Validate columns
            required_cols = ['match_id', 'batting_team', 'bowling_team', 'striker', 
                           'bowler', 'total_runs', 'venue']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                st.info("Please ensure your CSV has all required columns.")
                return
            
            # Show dataset preview
            st.markdown("### 📊 Dataset Preview")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Balls", f"{len(df):,}")
            with col2:
                st.metric("Matches", df['match_id'].nunique())
            with col3:
                st.metric("Teams", df['batting_team'].nunique())
            with col4:
                st.metric("Players", len(set(df['striker'].unique()) | set(df['bowler'].unique())))
            
            st.markdown("**First few rows:**")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Confirm button
            st.markdown("---")
            if st.button("✅ Confirm & Proceed to Training", type="primary", use_container_width=True):
                st.session_state.uploaded_data = df
                st.session_state.current_league = league_name
                st.session_state.model_trained = False
                st.session_state.data_loaded = False
                st.success(f"Dataset loaded! Proceeding to model training...")
                st.rerun()
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.info("Please ensure the file is a valid CSV format.")


def show_training_page():
    """Model training page with progress."""
    st.markdown(f"## ⚙️ Train Model - {st.session_state.current_league}")
    
    if st.session_state.uploaded_data is None:
        st.error("No dataset uploaded!")
        return
    
    df = st.session_state.uploaded_data
    
    # Show dataset info
    st.markdown("""
    <div class="premium-banner">
        <h3>Ready to Train</h3>
        <p>Your dataset is loaded and validated. Click below to start training.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Dataset Size", f"{len(df):,} balls")
    with col2:
        st.metric("Matches", df['match_id'].nunique())
    with col3:
        st.metric("Teams", df['batting_team'].nunique())
    
    st.markdown("---")
    
    # Training scope selection
    st.markdown("### ⚙️ Training Configuration")
    
    total_matches = df['match_id'].nunique()
    
    training_scope = st.radio(
        "Training Scope",
        options=["All Matches", "Recent Matches Only"],
        help="Train on all matches for better accuracy, or recent matches for faster training"
    )
    
    max_matches = None
    if training_scope == "Recent Matches Only":
        max_matches = st.slider(
            "Number of Recent Matches",
            min_value=20,
            max_value=min(100, total_matches),
            value=50,
            step=10,
            help="Limit training to recent matches for faster performance"
        )
        st.info(f"⚡ Will train on last {max_matches} of {total_matches} matches (Faster)")
    else:
        st.info(f"📊 Will train on all {total_matches} matches (Better accuracy, may take longer)")
    
    st.markdown("---")
    
    # Training button
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        from src.ml.trainer import train_model_from_dataframe
        
        # Progress containers
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(message, percent=None):
            status_text.write(f"**{message}**")
            if percent is not None:
                progress_bar.progress(percent / 100)
        
        try:
            # Train model
            with st.spinner("Training in progress..."):
                model, feature_names, model_info = train_model_from_dataframe(
                    df, 
                    progress_callback=update_progress,
                    league_name=st.session_state.current_league,
                    max_matches=max_matches  # Pass the user's choice
                )
            
            # Show results
            st.success("🎉 Training Complete!")
            
            st.markdown("### 📈 Model Performance")
            best_model = model_info['best_model']
            scores = model_info['model_scores'][best_model]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Best Model", best_model)
            with col2:
                st.metric("R² Score", f"{scores['r2']:.3f}")
            with col3:
                st.metric("MAE", f"{scores['mae']:.2f}")
            
            # Save model info
            st.session_state.model_info = model_info
            st.session_state.model_trained = True
            
            # Option to save model to library
            st.markdown("---")
            st.markdown("### 💾 Save Model for Later")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                model_save_name = st.text_input(
                    "Model Name",
                    value=f"{st.session_state.current_league}_{datetime.now().strftime('%Y%m%d')}",
                    help="Enter a unique name to save this trained model"
                )
            with col2:
                st.write("")
                st.write("")
                if st.button("💾 Save Model", use_container_width=True):
                    from src.ml.model_library import ModelLibrary
                    library = ModelLibrary()
                    
                    try:
                        library.save_model(model, feature_names, model_info, model_save_name)
                        st.success(f"✅ Model saved as '{model_save_name}'")
                        st.info("You can load this model later from the Model Library")
                    except Exception as e:
                        st.error(f"Failed to save: {str(e)}")
            
            st.markdown("---")
            if st.button("✅ Proceed to App", type="primary", use_container_width=True):
                st.rerun()
                
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            st.info("Please check your dataset format and try again.")
    
    # Option to go back
    st.markdown("---")
    if st.button("⬅️ Back to Upload", use_container_width=True):
        st.session_state.uploaded_data = None
        st.rerun()


def show_model_library_page():
    """Model library management page."""
    st.markdown("## 📚 Model Library")
    
    st.markdown("""
    <div class="premium-banner">
        <h3>Saved Models</h3>
        <p>Load previously trained models or manage your model library</p>
    </div>
    """, unsafe_allow_html=True)
    
    from src.ml.model_library import ModelLibrary
    library = ModelLibrary()
    
    saved_models = library.list_models()
    
    if not saved_models:
        st.info("📭 No saved models yet. Train a model and save it to access later!")
        return
    
    st.markdown(f"### {len(saved_models)} Saved Models")
    
    for model in saved_models:
        with st.expander(f"🏏 {model['league_name']} - {model['model_name']}"):
            col1, col2,col3 = st.columns(3)
            
            with col1:
                st.metric("Matches", model['n_matches'])
                st.metric("Teams", model['n_teams'])
            
            with col2:
                st.metric("Best Model", model['best_model'])
                st.metric("R² Score", f"{model['r2_score']:.3f}")
            
            with col3:
                st.metric("Saved At", model['saved_at'].split()[0])
                st.write("")
            
            # Action buttons
            col_load, col_delete = st.columns(2)
            
            with col_load:
                if st.button(f"📥 Load Model", key=f"load_{model['model_name']}", use_container_width=True):
                    try:
                        # Load model
                        loaded_model, feature_names, model_info = library.load_model(model['model_name'])
                        
                        # Also load model to active location for predictor
                        import joblib
                        joblib.dump(loaded_model, 'models/fantasy_predictor.pkl')
                        joblib.dump(feature_names, 'models/feature_names.pkl')
                        
                        # Update session state
                        st.session_state.current_league = model['league_name']
                        st.session_state.model_info = model_info
                        st.session_state.model_trained = True
                        st.session_state.loaded_model_name = model['model_name']
                        
                        # If dataset was uploaded for this, load it
                        # Otherwise just mark the model as loaded
                        
                        st.success(f"✅ Loaded '{model['model_name']}'")
                        st.info("Note: Make sure to upload the corresponding dataset for this league")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to load: {str(e)}")
            
            with col_delete:
                if st.button(f"🗑️ Delete", key=f"del_{model['model_name']}", use_container_width=True, type="secondary"):
                    try:
                        library.delete_model(model['model_name'])
                        st.success(f"Deleted '{model['model_name']}'")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to delete: {str(e)}")
    
    # Show currently loaded model
    if st.session_state.loaded_model_name:
        st.markdown("---")
        st.info(f"**Currently Active:** {st.session_state.loaded_model_name} ({st.session_state.current_league})")


def show_home_page():
    """Display home page."""
    # Hero Banner
    st.markdown("""
    <div class="premium-banner">
        <h2 style="font-size: 2rem; margin: 0; font-weight: 700;">🏏 Fantasy Cricket Predictor</h2>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">
            Data-Driven Team Selection for Dream11 | Powered by Machine Learning
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🏏 Teams", "8", delta="BBL Women's")
    
    with col2:
        st.metric("🏟️ Venues", "16", delta="Grounds")
    
    with col3:
        st.metric("📊 Matches", f"{st.session_state.df['match_id'].nunique()}", delta="Analyzed")
    
    st.markdown("---")
    st.markdown("## 🚀 How It Works")
    st.markdown("**Step-by-step process to build your fantasy team:**")
    st.markdown("1. **Select Teams** - Choose two competing teams")
    st.markdown("2. **Pick Ground** - Select match venue")  
    st.markdown("3. **Build Squad** - Select 22 players from any team")
    st.markdown("4. **Get Predictions** - ML analyzes performance")
    st.markdown("5. **Fantasy Team** - Receive optimized 14-player squad")
    
    st.markdown("---")
    st.markdown("## 🎯 Features")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🤖 Machine Learning**")
        st.markdown("- Random Forest & XGBoost models")
        st.markdown("- 65K+ ball records analyzed")
        st.markdown("- Venue-specific predictions")
        
        st.markdown("**📊 Statistical Analysis**")
        st.markdown("- Batting & bowling averages")
        st.markdown("- Recent form tracking")
        st.markdown("- Consistency metrics")
    
    with col2:
        st.markdown("**🏆 Dream11 Scoring**")
        st.markdown("- Official point system")
        st.markdown("- Captain 2x multiplier")
        st.markdown("- Vice-captain 1.5x multiplier")
        
        st.markdown("**✨ Smart Selection**")
        st.markdown("- Optimal 14-player teams")
        st.markdown("- Role-based composition")
        st.markdown("- Performance insights")


def show_team_selection_page():
    """Team selection interface."""
    st.markdown("## 🎯 Select Teams")
    st.markdown("Choose two teams that will compete in the match")
    
    teams = st.session_state.loader.get_teams()
    
    col1, col2 = st.columns(2)
    
    with col1:
        team1 = st.selectbox(
            "Team 1",
            options=teams,
            index=teams.index(st.session_state.selected_team1) if st.session_state.selected_team1 in teams else 0
        )
    
    with col2:
        available_teams = [t for t in teams if t != team1]
        team2 = st.selectbox(
            "Team 2",
            options=available_teams,
            index=available_teams.index(st.session_state.selected_team2) if st.session_state.selected_team2 in available_teams else 0
        )
    
    if st.button("Confirm Team Selection", type="primary"):
        st.session_state.selected_team1 = team1
        st.session_state.selected_team2 = team2
        st.success(f"✅ Selected: {team1} vs {team2}")
        st.balloons()


def show_ground_selection_page():
    """Ground selection interface."""
    st.markdown("## 🏟️ Select Ground")
    
    if not st.session_state.selected_team1 or not st.session_state.selected_team2:
        st.warning("⚠️ Please select teams first!")
        return
    
    st.markdown(f"Match: **{st.session_state.selected_team1}** vs **{st.session_state.selected_team2}**")
    
    venues = st.session_state.loader.get_venues()
    
    venue = st.selectbox(
        "Select Venue",
        options=venues,
        index=venues.index(st.session_state.selected_venue) if st.session_state.selected_venue in venues else 0
    )
    
    # Show ground stats
    venue_df = st.session_state.df[st.session_state.df['venue'] == venue]
    if len(venue_df) > 0:
        st.markdown("### Ground Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            matches_at_venue = venue_df['match_id'].nunique()
            st.metric("Matches Played", matches_at_venue)
        
        with col2:
            avg_score = venue_df.groupby('match_id')['total_runs'].sum().mean()
            st.metric("Avg Match Runs", f"{avg_score:.0f}")
        
        with col3:
            wickets = venue_df[venue_df['is_wicket'] == True]['match_id'].count()
            st.metric("Total Wickets", wickets)
    
    if st.button("Confirm Ground Selection", type="primary"):
        st.session_state.selected_venue = venue
        st.success(f"✅ Selected venue: {venue}")


def show_player_pool_page():
    """Player pool selection and tagging."""
    st.markdown("## 👥 Player Pool & Team Assignment")
    
    if not st.session_state.selected_team1 or not st.session_state.selected_team2:
        st.warning("⚠️ Please select teams first!")
        return
    
    team1 = st.session_state.selected_team1
    team2 = st.session_state.selected_team2
    
    # Get players for ALL teams (expanded pool)
    all_team_players = st.session_state.loader.get_players()
    all_unique_players = sorted(list(set().union(*all_team_players.values())))
    
    # Get players for selected teams (for default tagging)
    match_team_players = st.session_state.loader.get_players([team1, team2])
    
    st.markdown(f"**Available Players:** {len(all_unique_players)} (All Teams)")
    st.info("ℹ️ Select exactly 22 players. You can search for players from any team.")
    
    # Search bar
    search_query = st.text_input("🔍 Search Player Pool", placeholder="Type player name...")
    
    # --- NEW SELECTION LOGIC ---
    
    # 1. Search and Add Interface
    col_search, col_add = st.columns([3, 1])
    
    with col_search:
        # Filter out already selected players
        current_selection = st.session_state.selected_players
        available_players = [p for p in all_unique_players if p not in current_selection]
        
        # Search box
        player_to_add = st.selectbox(
            "Search and Select Player",
            options=available_players,
            index=None,
            placeholder="Type to search player...",
            key="player_search_box"
        )
        
    with col_add:
        st.write("") # Spacer
        st.write("") # Spacer
        if st.button("➕ Add Player", type="primary", disabled=not player_to_add):
            if player_to_add and player_to_add not in st.session_state.selected_players:
                st.session_state.selected_players.append(player_to_add)
                st.rerun()
    
    # 2. Selected Players List (Below)
    st.markdown("---")
    st.markdown(f"### Selected Squad ({len(st.session_state.selected_players)}/22)")
    
    if not st.session_state.selected_players:
        st.info("No players selected yet. Search and add players above.")
    else:
        # Header
        h1, h2, h3 = st.columns([3, 2, 1])
        h1.markdown("**Player Name**")
        h2.markdown("**Team Assignment**")
        h3.markdown("**Action**")
        
        players_to_remove = []
        player_tags = st.session_state.player_team_tags
        
        for i, player in enumerate(st.session_state.selected_players):
            c1, c2, c3 = st.columns([3, 2, 1])
            
            with c1:
                st.write(f"{i+1}. {player}")
                
            with c2:
                # Default assignment logic
                default_team = team1 if player in match_team_players.get(team1, set()) else team2
                # Use existing tag if available, else default
                current_tag = player_tags.get(player, default_team)
                
                new_tag = c2.selectbox(
                    f"Team for {player}",
                    options=[team1, team2],
                    index=0 if current_tag == team1 else 1,
                    key=f"tag_{player}",
                    label_visibility="collapsed"
                )
                player_tags[player] = new_tag
                
            with c3:
                if c3.button("🗑️", key=f"remove_{player}"):
                    players_to_remove.append(player)
        
        # Process removals
        if players_to_remove:
            for p in players_to_remove:
                st.session_state.selected_players.remove(p)
                if p in player_tags:
                    del player_tags[p]
            st.session_state.player_team_tags = player_tags
            st.rerun()
            
        # Update tags in session state
        st.session_state.player_team_tags = player_tags

    # 3. Confirmation
    st.markdown("---")
    if len(st.session_state.selected_players) == 22:
        st.success("✅ Squad Complete! Ready for predictions.")
    elif len(st.session_state.selected_players) > 22:
        st.error(f"⚠️ Too many players! Remove {len(st.session_state.selected_players) - 22} players.")
    else:
        st.warning(f"⚠️ Select {22 - len(st.session_state.selected_players)} more players")


def show_predictions_page():
    """ML predictions and fantasy team display."""
    st.markdown("## 📊 Fantasy Team Predictions")
    
    # Validate prerequisites
    if not st.session_state.selected_players or len(st.session_state.selected_players) != 22:
        st.warning("⚠️ Please select 22 players first!")
        return
    
    if not st.session_state.selected_venue:
        st.warning("⚠️ Please select a venue!")
        return
    
    # Check if model exists
    if not os.path.exists('models/fantasy_predictor.pkl'):
        st.error("❌ ML Model not found! Please train the model first.")
        st.code("python scripts/train_model.py")
        return
    
    team1 = st.session_state.selected_team1
    team2 = st.session_state.selected_team2
    venue = st.session_state.selected_venue
    players = st.session_state.selected_players
    
    st.markdown(f"**Match:** {team1} vs {team2}")
    st.markdown(f"**Venue:** {venue}")
    st.markdown(f"**Players:** {len(players)}")
    
    if st.button("🚀 Generate Predictions", type="primary"):
        with st.spinner("Running ML predictions..."):
            try:
                from src.ml.predictor import load_predictor
                from src.optimization.team_selector import TeamSelector
                
                # Load predictor
                predictor = load_predictor()
                
                # Generate predictions
                predictions_df = predictor.predict_all_players(
                    players, team1, team2, venue, st.session_state.df
                )
                
                # Select fantasy team
                selector = TeamSelector()
                fantasy_team = selector.select_fantasy_team(
                    predictions_df, st.session_state.player_team_tags
                )
                
                # Get captain/VC
                captain, vc = selector.suggest_captain_vice_captain(fantasy_team)
                
                # Display results
                st.success("✅ Predictions Generated!")
                
                # Fantasy Team
                st.markdown("### 🏆 Recommended Fantasy Team (Top 14)")
                
                fantasy_display = fantasy_team[['player', 'team', 'role', 'predicted_points']].copy()
                fantasy_display['predicted_points'] = fantasy_display['predicted_points'].round(1)
                fantasy_display.index = range(1, len(fantasy_display) + 1)
                
                # Highlight captain and VC
                def highlight_captain(row):
                    if row['player'] == captain:
                        return ['background-color: #ffd700'] * len(row)
                    elif row['player'] == vc:
                        return ['background-color: #c0c0c0'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(fantasy_display, use_container_width=True)
                
                # Captain/VC
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"### 👑 Captain: {captain}")
                    st.markdown(f"**Predicted Points (2x):** {fantasy_team[fantasy_team['player'] == captain]['predicted_points'].values[0] * 2:.1f}")
                
                with col2:
                    st.markdown(f"### 🥈 Vice-Captain: {vc}")
                    st.markdown(f"**Predicted Points (1.5x):** {fantasy_team[fantasy_team['player'] == vc]['predicted_points'].values[0] * 1.5:.1f}")
                
                # Total expected points
                total_points = selector.calculate_expected_team_points(fantasy_team, captain, vc)
                st.markdown(f"### 💯 Expected Team Total: {total_points:.1f} points")
                
                # All players ranking
                st.markdown("### 📋 All 22 Players - Performance Ranking")
                all_display = predictions_df[['player', 'team', 'predicted_points']].copy()
                all_display['predicted_points'] = all_display['predicted_points'].round(1)
                all_display['rank'] = range(1, len(all_display) + 1)
                all_display = all_display[['rank', 'player', 'team', 'predicted_points']]
                
                st.dataframe(all_display, use_container_width=True, hide_index=True)
                
            except Exception as e:
                st.error(f"Error generating predictions: {e}")
                import traceback
                st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
