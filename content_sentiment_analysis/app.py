from typing import Dict, List, Tuple
import streamlit as st
import aivm_client as aic
import torch
import time
import plotly.graph_objects as go
import pandas as pd
from dataclasses import dataclass
from functools import lru_cache

@dataclass
class Config:
    """Application configuration settings"""
    MODEL_NAME: str = "BertTinySentimentTwitter"
    TOXICITY_THRESHOLDS: Dict[str, float] = None
    SENTIMENT_CLASSES: List[str] = None
    
    def __post_init__(self):
        self.TOXICITY_THRESHOLDS = {
            "HIGH": 0.7,
            "MODERATE": 0.4,
            "LOW": 0.0
        }
        self.SENTIMENT_CLASSES = ["Negative", "Neutral", "Positive"]

config = Config()

def setup_page():
    """Configure page settings and custom styling"""
    st.set_page_config(
        page_title="Content Sentiment Analyzer",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Enable smooth scrolling and other improvements
    st.markdown("""
        <script>
            // Enable smooth scrolling
            document.documentElement.style.scrollBehavior = 'smooth';
            
            // Function to scroll to results
            function scrollToResults() {
                const resultsSection = document.getElementById('results-section');
                if (resultsSection) {
                    resultsSection.scrollIntoView({ 
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            }
            
            // Call scroll function when results are ready
            window.addEventListener('load', function() {
                if (document.getElementById('results-section')) {
                    setTimeout(scrollToResults, 100);
                }
            });
        </script>

        <style>
        .main {
            padding: 0rem 1rem;
        }
        .stTextArea textarea {
            border-radius: 10px;
            max-height: 150px;
            min-height: 70px;
            margin-bottom: 0 !important;
        }
        .stButton>button {
            border-radius: 10px;
            padding: 0.5rem 1rem;
            width: auto;
            min-width: 200px;
        }
        div.block-container {
            padding-top: 1rem;
            max-width: 100%;
        }
        .sentiment-box {
            padding: 1.5rem;
            border-radius: 10px;
            margin: 0.5rem 0;
            text-align: center;
            height: 150px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        .sentiment-box h2 {
            margin: 0.5rem 0;
            font-size: 1.5rem;
            line-height: 1.0;
        }
        .sentiment-box h3 {
            margin: 0;
            font-size: 1.2rem;
            line-height: 1.0;
        }
        .sentiment-box p {
            margin: 0.5rem 0 0 0;
            font-size: 0.9rem;
            line-height: 1.0;
            opacity: 0.8;
        }
        .positive {
            background-color: #dcfce7;
            color: #166534;
        }
        .negative {
            background-color: #fee2e2;
            color: #991b1b;
        }
        .neutral {
            background-color: #fef9c3;
            color: #854d0e;
        }
        .stPlotlyChart {
            width: 100%;
            margin: 1rem 0;
        }
        div[data-testid="metric-container"] {
            background-color: #f8fafc;
            border-radius: 10px;
            padding: 1rem;
            margin: 0rem;
        }
        div[data-testid="stExpander"] {
            border-radius: 10px;
            margin: 1rem 0;
        }
        div[data-testid="column"] {
            padding: 0.5rem;
            margin-top: 0.5rem !important;
        }
        div[data-testid="stMetricLabel"] > div {
            font-size: 1rem;
        }
        div[data-testid="stMetricValue"] > div {
            font-size: 1.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def check_model_exists() -> bool:
    """Check if required model exists in AIVM."""
    try:
        available_models = aic.get_supported_models()
        return config.MODEL_NAME in available_models
    except Exception as e:
        st.error(f"Error checking model availability: {str(e)}. Please ensure AIVM service is running.")
        return False

@st.cache_data(ttl=60)
def analyze_text(text: str) -> Dict:
    """Analyze text sentiment using AIVM model."""
    try:
        tokenized_inputs = aic.tokenize(text)
        encrypted_inputs = aic.BertTinyCryptensor(tokenized_inputs[0], tokenized_inputs[1])
        result = aic.get_prediction(encrypted_inputs, config.MODEL_NAME)
        
        logits = result[0]
        probabilities = torch.nn.functional.softmax(logits, dim=0)
        
        predicted_class = torch.argmax(logits).item()
        prob_list = probabilities.tolist()
        
        sorted_probs, _ = torch.sort(probabilities, descending=True)
        confidence = (sorted_probs[0] / (sorted_probs[0] + sorted_probs[1])).item()
        
        neg_prob, neut_prob, pos_prob = prob_list
        
        sentiment = determine_sentiment(neg_prob, neut_prob, pos_prob)
        toxicity_score = neg_prob
        
        return {
            'sentiment': sentiment,
            'toxicity_score': toxicity_score,
            'toxicity_level': get_toxicity_level(toxicity_score),
            'confidence': confidence,
            'suggestions': generate_suggestions(toxicity_score),
            'probabilities': prob_list,
            'raw_scores': {
                'Negative': f"{neg_prob:.1%}",
                'Neutral': f"{neut_prob:.1%}",
                'Positive': f"{pos_prob:.1%}"
            }
        }
        
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}. Please try again or contact support if the issue persists.")
        return None

def determine_sentiment(neg_prob: float, neut_prob: float, pos_prob: float) -> str:
    """Determine sentiment based on class probabilities."""
    if neg_prob > 0.4 and neg_prob > pos_prob:
        return "Negative"
    elif pos_prob > 0.4 and pos_prob > neg_prob:
        return "Positive"
    return "Neutral"

def get_toxicity_level(score: float) -> str:
    """Get toxicity level based on score thresholds."""
    if score > config.TOXICITY_THRESHOLDS["HIGH"]:
        return "High"
    elif score > config.TOXICITY_THRESHOLDS["MODERATE"]:
        return "Moderate"
    return "Low"

def generate_suggestions(score: float) -> str:
    """Generate content improvement suggestions based on toxicity score."""
    if score > config.TOXICITY_THRESHOLDS["HIGH"]:
        return ("⚠️ High toxicity detected. Consider:\n"
               "- Rephrasing more constructively\n"
               "- Using less confrontational language\n"
               "- Focusing on solutions rather than problems")
    elif score > config.TOXICITY_THRESHOLDS["MODERATE"]:
        return ("⚠️ Moderate toxicity detected. Consider:\n"
               "- Adding more balanced viewpoints\n"
               "- Using more neutral language\n"
               "- Clarifying your intent")
    return "✅ This message appears appropriate and constructive."

def create_gauge_chart(score: float, toxicity_level: str) -> go.Figure:
    """Create a responsive gauge chart for toxicity visualization."""
    colors = {
        "Low": "#dcfce7",
        "Moderate": "#fef9c3",
        "High": "#fee2e2"
    }
    
    text_colors = {
        "Low": "#166534",
        "Moderate": "#854d0e",
        "High": "#991b1b"
    }
    
    fig = go.Figure(go.Indicator(
        mode="gauge",
        value=score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': "Content Toxicity Assessment",
            'font': {'size': 20, 'color': '#1f2937'}
        },
        gauge={
            'axis': {
                'range': [None, 100],
                'tickwidth': 1,
                'tickfont': {'size': 12},
                'tickcolor': "#6b7280"
            },
            'bar': {'color': "rgba(0,0,0,0)"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e5e7eb",
            'steps': [
                {'range': [0, 40], 'color': "#dcfce7"},
                {'range': [40, 70], 'color': "#fef9c3"},
                {'range': [70, 100], 'color': "#fee2e2"}
            ],
            'threshold': {
                'line': {'color': text_colors[toxicity_level], 'width': 4},
                'thickness': 0.8,
                'value': score * 100
            }
        }
    ))

    fig.add_annotation(
        text=toxicity_level,
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(
            size=24,
            color=text_colors[toxicity_level]
        )
    )

    fig.update_layout(
        autosize=True,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='white',
        plot_bgcolor='white',
        font={'color': "#1f2937"},
        height=200
    )

    return fig

def display_results(result: Dict):
    """Display analysis results in the Streamlit UI."""
    # Create an anchor for scrolling
    anchor = st.empty()
    st.markdown("""
        <div id="results-section"></div>
        <script>
            document.getElementById('results-section').scrollIntoView({ behavior: 'smooth' });
        </script>
    """, unsafe_allow_html=True)
    
    main_container = st.container()
    
    with main_container:
        # Top metrics in two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
                <div class="sentiment-box {result['sentiment'].lower()}">
                    <h3>Sentiment</h3>
                    <h2>{result['sentiment']}</h2>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="sentiment-box neutral">
                    <h3>Confidence</h3>
                    <h2>{result['confidence']:.1%}</h2>
                    <p>Margin between top predictions</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Gauge chart
        st.plotly_chart(
            create_gauge_chart(
                result['toxicity_score'],
                result['toxicity_level']
            ),
            use_container_width=True,
            config={'displayModeBar': False}
        )
        
        # Sentiment distribution
        st.markdown("#### Sentiment Distribution")
        metrics_container = st.container()
        prob_cols = metrics_container.columns(3)
        for i, (label, prob) in enumerate(result['raw_scores'].items()):
            with prob_cols[i]:
                st.metric(label, prob)
        
        # Suggestions
        if result['toxicity_level'] != 'Low':
            st.warning(result['suggestions'], icon="⚠️")
        else:
            st.success(result['suggestions'], icon="✅")

def update_history(text: str, result: Dict):
    """Update analysis history in session state."""
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    st.session_state.history.append({
        'text': text[:50] + '...' if len(text) > 50 else text,
        'sentiment': result['sentiment'],
        'toxicity_level': result['toxicity_level'],
        'confidence': f"{result['confidence']:.1%}"
    })

def main():
    """Main application entry point"""
    setup_page()
    
    # Main content container
    main_container = st.container()
    
    with main_container:
        st.title("🛡️ Content Sentiment Analyzer")
        st.markdown("Analyze the sentiment and toxicity of your content using privacy-preserving AI")
        
        if not check_model_exists():
            st.error("Required model not found. Please ensure the model is properly uploaded to AIVM.")
            st.stop()
        
        # Input section with responsive width
        col1, col2 = st.columns([3, 1])
        with col1:
            text = st.text_area(
                "Enter your text for analysis",
                placeholder="Type or paste your content here..."
            )
        
        with col2:
            analyze_button = st.button(
                "Analyze Content",
                disabled=not text,
                use_container_width=True
            )
        
        if analyze_button and text:
            with st.spinner("Analyzing content..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                result = analyze_text(text)
                progress_bar.empty()
                
                if result:
                    # Add a small delay to ensure smooth scrolling
                    time.sleep(0.1)
                    display_results(result)
                    update_history(text, result)
                    
                    # History section in collapsible container
                    with st.expander("View Analysis History"):
                        st.dataframe(
                            pd.DataFrame(st.session_state.history),
                            use_container_width=True,
                            height=200
                        )

if __name__ == "__main__":
    main()