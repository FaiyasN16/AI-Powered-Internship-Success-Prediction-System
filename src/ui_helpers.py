"""
Streamlit UI and Styling Helper Module.

Provides dark-purple theme CSS, custom card components, probability gauges,
and standard headers for a consistent user experience across all dashboard pages.
"""

import streamlit as st
import plotly.graph_objects as go


CUSTOM_CSS = """
<style>
    /* Global application styling */
    .stApp {
        background-color: #100D1C;
        color: #F5F3FF;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Sidebar customization */
    [data-testid="stSidebar"] {
        background-color: #161226 !important;
        border-right: 1px solid #282245;
    }
    [data-testid="stSidebarNav"] {
        padding-top: 1.5rem;
    }

    /* Cards and Containers */
    .dashboard-card {
        background-color: #1C1730;
        border: 1px solid #2D254D;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .dashboard-card:hover {
        border-color: #7C3AED;
    }

    .metric-card {
        background: linear-gradient(135deg, #1C1730 0%, #241D40 100%);
        border: 1px solid #3B2E68;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        margin-bottom: 0.8rem;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #A78BFA;
        margin: 0.3rem 0;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #B8B1CC;
        font-weight: 600;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #948BA8;
    }

    /* Badges */
    .badge-selected {
        display: inline-block;
        background: linear-gradient(135deg, #059669 0%, #10B981 100%);
        color: #FFFFFF;
        font-size: 1.2rem;
        font-weight: 700;
        padding: 0.5rem 1.6rem;
        border-radius: 9999px;
        letter-spacing: 0.05em;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.4);
    }
    .badge-not-selected {
        display: inline-block;
        background: linear-gradient(135deg, #DC2626 0%, #EF4444 100%);
        color: #FFFFFF;
        font-size: 1.2rem;
        font-weight: 700;
        padding: 0.5rem 1.6rem;
        border-radius: 9999px;
        letter-spacing: 0.05em;
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.4);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #7C3AED 0%, #6D28D9 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid #A78BFA !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.8rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 14px rgba(124, 58, 237, 0.35) !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #8B5CF6 0%, #7C3AED 100%) !important;
        box-shadow: 0 6px 20px rgba(167, 139, 250, 0.5) !important;
        transform: translateY(-1px);
    }

    /* Form Inputs */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div {
        background-color: #1C1730 !important;
        color: #F5F3FF !important;
        border: 1px solid #3B2E68 !important;
        border-radius: 8px !important;
    }

    /* Header Accent */
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #F5F3FF 30%, #A78BFA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
    }
    .hero-subtitle {
        color: #B8B1CC;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    /* Disclaimer box */
    .disclaimer-box {
        background-color: rgba(124, 58, 237, 0.08);
        border-left: 4px solid #7C3AED;
        padding: 0.8rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        color: #B8B1CC;
        font-size: 0.85rem;
    }
</style>
"""


def apply_custom_css():
    """Inject custom dark-purple styling into Streamlit application."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_header(title: str, subtitle: str, icon: str = "🎓"):
    """Render consistent hero title header."""
    st.markdown(
        f"""
        <div style="margin-bottom: 1.5rem;">
            <div class="hero-title">{icon} {title}</div>
            <div class="hero-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_disclaimer():
    """Render the standard academic decision-support prototype disclaimer."""
    st.markdown(
        """
        <div class="disclaimer-box">
            ⚖️ <strong>Academic Decision-Support Prototype:</strong> This application utilizes a Random Forest
            machine learning model trained on reproducible synthetic student profiles. Predictions and scores are
            intended strictly for developmental feedback and career planning, not as an official employment guarantee
            or recruitment decision-maker.
        </div>
        """,
        unsafe_allow_html=True
    )


def create_probability_gauge_plotly(prob_selected: float) -> go.Figure:
    """
    Render a clean radial gauge chart for selection probability.
    """
    percentage = round(prob_selected * 100, 1)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percentage,
        number={"suffix": "%", "font": {"size": 42, "color": "#F5F3FF", "family": "Inter"}},
        title={"text": "<b>Estimated Selection Probability</b><br><span style='font-size:0.8em;color:#B8B1CC'>Random Forest Output</span>", "font": {"size": 15, "color": "#B8B1CC"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#B8B1CC"},
            "bar": {"color": "#A78BFA", "thickness": 0.28},
            "bgcolor": "#1C1730",
            "borderwidth": 1,
            "bordercolor": "#2D254D",
            "steps": [
                {"range": [0, 40], "color": "rgba(239, 68, 68, 0.2)"},
                {"range": [40, 70], "color": "rgba(245, 158, 11, 0.2)"},
                {"range": [70, 100], "color": "rgba(16, 185, 129, 0.2)"},
            ],
            "threshold": {
                "line": {"color": "#10B981", "width": 4},
                "thickness": 0.75,
                "value": 50.0,
            },
        }
    ))

    fig.update_layout(
        paper_bgcolor="#161226",
        plot_bgcolor="#161226",
        font={"color": "#F5F3FF", "family": "Inter"},
        height=280,
        margin=dict(l=25, r=25, t=50, b=25),
    )
    return fig

