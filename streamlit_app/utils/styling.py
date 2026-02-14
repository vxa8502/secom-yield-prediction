"""
Streamlit styling configuration
Author: Victoria A.

Clean, professional dashboard styling with Roboto font.
Cohesive blue theme with minimal color distractions.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

# Centralized path setup
from streamlit_app import setup_project_path
setup_project_path()

from src.config import COLORS, VIZ_CONFIG


# Title styling kwargs for matplotlib (DRY: used across all dashboard plots)
TITLE_STYLE: dict[str, str | int] = {
    'fontsize': VIZ_CONFIG['title_fontsize'],
    'fontweight': 'bold',
    'color': COLORS['primary'],
}

# Page configuration
PAGE_CONFIG = {
    "page_title": "SECOM Defect Prediction",
    "page_icon": "🔬",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Clean CSS with Roboto font and cohesive blue theme
STYLE_CSS = f"""
    /* Import Roboto font */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

    /* Global font application */
    html, body, [class*="css"], .stMarkdown, .stText, p, span, div,
    .stSelectbox, .stMultiSelect, .stTextInput, .stNumberInput,
    .stRadio, .stCheckbox, label, .stButton button {{
        font-family: 'Roboto', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }}

    /* Main container */
    .main {{
        padding: 1.5rem 2rem;
        background-color: #fafbfc;
    }}

    /* Sidebar - light theme with blue accent */
    [data-testid="stSidebar"] {{
        background-color: #f8f9fa;
        border-right: 3px solid {COLORS['primary']};
    }}

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span {{
        color: #333;
    }}

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{
        color: {COLORS['primary']} !important;
    }}

    /* Headers */
    h1 {{
        color: {COLORS['primary']} !important;
        font-family: 'Roboto', sans-serif !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid {COLORS['primary']};
        margin-bottom: 1rem;
    }}

    h2 {{
        color: {COLORS['primary']} !important;
        font-family: 'Roboto', sans-serif !important;
        font-weight: 500 !important;
        font-size: 1.4rem !important;
        padding-top: 0.8rem;
        margin-bottom: 0.5rem;
    }}

    h3 {{
        color: {COLORS['secondary']} !important;
        font-family: 'Roboto', sans-serif !important;
        font-weight: 500 !important;
        font-size: 1.1rem !important;
    }}

    /* Metric cards - blue theme */
    [data-testid="stMetricValue"] {{
        font-family: 'Roboto', sans-serif !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: {COLORS['primary']} !important;
    }}

    [data-testid="stMetricLabel"] {{
        font-family: 'Roboto', sans-serif !important;
        font-weight: 500 !important;
        color: #444 !important;
    }}

    [data-testid="stMetricDelta"] {{
        font-family: 'Roboto', sans-serif !important;
    }}

    /* ALL alert/callout boxes - cohesive blue theme */
    /* Remove jarring colored backgrounds, use subtle blue styling */
    .stAlert, div[data-testid="stAlert"] {{
        background-color: {COLORS['primary']}08 !important;
        border: 1px solid {COLORS['primary']}30 !important;
        border-left: 4px solid {COLORS['primary']} !important;
        border-radius: 0.5rem !important;
        color: #333 !important;
    }}

    .stAlert p, div[data-testid="stAlert"] p {{
        color: #333 !important;
    }}

    /* Info boxes - blue accent */
    div[data-baseweb="notification"][kind="info"],
    .element-container:has(.stAlert) {{
        background-color: {COLORS['primary']}08 !important;
    }}

    /* Success messages - subtle teal accent but not jarring */
    .stSuccess {{
        background-color: {COLORS['pass']}10 !important;
        border: 1px solid {COLORS['pass']}40 !important;
        border-left: 4px solid {COLORS['pass']} !important;
    }}

    /* Warning/Error - subtle orange accent */
    .stWarning, .stError {{
        background-color: {COLORS['fail']}08 !important;
        border: 1px solid {COLORS['fail']}30 !important;
        border-left: 4px solid {COLORS['fail']} !important;
    }}

    /* Primary buttons - blue */
    .stButton > button {{
        background-color: {COLORS['primary']} !important;
        color: white !important;
        border: none !important;
        border-radius: 0.5rem !important;
        font-family: 'Roboto', sans-serif !important;
        font-weight: 500 !important;
        padding: 0.5rem 1.2rem !important;
        transition: all 0.2s ease !important;
    }}

    .stButton > button:hover {{
        background-color: #0d1a6b !important;
        box-shadow: 0 2px 8px rgba(20, 40, 160, 0.25) !important;
    }}

    /* Form submit button */
    .stFormSubmitButton > button {{
        background-color: {COLORS['primary']} !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 0.5rem !important;
    }}

    /* Tabs - blue active state */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.25rem;
        background-color: transparent;
    }}

    .stTabs [data-baseweb="tab"] {{
        font-family: 'Roboto', sans-serif !important;
        font-weight: 500 !important;
        color: {COLORS['neutral']} !important;
        border-radius: 0.5rem 0.5rem 0 0;
        padding: 0.5rem 1rem;
        background-color: #f0f2f6;
        border: 1px solid #e0e0e0;
        border-bottom: none;
    }}

    .stTabs [aria-selected="true"] {{
        background-color: {COLORS['primary']} !important;
        color: white !important;
        border-color: {COLORS['primary']} !important;
    }}

    /* Selectbox - blue focus */
    .stSelectbox [data-baseweb="select"] > div {{
        border-radius: 0.5rem !important;
        font-family: 'Roboto', sans-serif !important;
    }}

    .stSelectbox [data-baseweb="select"] > div:focus-within {{
        border-color: {COLORS['primary']} !important;
        box-shadow: 0 0 0 1px {COLORS['primary']} !important;
    }}

    /* Slider - blue */
    .stSlider [data-baseweb="slider"] [role="slider"] {{
        background-color: {COLORS['primary']} !important;
    }}

    /* Number input */
    .stNumberInput input {{
        border-radius: 0.5rem !important;
        font-family: 'Roboto', sans-serif !important;
    }}

    .stNumberInput input:focus {{
        border-color: {COLORS['primary']} !important;
        box-shadow: 0 0 0 1px {COLORS['primary']} !important;
    }}

    /* DataFrame/Table - blue header */
    .stDataFrame thead tr th {{
        background-color: {COLORS['primary']} !important;
        color: white !important;
        font-weight: 500 !important;
        font-family: 'Roboto', sans-serif !important;
    }}

    .dataframe {{
        font-family: 'Roboto', sans-serif !important;
        font-size: 0.9rem;
    }}

    /* Expander - blue accent */
    .streamlit-expanderHeader {{
        font-family: 'Roboto', sans-serif !important;
        font-weight: 500 !important;
        color: {COLORS['primary']} !important;
        background-color: {COLORS['primary']}05;
        border-radius: 0.5rem;
    }}

    /* Code blocks */
    code {{
        background-color: #f4f4f4;
        padding: 0.2rem 0.4rem;
        border-radius: 0.3rem;
        font-size: 0.85rem;
    }}

    /* Divider */
    hr {{
        border: none;
        border-top: 1px solid {COLORS['primary']}20;
        margin: 1.5rem 0;
    }}

    /* Caption */
    .stCaption {{
        color: {COLORS['neutral']} !important;
        font-family: 'Roboto', sans-serif !important;
    }}

    /* Radio buttons */
    .stRadio [role="radiogroup"] label {{
        font-family: 'Roboto', sans-serif !important;
    }}

    /* File uploader - blue accent */
    .stFileUploader {{
        border: 2px dashed {COLORS['primary']}40 !important;
        border-radius: 0.5rem !important;
    }}

    .stFileUploader:hover {{
        border-color: {COLORS['primary']} !important;
    }}

    /* Progress bar - blue */
    .stProgress > div > div > div {{
        background-color: {COLORS['primary']} !important;
    }}

    /* Download button - secondary blue */
    .stDownloadButton > button {{
        background-color: {COLORS['secondary']} !important;
        color: white !important;
    }}

    .stDownloadButton > button:hover {{
        background-color: #1e7a9c !important;
    }}

    /* Remove default colored backgrounds from markdown callouts */
    .stMarkdown div[data-testid="stMarkdownContainer"] > div {{
        background-color: transparent !important;
    }}
"""


def apply_styling(st_module) -> None:
    """
    Apply custom CSS styling to Streamlit app.

    Injects the STYLE_CSS into the page using st.markdown with unsafe_allow_html.
    Call this once at the top of each page after st.set_page_config().

    Args:
        st_module: Streamlit module instance
    """
    st_module.markdown(f"<style>{STYLE_CSS}</style>", unsafe_allow_html=True)


def show_figure(fig: plt.Figure) -> None:
    """
    Finalize and display a matplotlib figure in Streamlit.

    Consolidates the repeated pattern of tight_layout + pyplot + close
    used across all dashboard pages.

    Args:
        fig: Matplotlib figure to display
    """
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
