import streamlit as st
import pandas as pd
import datetime
import requests
import os
import time
from typing import Dict, Any, Optional, List
import json
from urllib.parse import urljoin


BACKEND_BASE_URL = os.getenv("BANK_AUCTION_INSIGHTS_API_URL", "http://localhost:8000")
AUCTION_INSIGHTS_ENDPOINT = "/auction-insights"


st.set_page_config(
    page_title="Auction Portal India",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: bold;
        }
        .metric-tile {
            background: linear-gradient(135deg, #ff6b6b 0%, #ff8e53 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin: 0.5rem 0;
        }
        .insight-section {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .insight-section h3 {
            color: #1f77b4;
            border-bottom: 1px solid #ddd;
            padding-bottom: 0.5rem;
        }
        .connection-status {
            padding: 0.5rem;
            border-radius: 5px;
            margin-bottom: 1rem;
        }
        .connection-success {
            background-color: #e6f7e6;
            color: #2e7d32;
        }
        .connection-error {
            background-color: #ffebee;
            color: #c62828;
        }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("🏛️ Auction Portal")
page = st.sidebar.radio(
    "Navigate to:",
    ["🏠 Dashboard", "🔍 Search Analytics", "🧠 AI Auction Insights"],
    index=0
)

def check_backend_connection() -> bool:
    """Check if backend is reachable with retry logic."""
    retries = 3
    for attempt in range(retries):
        try:
            response = requests.get(
                urljoin(BACKEND_BASE_URL, "/ping"),
                timeout=5
            )
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            if attempt == retries - 1: 
                return False
            time.sleep(1)  
    return False

def safe_get_column(df: pd.DataFrame, column_name: str, default=None):
    """Safely get a column from DataFrame, returning default if column doesn't exist."""
    return df[column_name] if column_name in df.columns else pd.Series([default] * len(df))

@st.cache_data
def load_auction_data():
    """Load and preprocess auction data from CSV with robust error handling."""
    csv_path = r"auction_exports/combined_auctions_20250719.csv"
    
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            st.error(" CSV file is empty.")
            return None, None
        
        # Standardize column names (lowercase and underscores)
        df.columns = df.columns.str.strip().str.lower().str.replace(r"[^\w]+", "_", regex=True)

        
        # Ensure required columns exist
        required_columns = {
            'auction_id': ['auction_id', 'cin_llpin', 'unique_number'],
            'location_of_assets': ['location_of_assets', 'address', 'city']
        }
        
        # Map alternative column names
        for target_col, possible_cols in required_columns.items():
            if target_col not in df.columns:
                for col in possible_cols:
                    if col in df.columns:
                        df[target_col] = df[col]
                        break
                else:
                    df[target_col] = None
        
        # Date processing with proper format handling
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for col in date_columns:
            try:
                # Try parsing with dayfirst=True for dd-mm-yyyy format
                df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
            except:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Calculate days until submission if we have auction dates
        if 'auction_date_pdf' in df.columns:
            today = pd.Timestamp.now().normalize()
            df['days_until_submission'] = (df['auction_date_pdf'] - today).dt.days
        
        return df, csv_path
    
    except Exception as e:
        st.error(f" Failed to load data: {str(e)}")
        return None, None

def get_display_columns(dataframe: pd.DataFrame) -> list:
    """Get columns to display based on available data."""
    possible_cols = [
        'auction_id', 'auction_date_pdf', 'reserve_price_pdf', 
        'emd_amount_pdf', 'location_of_assets', 'auction_notice_url',
        'days_until_submission', 'emd_percent_category'
    ]
    return [col for col in possible_cols if col in dataframe.columns]

def get_auction_insights(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Get AI insights from backend with proper error handling."""
    try:
        response = requests.post(
            urljoin(BACKEND_BASE_URL, AUCTION_INSIGHTS_ENDPOINT),
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Backend connection error: {str(e)}")
        return None
    except json.JSONDecodeError:
        st.error("Invalid response format from backend")
        return None

def display_insights(insights: dict):
    """Display the insights in a structured format."""
    st.success(" Insights generated successfully!")
    
    # Main summary section
    with st.expander("Auction Summary", expanded=True):
        if 'corporate_debtor' in insights:
            st.markdown(f"**Corporate Debtor:** {insights['corporate_debtor']}")
        
        if 'assets' in insights:
            st.subheader("Assets Information")
            col1, col2 = st.columns(2)
            with col1:
                if 'description' in insights['assets']:
                    st.markdown(f"**Description:** {insights['assets']['description']}")
            with col2:
                if 'reserve_price' in insights['assets']:
                    st.markdown(f"**Reserve Price:** {insights['assets']['reserve_price']}")
    
    # Financial terms section
    if 'financial_terms' in insights:
        with st.expander("💰 Financial Terms", expanded=False):
            terms = insights['financial_terms']
            if 'emd' in terms:
                st.markdown(f"**EMD Amount:** {terms['emd']}")
            if 'bid_increments' in terms and terms['bid_increments']:
                st.markdown("**Bid Increments:**")
                for increment in terms['bid_increments']:
                    st.markdown(f"- {increment}")
    
    # Timeline section
    if 'timeline' in insights:
        with st.expander("⏰ Timeline", expanded=False):
            timeline = insights['timeline']
            if 'auction_date' in timeline:
                st.markdown(f"**Auction Date:** {timeline['auction_date']}")
            if 'inspection_period' in timeline:
                st.markdown(f"**Inspection Period:** {timeline['inspection_period']}")
    
    # Validation section
    if 'validation' in insights:
        with st.expander("🔍 Data Validation", expanded=False):
            validation = insights['validation']
            if 'debtor_match' in validation:
                st.markdown(f"**Debtor Match:** {'✅ Valid' if validation['debtor_match'] else '❌ Mismatch'}")
            if 'discrepancies' in validation and validation['discrepancies']:
                st.markdown("**Discrepancies Found:**")
                for field, details in validation['discrepancies'].items():
                    st.markdown(f"- {field}: Notice={details.get('notice_value', 'N/A')}, CSV={details.get('csv_value', 'N/A')}")

connection_status = check_backend_connection()
if connection_status:
    st.sidebar.markdown(
        '<div class="connection-status connection-success">✓ Backend Connected</div>',
        unsafe_allow_html=True
    )
else:
    st.sidebar.markdown(
        '<div class="connection-status connection-error">✗ Backend Disconnected</div>',
        unsafe_allow_html=True
    )
    st.sidebar.error(f"Ensure backend is running at: {BACKEND_BASE_URL}")

# Load data
df, latest_csv = load_auction_data()

# Dashboard Page
if page == "🏠 Dashboard" and df is not None:
    st.markdown('<div class="main-header">🏛️ Auction Portal India</div>', unsafe_allow_html=True)
    
    if 'days_until_submission' in df.columns:
        future_auctions = df[df['days_until_submission'] >= 0]
        st.metric("Upcoming Auctions", len(future_auctions))
        
        if not future_auctions.empty:
            st.dataframe(
                future_auctions[get_display_columns(future_auctions)],
                use_container_width=True
            )
        else:
            st.info("No upcoming auctions found.")
    else:
        st.warning("Could not calculate upcoming auctions - missing date information")

# Search Analytics Page
elif page == "🔍 Search Analytics" and df is not None:
    st.markdown('<div class="main-header">🔍 Search Analytics</div>', unsafe_allow_html=True)
    
    # Location filter (only show if column exists)
    if 'location_of_assets' in df.columns:
        unique_locations = df['location_of_assets'].dropna().unique()
        location_filter = st.multiselect(
            "Filter by Location",
            options=unique_locations
        )
    else:
        location_filter = []
        st.warning("Location information not available in this dataset")
    
    # Days until auction filter (only show if column exists)
    if 'days_until_submission' in df.columns:
        min_days = int(df['days_until_submission'].min())
        max_days = int(df['days_until_submission'].max())
        day_range = st.slider(
            "Days Until Auction",
            min_value=min_days,
            max_value=max_days,
            value=(0, max_days)
        )
    else:
        day_range = (0, 0)
        st.warning("Could not filter by days - missing date information")
    
    # Apply filters
    filtered_df = df.copy()
    if location_filter and 'location_of_assets' in df.columns:
        filtered_df = filtered_df[filtered_df['location_of_assets'].isin(location_filter)]
    if 'days_until_submission' in df.columns:
        filtered_df = filtered_df[
            (filtered_df['days_until_submission'] >= day_range[0]) & 
            (filtered_df['days_until_submission'] <= day_range[1])
        ]
    
    # Display results
    st.metric("Filtered Auctions", len(filtered_df))
    if not filtered_df.empty:
        st.dataframe(
            filtered_df[get_display_columns(filtered_df)],
            use_container_width=True
        )
    else:
        st.info("No auctions match your filters")

elif page == "🧠 AI Auction Insights":
    st.markdown('<div class="main-header">🧠 AI Auction Insights</div>', unsafe_allow_html=True)

    if not connection_status:
        st.error("Cannot generate insights - backend connection unavailable")
        st.stop()

    if df is None:
        st.error("No auction data loaded")
        st.stop()

    # Clean and standardize column names for safety
    df.columns = df.columns.str.strip().str.lower().str.replace(r"[^\w]+", "_", regex=True)

    # Use CIN/LLPIN column as Auction ID selector
    if 'cin_llpin' not in df.columns:
        st.error("Column 'CIN/LLPIN' (cin_llpin) not found in the uploaded data.")
        st.stop()

    auction_ids = df['cin_llpin'].dropna().unique()
    selected_id = st.selectbox("Select Auction ID (from CIN/LLPIN)", options=[""] + list(auction_ids))

    if selected_id:
        selected_row = df[df['cin_llpin'] == selected_id]
        if selected_row.empty:
            st.warning("Selected Auction ID not found in the data.")
            st.stop()

        auction_data = selected_row.iloc[0].to_dict()

        st.write("Auction data keys:", auction_data.keys())


        # Use the actual cleaned column names
        corporate_debtor = auction_data.get('name_of_corporate_debtor_pdf_', '')
        auction_notice_url = auction_data.get('auction_notice_url', '')

        
        st.write("Data sent to backend:", {
            "corporate_debtor": corporate_debtor,
            "auction_notice_url": auction_notice_url
        })


        if not corporate_debtor or not auction_notice_url:
            st.warning("Corporate Debtor name or Auction Notice URL missing for selected Auction ID.")
            st.stop()

        if st.button("Generate Insights"):
            payload = {
                "Name of Corporate Debtor (PDF)": str(corporate_debtor),
                "Auction Notice URL": str(auction_notice_url),
                "Reserve Price (PDF)": str(auction_data.get('reserve_price_pdf', '')),
                "EMD Amount (PDF)": str(auction_data.get('emd_amount_pdf', '')),
                "Date of Auction (PDF)": pd.to_datetime(auction_data.get('auction_date_pdf')).strftime("%Y-%m-%d") if pd.notna(auction_data.get('auction_date_pdf')) else "",
                "Name of IP (PDF)": str(auction_data.get('name_of_ip_pdf', '')),
                "IP Registration Number": str(auction_data.get('ip_registration_number', '')),
                "Unique Number": str(auction_data.get('unique_number', '')),
                "Auction Platform": str(auction_data.get('auction_platform', '')),
                "Details URL": str(auction_data.get('details_url', '')),
                "CIN/LLPIN": str(auction_data.get('cin_llpin', ''))
            }

            payload = {k: v for k, v in payload.items() if v}

            
            st.write("Final payload being sent to backend:", payload)

            with st.spinner("Generating insights..."):
                insights = get_auction_insights(payload)

                st.write("Backend Response:", insights)


            if insights and "insights" in insights:
                    insight_data = insights["insights"]
                    if isinstance(insight_data, dict):
                        display_insights(insight_data) 
                    else:
                        st.markdown(insight_data) 
            else:
                st.error("Could not fetch insights from backend")

