import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import requests
import io
from docx import Document
from PyPDF2 import PdfReader
import datetime as dt
import os
from dotenv import load_dotenv
import glob
import time
from typing import Dict, Any, Optional, List
import json
from urllib.parse import urljoin



# Try optional AI deps (app continues even if missing)
try:
    from langchain_core.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain_huggingface import HuggingFaceEndpoint
    HAS_LANGCHAIN = True
except Exception:
    HAS_LANGCHAIN = False

BACKEND_BASE_URL = os.getenv("BANK_AUCTION_INSIGHTS_API_URL", "http://localhost:8000")
AUCTION_INSIGHTS_ENDPOINT = "/auction-insights"

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
    





# Load environment variables
load_dotenv()
HF_API_KEY = (os.getenv("HF_API_KEY") or "").strip()

st.set_page_config(
    page_title="Auction Portal India",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
        .analytics-header {
            font-size: 1.8rem;
            color: #2e7d32;
            margin-bottom: 1rem;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("🏛️ Auction Portal")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate to:",
    ["🏠 Dashboard", "🔍 Search Analytics", "📊 Basic Analytics", "🤖 AI Analysis"],
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

#####################################################################################################################################################################
########################################################################################################################################################################
# Load data
@st.cache_data
def load_auction_data():
    csv_path = r"C:\Users\Amit Sharma\ai-platform\frontend\auction_exports\combined_auctions_20250819_154419.csv"
    try:
        # Get list of CSV files
        csv_files = glob.glob("auction_exports/combined_auctions_*.csv")
        
        if not csv_files:
            st.error("❌ No CSV files found in auction_exports folder.")
            return None, None
        
        # Pick the latest file by modification time
        latest_file = max(csv_files, key=os.path.getmtime)
        df = pd.read_csv(latest_file)
        st.success(f"✅ Loaded data from {latest_file} with {len(df)} records.")
        #st.write(df.dtypes )
        
        

        
        # Rename columns for clarity
        df = df.rename(columns={
            'Auction ID/CIN/LLPIN': 'Auction ID',
            'Bank/Organisation Name': 'Bank',
            'Location-City/District/address': 'Location',
            '_Auction date': 'Auction Date',
            '_Last Date of EMD Submission': 'EMD Submission Date',
            '_Reserve Price': '₹Reserve Price',
            'EMD Amount': '₹EMD Amount',
            'Nature of Assets': 'Nature of Assets',
            'Details URL': 'Details URL',
            'Auction Notice URL': 'Notice URL',
            'Source': 'Source'
        })
        # Convert date columns to datetime64[ns] and create duplicate columns for filtering
        df['EMD Submission Date_dt'] = pd.to_datetime(df['EMD Submission Date'], format="%d-%m-%Y", errors='coerce')
        df['Auction Date_dt'] = pd.to_datetime(df['Auction Date'], format="%d-%m-%Y", errors='coerce')

        # Convert date columns to datetime64[ns] and format as strings for display
        df['EMD Submission Date'] = pd.to_datetime(df['EMD Submission Date'], format="%d-%m-%Y", errors='coerce')
        df['Auction Date'] = pd.to_datetime(df['Auction Date'], format="%d-%m-%Y", errors='coerce')

        # Convert to string format to avoid Arrow conversion issues (only date part)
        df['EMD Submission Date'] = df['EMD Submission Date'].dt.strftime('%d-%m-%Y')
        df['Auction Date'] = df['Auction Date'].dt.strftime('%d-%m-%Y')

        # Use tz-naive date for "today" (as datetime object for consistency in calculations)
        today_date = pd.Timestamp.now(tz=None).date()

        # Calculate days_until_submission safely
        if 'days_until_submission' not in df.columns:
            df['days_until_submission'] = df['EMD Submission Date'].apply(
                lambda x: (pd.to_datetime(x).date() - today_date).days if pd.notna(x) and x != '' else -999
            )
        # Clean numeric columns
        df['₹Reserve Price'] = pd.to_numeric(df['₹Reserve Price'].astype(str).str.replace(r'[,₹\s]', '', regex=True), errors='coerce')
        df['₹EMD Amount'] = pd.to_numeric(df['₹EMD Amount'].astype(str).str.replace(r'[,₹\s]', '', regex=True), errors='coerce')

        # Calculate EMD % and categorize
        # Calculate EMD %
        df['EMD %'] = (df['₹EMD Amount'] / df['₹Reserve Price'] * 100).round(2)

        # Define bins and labels
        bins = [-float("inf"), 5, 10, 15, 20, float("inf")]
        labels = ["<5%", "5-10%", "10-15%", "15-20%", ">20%"]

        # Categorize into bins
        df['EMD % Category'] = pd.cut(df['EMD %'], bins=bins, labels=labels, right=False)


        if df['EMD Submission Date'].isna().any():
            st.warning("⚠️ Some EMD Submission Dates could not be parsed and are set to NaT. These rows may have invalid data.")

        return df, csv_path
    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")
        return None, None






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


#####################################################################################################################################################################
########################################################################################################################################################################







# Load data
df, latest_csv = load_auction_data()

# Dashboard Page
if page == "🏠 Dashboard" and df is not None:
    st.markdown('<div class="main-header">🏛️ Auction Portal India</div>', unsafe_allow_html=True)
    st.markdown(f"**Last Updated:** {latest_csv.split('_')[-1].split('.')[0] if latest_csv else 'Unknown'}")
    

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_auctions = len(df)
        st.metric("Total Auctions", total_auctions)
    
    with col2:
        invalid_count = df['EMD Submission Date'].isna().sum()
        st.metric("Invalid EMD Dates", invalid_count)
    
    with col3:
        active_auctions = len(df[df['days_until_submission'] >= 0])
        st.metric("Active Auctions", active_auctions)
    
    from babel.numbers import format_currency
    
    with col4:
        avg_reserve = df[df['days_until_submission'] >= 0]['₹Reserve Price'].mean()
        if not pd.isna(avg_reserve):
            formatted_value = format_currency(avg_reserve, "INR", locale="en_IN")
        else:
            formatted_value = "N/A"

    st.metric("Avg Reserve Price of active functions", formatted_value)
    # Display filtered data
    filtered_df = df[df['days_until_submission'] >= 0]
    if not filtered_df.empty:
        st.dataframe(filtered_df[['Auction ID', 'Bank', 'Location', 'Auction Date', 'EMD Submission Date',
                                 '₹Reserve Price', '₹EMD Amount', 'EMD %', 'EMD % Category', 'Nature of Assets'
                                 ,   'days_until_submission']],
                     use_container_width=True)
        st.write(f"**Total Auctions (Today or Future):** {len(filtered_df)}")
 
    else:
        st.info("✅ No auctions found for day or future dates.")
        








#####################################################################################################################################################################
########################################################################################################################################################################








# Search Analytics Page
elif page == "🔍 Search Analytics" and df is not None:
    st.markdown('<div class="main-header">🔍 Search Analytics</div>', unsafe_allow_html=True)
    st.markdown(f"**Last Updated:** {latest_csv.split('_')[-1].split('.')[0] if latest_csv else 'Unknown'}")
    

    invalid_count = df['EMD Submission Date'].isna().sum()
    st.markdown(f"""
        <div class="metric-tile">
            <h3>{invalid_count}</h3>
            <p>Invalid EMD Submission Dates</p>
        </div>
    """, unsafe_allow_html=True)

    filtered_df = df[df['days_until_submission'] >= 0].copy()
    

    # Location Filter
    use_location_filter = st.checkbox("Use Location Filter", value=False)
    if use_location_filter:
        unique_locations = sorted(filtered_df['Location'].dropna().unique())
        selected_locations = st.multiselect(
            "Select Locations",
            options=unique_locations,
            default=None
        )
        if selected_locations:
            filtered_df = filtered_df[filtered_df['Location'].isin(selected_locations)]

    # Range Slider for days_until_submission
    use_days_filter = st.checkbox("Use Days Until Submission Filter", value=False)
    if use_days_filter and not filtered_df.empty:
        min_days = int(filtered_df['days_until_submission'].min())
        max_days = int(filtered_df['days_until_submission'].max())
        days_range = st.slider(
            "Filter by Days Until Submission",
            min_value=min_days,
            max_value=max_days,
            value=(min_days, max_days)
        )
        filtered_df = filtered_df[
            (filtered_df['days_until_submission'] >= days_range[0]) &
            (filtered_df['days_until_submission'] <= days_range[1])
        ]

    # Checkbox and Date Input for EMD Submission Date
    use_date_filter = st.checkbox("Use EMD Submission Date Filter", value=False)
    if use_date_filter:
        selected_date = st.date_input("Select EMD Submission Date", value=pd.Timestamp.now(tz=None).date(), disabled=not use_date_filter)
        filtered_df = filtered_df[filtered_df['EMD Submission Date_dt'].dt.date == selected_date]

   # EMD % Filter
    use_emd_percent_filter = st.checkbox("Use EMD % Filter", value=False)
    if use_emd_percent_filter:
        emd_options = ["<5%", "5-10%", "10-15%", "15-20%", ">20%"]
        selected_emd = st.multiselect(
            "Select EMD % Category",
            options=emd_options,
            default=None
        )
        if selected_emd:
            mask = filtered_df['EMD % Category'].str.contains('|'.join(selected_emd), na=False).fillna(False)
            filtered_df = filtered_df[mask]

    # Drop rows with any NaN values across all columns
    filtered_df = filtered_df.dropna()

    if not filtered_df.empty:
        st.dataframe(filtered_df[['Auction ID', 'Bank', 'Location', 'Auction Date', 'EMD Submission Date',
                                 '₹Reserve Price', '₹EMD Amount', 'EMD %', 'EMD % Category', 'Nature of Assets'
                                 , 'days_until_submission']],
                     use_container_width=True)
        st.write(f"**Total Auctions:** {len(filtered_df)}")
    else:
        st.info("✅ No auctions found with the selected filters.")










#####################################################################################################################################################################
########################################################################################################################################################################



# Basic Analytics Page
elif page == "📊 Basic Analytics" and df is not None:
    st.markdown('<div class="main-header">📊 Basic Analytics</div>', unsafe_allow_html=True)
    
    # Inject custom CSS for improved metric tiles
    st.markdown("""
        <style>
            .metric-grid {
                display: flex;
                flex-wrap: wrap;
                gap: 15px;
                padding: 15px;
            }
            .metric-tile {
                background-color: #ffffff;
                border-radius: 10px;
                padding: 15px;
                text-align: center;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                transition: transform 0.2s, box-shadow 0.2s;
                border: 1px solid #e0e0e0;
                flex: 1;
                min-width: 200px;
            }
            .metric-tile:hover {
                transform: translateY(-5px);
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
            }
            .metric-tile h3 {
                font-size: 1.5em;
                margin: 0;
                color: #1a73e8;
                font-weight: bold;
            }
            .metric-tile p {
                font-size: 0.9em;
                margin: 5px 0 0 0;
                color: #5f6368;
                font-weight: 500;
            }
        </style>
    """, unsafe_allow_html=True)

    # Row 1: Total Auctions and Active Auctions
    col1_row1, col2_row1 = st.columns(2)
    with col1_row1:
        st.markdown("""
            <div class="metric-tile">
                <h3>{}</h3>
                <p>Total Auctions</p>
            </div>
        """.format(len(df)), unsafe_allow_html=True)
    with col2_row1:
        active_auctions = len(df[df['days_until_submission'] >= 0])
        st.markdown("""
            <div class="metric-tile">
                <h3>{}</h3>
                <p>Active Auctions</p>
            </div>
        """.format(active_auctions), unsafe_allow_html=True)

    # Row 2: Avg Reserve Price (All) and Avg Reserve Price of Active Auctions
    col1_row2, col2_row2 = st.columns(2)
    with col1_row2:
        from babel.numbers import format_currency
        avg_reserve_all = int(df['₹Reserve Price'].mean()) if not pd.isna(df['₹Reserve Price'].mean()) else 0
        formatted_value_all = format_currency(avg_reserve_all, "INR", locale="en_IN")
        st.markdown("""
            <div class="metric-tile">
                <h3>{}</h3>
                <p>Avg Reserve Price (All)</p>
            </div>
        """.format(formatted_value_all), unsafe_allow_html=True)
    with col2_row2:
        avg_reserve_active = int(df[df['days_until_submission'] >= 0]['₹Reserve Price'].mean()) if not pd.isna(df[df['days_until_submission'] >= 0]['₹Reserve Price'].mean()) else 0
        formatted_value_active = format_currency(avg_reserve_active, "INR", locale="en_IN")
        st.markdown("""
            <div class="metric-tile">
                <h3>{}</h3>
                <p>Avg Reserve Price of Active Auctions</p>
            </div>
        """.format(formatted_value_active), unsafe_allow_html=True)

    # Row 3: Sum of Reserve Price (All) and Sum of Reserve Price of Active Auctions
    col1_row3, col2_row3 = st.columns(2)
    with col1_row3:
        sum_reserve_all = int(df['₹Reserve Price'].sum()) if not pd.isna(df['₹Reserve Price'].sum()) else 0
        formatted_value_sum_all = format_currency(sum_reserve_all, "INR", locale="en_IN")
        st.markdown("""
            <div class="metric-tile">
                <h3>{}</h3>
                <p>Sum of Reserve Price (All)</p>
            </div>
        """.format(formatted_value_sum_all), unsafe_allow_html=True)
    with col2_row3:
        sum_reserve_active = int(df[df['days_until_submission'] >= 0]['₹Reserve Price'].sum()) if not pd.isna(df[df['days_until_submission'] >= 0]['₹Reserve Price'].sum()) else 0
        formatted_value_sum_active = format_currency(sum_reserve_active, "INR", locale="en_IN")
        st.markdown("""
            <div class="metric-tile">
                <h3>{}</h3>
                <p>Sum of Reserve Price of Active Auctions</p>
            </div>
        """.format(formatted_value_sum_active), unsafe_allow_html=True)

       # Row 4: Total Banks and Top 5 Banks
    col1_row4, col2_row4 = st.columns(2)  # Changed to 2 columns to accommodate both cards
    with col1_row4:
        total_banks = df['Bank'].nunique()
        st.markdown("""
            <div class="metric-tile">
                <h3>{}</h3>
                <p>Total Banks</p>
            </div>
        """.format(total_banks), unsafe_allow_html=True)
    with col2_row4:
        # Calculate top 5 banks by count
        top_banks = df['Bank'].value_counts().head(5).to_dict()
        bank_list = "<ul>" + "".join([f"<li>{bank}: {count}</li>" for bank, count in top_banks.items()]) + "</ul>"
        st.markdown("""
            <div class="metric-tile">
                <h3>Top 5 Banks</h3>
                <p>{}</p>
            </div>
        """.format(bank_list), unsafe_allow_html=True)

    # Row 5: Min and Max of Reserve Price of Active Auctions
    col1_row5, col2_row5 = st.columns(2)
    with col1_row5:
        min_reserve_active = int(df[df['days_until_submission'] >= 0]['₹Reserve Price'].min()) if not pd.isna(df[df['days_until_submission'] >= 0]['₹Reserve Price'].min()) else 0
        formatted_min_active = format_currency(min_reserve_active, "INR", locale="en_IN")
        st.markdown("""
            <div class="metric-tile">
                <h3>{}</h3>
                <p>Min of Reserve Price of Active Auctions</p>
            </div>
        """.format(formatted_min_active), unsafe_allow_html=True)
    with col2_row5:
        max_reserve_active = int(df[df['days_until_submission'] >= 0]['₹Reserve Price'].max()) if not pd.isna(df[df['days_until_submission'] >= 0]['₹Reserve Price'].max()) else 0
        formatted_max_active = format_currency(max_reserve_active, "INR", locale="en_IN")
        st.markdown("""
            <div class="metric-tile">
                <h3>{}</h3>
                <p>Max of Reserve Price of Active Auctions</p>
            </div>
        """.format(formatted_max_active), unsafe_allow_html=True)

    st.markdown("---")
    # Chart 1: Top 10 Banks by Auction Count
    st.subheader("📈 Top 10 Banks by Auction Count")
    bank_counts = df['Bank'].value_counts().head(10)
    fig1 = px.bar(
        x=bank_counts.values,
        y=bank_counts.index,
        orientation='h',
        title="Top 10 Banks by Auction Count",
        labels={'x': 'Number of Auctions', 'y': 'Bank'},
        color=bank_counts.values,
        color_continuous_scale='viridis'
    )
    fig1.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

    # Chart 2: Average Reserve Price by Location (Top 10)
    st.subheader("💰 Top 10 Locations by Average Reserve Price")
    location_avg = df.groupby('Location')['₹Reserve Price'].mean().sort_values(ascending=False).head(10)
    fig2 = px.bar(
        x=location_avg.values,
        y=location_avg.index,
        orientation='h',
        title="Top 10 Locations by Average Reserve Price",
        labels={'x': 'Average Reserve Price (₹)', 'y': 'Location'},
        color=location_avg.values,
        color_continuous_scale='plasma'
    )
    fig2.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    # Chart 3: EMD Percentage Distribution
    st.subheader("📊 EMD Percentage Distribution")
    emd_dist = df['EMD %'].dropna()
    fig3 = px.histogram(
        emd_dist,
        title="Distribution of EMD Percentages",
        labels={'value': 'EMD %', 'count': 'Frequency'},
        nbins=50,
        color_discrete_sequence=['#ff6b6b']
    )
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)

    # Chart 4: Auctions Over Time
    st.subheader("📅 Auction Trends Over Time")
    if not df['Auction Date_dt'].isna().all():
        df_time = df.dropna(subset=['Auction Date_dt']).copy()
        df_time['Month'] = df_time['Auction Date_dt'].dt.to_period('M').dt.to_timestamp()
        monthly_auctions = df_time.groupby('Month').size().reset_index(name='Count')
        
        fig4 = px.line(
            monthly_auctions,
            x='Month',
            y='Count',
            title="Number of Auctions per Month",
            labels={'Count': 'Number of Auctions', 'Month': 'Month'}
        )
        fig4.update_traces(line_color='#2e7d32', line_width=3)
        fig4.update_layout(height=400)
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("No valid auction dates available for time trend analysis.")

    # Chart 5: Reserve Price vs EMD Amount Scatter
    st.subheader("💸 Reserve Price vs EMD Amount")
    scatter_data = df[df['days_until_submission'] >= 0].dropna(subset=['₹Reserve Price', '₹EMD Amount'])
    if not scatter_data.empty:
        fig5 = px.scatter(
            scatter_data,
            x='₹Reserve Price',
            y='₹EMD Amount',
            title="Reserve Price vs EMD Amount",
            labels={'Reserve Price': 'Reserve Price (₹)', 'EMD Amount': 'EMD Amount (₹)'},
            opacity=0.6,
            color='EMD %',
            color_continuous_scale='viridis'
        )
        fig5.update_layout(height=500)
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info("No valid price data available for scatter plot.")





#####################################################################################################################################################################
########################################################################################################################################################################      








# AI Analysis Page
elif page == "🤖 AI Analysis":
    st.markdown('<div class="main-header">🤖 AI Analysis</div>', unsafe_allow_html=True)

    if not connection_status:
        st.error("Cannot generate insights - backend connection unavailable")
        st.stop()

    if df is None:
        st.error("No auction data loaded")
        st.stop()

    # Clean and standardize column names for safety
    df.columns = df.columns.str.strip().str.lower().str.replace(r"[^\w]+", "_", regex=True)

    # Use CIN/LLPIN column as Auction ID selector
    if 'auction_id' not in df.columns:
        st.error("Column 'Auction ID' (auction_id) not found in the uploaded data.")
        st.stop()

    auction_ids = df['auction_id'].dropna().unique()
    selected_id = st.selectbox("Select Auction ID (from CIN/LLPIN)", options=[""] + list(auction_ids))

    if selected_id:
        selected_row = df[df['auction_id'] == selected_id]
        if selected_row.empty:
            st.warning("Selected Auction ID not found in the data.")
            st.stop()

        auction_data = selected_row.iloc[0].to_dict()

        st.write("Auction data keys:", auction_data.keys())


        # Use the actual cleaned column names
        corporate_debtor = auction_data.get('bank', '')
        auction_notice_url = auction_data.get('notice_url', '')

        
        st.write("Data sent to backend:", {
            "corporate_debtor": corporate_debtor,
            "auction_notice_url": auction_notice_url
        })


        if not corporate_debtor or not auction_notice_url:
            st.warning("Corporate Debtor name or Auction Notice URL missing for selected Auction ID.")
            st.stop()

        if st.button("Generate Insights"):
            payload = {
                "Name of Corporate Debtor (PDF)": str(auction_data.get('bank', '')),
                "Auction Notice URL": str(auction_data.get('notice_url', '')),
                "Reserve Price (PDF)": str(auction_data.get('reserve_price', '')),
                "EMD Amount (PDF)": str(auction_data.get('emd_amount', '')),
                "Date of Auction (PDF)": pd.to_datetime(auction_data.get('auction_date')).strftime("%Y-%m-%d") if pd.notna(auction_data.get('auction_date')) else "",
                "Name of IP (PDF)": "",
                "IP Registration Number": "",
                "Unique Number": "",
                "Auction Platform": "",
                "Details URL": str(auction_data.get('details_url', '')),
               "CIN/LLPIN": str(auction_data.get('auction_id', ''))
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



