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
    ["🏠 Dashboard", "🔍 Search Analytics", "📊 Basic Analytics","📈 KPI Analytics", "🤖 AI Analysis"],
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
            'Source': 'Source',
            'Notice_date': 'Notice_date'
        })
        # Convert date columns to datetime64[ns] and create duplicate columns for filtering
        df['EMD Submission Date_dt'] = pd.to_datetime(df['EMD Submission Date'], format="%d-%m-%Y", errors='coerce')
        df['Auction Date_dt'] = pd.to_datetime(df['Auction Date'], format="%d-%m-%Y", errors='coerce')
        df['Notice_date'] = pd.to_datetime(df['Notice_date'], format="%d/%m/%Y", errors='coerce')

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
    import pandas as pd

    def format_indian_currency(value):
        if pd.isna(value) or value <= 0:
            return "N/A"
        # Convert to lakhs or crores
        if value >= 10000000:  # 1 crore = 10 million
            formatted = value / 10000000
            return f"{formatted:.2f} cr"
        elif value >= 100000:  # 1 lakh = 100,000
            formatted = value / 100000
            return f"{formatted:.2f} lakhs"
        else:
            return f"{value:.2f}"

    with col4:
        avg_reserve = df[df['days_until_submission'] >= 0]['₹Reserve Price'].mean()
        formatted_value = format_indian_currency(avg_reserve)
        st.metric("Avg Reserve Price of active auctions", formatted_value)
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
        import textwrap
        def format_indian_currency(value):
            if pd.isna(value) or value <= 0:
                return "N/A"
            if value >= 10000000:  # 1 crore = 10 million
                formatted = value / 10000000
                return f"{formatted:.2f} cr"
            elif value >= 100000:  # 1 lakh = 100,000
                formatted = value / 100000
                return f"{formatted:.2f} lakhs"
            else:
                return f"{value:.2f}"
        
        avg_reserve_all = df['₹Reserve Price'].mean()
        formatted_value_all = format_indian_currency(avg_reserve_all)
        st.markdown("""
            <div class="metric-tile">
                <h3>{}</h3>
                <p>Avg Reserve Price (All)</p>
            </div>
        """.format(formatted_value_all), unsafe_allow_html=True)
    with col2_row2:
        avg_reserve_active = df[df['days_until_submission'] >= 0]['₹Reserve Price'].mean()
        formatted_value_active = format_indian_currency(avg_reserve_active)
        st.markdown("""
            <div class="metric-tile">
                <h3>{}</h3>
                <p>Avg Reserve Price of Active Auctions</p>
            </div>
        """.format(formatted_value_active), unsafe_allow_html=True)

    # Row 3: Sum of Reserve Price (All) and Sum of Reserve Price of Active Auctions
    col1_row3, col2_row3 = st.columns(2)
    with col1_row3:
        sum_reserve_all = df['₹Reserve Price'].sum()
        formatted_value_sum_all = format_indian_currency(sum_reserve_all)
        st.markdown("""
            <div class="metric-tile">
                <h3>{}</h3>
                <p>Sum of Reserve Price (All)</p>
            </div>
        """.format(formatted_value_sum_all), unsafe_allow_html=True)
    with col2_row3:
        sum_reserve_active = df[df['days_until_submission'] >= 0]['₹Reserve Price'].sum()
        formatted_value_sum_active = format_indian_currency(sum_reserve_active)
        st.markdown("""
            <div class="metric-tile">
                <h3>{}</h3>
                <p>Sum of Reserve Price of Active Auctions</p>
            </div>
        """.format(formatted_value_sum_active), unsafe_allow_html=True)

    # Row 5: Min and Max of Reserve Price of Active Auctions
    col1_row5, col2_row5 = st.columns(2)
    with col1_row5:
        min_reserve_active = df[df['days_until_submission'] >= 0]['₹Reserve Price']
        if not min_reserve_active.empty:
            min_reserve_active = min_reserve_active[min_reserve_active > 0].min() if (min_reserve_active > 0).any() else float('nan')
        else:
            min_reserve_active = float('nan')
        formatted_min_active = format_indian_currency(min_reserve_active)
        st.markdown("""
            <div class="metric-tile">
                <h3>{}</h3>
                <p>Min of Reserve Price of Active Auctions</p>
            </div>
        """.format(formatted_min_active), unsafe_allow_html=True)
    with col2_row5:
        max_reserve_active = df[df['days_until_submission'] >= 0]['₹Reserve Price'].max()
        formatted_max_active = format_indian_currency(max_reserve_active)
        st.markdown("""
            <div class="metric-tile">
                <h3>{}</h3>
                <p>Max of Reserve Price of Active Auctions</p>
            </div>
        """.format(formatted_max_active), unsafe_allow_html=True)

    st.markdown("---")

    # Top 5 Banks with Min and Max Reserve Price as a DataFrame
    top_banks = df['Bank'].value_counts().head(5).index
    active_df = df[df['days_until_submission'] >= 0]
    bank_stats = []
    for bank in top_banks:
        bank_data = active_df[active_df['Bank'] == bank]['₹Reserve Price']
        min_price = bank_data[bank_data > 0].min() if (bank_data > 0).any() else float('nan')
        max_price = bank_data[bank_data > 0].max() if (bank_data > 0).any() else float('nan')
        bank_stats.append({
            'Bank': bank,
            'Min Reserve Price': min_price,
            'Max Reserve Price': max_price
        })
    bank_df = pd.DataFrame(bank_stats)
    bank_df['Min Reserve Price'] = bank_df['Min Reserve Price'].apply(format_indian_currency)
    bank_df['Max Reserve Price'] = bank_df['Max Reserve Price'].apply(format_indian_currency)
    st.subheader("📈 Top 5 Banks by Reserve Price ")
    st.dataframe(bank_df)

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
        x=location_avg.apply(format_indian_currency),
        y=location_avg.index,
        orientation='h',
        title="Top 10 Locations by Average Reserve Price",
        labels={'x': 'Average Reserve Price', 'y': 'Location'},
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
            labels={'x': 'Reserve Price (₹)', 'y': 'EMD Amount (₹)'},
            opacity=0.6,
            color='EMD %',
            color_continuous_scale='viridis'
        )
        combined_text = scatter_data.apply(lambda row: f"{format_indian_currency(row['₹Reserve Price'])} / {format_indian_currency(row['₹EMD Amount'])}", axis=1)
        fig5.update_traces(text=combined_text, textposition='top center')
        fig5.update_layout(height=500)
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info("No valid price data available for scatter plot.")

#####################################################################################################################################################################
########################################################################################################################################################################      

# Sidebar Navigation (update the radio options)


# ... (existing code for other pages)

# KPI Analytics Page
elif page == "📈 KPI Analytics" and df is not None:
    st.markdown('<div class="main-header">📈 KPI Analytics</div>', unsafe_allow_html=True)
    
    # Filter for active auctions
    active_df = df[df['days_until_submission'] >= 0]
    active_df1=active_df[active_df["Source"]!="Albion"]
    
    if not active_df.empty:

        # notice compliance rate (proxy)
        total_auctions = len(active_df1)
        compliant_auctions = len(active_df1[active_df1['Notice URL'] != 'URL 2_if available'])
        notice_compliance_rate = (compliant_auctions / total_auctions * 100) if total_auctions > 0 else 0


        # Compute Disclosure Timeliness (proxy: auction_date - emd_submission_date)
        active_df1['timeliness_days'] = (active_df1['Auction Date_dt'] - active_df['Notice_date']).dt.days
        min_days = active_df1['timeliness_days'].min()
        median_days = active_df1['timeliness_days'].median()
        p95_days = active_df1['timeliness_days'].quantile(0.95)
        
        # Compute Data Quality Error Rate
        error_rate = (active_df.isna().any(axis=1).sum() / len(active_df)) * 100
        
        # Display in cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
                <div class="metric-tile">
                    <h3>Disclosure Timeliness (Days)</h3>
                    <p>Min: {min_days}, Median: {median_days}, P95: {p95_days}</p>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
                <div class="metric-tile">
                    <h3>Notice Compliance Rate (Proxy)</h3>
                    <p>{notice_compliance_rate:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
                <div class="metric-tile">
                    <h3>Data Quality Error Rate</h3>
                    <p>{error_rate:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.write(f"**Active Auctions Analyzed:** {len(active_df)}")
    else:
        st.info("No active auctions available for KPI calculation.")
    
    st.markdown("---")




#####################################################################################################################################################################
########################################################################################################################################################################  
