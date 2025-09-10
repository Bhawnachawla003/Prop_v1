import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import requests
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
import os
import fitz # PyMuPDF
import pytesseract
from PIL import Image, ImageOps
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import Optional, List, Tuple, Dict, Any
import logging
from io import BytesIO
import re
import json
from pydantic import BaseModel, Field
import pdfplumber
import camelot
from pdf2image import convert_from_bytes
import io
from bs4 import BeautifulSoup



# Try optional AI deps (app continues even if missing)
try:
    from langchain_core.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain_huggingface import HuggingFaceEndpoint
    HAS_LANGCHAIN = True
except Exception:
    HAS_LANGCHAIN = False

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

df, latest_csv = load_auction_data()

LANDING_API_URL = "https://api.va.landing.ai/v1/tools/agentic-document-analysis"
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
VA_API_KEY = st.secrets["VA_API_KEY"]

def display_insights(insights: dict):
    """Display auction insights in Streamlit directly, without expanders."""
    st.success("Insights generated successfully!")

    # Summary section
    st.markdown("### Auction Summary")
    st.markdown(f"**Corporate Debtor:** {insights.get('corporate_debtor', '')}")
    st.markdown(f"**Auction Date:** {insights.get('auction_date', '')}")
    st.markdown(f"**Auction Time:** {insights.get('auction_time', '')}")
    st.markdown(f"**Inspection Date:** {insights.get('inspection_date', '')}")
    st.markdown(f"**Inspection Time:** {insights.get('inspection_time', '')}")
    st.markdown(f"**Auction Platform:** {insights.get('auction_platform', '')}")
    st.markdown(f"**Contact Email:** {insights.get('contact_email', '')}")
    st.markdown(f"**Contact Mobile:** {insights.get('contact_mobile', '')}")

    # Assets
    assets = insights.get("assets", [])
    if assets:
        st.markdown("### Assets Information")
        for asset in assets:
            st.markdown(f"**Block Name:** {asset.get('block_name', '')}")
            st.markdown(f"**Description:** {asset.get('asset_description', '')}")
            st.markdown(f"**Reserve Price:** {asset.get('reserve_price', '')}")
            st.markdown(f"**EMD Amount:** {asset.get('emd_amount', '')}")
            st.markdown(f"**Incremental Bid Amount:** {asset.get('incremental_bid_amount', '')}")
            st.markdown("---")

    # Financial Terms
    financial = insights.get("financial_terms", {})
    if financial:
        st.markdown("### Financial Terms")
        st.markdown(f"**EMD Amount:** {financial.get('emd', '')}")
        bid_increments = financial.get("bid_increments", [])
        if bid_increments:
            st.markdown("**Bid Increments:**")
            for inc in bid_increments:
                st.markdown(f"- {inc}")

    # Timeline
    timeline = insights.get("timeline", {})
    if timeline:
        st.markdown("### Timeline")
        st.markdown(f"**Auction Date:** {timeline.get('auction_date', '')}")
        st.markdown(f"**Inspection Period:** {timeline.get('inspection_period', '')}")

    # Ranking
    ranking = insights.get("ranking", insights)
    if ranking:
        st.markdown("### Auction Ranking")
        st.markdown(f"**Legal Compliance Score:** {ranking.get('legal_compliance_score', 0)}")
        st.markdown(f"**Economical Score:** {ranking.get('economical_score', 0)}")
        st.markdown(f"**Market Trends Score:** {ranking.get('market_trends_score', 0)}")
        st.markdown(f"**Final Score:** {ranking.get('final_score', 0)}")
        st.markdown(f"**Risk Summary:** {ranking.get('risk_summary', '')}")
        references = ranking.get("reference_summary") or insights.get("referance summary")
        if references:
            st.markdown("**Reference Summary:**")
            if isinstance(references, str):
                st.markdown(references.replace("\n", " "))
            elif isinstance(references, list):
                for ref in references:
                    st.markdown(f"- {str(ref)}")

def normalize_keys(obj):
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            new_key = k.strip().lower().replace(" ", "_")
            new_obj[new_key] = normalize_keys(v)
        return new_obj
    elif isinstance(obj, list):
        return [normalize_keys(i) for i in obj]
    else:
        return obj

# Extract structured details from PDF using Landing.ai
def extract_pdf_details(pdf_url: str) -> dict:
    response = requests.get(pdf_url)
    response.raise_for_status()

    files = [("pdf", ("document.pdf", response.content, "application/pdf"))]

    schema = {
        "Corporate Debtor": "string",
        "Auction Date": "string",
        "Auction Time": "string",
        "Last Date for EMD Submission": "string",
        "Inspection Date": "string",
        "Inspection Time": "string",
        "Property Description": "string",
        "Auction Platform": "string",
        "Contact Email": "string",
        "Contact Mobile": "string",
        "Assets": [
            {
                "Block Name": "string",
                "Asset Description": "string",
                "Reserve Price": "string",
                "EMD Amount": "string",
                "Incremental Bid Amount": "string"
            }
        ]
    }

    payload = {"fields_schema": schema}
    headers = {"Authorization": f"Bearer {VA_API_KEY}", "Accept": "application/json"}

    r = requests.post(
        LANDING_API_URL,
        headers=headers,
        files=files,
        data={"payload": json.dumps(payload)}
    )
    r.raise_for_status()
    response_json = r.json()

    # Extract both structured schema & raw markdown
    raw_data = response_json.get("data", {})
    markdown = raw_data.get("markdown", "")
    chunks = raw_data.get("chunks", [])

    # st.subheader("Raw Extracted JSON")
    # st.json(raw_data)

    return {
        "structured": raw_data,
        "markdown": markdown,
        "chunks": chunks,
    }

def regex_preparser(markdown: str, chunks: list) -> dict:
    parsed = {}

    # Corporate Debtor
    debtor_match = re.search(
        r'([A-Z][A-Za-z0-9\s&().,-]*(?:LIMITED|LTD)(?:\s*\(.*?\))?)',
        markdown,
        re.IGNORECASE
    )
    if debtor_match:
        parsed["corporate_debtor"] = debtor_match.group(1).strip()

    # Auction Date 
    auction_date_match = re.search(
        r'(?:Date(?: and Time)? of)?\s*(?:E-?)?Auction[:\-\s]*?(\d{1,2}[./]\d{1,2}[./]\d{4}|\d{1,2}(st|nd|rd|th)?\s+\w+\s*,?\s*\d{4})',
        markdown,
        re.IGNORECASE
    )
    if auction_date_match:
        parsed["auction_date"] = auction_date_match.group(1).strip()
    else:
       fallback_date_match = re.search(
            r'(?:auction.*?)(\d{1,2}[./]\d{1,2}[./]\d{4}|\d{1,2}(st|nd|rd|th)?\s+\w+\s*,?\s*\d{4})',
            markdown,
            re.IGNORECASE
        )
       if fallback_date_match:
            parsed["auction_date"] = fallback_date_match.group(1).strip()

    # Auction Time 
    time_match = re.search(
        r'(Auction\s*Time[:\-]?\s*)?'
        r'((?:from\s*)?\d{1,2}(?::\d{2}|\.\d{2})?\s*(?:AM|PM|A\.M\.|P\.M\.)'
        r'(?:\s*(?:to|–|-)\s*'
        r'\d{1,2}(?::\d{2}|\.\d{2})?\s*(?:AM|PM|A\.M\.|P\.M\.))?)', 
        markdown,
        re.IGNORECASE
    )
    if time_match:
        time_text = time_match.group(2).strip()
        time_text = re.sub(r'(?i)\b(a\.m\.|p\.m\.)\b',
                           lambda m: m.group(1).replace('.', '').upper(),
                           time_text)
        parsed["auction_time"] = time_text

    # Emails
    emails = list(set(re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", markdown)))
    if emails:
        parsed["contact_email"] = emails

    # Phone numbers
    phone_match = re.search(
        r'(?:Ph[:\s]*|Phone[:\s]*|Contact[:\s]*|Mob[:\s]*)?(\b[6-9]\d{9}\b)',
        markdown
    )
    if phone_match:
        parsed["contact_mobile"] = phone_match.group(1)

    # Inspection Date
    insp_date_match = re.search(r'Inspection Date[:\-]?\s*(.*?)\n', markdown, re.IGNORECASE)
    if insp_date_match:
        parsed["inspection_date"] = insp_date_match.group(1).strip()

    # Inspection Time
    insp_time_match = re.search(r'Inspection Time[:\-]?\s*(.*?)\n', markdown, re.IGNORECASE)
    if insp_time_match:
        parsed["inspection_time"] = insp_time_match.group(1).strip()

    # Auction Platform
    platform_match = re.search(r'(E-Auction Platform.*?)\n', markdown, re.IGNORECASE)
    if platform_match:
        parsed["auction_platform"] = platform_match.group(1).strip()

    # Parse Assets 
    assets = []
    for c in chunks:
        if c.get("chunk_type") == "table":
            try:
                soup = BeautifulSoup(c["text"], "html.parser")
                rows = soup.find_all("tr")
                if not rows:
                    continue

                # headers
                headers = [h.get_text(" ", strip=True) for h in rows[0].find_all(["td", "th"])]

                for r in rows[1:]:
                    cols = [col.get_text(" ", strip=True) for col in r.find_all("td")]
                    if not cols:
                        continue
                    
                    # Create a temporary dictionary for the current row
                    row_data = {headers[i]: cols[i] for i in range(len(headers))}
                    
                    asset_entry = {}
                    
                    # Mapping logic based on column headers
                    for header, value in row_data.items():
                        lower_header = header.lower()
                        
                        # Extract and normalize the value
                        unit_match = re.search(r'\(([^)]+)\)', lower_header)
                        unit = unit_match.group(1) if unit_match else ""
                        value_with_unit = f"{value.strip()} {unit.strip()}" if unit else value.strip()
                        
                        if "lot" in lower_header or "block" in lower_header or "sr" in lower_header:
                            asset_entry["block_name"] = value_with_unit
                        elif "asset" in lower_header or "details" in lower_header or "description" in lower_header:
                            asset_entry["asset_description"] = value_with_unit
                        elif "reserve" in lower_header:
                            asset_entry["reserve_price"] = value_with_unit
                        elif "emd" in lower_header:
                            asset_entry["emd_amount"] = value_with_unit
                        elif "increment" in lower_header or "bid" in lower_header:
                            asset_entry["incremental_bid_amount"] = value_with_unit
                        elif "auction" in lower_header and "time" in lower_header:
                            asset_entry["auction_time"] = value_with_unit
                        elif "quantity" in lower_header:
                            asset_entry["quantity"] = value_with_unit
                        elif "location" in lower_header:
                            asset_entry["location"] = value_with_unit
                        else:
                            asset_entry[lower_header.replace(" ", "_")] = value_with_unit
                            
                    assets.append(asset_entry)

            except Exception as e:
                print(f"Error parsing table: {e}")
                continue

    if assets:
        parsed["assets"] = assets

    return parsed

# Risk scoring via Groq LLM# Risk scoring via Groq LLM
def generate_risk_insights(auction_json: dict, llm) -> dict:
    import json

    # Extract pre-parsed values 
    pre_parsed = {
        k: v for k, v in auction_json.items()
        if k not in ["markdown", "structured", "chunks"]
    }

    # Simplify structured JSON 
    structured = auction_json.get("structured", {})
    slim_structured = {
        k: v for k, v in structured.items()
        if isinstance(v, (str, int, float)) or (isinstance(v, list) and len(v) < 10)
    }

    prompt = f"""
You are an expert financial analyst specializing in Indian auction notices.

Here are the pre-parsed values (from regex):
{json.dumps(pre_parsed, indent=2)}

Here is a simplified version of structured JSON:
{json.dumps(slim_structured, indent=2)}

Analyze this information. Extract missing fields if possible, normalize,
and then apply the RISK SCORING FRAMEWORK.
Return JSON with:
- corporate_debtor
- auction_date
- auction_time (only if explicitly mentioned in the notice, otherwise leave null or omit)
- inspection_date
- inspection_time
- auction_platform
- contact_email
- contact_mobile
- assets (list with block_name, description, reserve_price, emd_amount, incremental_bid_amount)
* For the financial fields (reserve_price, emd_amount, incremental_bid_amount), **ensure the units (e.g., 'Lacs') are included** along with the numerical value.
* If incremental_bid_amount is not explicitly mentioned in the notice, return null or leave it blank
* If EMD amount is not explicitly mentioned in the notice, return null or leave it blank
# RISK SCORING FRAMEWORK (Use this internally for scoring)
## HIGH RISK (Block/Hold - Score 0-3)
Assign this risk level if ANY of the following Critical Defects are present. If a High Risk item is found, the Legal Compliance Score MUST be in the 0-3 range.
- **Statutory Defects:** Notice has legal defects (e.g., notice period shorter than mandated; missing authorized officer name/signature/seal).
- **Critical Mismatch:** Key details (e.g., property size/reserve price) differ significantly between the official notice PDF and the listing data.
- **Missing Core Docs:** Critical artifacts are missing (Sale Notice PDF, Valuation Report, Title documents).
- **Expired Valuation:** The valuation report date is older than 6-12 months (stale valuation).
- **Extreme Price Outlier:** Reserve price is an extreme outlier (e.g., > +50% or < -40% vs. comparable properties/norms).
- **Known Litigation:** Known, unresolved litigation (lis pendens, stay order) is disclosed.
- **Process Integrity:** Frequent re-schedules (>=3) without adequate cause, OR outcome anomalies (postings contradict terms).

## AVERAGE RISK (Warn Users - Score 4-7)
Assign this risk level if NO High Risk items are present, but ANY of the following Moderate Defects are found. The Legal Compliance Score MUST be in the 4-7 range.
- **Ambiguity:** Property description is ambiguous or inconsistent across documents.
- **Missing Minor Annexures:** Minor supporting documents (e.g., uncertified translations) are missing.
- **Low Quality:** Low photo count (<=3 .photos) or poor readability of scanned documents.
- **Mild Price Outlier:** Reserve price is a mild outlier (e.g., 10-25% out of band).
- **Short EMD Window:** Tight gap between notice and EMD close date.
- **Multiple Re-Auctions:** Listing mentions multiple prior re-auctions with ad-hoc reserve changes (no method cited).
- **Non-Standard Contact:** Personal emails/phones used in notices instead of official domains.

## LOW / NO RISK (Informational - Score 8-10)
Assign this risk level if NO High Risk or Average Risk items are found. The Legal Compliance Score MUST be in the 8-10 range.
- **Minor Typos:** Only minor typos/formatting errors that don't change legal meaning.
- **Normal Dynamics:** Events like last-minute bidding or re-auction due to reserve not met (1-2 cycles).

Rank the Auction using the provided **RISK SCORING FRAMEWORK** and the three components:
- Legal Compliance (Score 0-10, based on the Framework)
- Economical Point of View (Score 0-10, based on asset value and market context)
- Market Trends (Score 0-10, based on timing and location factors)

Provide:
- Individual scores for each component (0–10).
- A final score (simple average of the three components).
- A single-line summary of risk: "High Risk", "Average Risk", or "Low/No Risk" based on the highest risk category found.
- A **Reference Summary** that consists of **exactly 8 bullet points** in the JSON array. This summary must be a **detailed, evidence-based audit report** that uses plain, easy-to-understand language. **For every point, you MUST include the specific data/text from the notice that justifies the conclusion, and explicitly state the legal or market standard where applicable.**

1. **Primary Risk & Evidence:** State the assigned risk level and the single most critical issue found. **DO NOT copy text from other points.** (Example: 'AVERAGE RISK: The contact email "anilgoel@aaainsolvency.com" is non-standard.')
2. **Justification of Primary Risk:** Explain the risk. (Example: This email uses a non-institutional domain, raising a minor integrity concern over accountability.)
3. **Statutory Compliance Check:** Report on legal defects by **citing the legal standard and the full period**. (Example: Statutory defects were cleared. The 21-day notice period rule is met, as the period from [Notice Date] to [Auction Date] is compliant.)
4. **Authorization/Evidence Check:** Report on authorization and evidence by citing the document/reference. (Example: Valid authorization evidence is present, citing the NCLT order/Resolution reference from the text, ensuring the sale is legally sound.)
5. **Artifacts Check:** Report on critical and minor documents. (Example: All critical documents are present. Minor annexures (like uncertified translations) are missing, which is an Average Risk.)
6. **Valuation Check:** Report on price outlier and valuation currency. (Example: The Reserve Price of [Price] is acceptable based on market norms. The valuation report date of [Valuation Date] is current and not expired, meeting the 6-12 month policy window.)
7. **Process/Timeline Check:** Report on EMD window/re-auctions. (Example: The EMD window from [Notice Date] to [EMD Date] is adequate. No signs of multiple re-auctions or ad-hoc reserve changes were noted.)
8. **Listing Quality Warning:** Report on photos/description/ambiguity. (Example: Listing quality is low. Property photos and detailed descriptions require significant improvement for better transparency.)



OUTPUT INSTRUCTIONS (IMPORTANT):
Your entire response MUST be a single valid JSON object with this structure:

{{
  "corporate_debtor": "string",
  "auction_date": "string",
  "auction_time": "string or null",
  "inspection_date": "string or null",
  "inspection_time": "string or null",
  "auction_platform": "string",
  "contact_email": ["list of strings"],
  "contact_mobile": "string",
  "assets": [
    {{
      "block_name": "string",
      "asset_description": "string",
      "reserve_price": "string",
      "emd_amount": "string or null",
      "incremental_bid_amount": "string or null"
    }}
  ],
  "ranking": {{
    "legal_compliance_score": int,
    "economical_score": int,
    "market_trends_score": int,
    "final_score": int,
    "risk_summary": "string",
    "reference_summary": ["list of 8 strings"]
  }}
}}
Do not include any text outside of this JSON.
"""

    # Call LLM
    messages = [{"role": "user", "content": prompt}]
    resp = llm.invoke(messages, response_format={"type": "json_object"})
    raw_output = resp.content

    try:
        parsed = json.loads(raw_output)

        # Ensure auction_time is present 
        if not pre_parsed.get("auction_time") and "auction_time" not in parsed:
            parsed["auction_time"] = None

        # Enforce defaults for ranking
        parsed.setdefault("ranking", {})
        ranking = parsed["ranking"]
        ranking.setdefault("legal_compliance_score", 0)
        ranking.setdefault("economical_score", 0)
        ranking.setdefault("market_trends_score", 0)
        ranking.setdefault("final_score", 0)
        ranking.setdefault("risk_summary", "Not available")
        ranking.setdefault("reference_summary", [])

        return parsed

    except Exception:
        return {"error": "Invalid JSON", "raw": raw_output}


# integrate Landing.ai + Groq
def generate_auction_insights(corporate_debtor: str, auction_notice_url: str, llm, include_markdown: bool = False) -> dict:
    try:
        details = extract_pdf_details(auction_notice_url)

        # Run regex pre-parser on markdown + chunks
        pre_parsed = regex_preparser(details.get("markdown", ""), details.get("chunks", []))

        if corporate_debtor:
            pre_parsed["corporate_debtor"] = corporate_debtor.strip()

        # Merge regex-parsed fields into details
        merged = {**details, **pre_parsed}
        merged = normalize_keys(merged)

        if not include_markdown:
            merged.pop("markdown", None)
            merged.pop("chunks", None)

        insights = generate_risk_insights(merged, llm)
        insights = normalize_keys(insights)

        if corporate_debtor:
            insights["corporate_debtor"] = corporate_debtor.strip()

        merged.update(insights)
        return {"status": "success", "insights": merged}

    except Exception as e:
        return {"status": "error", "message": str(e)}

# 🤖 AI Analysis Page
if page == "🤖 AI Analysis":
    st.markdown('<div class="main-header">🤖 AI Analysis</div>', unsafe_allow_html=True)

    if df is None:
        st.error("No auction data loaded")
        st.stop()

    # Normalize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.strip("_")
    )

    # Check for required columns
    required_columns = ['auction_id', 'emd_submission_date', 'notice_url'] 
    if not all(col in df.columns for col in required_columns):
        st.error(f"Required columns not found in the data. Make sure your CSV contains: {', '.join(required_columns)}")
        st.stop()
    
    # convert the EMD submission date column to a datetime object
    df['emd_submission_date_dt'] = pd.to_datetime(df['emd_submission_date'], format='%d-%m-%Y', errors='coerce')

    
    # Filter out rows with 'URL 2_if available' in the notice_url column
    df_filtered = df[~df['notice_url'].str.contains('URL 2_if available', case=False, na=False)]
    
    # Filter to show only today's or future EMD dates
    df_filtered = df_filtered[df_filtered['emd_submission_date_dt'].notna()]
    today = datetime.today().date()
    df_filtered = df_filtered[df_filtered['emd_submission_date_dt'].dt.date >= today]
    
    # Use the Filtered DataFrame to populate the selectbox
    auction_ids = df_filtered['auction_id'].dropna().unique()
    selected_id = st.selectbox("Select Auction ID (from CIN/LLPIN)", options=[""] + list(auction_ids))
    
    if selected_id:
        # Use the Filtered DataFrame to get the selected row
        selected_row = df_filtered[df_filtered['auction_id'] == selected_id]
        if selected_row.empty:
            st.warning("Selected Auction ID not found in the filtered data.")
            st.stop()

        auction_data = selected_row.iloc[0].to_dict()
        corporate_debtor = auction_data.get('bank', '')
        auction_notice_url = auction_data.get('notice_url', '')

        if not corporate_debtor or not auction_notice_url:
            st.warning("Corporate Debtor name or Auction Notice URL missing for selected Auction ID.")
            st.stop()

        @st.cache_resource
        def initialize_llm():
            return ChatGroq(
                model="deepseek-r1-distill-llama-70b",
                temperature=0,
                api_key=GROQ_API_KEY,
            )

        llm = initialize_llm()

        if st.button("Generate Insights", use_container_width=True):
            if not llm:
                st.error("LLM failed to initialize. Check your GROQ_API_KEY secret.")
                st.stop()

            with st.spinner("Generating insights (Landing.ai + Groq)..."):
                insights_result = generate_auction_insights(corporate_debtor, auction_notice_url, llm)

                if insights_result["status"] == "success":
                    insight_data = insights_result["insights"]
                    if isinstance(insight_data, dict):
                        display_insights(insight_data)
                    else:
                        st.markdown(insight_data)
                else:
                    st.error("Analysis Failed")
                    st.exception(Exception(insights_result["message"]))
