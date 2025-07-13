import pandas as pd
import glob
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_and_combine():
    """Combine the latest data from albion_combined_t2.py and ibbi_gov.py."""
    combined_data = []
    output_dir = "auction_exports"
    os.makedirs(output_dir, exist_ok=True)

    # Standard columns for combined output
    standard_columns = [
        "Auction ID", "Bank/Organisation Name", "City/District/Location",
        "Auction Date", "Reserve Price", "Source"
    ]

    # Find latest files
    albion_files = glob.glob(f"{output_dir}/albion_master_*.csv")
    ibbi_files = glob.glob(f"{output_dir}/ibbi_auctions_enriched_*.csv")

    # Process Albion data
    if albion_files:
        try:
            latest_albion = max(albion_files, key=os.path.getctime)
            albion_df = pd.read_csv(latest_albion)
            logger.info(f"Loaded Albion data: {latest_albion} with {len(albion_df)} records")

            # Map columns
            albion_df = albion_df.rename(columns={
                "Auction ID": "Auction ID",
                "Bank Name": "Bank/Organisation Name",
                "Location": "City/District/Location",
                "Auction Date": "Auction Date",
                "Reserve Price": "Reserve Price"
            })
            albion_df["Source"] = "Albion"
            # Select standard columns, fill missing with "-"
            albion_df = albion_df.reindex(columns=standard_columns, fill_value="-")
            combined_data.append(albion_df)
        except Exception as e:
            logger.error(f"Failed to process Albion data: {e}")

    # Process IBBI data
    if ibbi_files:
        try:
            latest_ibbi = max(ibbi_files, key=os.path.getctime)
            ibbi_df = pd.read_csv(latest_ibbi)
            logger.info(f"Loaded IBBI data: {latest_ibbi} with {len(ibbi_df)} records")

            # Map columns
            ibbi_df = ibbi_df.rename(columns={
                "CIN/LLPIN": "Auction ID",
                "Name of Corporate Debtor (Table)": "Bank/Organisation Name",
                "Location of Assets": "City/District/Location",
                "Date of Auction (Table)": "Auction Date",
                "Reserve Price (Table)": "Reserve Price"
            })
            ibbi_df["Source"] = "IBBI"
            # Select standard columns, fill missing with "-"
            ibbi_df = ibbi_df.reindex(columns=standard_columns, fill_value="-")
            combined_data.append(ibbi_df)
        except Exception as e:
            logger.error(f"Failed to process IBBI data: {e}")

    # Combine and save
    if combined_data:
        final_df = pd.concat(combined_data, ignore_index=True)
        today_str = datetime.now().strftime('%Y%m%d')
        output_file = f"{output_dir}/combined_auctions_{today_str}.csv"
        final_df.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"Combined data saved to: {output_file} with {len(final_df)} records")
        return output_file
    else:
        logger.error("No data to combine.")
        return None

if __name__ == "__main__":
    process_and_combine()