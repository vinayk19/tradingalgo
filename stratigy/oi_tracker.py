"""
This script tracks the Open Interest (OI) changes for NIFTY options
and displays them in a live-updating table.
"""
import os
import time
import logging
import sys
from datetime import datetime, timedelta
from collections import deque
import pandas as pd
import numpy as np
import beepy as bp
from kiteconnect import KiteConnect

# --- Configuration ---
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- User Configuration ---
# IMPORTANT: Fill in your API key and secret here
API_KEY = "YOUR_API_KEY"
API_SECRET = "YOUR_API_SECRET"

# Path to store the access token
TOKEN_PATH = "access_token.txt"

# --- Constants ---
NIFTY_INSTRUMENT = "NIFTY 50"
NIFTY_EXCHANGE = "NSE"
OPTIONS_EXCHANGE = "NFO"
STRIKE_DIFFERENCE = 50
NUM_STRIKES = 2  # Number of strikes to fetch above and below ATM (ATM-2, ATM-1, ATM, ATM+1, ATM+2)


def authenticate_kite():
    """Handles the Kite Connect authentication flow."""
    if API_KEY == "YOUR_API_KEY" or API_SECRET == "YOUR_API_SECRET":
        logging.error("API_KEY or API_SECRET is not set. Please update the configuration.")
        sys.exit(1)

    kite = KiteConnect(api_key=API_KEY)

    # Check if access token is already available
    if os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH, 'r') as f:
            access_token = f.read()
        try:
            kite.set_access_token(access_token)
            # Check if the session is still valid
            profile = kite.profile()
            logging.info(f"Successfully logged in as {profile['user_name']}.")
            return kite
        except Exception as e:
            logging.warning(f"Could not connect with existing access token: {e}. Re-authenticating.")
            # Clear the invalid token
            os.remove(TOKEN_PATH)

    # If no valid token, start the login flow
    logging.info("Starting new authentication process.")
    print("Please login to Kite and get the request_token.")
    print(f"Login URL: {kite.login_url()}")
    request_token = input("Enter the request_token from the redirect URL: ")

    try:
        data = kite.generate_session(request_token, api_secret=API_SECRET)
        access_token = data["access_token"]
        kite.set_access_token(access_token)

        # Save the access token for future use
        with open(TOKEN_PATH, 'w') as f:
            f.write(access_token)

        profile = kite.profile()
        logging.info(f"Authentication successful. Logged in as {profile['user_name']}.")
        return kite
    except Exception as e:
        logging.error(f"Authentication failed: {e}")
        sys.exit(1)


def get_instrument_df(kite):
    """Downloads and caches the instrument list."""
    instrument_dump_path = "instruments.csv"
    if os.path.exists(instrument_dump_path):
        logging.info("Loading instruments from local cache.")
        return pd.read_csv(instrument_dump_path)
    else:
        logging.info("Downloading instruments from Kite.")
        instrument_dump = kite.instruments(OPTIONS_EXCHANGE)
        instrument_df = pd.DataFrame(instrument_dump)
        instrument_df.to_csv(instrument_dump_path, index=False)
        return instrument_df


def get_nifty_options(instrument_df, nifty_ltp):
    """Finds the relevant NIFTY option contracts."""
    # Find ATM strike
    atm_strike = round(nifty_ltp / STRIKE_DIFFERENCE) * STRIKE_DIFFERENCE

    # Get all NIFTY options
    nifty_options = instrument_df[
        (instrument_df['name'] == 'NIFTY') &
        (instrument_df['segment'] == OPTIONS_EXCHANGE)
    ]

    # Find the nearest expiry date (weekly options)
    nifty_options.loc[:, 'expiry'] = pd.to_datetime(nifty_options['expiry']).dt.date
    today = datetime.now().date()
    # Find the next expiry date that is not today
    future_expiries = nifty_options[nifty_options['expiry'] > today]['expiry'].unique()
    if not future_expiries.any():
        logging.error("No future expiry dates found for NIFTY options.")
        return [], []

    nearest_expiry = min(future_expiries)

    # Get strikes around ATM
    strikes = [atm_strike + (i * STRIKE_DIFFERENCE) for i in range(-NUM_STRIKES, NUM_STRIKES + 1)]

    # Filter for the nearest expiry and the required strikes
    options = nifty_options[
        (nifty_options['expiry'] == nearest_expiry) &
        (nifty_options['strike'].isin(strikes))
    ]

    call_options = options[options['instrument_type'] == 'CE'].sort_values('strike')
    put_options = options[options['instrument_type'] == 'PE'].sort_values('strike')

    return call_options, put_options


# --- Data Storage ---
# Store (timestamp, value) tuples
MAX_LEN = 181 # 3 hours of data + 1 for buffer
nifty_history = deque(maxlen=MAX_LEN)
oi_history = {
    "CE": {i: deque(maxlen=MAX_LEN) for i in range(2 * NUM_STRIKES + 1)},
    "PE": {i: deque(maxlen=MAX_LEN) for i in range(2 * NUM_STRIKES + 1)}
}


# --- Calculation and Display ---
TIME_INTERVALS = [3, 5, 10, 15, 30, 180] # in minutes

def get_change(history, interval_mins):
    """Calculates change for a given time interval."""
    if len(history) < interval_mins + 1:
        return np.nan, np.nan # Not enough data

    now_ts, now_val = history[-1]
    past_ts_target = now_ts - timedelta(minutes=interval_mins)

    # Find the closest data point to the target time
    closest_past_point = min(history, key=lambda x: abs(x[0] - past_ts_target))
    past_val = closest_past_point[1]

    if past_val == 0: # Avoid division by zero
        return np.inf, now_val - past_val

    abs_change = now_val - past_val
    pct_change = (abs_change / past_val) * 100
    return pct_change, abs_change


def is_oi_change_significant(pct_change, interval):
    """Checks if the OI percentage change is significant for the given interval."""
    if pd.isna(pct_change):
        return False
    thresholds = {3: 10, 5: 12, 10: 15, 15: 30, 30: 30, 180: 100}
    return pct_change > thresholds.get(interval, np.inf)

def create_oi_table(option_df, option_type):
    """Creates the OI table for Calls or Puts and returns the data and styling info."""
    rows = []
    style_info = {} # (row_idx, col_name) -> is_significant
    red_count = 0

    option_df = option_df.reset_index(drop=True)

    for i, row in option_df.iterrows():
        strike = row['strike']
        history = oi_history[option_type][i]

        table_row = {'Strike': strike}

        if history:
            table_row['Current OI'] = int(history[-1][1])
            for interval in TIME_INTERVALS:
                pct, abso = get_change(history, interval)
                col_name = f'{interval} min'
                if pd.notna(pct):
                    table_row[col_name] = f"{pct:.2f}% ({int(abso)})"
                    if is_oi_change_significant(pct, interval):
                        red_count += 1
                        style_info[(i, col_name)] = True
                else:
                    table_row[col_name] = "N/A"
        else:
            table_row['Current OI'] = "N/A"
            for interval in TIME_INTERVALS:
                table_row[f'{interval} min'] = "N/A"

        rows.append(table_row)

    if not rows:
        return pd.DataFrame(), {}, 0

    df = pd.DataFrame(rows)
    return df, style_info, red_count


def create_nifty_table():
    """Creates and styles the NIFTY table."""
    if not nifty_history:
        return None

    current_price = nifty_history[-1][1]
    row = {'NIFTY Price': f"{current_price:.2f}"}

    for interval in TIME_INTERVALS:
        pct, abso = get_change(nifty_history, interval)
        if pd.notna(pct):
            row[f'{interval} min'] = f"{pct:.2f}% ({abso:.2f})"
        else:
            row[f'{interval} min'] = "N/A"

    return pd.DataFrame([row])


def check_alerts(call_reds, put_reds, call_df, put_df):
    """Checks if alert condition is met."""
    if not call_df.empty:
        call_total_cells = (len(call_df.index) * len(TIME_INTERVALS))
        if call_total_cells > 0 and (call_reds / call_total_cells) > 0.3:
            logging.warning("ALERT: Over 30% of cells in the CALL table are red!")
            bp.beep(sound="ping")

    if not put_df.empty:
        put_total_cells = (len(put_df.index) * len(TIME_INTERVALS))
        if put_total_cells > 0 and (put_reds / put_total_cells) > 0.3:
            logging.warning("ALERT: Over 30% of cells in the PUT table are red!")
            bp.beep(sound="ping")


def print_colored_df(df, style_info):
    """Prints a DataFrame with ANSI color codes for terminal output."""
    if df.empty:
        return

    RED_BG = "\033[41m"
    RESET = "\033[0m"

    # Calculate column widths
    col_widths = {col: max(df[col].astype(str).map(len).max(), len(col)) for col in df.columns}

    # Print header
    header = " | ".join(f"{col:<{col_widths[col]}}" for col in df.columns)
    print(header)
    print("-" * len(header))

    # Print rows
    for i, row in df.iterrows():
        row_str = []
        for col in df.columns:
            val = str(row[col])
            if style_info.get((i, col), False):
                row_str.append(f"{RED_BG}{val:<{col_widths[col]}}{RESET}")
            else:
                row_str.append(f"{val:<{col_widths[col]}}")
        print(" | ".join(row_str))


# --- Main Execution ---
def run_tracker(kite, instrument_df):
    """The main loop to run the OI tracker."""
    while True:
        try:
            # 1. Get NIFTY LTP
            nifty_quote = kite.quote(f"{NIFTY_EXCHANGE}:{NIFTY_INSTRUMENT}")
            nifty_ltp = nifty_quote[f"{NIFTY_EXCHANGE}:{NIFTY_INSTRUMENT}"]["last_price"]
            nifty_history.append((datetime.now(), nifty_ltp))

            # 2. Get relevant option contracts
            call_options, put_options = get_nifty_options(instrument_df, nifty_ltp)
            if call_options.empty or put_options.empty:
                logging.warning("Could not find call or put options. Waiting for next cycle.")
                time.sleep(60)
                continue

            # 3. Get quotes for options
            call_symbols = list(call_options['tradingsymbol'])
            put_symbols = list(put_options['tradingsymbol'])
            all_symbols = [f"{OPTIONS_EXCHANGE}:{s}" for s in call_symbols + put_symbols]
            quotes = kite.quote(all_symbols)

            # 4. Store OI data
            now = datetime.now()
            call_options_reset = call_options.reset_index(drop=True)
            put_options_reset = put_options.reset_index(drop=True)
            for i, symbol in enumerate(call_symbols):
                oi = quotes[f"{OPTIONS_EXCHANGE}:{symbol}"]["oi"]
                oi_history["CE"][i].append((now, oi))
            for i, symbol in enumerate(put_symbols):
                oi = quotes[f"{OPTIONS_EXCHANGE}:{symbol}"]["oi"]
                oi_history["PE"][i].append((now, oi))

            # 5. Create and display tables
            sys.stdout.write("\033[H\033[J") # Clear console

            print("--- CALL OPTIONS ---")
            call_df, call_style, call_reds = create_oi_table(call_options, "CE")
            print_colored_df(call_df, call_style)

            print("\n--- PUT OPTIONS ---")
            put_df, put_style, put_reds = create_oi_table(put_options, "PE")
            print_colored_df(put_df, put_style)

            print("\n--- NIFTY ---")
            nifty_table = create_nifty_table()
            if nifty_table is not None:
                print(nifty_table.to_string(index=False))

            # 6. Check for alerts
            check_alerts(call_reds, put_reds, call_df, put_df)

            # 7. Wait for the next minute
            logging.info("Update complete. Waiting for 60 seconds...")
            time.sleep(60)

        except Exception as e:
            logging.error(f"An error occurred in the main loop: {e}")
            time.sleep(60)


if __name__ == "__main__":
    # Authenticate and get Kite object
    kite_session = authenticate_kite()

    # Get instrument data
    instruments = get_instrument_df(kite_session)

    # Start the tracker
    run_tracker(kite_session, instruments)
