import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
import os

warnings.filterwarnings("ignore")

# --- Page Config ---
st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="✈️",
    layout="wide"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .boarding-pass {
        background: white;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        padding: 20px;
        border-left: 10px solid #2e7d32;
        margin-top: 20px;
        color: black;
    }
    .bp-header { font-weight: bold; color: #888; text-transform: uppercase; font-size: 12px; }
    .bp-city { font-size: 24px; font-weight: bold; color: #1f1f1f; }
    .bp-price { font-size: 32px; font-weight: 800; color: #2e7d32; }
    .bp-plane-icon { font-size: 24px; margin: 0 15px; color: #007bff; }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def parse_duration(duration_str):
    if pd.isna(duration_str) or str(duration_str).strip() == "": return 0
    hours = minutes = 0
    parts = str(duration_str).split()
    for part in parts:
        if 'h' in part:
            h_str = part.replace('h', '').strip()
            hours = int(h_str) if h_str.isdigit() else 0
        elif 'm' in part:
            m_str = part.replace('m', '').strip()
            minutes = int(m_str) if m_str.isdigit() else 0
    return hours * 60 + minutes

def stops_to_num(val):
    val_str = str(val).lower().strip()
    d = {'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}
    return d.get(val_str, 0)

def preprocess_data(df: pd.DataFrame, training_columns=None) -> pd.DataFrame:
    df = df.copy()
    df.dropna(inplace=True)
    
    # Date Handling
    df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'], dayfirst=True)
    df['Journey_Day'] = df['Date_of_Journey'].dt.day
    df['Journey_Month'] = df['Date_of_Journey'].dt.month
    
    # Time Handling
    df['Dep_Hour'] = pd.to_datetime(df['Dep_Time']).dt.hour
    df['Dep_Min'] = pd.to_datetime(df['Dep_Time']).dt.minute
    df['Arr_Hour'] = pd.to_datetime(df['Arrival_Time']).dt.hour
    df['Arr_Min'] = pd.to_datetime(df['Arrival_Time']).dt.minute
    
    # Duration and Stops
    df['Duration_Min'] = df['Duration'].apply(parse_duration)
    df['Stops'] = df['Total_Stops'].apply(stops_to_num)
    
    # Drop raw columns
    cols_to_drop = ['Date_of_Journey', 'Dep_Time', 'Arrival_Time', 'Duration', 'Total_Stops', 'Route']
    df.drop([c for c in cols_to_drop if c in df.columns], axis=1, inplace=True)
    
    # Encoding
    cat_cols = ['Airline', 'Source', 'Destination', 'Additional_Info']
    df = pd.get_dummies(df, columns=[c for c in cat_cols if c in df.columns])
    
    # Align columns
    if training_columns is not None:
        for col in training_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[training_columns]
        
    return df

# --- Model Logic ---
@st.cache_resource(show_spinner=False)
def train_model(file_path):
    df_raw = pd.read_excel(file_path)
    df_raw.dropna(subset=['Total_Stops', 'Airline', 'Source', 'Destination', 'Price'], inplace=True)
    
    df_proc = preprocess_data(df_raw)
    X = df_proc.drop('Price', axis=1)
    y = df_proc['Price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    score = r2_score(y_test, model.predict(X_test))
    mae = mean_absolute_error(y_test, model.predict(X_test))
    
    return model, X.columns.tolist(), df_raw, score, mae

# --- Sidebar ---
st.sidebar.title("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Data_Train.xlsx", type=["xlsx"])

DATA_PATH = None
if uploaded_file:
    DATA_PATH = uploaded_file
elif os.path.exists("Data_Train.xlsx"):
    DATA_PATH = "Data_Train.xlsx"

# --- Main App ---
st.title("✈️ Flight Price Predictor")

if DATA_PATH:
    try:
        with st.spinner("Training Model..."):
            model, feat_cols, df_orig, r2, mae = train_model(DATA_PATH)
        
        st.success(f"Model Ready! Accuracy: {r2:.2%} | Error: ₹{mae:.0f}")

        # Fix for the TypeError: Convert all unique values to strings and drop NaN before sorting
        def get_sorted_list(column_name):
            vals = df_orig[column_name].dropna().unique()
            return sorted([str(x) for x in vals])

        with st.form("prediction_form"):
            st.subheader("Flight Details")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                airline = st.selectbox("Airline", get_sorted_list('Airline'))
                source = st.selectbox("Source City", get_sorted_list('Source'))
                
            with col2:
                dest = st.selectbox("Destination City", get_sorted_list('Destination'))
                date = st.date_input("Journey Date")
                
            with col3:
                # This was the line causing the TypeError
                stops = st.selectbox("Stops", get_sorted_list('Total_Stops'))
                add_info = st.selectbox("Additional Info", get_sorted_list('Additional_Info'))
            
            col4, col5, col6 = st.columns(3)
            with col4: dep_t = st.time_input("Departure Time")
            with col5: arr_t = st.time_input("Arrival Time")
            with col6: dur = st.text_input("Duration (e.g., 2h 30m)", "2h 30m")
            
            # This button MUST be inside the 'with st.form' block
            submit_btn = st.form_submit_button("Predict Price")

        if submit_btn:
            if source == dest:
                st.error("Source and Destination cannot be the same.")
            else:
                input_data = pd.DataFrame([{
                    'Airline': airline, 'Source': source, 'Destination': dest,
                    'Total_Stops': stops, 'Additional_Info': add_info,
                    'Date_of_Journey': date.strftime("%d/%m/%Y"),
                    'Dep_Time': dep_t.strftime("%H:%M"),
                    'Arrival_Time': arr_t.strftime("%H:%M"),
                    'Duration': dur
                }])
                
                proc_input = preprocess_data(input_data, training_columns=feat_cols)
                prediction = model.predict(proc_input)[0]
                
                st.markdown(f"""
                <div class="boarding-pass">
                    <div class="bp-header">Ticket Estimate</div>
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div>
                            <span class="bp-city">{source}</span>
                            <span class="bp-plane-icon">✈</span>
                            <span class="bp-city">{dest}</span>
                            <p>{airline} • {stops}</p>
                        </div>
                        <div style="text-align: right;">
                            <div class="bp-header">Predicted Fare</div>
                            <div class="bp-price">₹{prediction:,.0f}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
    except Exception as e:
        st.error(f"Error processing data: {e}")
else:
    st.info("Please upload your training dataset (Excel) in the sidebar to start.")
