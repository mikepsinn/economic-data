import os
import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv('FRED_API_KEY')

# FRED API configuration
FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
SERIES_ID = "CPIAUCSL"  # Consumer Price Index for All Urban Consumers
OUTPUT_FILE = "data/us_dollar_value.csv"

def download_fred_data():
    """Download CPI data from FRED API and calculate purchasing power."""
    try:
        params = {
            'series_id': SERIES_ID,
            'api_key': API_KEY,
            'file_type': 'json',
            'observation_start': '1913-01-01'  # Earliest CPI data
        }
        
        response = requests.get(FRED_URL, params=params)
        response.raise_for_status()
        
        data = response.json()
        observations = data['observations']
        
        # Convert to DataFrame and save as CSV
        df = pd.DataFrame(observations)
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        # Rename columns and calculate purchasing power
        df = df.rename(columns={'value': 'cpi'})
        df['purchasing_power'] = 100 / df['cpi']
        df = df[['date', 'cpi', 'purchasing_power']]
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        df.to_csv(OUTPUT_FILE, index=False)
        
        st.success(f"Successfully downloaded {len(df)} records of CPI data")
        return df
        
    except Exception as e:
        st.error(f"Error downloading data: {str(e)}")
        return None

def calculate_dollar_value(params, years):
    """Calculate projected dollar value based on economic parameters."""
    # Initial value
    base_value = 100
    
    # Monthly timesteps
    months = years * 12
    timeline = np.array([START_YEAR + i/12 for i in range(months)])
    
    # Factor weights (simplified model)
    inflation_impact = -0.8
    interest_rate_impact = 0.4
    gdp_impact = 0.3
    debt_impact = -0.2
    trade_impact = -0.3
    
    # Calculate combined monthly effect
    monthly_inflation = (1 + params['inflation'] / 100) ** (1/12) - 1
    monthly_interest = (1 + params['interest'] / 100) ** (1/12) - 1
    monthly_gdp = (1 + params['gdp'] / 100) ** (1/12) - 1
    
    # Combined monthly effect
    monthly_effect = (
        monthly_inflation * inflation_impact +
        monthly_interest * interest_rate_impact +
        monthly_gdp * gdp_impact +
        (params['debt'] / 100) * debt_impact / 12 +
        (params['trade'] / 100) * trade_impact / 12
    )
    
    # Calculate cumulative effect
    values = base_value * (1 + monthly_effect) ** np.array(range(months))
    
    return timeline, values

# Set page config
st.set_page_config(
    page_title="USD Value Analysis",
    page_icon="💵",
    layout="wide"
)

# Add tabs for different views
tab1, tab2 = st.tabs(["Future Projections", "Historical Analysis (1913-Present)"])

with tab1:
    # Title and description for projections
    st.title("🔮 USD Value Projections")
    st.markdown("""
    This tool combines historical CPI data with future value predictions:
    1. Historical data from FRED (Federal Reserve Economic Data)
    2. Future projections based on adjustable economic parameters
    """)

    # Add data refresh button
    if st.button("Refresh FRED Data"):
        download_fred_data()

    # Constants
    START_YEAR = 2000
    CURRENT_YEAR = datetime.now().year

    # Sidebar controls
    st.sidebar.header("Economic Parameters")

    # Simulation period
    end_year = st.sidebar.slider(
        "End Year",
        min_value=START_YEAR + 1,
        max_value=START_YEAR + 30,
        value=START_YEAR + 5,
        step=1
    )

    # Calculate number of years
    years = end_year - START_YEAR

    # Economic parameters
    inflation_rate = st.sidebar.slider(
        "Annual Inflation Rate (%)",
        min_value=0.0,
        max_value=15.0,
        value=3.0,
        step=0.1,
        help="Expected annual inflation rate"
    )

    interest_rate = st.sidebar.slider(
        "Federal Funds Rate (%)",
        min_value=0.0,
        max_value=10.0,
        value=5.25,
        step=0.25,
        help="Federal Reserve's target interest rate"
    )

    gdp_growth = st.sidebar.slider(
        "GDP Growth Rate (%)",
        min_value=-5.0,
        max_value=10.0,
        value=2.0,
        step=0.1,
        help="Expected annual GDP growth rate"
    )

    debt_gdp = st.sidebar.slider(
        "Debt to GDP Ratio (%)",
        min_value=50.0,
        max_value=200.0,
        value=120.0,
        step=5.0,
        help="Government debt as percentage of GDP"
    )

    trade_deficit = st.sidebar.slider(
        "Trade Deficit (% of GDP)",
        min_value=0.0,
        max_value=10.0,
        value=3.0,
        step=0.1,
        help="Trade deficit as percentage of GDP"
    )

    # Run simulation
    params = {
        'inflation': inflation_rate,
        'interest': interest_rate,
        'gdp': gdp_growth,
        'debt': debt_gdp,
        'trade': trade_deficit
    }

    timeline, values = calculate_dollar_value(params, years)

    # Create plot
    fig = go.Figure()

    # Add historical data if available
    try:
        historical_data = pd.read_csv(OUTPUT_FILE)
        historical_data['date'] = pd.to_datetime(historical_data['date'])
        mask = historical_data['date'].dt.year >= START_YEAR
        historical_data = historical_data[mask]
        
        fig.add_trace(go.Scatter(
            x=historical_data['date'],
            y=historical_data['purchasing_power'],
            mode='lines',
            name='Historical',
            line=dict(color='#2ecc71', width=2)
        ))
    except Exception as e:
        st.warning("Historical data not available. Use the 'Refresh FRED Data' button to download it.")

    # Add projected values
    fig.add_trace(go.Scatter(
        x=timeline,
        y=values,
        mode='lines',
        name='Projected',
        line=dict(color='#1f77b4', width=2)
    ))

    # Customize layout
    fig.update_layout(
        title='USD Value Over Time (Historical & Projected)',
        xaxis_title='Year',
        yaxis_title='USD Value (Year 2000 = 100)',
        hovermode='x unified',
        height=500
    )

    # Add reference line
    fig.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.5)

    # Update x-axis to show years clearly
    fig.update_xaxes(dtick=1)

    # Display plot
    st.plotly_chart(fig, use_container_width=True)

    # Display key metrics
    final_value = values[-1]
    total_change = ((final_value - 100) / 100) * 100
    annual_change = ((final_value / 100) ** (1/years) - 1) * 100

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Projected Final Value",
            f"{final_value:.2f}",
            f"{total_change:+.2f}% total"
        )

    with col2:
        st.metric(
            "Annual Rate of Change",
            f"{annual_change:+.2f}%/year"
        )

    with col3:
        st.metric(
            "Time Period",
            f"{START_YEAR} to {end_year}"
        )

with tab2:
    # Title and description for historical analysis
    st.title("📈 Historical USD Value Analysis (1913-Present)")
    st.markdown("""
    This view shows the long-term historical trends in the US dollar's value:
    1. Consumer Price Index (CPI) growth over time
    2. Corresponding decline in purchasing power
    """)
    
    try:
        # Load historical data
        historical_data = pd.read_csv(OUTPUT_FILE)
        historical_data['date'] = pd.to_datetime(historical_data['date'])
        
        # Create two plots
        fig1 = go.Figure()
        fig2 = go.Figure()
        
        # Plot CPI
        fig1.add_trace(go.Scatter(
            x=historical_data['date'],
            y=historical_data['cpi'],
            mode='lines',
            name='CPI',
            line=dict(color='blue', width=2)
        ))
        
        fig1.update_layout(
            title='Consumer Price Index (1913-Present)',
            xaxis_title='Year',
            yaxis_title='CPI Value',
            height=400
        )
        
        # Plot Purchasing Power
        fig2.add_trace(go.Scatter(
            x=historical_data['date'],
            y=historical_data['purchasing_power'],
            mode='lines',
            name='Purchasing Power',
            line=dict(color='red', width=2)
        ))
        
        fig2.update_layout(
            title='US Dollar Purchasing Power (1913-Present)',
            xaxis_title='Year',
            yaxis_title='Purchasing Power',
            yaxis_type="log",
            height=400
        )
        
        # Display plots
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
        
    except Exception as e:
        st.error("Please download the historical data first using the 'Refresh FRED Data' button in the Projections tab.")

# Add explanatory notes
st.markdown("""
### How to Interpret the Results

- Historical data (green line) shows actual purchasing power based on CPI
- Projected values (blue line) show simulated future scenarios
- The simulation considers these key factors:
  - Inflation rate (negative impact)
  - Interest rates (positive impact)
  - GDP growth (positive impact)
  - Debt to GDP ratio (negative impact)
  - Trade deficit (negative impact)

### Model Limitations

- This is a simplified model for educational purposes
- Real currency values are affected by many more factors
- International events, policy changes, and market sentiment are not included
- Past performance does not guarantee future results
""") 