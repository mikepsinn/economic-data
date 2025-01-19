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
OUTPUT_FILE = "us_dollar_value.csv"

# Scenario presets
SCENARIOS = {
    "Custom": {
        "inflation": 3.0,
        "interest": 5.25,
        "gdp": 2.0,
        "debt": 120.0,
        "trade": 3.0,
        "description": "Custom scenario with user-defined parameters"
    },
    "Russian Crisis (1998)": {
        "inflation": 84.4,  # Peak inflation during the crisis
        "interest": 150.0,  # Emergency rate hike
        "gdp": -5.3,      # GDP contraction
        "debt": 130.0,    # High debt levels
        "trade": 8.0,     # Trade imbalances
        "description": """Simulates conditions similar to the 1998 Russian financial crisis:
        - Hyperinflation following currency collapse
        - Emergency interest rate hikes
        - Severe economic contraction
        - High government debt
        - Trade imbalances"""
    },
    "Moderate Decline": {
        "inflation": 7.0,
        "interest": 3.0,
        "gdp": 1.0,
        "debt": 150.0,
        "trade": 5.0,
        "description": "Gradual economic deterioration with above-target inflation"
    },
    "Stagflation": {
        "inflation": 12.0,
        "interest": 8.0,
        "gdp": -1.0,
        "debt": 140.0,
        "trade": 4.0,
        "description": "High inflation combined with economic stagnation, similar to the 1970s"
    }
}

def calculate_dollar_value(params, years, start_value, scenario_name="Custom"):
    """Calculate projected dollar value based on economic parameters."""
    # Initial value from the last historical point
    base_value = start_value
    
    # Monthly timesteps
    months = years * 12
    timeline = np.array([START_YEAR + i/12 for i in range(months)])
    
    # Enhanced factor weights based on scenario
    if scenario_name == "Russian Crisis (1998)":
        # More dramatic weights for crisis scenario
        inflation_impact = -1.2  # Stronger inflation impact
        interest_rate_impact = 0.3  # Reduced effectiveness of interest rates
        gdp_impact = 0.4  # Stronger GDP impact
        debt_impact = -0.4  # Stronger debt impact
        trade_impact = -0.4  # Stronger trade impact
        
        # Add crisis acceleration factor
        crisis_factor = 1.5  # Accelerates the decline
    else:
        # Standard weights for other scenarios
        inflation_impact = -0.8
        interest_rate_impact = 0.4
        gdp_impact = 0.3
        debt_impact = -0.2
        trade_impact = -0.3
        crisis_factor = 1.0
    
    # Calculate monthly effects with non-linear relationships
    monthly_inflation = (1 + params['inflation'] / 100) ** (1/12) - 1
    monthly_interest = (1 + params['interest'] / 100) ** (1/12) - 1
    monthly_gdp = (1 + params['gdp'] / 100) ** (1/12) - 1
    
    # Enhanced combined monthly effect with non-linear relationships
    monthly_effect = (
        (monthly_inflation * inflation_impact * (1 + abs(monthly_inflation))) +  # Non-linear inflation impact
        (monthly_interest * interest_rate_impact * (1 - monthly_inflation)) +    # Interest rate effectiveness decreases with high inflation
        (monthly_gdp * gdp_impact) +
        (params['debt'] / 100) * debt_impact * (1 + params['inflation'] / 200) / 12 +  # Debt impact increases with inflation
        (params['trade'] / 100) * trade_impact / 12
    ) * crisis_factor
    
    # Calculate cumulative effect with potential for accelerated decline
    values = base_value * (1 + monthly_effect) ** np.array(range(months))
    
    # Add volatility for crisis scenario
    if scenario_name == "Russian Crisis (1998)":
        volatility = np.random.normal(0, 0.02, months)  # 2% monthly volatility
        values = values * (1 + volatility)
    
    return timeline, values

# Set page config
st.set_page_config(
    page_title="USD Value Analysis",
    page_icon="💵",
    layout="wide"
)

# Title and description
st.title("💵 US Dollar Value Analysis & Projection")
st.markdown("""
This tool shows the historical purchasing power of the US dollar and projects potential future scenarios based on economic parameters.
Values are normalized to 100 at the selected start year, showing the relative change in purchasing power over time.
""")

# Load historical data
try:
    historical_data = pd.read_csv("us_dollar_value.csv")
    historical_data['date'] = pd.to_datetime(historical_data['date'])
except Exception as e:
    st.error("Please make sure the historical data file 'us_dollar_value.csv' is available.")
    historical_data = None

if historical_data is not None:
    # Constants
    CURRENT_YEAR = datetime.now().year
    MIN_YEAR = historical_data['date'].dt.year.min()
    
    # Sidebar controls
    st.sidebar.header("Analysis Parameters")
    
    # Start year selection
    START_YEAR = st.sidebar.slider(
        "Start Year",
        min_value=MIN_YEAR,
        max_value=CURRENT_YEAR-1,
        value=2000,
        step=1,
        help="Year to start the analysis (normalized to 100)"
    )

    # Simulation period
    end_year = st.sidebar.slider(
        "Project Until Year",
        min_value=CURRENT_YEAR,
        max_value=CURRENT_YEAR + 30,
        value=CURRENT_YEAR + 5,
        step=1
    )

    st.sidebar.header("Scenario Selection")
    
    # Scenario selection
    selected_scenario = st.sidebar.selectbox(
        "Choose a Scenario",
        options=list(SCENARIOS.keys()),
        help="Select a predefined scenario or customize your own"
    )
    
    # Show scenario description
    st.sidebar.markdown(f"**Scenario Description:**\n{SCENARIOS[selected_scenario]['description']}")

    st.sidebar.header("Economic Parameters")
    
    # Calculate number of years
    years = end_year - CURRENT_YEAR

    # Economic parameters with preset values
    inflation_rate = st.sidebar.slider(
        "Annual Inflation Rate (%)",
        min_value=0.0,
        max_value=200.0,  # Increased for crisis scenarios
        value=SCENARIOS[selected_scenario]['inflation'],
        step=0.1,
        help="Expected annual inflation rate"
    )

    interest_rate = st.sidebar.slider(
        "Federal Funds Rate (%)",
        min_value=0.0,
        max_value=200.0,  # Increased for crisis scenarios
        value=SCENARIOS[selected_scenario]['interest'],
        step=0.25,
        help="Federal Reserve's target interest rate"
    )

    gdp_growth = st.sidebar.slider(
        "GDP Growth Rate (%)",
        min_value=-10.0,
        max_value=10.0,
        value=SCENARIOS[selected_scenario]['gdp'],
        step=0.1,
        help="Expected annual GDP growth rate"
    )

    debt_gdp = st.sidebar.slider(
        "Debt to GDP Ratio (%)",
        min_value=50.0,
        max_value=200.0,
        value=SCENARIOS[selected_scenario]['debt'],
        step=5.0,
        help="Government debt as percentage of GDP"
    )

    trade_deficit = st.sidebar.slider(
        "Trade Deficit (% of GDP)",
        min_value=0.0,
        max_value=20.0,  # Increased for crisis scenarios
        value=SCENARIOS[selected_scenario]['trade'],
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

    # Filter and normalize historical data
    mask = historical_data['date'].dt.year >= START_YEAR
    filtered_data = historical_data[mask].copy()
    
    # Find the first value to normalize against
    start_value = filtered_data.iloc[0]['purchasing_power']
    filtered_data['normalized_power'] = filtered_data['purchasing_power'] * (100 / start_value)

    # Get the last historical value to start projections from
    last_historical = filtered_data.iloc[-1]
    current_value = last_historical['normalized_power']
    timeline, values = calculate_dollar_value(params, years, current_value, selected_scenario)

    # Create plot
    fig = go.Figure()

    # Add historical data
    fig.add_trace(go.Scatter(
        x=filtered_data['date'],
        y=filtered_data['normalized_power'],
        mode='lines',
        name='Historical',
        line=dict(color='#2ecc71', width=2)
    ))

    # Create dates for projected values
    projection_dates = pd.date_range(
        start=last_historical['date'],
        periods=len(values),
        freq='M'
    )

    # Add projected values
    fig.add_trace(go.Scatter(
        x=projection_dates,
        y=values,
        mode='lines',
        name=f'Projected ({selected_scenario})',
        line=dict(color='#1f77b4', width=2, dash='dash')
    ))

    # Add reference line for start year baseline
    fig.add_hline(
        y=100, 
        line_dash="dot", 
        line_color="gray", 
        opacity=0.5,
        annotation_text=f"{START_YEAR} Baseline (100)",
        annotation_position="bottom right"
    )

    # Customize layout
    fig.update_layout(
        title=f'US Dollar Purchasing Power ({START_YEAR}-Present) with {selected_scenario} Scenario',
        xaxis_title='Year',
        yaxis_title=f'Purchasing Power ({START_YEAR} = 100)',
        hovermode='x unified',
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    # Display plot
    st.plotly_chart(fig, use_container_width=True)

    # Display key metrics
    final_value = values[-1]
    total_change = ((final_value - current_value) / current_value) * 100
    annual_change = ((final_value / current_value) ** (1/years) - 1) * 100

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Current Value",
            f"{current_value:.1f}",
            f"Relative to {START_YEAR} (100)"
        )

    with col2:
        st.metric(
            "Projected Value in " + str(end_year),
            f"{final_value:.1f}",
            f"{total_change:+.1f}% total change"
        )

    with col3:
        st.metric(
            "Projected Annual Change",
            f"{annual_change:+.1f}%/year",
            f"From {CURRENT_YEAR} to {end_year}"
        )

    # Add explanatory notes
    st.markdown("""
    ### How to Interpret the Results

    - **Historical Data** (solid green line) shows the actual purchasing power of the US dollar based on CPI data
    - **Projected Values** (dashed blue line) show simulated future scenarios based on the selected scenario and parameters
    - Values are normalized to 100 at the selected start year
    - A value of 50 means half the purchasing power compared to the start year

    ### Model Factors
    The projection considers these key economic factors with varying weights based on the scenario:
    - Inflation rate (stronger negative impact in crisis scenarios)
    - Interest rates (effectiveness decreases during high inflation)
    - GDP growth (increased impact during crises)
    - Debt to GDP ratio (impact increases with inflation)
    - Trade deficit (higher impact in crisis scenarios)

    ### Scenario Details
    - **Custom**: User-defined parameters for flexible analysis
    - **Russian Crisis**: Simulates severe economic conditions similar to the 1998 Russian financial crisis
    - **Moderate Decline**: Gradual deterioration with above-target inflation
    - **Stagflation**: High inflation with economic stagnation

    ### Limitations
    - This is a simplified model for educational purposes
    - Real currency crises can be more severe and unpredictable
    - International events, policy changes, and market sentiment are not fully captured
    - Past performance does not guarantee future results
    """)
else:
    st.error("Unable to load historical data. Please check the data file.") 