import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import date, timedelta
import re
import requests
from transformers import pipeline  # For local Hugging Face models

# Set page configuration
st.set_page_config(
    page_title="Financial Planner & Stock Recommender",
    page_icon="💰",
    layout="wide"
)

# Define navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Financial Planning", "Stock Analysis", "Investment Recommendations", "Financial Advisor Chat"])

# Currency conversion
if 'currency' not in st.session_state:
    st.session_state.currency = 'USD'

# Exchange rate (USD to INR)
def get_exchange_rate():
    try:
        # Cache the exchange rate to avoid too many API calls
        if 'exchange_rate' not in st.session_state:
            # Use a free exchange rate API (you may need to sign up for an API key)
            response = requests.get("https://open.er-api.com/v6/latest/USD")
            data = response.json()
            st.session_state.exchange_rate = data['rates']['INR']
        return st.session_state.exchange_rate
    except Exception as e:
        # Fallback to an approximate exchange rate if API fails
        st.warning(f"Using approximate exchange rate due to API error: {e}")
        return 83.0  # Approximate USD to INR rate (update as needed)

# Currency converter function
def convert_currency(amount, to_currency='INR'):
    if to_currency == 'USD':
        return amount
    elif to_currency == 'INR':
        exchange_rate = get_exchange_rate()
        return amount * exchange_rate
    return amount

# Format currency for display
def format_currency(amount, currency='USD'):
    if currency == 'USD':
        return f"${amount:,.2f}"
    elif currency == 'INR':
        return f"₹{amount:,.2f}"
    return f"{amount:,.2f}"

# Currency selector in sidebar
st.sidebar.title("Currency Settings")
currency_option = st.sidebar.selectbox(
    "Select Currency",
    ("USD", "INR"),
    index=0 if st.session_state.currency == 'USD' else 1
)

if currency_option != st.session_state.currency:
    st.session_state.currency = currency_option
    st.experimental_rerun()

# Global variables to store user data
if 'user_data' not in st.session_state:
    st.session_state.user_data = {
        'monthly_income': 0.0,
        'transportation_costs': 0.0,
        'food_costs': 0.0,
        'outing_expenses': 0.0,
        'other_costs': 0.0,
        'variable_costs': {},
        'savings': 0.0
    }

if 'investment_capacity' not in st.session_state:
    st.session_state.investment_capacity = 0

# Chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize the LLM for financial advice
@st.cache_resource
def load_llm_model():
    try:
        # Use a smaller model that can run locally
        # Options: "distilbert-base-uncased-finetuned-sst-2-english" for sentiment
        # or "gpt2" for text generation
        return pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    except Exception as e:
        st.error(f"Error loading LLM model: {e}")
        return None

# Months list for reference
months = [
    "January", "February", "March", "April", "May", "June", "July", "August", 
    "September", "October", "November", "December"
]

#####################################
# Financial Planning Functions
#####################################

def calculate_income_and_costs():
    """Calculates yearly income and expenses, including variable costs."""
    monthly_income = st.session_state.user_data['monthly_income']
    yearly_income = [monthly_income] * 12

    fixed_expenses = sum([
        st.session_state.user_data['transportation_costs'],
        st.session_state.user_data['food_costs'],
        st.session_state.user_data['outing_expenses'],
        st.session_state.user_data['other_costs']
    ])
    
    yearly_expenses = [fixed_expenses] * 12

    variable_costs = st.session_state.user_data['variable_costs']
    for idx, month in enumerate(months):
        yearly_expenses[idx] += variable_costs.get(month, 0)

    return yearly_income, yearly_expenses

def plot_financial_graphs():
    """Generates interactive graphs for income, costs, and investment capacity."""
    income, costs = calculate_income_and_costs()
    invest_capacity = [income[i] - costs[i] for i in range(len(months))]
    
    # Calculate yearly investment capacity and store it
    st.session_state.investment_capacity = sum(invest_capacity)

    fig = go.Figure()

    # Income bar
    fig.add_trace(go.Bar(
        x=months, y=income, name='Income', marker_color='green',
        hoverinfo='x+y', hovertemplate='%{y:,.2f}'
    ))

    # Costs bar
    fig.add_trace(go.Bar(
        x=months, y=costs, name='Costs', marker_color='red',
        hoverinfo='x+y', hovertemplate='%{y:,.2f}'
    ))

    # Investment capacity line
    fig.add_trace(go.Scatter(
        x=months, y=invest_capacity, mode='lines+markers', name='Investment Capacity',
        line=dict(color='blue', width=3), hoverinfo='x+y', hovertemplate='%{y:,.2f}'
    ))

    currency_symbol = "₹" if st.session_state.currency == "INR" else "$"
    
    fig.update_layout(
        title="Income, Costs, and Investment Capacity",
        xaxis_title="Months",
        yaxis_title=f"Value ({currency_symbol})",
        barmode='group',
        hovermode="x"
    )

    return fig

def plot_pie_chart():
    """Generates an interactive pie chart for cost distribution and investment capacity."""
    variable_costs = sum(st.session_state.user_data['variable_costs'].values())
    fixed_costs = sum([
        st.session_state.user_data['transportation_costs'],
        st.session_state.user_data['food_costs'],
        st.session_state.user_data['outing_expenses'],
        st.session_state.user_data['other_costs']
    ]) * 12

    total_income = sum(calculate_income_and_costs()[0])
    investment_capacity = total_income - (fixed_costs + variable_costs)

    labels = ["Variable Costs", "Fixed Costs", "Investment Capacity"]
    values = [variable_costs, fixed_costs, investment_capacity]
    colors = ["red", "orange", "green"]

    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=0.3,
        textinfo="label+percent", hoverinfo="label+value",
        marker=dict(colors=colors)
    )])

    fig.update_layout(title_text="Cost and Investment Capacity Distribution")
    return fig

#####################################
# Stock Analysis Functions
#####################################

def get_stock_data(ticker, period="1y"):
    """Fetches stock data for the given ticker."""
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period=period)
        
        if history.empty:
            return None
            
        # Financial statements
        income_statement = stock.income_stmt
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow
        
        return {
            "history": history,
            "income_statement": income_statement,
            "balance_sheet": balance_sheet,
            "cash_flow": cash_flow
        }
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

def calculate_financial_metrics(stock_data):
    """Calculate key financial metrics from stock data."""
    metrics = {}
    
    if stock_data is None:
        return metrics
    
    # Price metrics
    history = stock_data["history"]
    if not history.empty:
        current_price = history['Close'].iloc[-1]
        metrics["Current Price"] = round(current_price, 2)
        
        # Calculate price growth if we have enough data
        if len(history) > 20:
            start_price = history['Close'].iloc[0]
            price_change = ((current_price - start_price) / start_price) * 100
            metrics["Price Change (%)"] = round(price_change, 2)
    
    # Income statement metrics
    income = stock_data["income_statement"]
    balance = stock_data["balance_sheet"]
    
    if not income.empty and not balance.empty:
        # Use the most recent annual data
        try:
            # Get latest year data
            latest_income = income.iloc[:, 0]
            latest_balance = balance.iloc[:, 0]
            
            # Calculation of metrics
            if 'Total Revenue' in latest_income and 'Net Income' in latest_income:
                revenue = latest_income['Total Revenue']
                net_income = latest_income['Net Income']
                
                metrics["Revenue"] = revenue
                metrics["Net Income"] = net_income
                
                # Net Margin
                if revenue > 0:
                    net_margin = (net_income / revenue) * 100
                    metrics["Net Margin (%)"] = round(net_margin, 2)
            
            # ROE
            if 'Net Income' in latest_income and 'Total Stockholder Equity' in latest_balance:
                equity = latest_balance['Total Stockholder Equity']
                if equity > 0:
                    roe = (net_income / equity) * 100
                    metrics["ROE (%)"] = round(roe, 2)
            
            # Debt-to-Equity
            if 'Total Debt' in latest_balance and 'Total Stockholder Equity' in latest_balance:
                debt = latest_balance.get('Total Debt', 0)
                if equity > 0:
                    debt_to_equity = (debt / equity) * 100
                    metrics["Debt-to-Equity (%)"] = round(debt_to_equity, 2)
        
        except Exception as e:
            st.warning(f"Could not calculate all metrics: {e}")
    
    return metrics

def plot_stock_price_history(history):
    """Plot the stock price history."""
    if history is None or history.empty:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=history.index, 
        y=history['Close'], 
        mode='lines',
        name='Close Price',
        line=dict(color='blue', width=2)
    ))
    
    currency_symbol = "₹" if st.session_state.currency == "INR" else "$"
    
    fig.update_layout(
        title="Stock Price History",
        xaxis_title="Date",
        yaxis_title=f"Price ({currency_symbol})",
        hovermode="x unified"
    )
    
    return fig

def get_stock_recommendations(metrics_list, tickers):
    """Get stock recommendations based on financial metrics."""
    # Create a scoring system
    scored_stocks = []
    
    for ticker, metrics in zip(tickers, metrics_list):
        if not metrics:
            continue
            
        score = 0
        
        # Price change score
        if "Price Change (%)" in metrics:
            price_change = metrics["Price Change (%)"]
            # Higher price change is better, but cap at 20 points
            score += min(price_change / 5, 20) if price_change > 0 else max(price_change / 10, -10)
        
        # Net margin score
        if "Net Margin (%)" in metrics:
            net_margin = metrics["Net Margin (%)"]
            # Higher margins are better
            score += min(net_margin / 2, 30)
        
        # ROE score
        if "ROE (%)" in metrics:
            roe = metrics["ROE (%)"]
            # Higher ROE is better, but cap at 25 points
            score += min(roe / 2, 25)
        
        # Debt-to-Equity score (lower is better)
        if "Debt-to-Equity (%)" in metrics:
            dte = metrics["Debt-to-Equity (%)"]
            # Lower D/E is better, penalize high debt
            score -= min(dte / 10, 15)
        
        scored_stocks.append((ticker, round(score, 2), metrics))
    
    # Sort by score (highest first)
    scored_stocks.sort(key=lambda x: x[1], reverse=True)
    
    return scored_stocks

def allocate_investment(recommendations, investment_amount):
    """Allocate investment amount across recommended stocks."""
    if not recommendations:
        return {}
    
    # Number of stocks to invest in (top 3 or all if less than 3)
    num_stocks = min(3, len(recommendations))
    top_stocks = recommendations[:num_stocks]
    
    # Equal allocation
    amount_per_stock = investment_amount / num_stocks
    
    allocation = {}
    for ticker, score, metrics in top_stocks:
        if "Current Price" in metrics:
            price = metrics["Current Price"]
            shares = int(amount_per_stock / price)
            
            allocation[ticker] = {
                "Allocation ($)": round(amount_per_stock, 2),
                "Share Price ($)": price,
                "Shares to Buy": shares,
                "Total Cost ($)": round(shares * price, 2),
                "Score": score
            }
    
    return allocation

#####################################
# Financial Advisor Chat with Open Source LLM
#####################################

def get_financial_advice_llm(query):
    """Generate financial advice using open source LLM."""
    try:
        # Load the model
        model = load_llm_model()
        if model is None:
            return "I'm sorry, the language model could not be loaded. Please try again later."
        
        # Add context about the user's financial situation if available
        context = ""
        if st.session_state.user_data['monthly_income'] > 0:
            income = st.session_state.user_data['monthly_income']
            savings = st.session_state.user_data['savings']
            
            if st.session_state.currency == "INR":
                # Convert to INR for display
                income_inr = convert_currency(income, 'INR')
                savings_inr = convert_currency(savings, 'INR')
                context = f"With a monthly income of ₹{income_inr:,.2f} and savings of ₹{savings_inr:,.2f}: "
            else:
                context = f"With a monthly income of ${income:,.2f} and savings of ${savings:,.2f}: "

        # Create a prompt with financial advisor context
        prompt = f"""
        <|system|>
        You are a knowledgeable financial advisor assistant. Provide clear, accurate, and helpful advice on personal finance topics. 
        Keep responses concise but informative. Don't recommend specific investments or make promises about returns.
        </|system|>
        
        <|user|>
        {context}{query}
        </|user|>
        
        <|assistant|>
        """
        
        # Generate response
        response = model(prompt, max_length=400, temperature=0.7, num_return_sequences=1)
        
        # Extract the response text
        generated_text = response[0]['generated_text']
        
        # Extract only the assistant's response
        assistant_response = generated_text.split("<|assistant|>")[-1].strip()
        
        # Clean up the response if needed
        if "<|" in assistant_response:
            assistant_response = assistant_response.split("<|")[0].strip()
            
        return assistant_response
        
    except Exception as e:
        st.error(f"Error with LLM service: {e}")
        return "I'm sorry, I couldn't process your request. Please try again or ask a different question."

#####################################
# Page Content
#####################################

# Financial Planning Page
if page == "Financial Planning":
    st.title("📊 Personal Financial Planning")
    
    with st.form("financial_info_form"):
        st.subheader("Monthly Income & Expenses")
        col1, col2 = st.columns(2)
        
        with col1:
            monthly_income = st.number_input(
                f"Monthly Net Income ({st.session_state.currency})",
                min_value=0.0,
                value=st.session_state.user_data['monthly_income']
            )
            transportation = st.number_input(
                f"Transportation Costs ({st.session_state.currency})",
                min_value=0.0,
                value=st.session_state.user_data['transportation_costs']
            )
            food = st.number_input(
                f"Food Costs ({st.session_state.currency})",
                min_value=0.0,
                value=st.session_state.user_data['food_costs']
            )
        
        with col2:
            outings = st.number_input(
                f"Outing Expenses ({st.session_state.currency})",
                min_value=0.0,
                value=st.session_state.user_data['outing_expenses']
            )
            other = st.number_input(
                f"Other Fixed Costs ({st.session_state.currency})",
                min_value=0.0,
                value=st.session_state.user_data['other_costs']
            )
            savings = st.number_input(
                f"Available Savings ({st.session_state.currency})",
                min_value=0.0,
                value=st.session_state.user_data['savings']
            )
        
        st.subheader("Variable Monthly Costs")
        has_variable_costs = st.checkbox("I have variable costs")
        
        if has_variable_costs:
            variable_costs = {}
            
            # Create two columns for the months
            cols = st.columns(2)
            
            for i, month in enumerate(months):
                col_idx = i % 2
                with cols[col_idx]:
                    variable_costs[month] = st.number_input(
                        f"{month} ({st.session_state.currency})",
                        min_value=0.0,
                        value=st.session_state.user_data['variable_costs'].get(month, 0.0)
                    )
        else:
            variable_costs = {}
        
        # Submit button: user data is submitted only when this button is pressed
        submit_button = st.form_submit_button("Calculate Financial Summary")
        
        if submit_button:
            # Update session state
            st.session_state.user_data = {
                'monthly_income': monthly_income,
                'transportation_costs': transportation,
                'food_costs': food,
                'outing_expenses': outings,
                'other_costs': other,
                'variable_costs': variable_costs,
                'savings': savings
            }
    
    # Only show financial summary if we have income data
    if st.session_state.user_data['monthly_income'] > 0:
        st.subheader("Financial Summary")
        
        # Calculate monthly and yearly metrics
        yearly_income, yearly_expenses = calculate_income_and_costs()
        monthly_investment = sum(yearly_income) / 12 - sum(yearly_expenses) / 12
        yearly_investment = sum(yearly_income) - sum(yearly_expenses)
        
        # Store the investment capacity
        st.session_state.investment_capacity = yearly_investment
        
        # Currency symbol for display
        currency_symbol = "₹" if st.session_state.currency == "INR" else "$"
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Monthly Income", f"{currency_symbol}{st.session_state.user_data['monthly_income']:,.2f}")
        col2.metric("Monthly Expenses", f"{currency_symbol}{sum(yearly_expenses) / 12:,.2f}")
        col3.metric("Monthly Investment Capacity", f"{currency_symbol}{monthly_investment:,.2f}")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Yearly Income", f"{currency_symbol}{sum(yearly_income):,.2f}")
        col2.metric("Yearly Expenses", f"{currency_symbol}{sum(yearly_expenses):,.2f}")
        col3.metric("Yearly Investment Capacity", f"{currency_symbol}{yearly_investment:,.2f}")
        
        # Show graphs
        st.plotly_chart(plot_financial_graphs(), use_container_width=True)
        st.plotly_chart(plot_pie_chart(), use_container_width=True)

# Stock Analysis Page
elif page == "Stock Analysis":
    st.title("📈 Stock Analysis")
    
    # Stock ticker input
    ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL, MSFT, GOOGL)", "AAPL").upper()
    
    if ticker:
        with st.spinner(f"Fetching data for {ticker}..."):
            stock_data = get_stock_data(ticker)
            
            if stock_data:
                st.success(f"Successfully retrieved data for {ticker}")
                
                # Calculate metrics
                metrics = calculate_financial_metrics(stock_data)
                
                # Display stock price chart
                history_chart = plot_stock_price_history(stock_data["history"])
                if history_chart:
                    st.plotly_chart(history_chart, use_container_width=True)
                
                # Display key metrics
                st.subheader("Key Financial Metrics")
                
                if metrics:
                    metric_cols = st.columns(3)
                    
                    for i, (metric, value) in enumerate(metrics.items()):
                        col_idx = i % 3
                        
                        # Format the value based on its magnitude and currency
                        currency_symbol = "₹" if st.session_state.currency == "INR" else "$"
                        
                        if st.session_state.currency == "INR" and "%" not in metric:
                            # Convert to INR for display
                            value_converted = convert_currency(value, 'INR') if isinstance(value, (int, float)) else value
                        else:
                            value_converted = value
                        
                        if isinstance(value_converted, (int, float)) and abs(value_converted) > 1000000:
                            display_value = f"{currency_symbol}{value_converted/1000000:,.2f}M"
                        elif isinstance(value_converted, (int, float)) and "%" not in metric:
                            display_value = f"{currency_symbol}{value_converted:,.2f}"
                        else:
                            display_value = f"{value_converted}"
                            
                        metric_cols[col_idx].metric(metric, display_value)
                else:
                    st.warning("Could not calculate financial metrics for this stock.")
                
                # Company information
                st.subheader("Company Information")
                try:
                    info = yf.Ticker(ticker).info
                    if 'longBusinessSummary' in info:
                        st.info(info['longBusinessSummary'])
                    else:
                        st.info("No company information available.")
                except:
                    st.info("No company information available.")
            else:
                st.error(f"Could not retrieve data for {ticker}. Please check the ticker symbol.")

# Investment Recommendations Page
elif page == "Investment Recommendations":
    st.title("🔍 Investment Recommendations")
    
    # Check if we have investment capacity
    if st.session_state.investment_capacity <= 0:
        st.warning("Please complete the Financial Planning section first to calculate your investment capacity.")
        if st.button("Go to Financial Planning"):
            st.session_state.page = "Financial Planning"
            st.experimental_rerun()
    else:
        currency_symbol = "₹" if st.session_state.currency == "INR" else "$"
        
        # Display investment capacity in selected currency
        if st.session_state.currency == "INR":
            investment_capacity_display = convert_currency(st.session_state.investment_capacity, 'INR')
        else:
            investment_capacity_display = st.session_state.investment_capacity
            
        st.subheader(f"Investment Capacity: {currency_symbol}{investment_capacity_display:,.2f}")
        
        # List of sample tickers to analyze
        default_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA","META", "JPM", "V"]
        
        # Let user customize tickers
        st.write("Enter stock tickers to analyze (comma-separated):")
        tickers_input = st.text_input("Tickers", ", ".join(default_tickers))
        
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        
        if st.button("Analyze Stocks and Get Recommendations"):
            if not tickers:
                st.error("Please enter at least one ticker symbol.")
            else:
                with st.spinner("Analyzing stocks... This may take a moment."):
                    # Get data for each ticker
                    all_metrics = []
                    valid_tickers = []
                    
                    progress_bar = st.progress(0)
                    
                    for i, ticker in enumerate(tickers):
                        progress_bar.progress((i + 1) / len(tickers))
                        stock_data = get_stock_data(ticker)
                        
                        if stock_data:
                            metrics = calculate_financial_metrics(stock_data)
                            if metrics:
                                all_metrics.append(metrics)
                                valid_tickers.append(ticker)
                    
                    # Get recommendations
                    recommendations = get_stock_recommendations(all_metrics, valid_tickers)
                    
                    if recommendations:
                        st.success("Analysis complete!")
                        
                        # Display top recommendations
                        st.subheader("Top Stock Recommendations")
                        
                        for i, (ticker, score, metrics) in enumerate(recommendations[:5]):
                            with st.expander(f"{i+1}. {ticker} - Score: {score}"):
                                cols = st.columns(4)
                                
                                for j, (metric, value) in enumerate(metrics.items()):
                                    col_idx = j % 4
                                    
                                    # Format the value with currency conversion if needed
                                    currency_symbol = "₹" if st.session_state.currency == "INR" else "$"
                                    
                                    if st.session_state.currency == "INR" and "%" not in metric:
                                        # Convert to INR for display
                                        value_converted = convert_currency(value, 'INR') if isinstance(value, (int, float)) else value
                                    else:
                                        value_converted = value
                                    
                                    if isinstance(value_converted, (int, float)) and abs(value_converted) > 1000000:
                                        display_value = f"{currency_symbol}{value_converted/1000000:,.2f}M"
                                    elif isinstance(value_converted, (int, float)) and "%" not in metric:
                                        display_value = f"{currency_symbol}{value_converted:,.2f}"
                                    else:
                                        display_value = f"{value_converted}"
                                        
                                    cols[col_idx].metric(metric, display_value)
                        
                        # Calculate investment allocation
                        st.subheader("Recommended Investment Allocation")
                        allocations = allocate_investment(recommendations, st.session_state.investment_capacity)
                        
                        if allocations:
                            allocation_data = []
                            for ticker, details in allocations.items():
                                # Convert currency values if needed
                                if st.session_state.currency == "INR":
                                    allocation_amount = convert_currency(details["Allocation ($)"], 'INR')
                                    share_price = convert_currency(details["Share Price ($)"], 'INR')
                                    total_cost = convert_currency(details["Total Cost ($)"], 'INR')
                                    currency_label = "Allocation (₹)"
                                    price_label = "Share Price (₹)"
                                    cost_label = "Total Cost (₹)"
                                else:
                                    allocation_amount = details["Allocation ($)"]
                                    share_price = details["Share Price ($)"]
                                    total_cost = details["Total Cost ($)"]
                                    currency_label = "Allocation ($)"
                                    price_label = "Share Price ($)"
                                    cost_label = "Total Cost ($)"
                                    
                                allocation_data.append({
                                    "Ticker": ticker,
                                    currency_label: allocation_amount,
                                    price_label: share_price,
                                    "Shares to Buy": details["Shares to Buy"],
                                    cost_label: total_cost,
                                    "Score": details["Score"]
                                })
                            
                            st.table(pd.DataFrame(allocation_data))
                            
                            # Pie chart for allocation
                            allocation_fig = go.Figure(data=[go.Pie(
                                labels=list(allocations.keys()),
                                values=[details["Allocation ($)"] for details in allocations.values()],
                                hole=0.4,
                                textinfo="label+percent"
                            )])
                            
                            allocation_fig.update_layout(title_text=f"Investment Allocation ({st.session_state.currency})")
                            st.plotly_chart(allocation_fig, use_container_width=True)
                        else:
                            st.warning("Could not calculate investment allocations.")
                    else:
                        st.warning("Could not generate recommendations. Please check the ticker symbols.")

# Financial Advisor Chat Page
elif page == "Financial Advisor Chat":
    st.title("💬 Financial Advisor Chat")
    
    st.write("""
    Ask me anything about personal finance, investing, retirement planning, or financial terms!
    """)
    
    # Create a container for the chat
    chat_container = st.container()
    
    # Create a form for user input
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Your question:", key="user_message")
        submit_chat = st.form_submit_button("Send")
    
    # Show example questions
    with st.expander("Example questions you can ask"):
        st.markdown("""
        - What is a Roth IRA?
        - How much should I save for retirement?
        - What's a good investment strategy for beginners?
        - Should I pay off debt or invest?
        - What is an ETF?
        - How should I build an emergency fund?
        """)
    
    # If user submits a question
    if submit_chat and user_input:
        # Append the user's message to chat history
        st.session_state.chat_history.append({"role": "user", "message": user_input})
        
        # Get the response from the open source LLM
        response = get_financial_advice_llm(user_input)
        
        # Append the assistant's response to chat history
        st.session_state.chat_history.append({"role": "assistant", "message": response})
        
        # Rerun to update the chat container with the new messages
        st.experimental_rerun()
    
    # Display the chat history in the chat container
    with chat_container:
        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                st.markdown(f"**You:** {chat['message']}")
            else:
                st.markdown(f"**Advisor:** {chat['message']}")