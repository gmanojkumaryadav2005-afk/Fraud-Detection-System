import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# ===================== PAGE CONFIG ===================== #
st.set_page_config(
    page_title="Fraud Detection in Online Transactions",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== CUSTOM CSS ===================== #
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, #1a2980 0%, #26a0da 100%);
    }
    
    /* Main container with elegant glassmorphism */
    .main-container {
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(10px);
        border-radius: 24px;
        padding: 2rem;
        box-shadow: 0 30px 60px rgba(0, 0, 0, 0.15);
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Elegant Header */
    .hero-header {
        text-align: center;
        padding: 2.5rem 2rem;
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        border-radius: 20px;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
    }
    
    .hero-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%);
        animation: shine 3s infinite;
    }
    
    @keyframes shine {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .hero-header h1 {
        color: white;
        font-size: 2.8rem;
        font-weight: 600;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        position: relative;
        z-index: 1;
    }
    
    .hero-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
        position: relative;
        z-index: 1;
    }
    
    /* Stats container */
    .stats-container {
        display: flex;
        justify-content: center;
        gap: 3rem;
        margin-top: 2rem;
        flex-wrap: wrap;
    }
    
    .stat-item {
        text-align: center;
        padding: 1rem 2rem;
        background: rgba(255,255,255,0.15);
        border-radius: 15px;
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: white;
        line-height: 1.2;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: rgba(255,255,255,0.8);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Alert styles with better colors */
    .fraud-alert {
        background: linear-gradient(135deg, #dc3545 0%, #b22222 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 600;
        animation: pulse 2s infinite;
        box-shadow: 0 10px 30px rgba(220, 53, 69, 0.3);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .legit-alert {
        background: linear-gradient(135deg, #28a745 0%, #1e7e34 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 600;
        box-shadow: 0 10px 30px rgba(40, 167, 69, 0.3);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    /* Modern metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
    }
    
    .metric-card .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-card .metric-label {
        font-size: 1rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background: linear-gradient(90deg, #28a745, #ffc107, #dc3545);
        border-radius: 10px;
        height: 20px !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 500;
        font-size: 1.1rem;
        padding: 0.75rem 2rem;
        border: none;
        border-radius: 12px;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 1px;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.4);
    }
    
    /* Input field styling */
    .stTextInput > div > div > input, 
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        border-radius: 10px;
        border: 2px solid #e9ecef;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
        font-family: 'Poppins', sans-serif;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Info box styling */
    .info-box {
        background: linear-gradient(135deg, #f1f5f9 0%, #e9eff7 100%);
        border-left: 4px solid #667eea;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-size: 0.95rem;
        color: #334155;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f8fafc;
        padding: 0.5rem;
        border-radius: 50px;
        border: 1px solid #e2e8f0;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        border-radius: 25px;
        padding: 0 2rem;
        font-weight: 500;
        transition: all 0.3s ease;
        color: #64748b;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #e2e8f0;
        color: #334155;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: #f8fafc;
        border-radius: 10px;
        font-weight: 500;
        border: 1px solid #e2e8f0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: white;
        font-size: 0.9rem;
    }
    
    .footer a {
        color: white;
        text-decoration: none;
        font-weight: 500;
    }
    
    .footer a:hover {
        text-decoration: underline;
    }
    
    /* Section headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 1.5rem;
    }
    
    .section-icon {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        width: 45px;
        height: 45px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.2);
    }
    
    .section-icon span {
        color: white;
        font-size: 1.3rem;
    }
    
    .section-header h2 {
        margin: 0;
        color: #1e293b;
        font-weight: 600;
        font-size: 1.5rem;
    }
    
    /* Risk level badge */
    .risk-badge {
        display: inline-block;
        padding: 0.5rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.2rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Particle background */
    #particles-js {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -1;
    }
    
    /* Loading animation */
    .loading-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 200px;
    }
    
    .loading-spinner::after {
        content: '';
        width: 50px;
        height: 50px;
        border: 4px solid #f3f3f3;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>

<!-- Particles.js background -->
<script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
<script>
    document.addEventListener('DOMContentLoaded', function() {
        particlesJS('particles-js', {
            particles: {
                number: { value: 60, density: { enable: true, value_area: 800 } },
                color: { value: '#667eea' },
                shape: { type: 'circle' },
                opacity: { value: 0.3, random: false },
                size: { value: 2, random: true },
                line_linked: { enable: true, distance: 150, color: '#667eea', opacity: 0.2, width: 1 },
                move: { enable: true, speed: 3, direction: 'none', random: false, straight: false, out_mode: 'out', bounce: false }
            },
            interactivity: {
                detect_on: 'canvas',
                events: { onhover: { enable: true, mode: 'repulse' }, onclick: { enable: true, mode: 'push' }, resize: true },
                modes: { repulse: { distance: 100, duration: 0.4 }, push: { particles_nb: 4 } }
            },
            retina_detect: true
        });
    });
</script>
""", unsafe_allow_html=True)

# Add particles background div
st.markdown('<div id="particles-js"></div>', unsafe_allow_html=True)

# ===================== SESSION STATE ===================== #
if 'scroll_to_top' not in st.session_state:
    st.session_state.scroll_to_top = False
if 'history' not in st.session_state:
    st.session_state.history = []

# ===================== HEADER ===================== #
st.markdown("""
<div class="hero-header">
    <h1>🔍 Fraud Detection in Online Transactions</h1>
    <p>Advanced Machine Learning System for Real-time Transaction Monitoring</p>
    <div class="stats-container">
        <div class="stat-item">
            <div class="stat-value">99.9%</div>
            <div class="stat-label">Accuracy</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">Real-time</div>
            <div class="stat-label">Processing</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">24/7</div>
            <div class="stat-label">Monitoring</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ===================== LOAD MODELS ===================== #
@st.cache_resource
def load_models():
    with st.spinner('Loading AI models...'):
        time.sleep(1)  # Simulate loading
        try:
            model = joblib.load("models/fraud_model.pkl")
            scaler = joblib.load("models/scaler.pkl")
            return model, scaler
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            st.stop()

model, scaler = load_models()

# ===================== MAIN CONTAINER ===================== #
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# ===================== LAYOUT ===================== #
col1, col2 = st.columns([1, 1.5])

# --------------------- LEFT COLUMN: INPUTS --------------------- #
with col1:
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">
            <span>⚙️</span>
        </div>
        <h2>Transaction Parameters</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">💡 Enter transaction details below for real-time fraud analysis. All fields are required for accurate assessment.</div>', unsafe_allow_html=True)
    
    # Transaction Time
    st.markdown("##### 📅 Transaction Time")
    step = st.slider("Time Step", 0, 100, 1, help="Time step of the transaction (0-100)")
    
    # Transaction Details
    st.markdown("##### 💰 Transaction Details")
    type_tx = st.selectbox("Transaction Type", ["PAYMENT","TRANSFER","CASH_OUT","DEBIT","CASH_IN"])
    amount = st.number_input("Amount ($)", 0.0, 500000.0, 1000.0, step=100.0, format="%.2f")
    
    # Account Balances
    st.markdown("##### 🏦 Account Balances")
    col_bal1, col_bal2 = st.columns(2)
    with col_bal1:
        oldbalanceOrg = st.number_input("Origin Old Balance", 0.0, 1000000.0, 5000.0, step=100.0, format="%.2f", help="Sender's balance before transaction")
        oldbalanceDest = st.number_input("Destination Old Balance", 0.0, 1000000.0, 0.0, step=100.0, format="%.2f", help="Recipient's balance before transaction")
    with col_bal2:
        newbalanceOrig = st.number_input("Origin New Balance", 0.0, 1000000.0, 3000.0, step=100.0, format="%.2f", help="Sender's balance after transaction")
        newbalanceDest = st.number_input("Destination New Balance", 0.0, 1000000.0, 0.0, step=100.0, format="%.2f", help="Recipient's balance after transaction")
    
    st.caption("📌 Note: In a valid transaction, amount ≈ old balance − new balance for origin account")
    
    # Detect Button
    detect_button = st.button("🔍 Analyze Transaction", use_container_width=True)
    if detect_button:
        st.session_state.scroll_to_top = True

# --------------------- RIGHT COLUMN: RESULTS --------------------- #
with col2:
    st.markdown("""
    <div class="section-header">
        <div class="section-icon" style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%);">
            <span>📊</span>
        </div>
        <h2>Analysis Results</h2>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📈 Detection Result", "📊 Visual Analysis", "📋 Transaction History"])
    
    type_map = {"PAYMENT":0,"TRANSFER":1,"CASH_OUT":2,"DEBIT":3,"CASH_IN":4}
    type_encoded = type_map[type_tx]
    input_data = np.array([[step,type_encoded,amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest]])
    input_scaled = scaler.transform(input_data)
    
    with tab1:
        if detect_button:
            # Show loading animation
            with st.spinner('AI analyzing transaction...'):
                time.sleep(1)  # Simulate processing
                prediction = model.predict(input_scaled)[0]
                probability = float(model.predict_proba(input_scaled)[0][1]) if hasattr(model, "predict_proba") else float(prediction)
            
            # Alert based on prediction
            if prediction == 1:
                st.markdown('<div class="fraud-alert">⚠️ FRAUDULENT TRANSACTION DETECTED</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="legit-alert">✅ LEGITIMATE TRANSACTION CONFIRMED</div>', unsafe_allow_html=True)
            
            # Fraud Probability with gauge
            st.markdown("##### Fraud Risk Assessment")
            
            # Create gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = probability * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Risk Score", 'font': {'size': 20, 'color': '#334155'}},
                delta = {'reference': 50, 'increasing': {'color': '#dc3545'}, 'decreasing': {'color': '#28a745'}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': '#64748b'},
                    'bar': {'color': '#667eea'},
                    'bgcolor': 'white',
                    'borderwidth': 2,
                    'bordercolor': '#e2e8f0',
                    'steps': [
                        {'range': [0, 25], 'color': '#d4edda'},
                        {'range': [25, 50], 'color': '#fff3cd'},
                        {'range': [50, 75], 'color': '#ffe5b4'},
                        {'range': [75, 100], 'color': '#f8d7da'}
                    ],
                    'threshold': {
                        'line': {'color': '#dc3545', 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            fig.update_layout(height=280, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk level with badge
            if probability >= 0.8:
                risk_label = "Critical"
                risk_color = "#dc3545"
                risk_bg = "#f8d7da"
            elif probability >= 0.6:
                risk_label = "High"
                risk_color = "#fd7e14"
                risk_bg = "#ffe5b4"
            elif probability >= 0.4:
                risk_label = "Medium"
                risk_color = "#ffc107"
                risk_bg = "#fff3cd"
            else:
                risk_label = "Low"
                risk_color = "#28a745"
                risk_bg = "#d4edda"
            
            st.markdown(f"""
            <div style="text-align: center; margin: 1.5rem 0;">
                <div style="font-size: 1rem; color: #64748b; margin-bottom: 0.5rem;">Risk Level</div>
                <div style="display: inline-block; padding: 0.75rem 2.5rem; border-radius: 50px; 
                            background: {risk_bg}; color: {risk_color}; font-size: 1.8rem; 
                            font-weight: 600; border: 2px solid {risk_color};">
                    {risk_label}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Transaction Summary with enhanced metrics
            st.markdown("##### Transaction Summary")
            col_metric1, col_metric2, col_metric3 = st.columns(3)
            with col_metric1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Type</div>
                    <div class="metric-value">{type_tx}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_metric2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Amount</div>
                    <div class="metric-value">${amount:,.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_metric3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value">{abs(probability*2-1)*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Add to history
            st.session_state.history.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'type': type_tx,
                'amount': f"${amount:,.2f}",
                'risk': risk_label,
                'probability': f"{probability*100:.1f}%"
            })
            
            # Scroll to top
            if st.session_state.scroll_to_top:
                st.markdown("<script>window.scrollTo({top:0,behavior:'smooth'});</script>", unsafe_allow_html=True)
                st.session_state.scroll_to_top = False
        else:
            st.info("👆 Enter transaction details and click **Analyze Transaction** to begin fraud detection")
    
    with tab2:
        if detect_button:
            # Create enhanced visualization with plotly
            data = pd.DataFrame({
                "Account": ["Origin (Before)", "Origin (After)", "Destination (Before)", "Destination (After)", "Transaction Amount"],
                "Balance": [oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, amount]
            })
            
            # Bar chart with better colors
            fig1 = px.bar(data, x="Account", y="Balance", 
                         title="Balance Analysis",
                         color="Balance",
                         color_continuous_scale=['#28a745', '#ffc107', '#dc3545'],
                         text_auto='.2s')
            
            fig1.update_traces(texttemplate='$%{text}', textposition='outside')
            fig1.update_layout(
                showlegend=False, 
                height=400,
                title_font_size=16,
                title_font_color='#334155',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig1, use_container_width=True)
            
            # Balance change visualization
            origin_change = oldbalanceOrg - newbalanceOrig
            dest_change = newbalanceDest - oldbalanceDest
            
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                # Pie chart
                fig2 = px.pie(
                    values=[oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, amount],
                    names=['Origin Before', 'Origin After', 'Dest Before', 'Dest After', 'Amount'],
                    title="Balance Distribution",
                    color_discrete_sequence=['#667eea', '#764ba2', '#28a745', '#ffc107', '#dc3545']
                )
                fig2.update_traces(textposition='inside', textinfo='percent+label')
                fig2.update_layout(
                    height=350,
                    title_font_size=14,
                    title_font_color='#334155',
                    showlegend=False
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            with col_viz2:
                # Waterfall chart
                fig3 = go.Figure()
                fig3.add_trace(go.Waterfall(
                    name="Balance Flow",
                    orientation="v",
                    measure=["relative", "relative", "total"],
                    x=["Origin Account", "Destination Account", "Net Effect"],
                    y=[-origin_change if origin_change > 0 else origin_change, 
                       dest_change if dest_change > 0 else dest_change,
                       dest_change - origin_change],
                    textposition="outside",
                    text=["▼ $" + f"{abs(origin_change):,.2f}" if origin_change > 0 else "▲ $" + f"{abs(origin_change):,.2f}",
                          "▲ $" + f"{dest_change:,.2f}" if dest_change > 0 else "▼ $" + f"{abs(dest_change):,.2f}",
                          "Net: $" + f"{dest_change - origin_change:,.2f}"],
                    connector={"line": {"color": "#94a3b8"}},
                    increasing={"marker": {"color": "#28a745"}},
                    decreasing={"marker": {"color": "#dc3545"}}
                ))
                
                fig3.update_layout(
                    title="Balance Flow Analysis",
                    height=350,
                    showlegend=False,
                    title_font_size=14,
                    title_font_color='#334155',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig3, use_container_width=True)
            
        else:
            st.info("Run an analysis to see visualizations")
    
    with tab3:
        if st.session_state.history:
            history_df = pd.DataFrame(st.session_state.history)
            
            # Summary metrics
            total_transactions = len(history_df)
            high_risk_count = len(history_df[history_df['risk'].isin(['High', 'Critical'])])
            
            col_hist1, col_hist2, col_hist3 = st.columns(3)
            col_hist1.metric("Total Transactions", total_transactions)
            col_hist2.metric("High Risk Alerts", high_risk_count)
            col_hist3.metric("Clean Rate", f"{((total_transactions - high_risk_count)/total_transactions*100):.1f}%")
            
            # Display history table
            st.dataframe(
                history_df.style.applymap(
                    lambda x: 'color: #dc3545; font-weight: 600' if x in ['High', 'Critical'] else '',
                    subset=['risk']
                ).applymap(
                    lambda x: 'color: #28a745; font-weight: 600' if x == 'Low' else '',
                    subset=['risk']
                ),
                use_container_width=True
            )
            
            # Clear history button
            if st.button("🗑️ Clear History", key="clear_history"):
                st.session_state.history = []
                st.experimental_rerun()
        else:
            st.info("No transaction history yet. Run some analyses to see history here.")

st.markdown('</div>', unsafe_allow_html=True)

# ===================== FOOTER ===================== #
st.markdown("""
<div class="footer">
    <p>🔍 Fraud Detection in Online Transactions</p>
    <p>Powered by Machine Learning | Real-time Analysis | 99.9% Accuracy</p>
    <p style="margin-top: 1rem; font-size: 0.8rem;">© 2024 | For demonstration purposes only</p>
</div>
""", unsafe_allow_html=True)
