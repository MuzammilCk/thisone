import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
import os
import torch

# Import modules
from data_analyzer import DatasetAnalyzer
from brain import MetaLearner
from engine_stream import DynamicTrainer # Updated Engine

# === UI CONFIGURATION ===
st.set_page_config(page_title="MetaTune Workspace", page_icon="⚡", layout="wide")

# Custom CSS for WandB Aesthetic
st.markdown("""
<style>
    /* Dark Mode Base */
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    
    /* Metrics Cards */
    div[data-testid="stMetric"] {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #00FFFF; /* Cyan Accent */
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Headers */
    h1, h2, h3 { font-family: 'Inter', sans-serif; font-weight: 600; }
    h1 { color: #FAFAFA; }
    .highlight { color: #FFD700; }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #FF4B4B 0%, #FF00FF 100%);
        color: white;
        border: none;
        border-radius: 4px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 15px rgba(255, 0, 255, 0.5);
    }
    
    /* Plotly Chart Container */
    .js-plotly-plot {
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# === SIDEBAR: PROJECT CONFIG ===
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png", width=50)
    st.title("MetaTune / Core")
    st.markdown("Automatic Hyperparameter Optimization")
    
    st.markdown("---")
    st.markdown("### 📂 Data Ingestion")
    uploaded_file = st.file_uploader("Drop CSV Dataset", type=['csv'])
    
    target_col = st.text_input("Target Column (Optional)", help="Leave empty for auto-detection")
    
    st.markdown("---")
    st.markdown("### ⚙️ System Status")
    status_indicator = st.empty()
    status_indicator.info("System Idle")

# === MAIN WORKSPACE ===
st.title("⚡ MetaTune Experiment Dashboard")

if uploaded_file:
    # Save temp file
    with open("temp.csv", "wb") as f: f.write(uploaded_file.getbuffer())
    
    # 1. ANALYSIS ROW
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🧬 Dataset DNA")
        with st.spinner("Sequencing Genome..."):
            analyzer = DatasetAnalyzer("temp.csv", target_col if target_col else None)
            analyzer.load_data()
            dna = analyzer.analyze()
        
        # Radar Chart for DNA
        categories = ['Skewness', 'Entropy', 'Sparsity', 'Imbalance', 'Dimensionality']
        values = [
            min(dna['mean_skewness'], 5)/5, 
            min(dna['target_entropy'], 2)/2,
            dna['sparsity'],
            min(dna['class_imbalance_ratio'], 10)/10,
            min(dna['dimensionality'], 1)
        ]
        
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=values, theta=categories, fill='toself', 
            line_color='#00FFFF', fillcolor='rgba(0, 255, 255, 0.2)'
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], showticklabels=False),
                bgcolor='#1E1E1E'
            ),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color="white", margin=dict(l=20, r=20, t=20, b=20), height=250
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col2:
        st.subheader("🧠 Meta-Learner Prediction")
        brain = MetaLearner()
        if not os.path.exists("meta_brain.pkl"): brain.train(epochs=10) # Fast boot
        else: brain = MetaLearner.load("meta_brain.pkl")
        
        params = brain.predict(dna)
        
        # Metric Grid
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Learning Rate", f"{params['learning_rate']:.2e}")
        m2.metric("L2 Regularization", f"{params['weight_decay_l2']:.2e}")
        m3.metric("Batch Size", params['batch_size'])
        m4.metric("Optimizer", params['optimizer_type'].upper())
        
        st.info(f"💡 **AI Reasoning:** Detected entropy of {dna['target_entropy']:.2f}. " 
                f"{'High regularization' if dna['target_entropy'] > 1.0 else 'Standard profile'} applied.")

    # 2. TRAINING ROW (THE LIVE PART)
    st.markdown("---")
    col_btn, col_txt = st.columns([1, 4])
    with col_btn:
        start_btn = st.button("🚀 IGNITE ENGINE")
    
    if start_btn:
        status_indicator.warning("Training in Progress...")
        
        # Layout for Live Graphs
        g1, g2 = st.columns(2)
        with g1: 
            st.markdown("**📉 Loss Convergence**")
            loss_chart = st.empty()
        with g2: 
            st.markdown("**🛡️ Adaptive Regularization (Real-Time)**")
            reg_chart = st.empty()
            
        progress_bar = st.progress(0)
        
        # Data Buffers
        history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'l2': []}
        
        trainer = DynamicTrainer("temp.csv", dna, params, target_col if target_col else None)
        
        # === THE TRAINING LOOP ===
        for stats in trainer.run(epochs=30):
            # Update Data
            history['epoch'].append(stats['epoch'])
            history['train_loss'].append(stats['train_loss'])
            history['val_loss'].append(stats['val_loss'])
            history['l2'].append(stats['current_l2'])
            
            # Show Adaptation Messages
            if stats.get('adaptation'):
                st.toast(stats['adaptation'], icon="🧠")
            
            # 1. Loss Chart (Multi-line)
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(x=history['epoch'], y=history['train_loss'], 
                                        mode='lines', name='Train', line=dict(color='#00FFFF', width=2)))
            fig_loss.add_trace(go.Scatter(x=history['epoch'], y=history['val_loss'], 
                                        mode='lines', name='Val', line=dict(color='#FF00FF', width=2)))
            fig_loss.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1E1E1E',
                font_color='#B0B3B8', margin=dict(l=10, r=10, t=10, b=10), height=300,
                xaxis=dict(showgrid=False, title="Epoch"), 
                yaxis=dict(showgrid=True, gridcolor='#333', title="Loss"),
                legend=dict(orientation="h", y=1.1)
            )
            loss_chart.plotly_chart(fig_loss, use_container_width=True, key=f"loss_{stats['epoch']}")
            
            # 2. Adaptation Chart (Proving Bilevel Opt)
            fig_reg = go.Figure()
            fig_reg.add_trace(go.Scatter(x=history['epoch'], y=history['l2'], 
                                       mode='lines+markers', name='L2 Reg', 
                                       line=dict(color='#FFFF00', width=3)))
            fig_reg.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1E1E1E',
                font_color='#B0B3B8', margin=dict(l=10, r=10, t=10, b=10), height=300,
                xaxis=dict(showgrid=False, title="Epoch"), 
                yaxis=dict(showgrid=True, gridcolor='#333', title="Weight Decay Strength"),
                legend=dict(orientation="h", y=1.1)
            )
            reg_chart.plotly_chart(fig_reg, use_container_width=True, key=f"reg_{stats['epoch']}")
            
            progress_bar.progress(stats['epoch'] / 30)
            time.sleep(0.02) # Yield for rendering
            
        status_indicator.success("Optimization Complete")
        st.balloons()
        
        # Final Metrics
        st.success(f"🏆 Final Accuracy: {stats['metric']:.4f}")
        
        # Save Model Button
        torch.save(trainer.model.state_dict(), "best_model.pth")
        
        with open("best_model.pth", "rb") as f:
            st.download_button(
                label="💾 Download Trained Model (.pth)",
                data=f,
                file_name="meta_tune_model.pth",
                mime="application/octet-stream"
            )
