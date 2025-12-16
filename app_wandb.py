import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import plotly.graph_objects as go
import plotly.express as px
import torch

# Import Modules
from data_analyzer import DatasetAnalyzer
from brain import MetaLearner
from engine_stream import DynamicTrainerStream

# === PAGE CONFIGURATION ===
st.set_page_config(
    page_title="MetaTune Mission Control",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === WANDB STYLE CSS ===
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Card Styling */
    .css-1r6slb0, .css-12oz5g7, .css-1ec096l {
        border-radius: 8px;
        padding: 20px;
        background-color: #2b303b;
        border: 1px solid #41444C;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        color: #00FFFF; /* Cyan */
    }
    div[data-testid="stMetricLabel"] {
        color: #B0B3B8;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #FAFAFA;
        font-family: 'Inter', sans-serif;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #00FFFF;
        color: #1a1c24;
        font-weight: bold;
        border-radius: 6px;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #00DDDD;
        color: #000000;
        transform: scale(1.02);
    }
    
    /* Custom Card Class */
    .wandb-card {
        background-color: #1a1c24; /* Darker for contrast */
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #2b303b;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# === SIDEBAR ===
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png", width=50)
    st.title("MetaTune")
    st.caption("AutoML Mission Control")
    st.markdown("---")
    
    st.subheader("⚙️ Run Configuration")
    run_name = st.text_input("Run Name", value=f"experiment-{int(time.time())}")
    st.info("System Ready")
    
    # Progress placeholder
    sidebar_progress = st.empty()

# === MAIN INTERFACE ===
col_logo, col_title = st.columns([1, 10])
with col_title:
    st.title("Mission Control Center")

uploaded_file = st.file_uploader("Iniitalize Datastream (CSV)", type=["csv"], help="Drop your dataset here to begin analysis")

if uploaded_file:
    # Save file
    with open("temp_data_stream.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.markdown("---")
    
    # === PHASE 1: DATA INTELLIGENCE (RADAR CHART) ===
    col_radar, col_stats = st.columns([1, 2])
    
    with st.spinner("Analyzing Dataset DNA..."):
        analyzer = DatasetAnalyzer("temp_data_stream.csv")
        analyzer.load_data()
        dna = analyzer.analyze()
        
    with col_radar:
        # Normalize for Radar Chart (0-1 scale for visualisation)
        categories = ['Sparsity', 'Entropy', 'Skewness', 'Imbalance', 'Dimensionality']
        values = [
            dna['sparsity'], 
            min(dna['target_entropy']/2.0, 1.0), # Normalize entropy approx
            min(abs(dna['mean_skewness'])/5.0, 1.0), 
            min(dna['class_imbalance_ratio']/10.0, 1.0),
            min(dna['dimensionality'], 1.0)
        ]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Dataset DNA',
            line_color='#00FFFF',
            fillcolor='rgba(0, 255, 255, 0.2)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, linecolor='#41444C'),
                bgcolor='#1a1c24'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            margin=dict(l=40, r=40, t=20, b=20),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_stats:
        st.subheader("🧬 Dataset DNA Extract")
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Instances", dna['n_instances'], delta="Rows")
        sc2.metric("Features", dna['n_features'], delta="Cols")
        sc3.metric("Entropy", f"{dna['target_entropy']:.3f}", delta_color="inverse")
        
        st.markdown(f"""
        <div class="wandb-card">
            <b>Diagnosis:</b> { "High Complexity" if dna['target_entropy'] > 1.0 else "Standard Complexity" } <br>
            <b>Task Type:</b> {dna['task_type'].upper()} <br>
            <b>Recommendation:</b> {"Deep Network" if dna['target_entropy'] > 1.0 else "Shallow Network"}
        </div>
        """, unsafe_allow_html=True)
    
    # === PHASE 2: HYPERPARAMETER TUNING ===
    st.markdown("### 🧠 Neural Configuration")
    
    if 'params' not in st.session_state:
        brain = MetaLearner()
        if os.path.exists("meta_brain.pkl"):
            brain = MetaLearner.load("meta_brain.pkl")
        else:
            brain.train(epochs=5) # Quick init
        st.session_state['params'] = brain.predict(dna)
        
    params = st.session_state['params']
    
    # Custom Card Layout for Params
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Learning Rate", f"{params['learning_rate']:.1e}")
    p2.metric("L2 Weight Decay", f"{params['weight_decay_l2']:.1e}")
    p3.metric("Batch Size", params['batch_size'])
    p4.metric("Dropout", f"{params['dropout']:.2f}")

    # === PHASE 3: LIVE TRAINING ===
    st.markdown("---")
    left_col, right_col = st.columns([3, 1])
    
    with right_col:
        st.markdown("<br><br>", unsafe_allow_html=True)
        start_btn = st.button("🚀 IGNITE ENGINE", use_container_width=True)
        
    if start_btn:
        st.session_state['training'] = True
        
    if st.session_state.get('training'):
        # Layout for Charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            loss_chart = st.empty()
        with chart_col2:
            acc_chart = st.empty()
            
        # Initialize Trainer
        trainer = DynamicTrainerStream("temp_data_stream.csv", dna, params)
        
        # Data Buffers
        epochs_x = []
        train_loss_y = []
        val_loss_y = []
        metric_y = []
        
        # STREAMING LOOP
        engine_gen = trainer.run(epochs=30)
        
        for epoch, total_epochs, t_loss, v_loss, metric, metric_name in engine_gen:
            # Update Data
            epochs_x.append(epoch)
            train_loss_y.append(t_loss)
            val_loss_y.append(v_loss)
            metric_y.append(metric)
            
            # --- LOSS CHART (WandB Style) ---
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(x=epochs_x, y=train_loss_y, mode='lines', name='Train Loss', line=dict(color='#00FFFF', width=3)))
            fig_loss.add_trace(go.Scatter(x=epochs_x, y=val_loss_y, mode='lines', name='Val Loss', line=dict(color='#FF00FF', width=3)))
            
            fig_loss.update_layout(
                title="<b>Loss Curves</b>",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                plot_bgcolor='#1a1c24',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                hovermode="x unified",
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig_loss.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#2b303b')
            fig_loss.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#2b303b')
            loss_chart.plotly_chart(fig_loss, use_container_width=True, key=f"loss_{epoch}")
            
            # --- METRIC CHART ---
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(x=epochs_x, y=metric_y, mode='lines+markers', name=metric_name, line=dict(color='#00FF00', width=2), fill='tozeroy', fillcolor='rgba(0, 255, 0, 0.1)'))
            
            fig_acc.update_layout(
                title=f"<b>{metric_name} Evolution</b>",
                xaxis_title="Epoch",
                yaxis_title=metric_name,
                plot_bgcolor='#1a1c24',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            fig_acc.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#2b303b')
            fig_acc.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#2b303b')
            acc_chart.plotly_chart(fig_acc, use_container_width=True, key=f"acc_{epoch}")
            
            # Sidebar Status
            sidebar_progress.progress(epoch/total_epochs)
            
            time.sleep(0.05) # Yield for rendering
            
        st.success(f"Run '{run_name}' Completed. Final {metric_name}: {metric:.4f}")
        st.balloons()
        
        # Save Model Button
        torch.save(trainer.model.state_dict(), f"{run_name}.pth")
        with open(f"{run_name}.pth", "rb") as f:
            st.download_button(
                label=f"💾 Download {run_name}.pth",
                data=f,
                file_name=f"{run_name}.pth",
                mime="application/octet-stream"
            )
