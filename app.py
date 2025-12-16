import streamlit as st
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
import torch

# Import your modules
from data_analyzer import DatasetAnalyzer
from brain import MetaLearner
from engine import DynamicTrainer

# === PAGE CONFIGURATION ===
st.set_page_config(
    page_title="MetaTune AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === CUSTOM CSS FOR "OUT OF WORLD" LOOK ===
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        border-radius: 10px;
    }
    .metric-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #41444C;
    }
    h1 { color: #FAFAFA; }
    h2, h3 { color: #FF4B4B; }
    </style>
    """, unsafe_allow_html=True)

# === SIDEBAR ===
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.title("MetaTune v1.0")
    st.markdown("### **AI-Driven Hyperparameter Optimization**")
    st.markdown("---")
    st.info("💡 **System Status:** Online")
    st.info("🛡️ **Bilevel Optimization:** Active")
    st.info("🧠 **Meta-Learning Core:** Ready")

# === MAIN LAYOUT ===
st.title("🧠 MetaTune: Dataset-Aware Optimization")
st.markdown("### Upload your dataset to activate the Neural Brain")

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:
    # Save file temporarily
    with open("temp_data.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # === PHASE 1: DIAGNOSIS ===
    st.markdown("---")
    st.header("1. 🧬 Forensic Data Diagnosis")
    
    with st.spinner('Scanning dataset DNA...'):
        analyzer = DatasetAnalyzer("temp_data.csv")
        analyzer.load_data()
        dna = analyzer.analyze()
        time.sleep(1) # Dramatic pause for effect
    
    # Display DNA Metrics in Columns
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows (Instances)", dna['n_instances'])
    col2.metric("Features", dna['n_features'])
    col3.metric("Complexity (Entropy)", f"{dna['target_entropy']:.3f}")
    col4.metric("Noise (Imbalance)", f"{dna['class_imbalance_ratio']:.2f}")
    
    # Show "Brain" Logic
    if dna['target_entropy'] > 1.0:
        st.warning(f"⚠️ High Entropy Detected ({dna['target_entropy']:.2f}). MetaTune will enforce stricter regularization.")
    else:
        st.success("✅ Dataset Stability Confirmed. Standard optimization protocols engaged.")

    # === PHASE 2: PRESCRIPTION ===
    st.markdown("---")
    st.header("2. 🧠 Neural Hyperparameter Prediction")
    
    if st.button("Query Meta-Learner"):
        with st.spinner('Meta-Brain is calculating optimal geometry...'):
            brain = MetaLearner()
            # Check for saved brain
            if not os.path.exists("meta_brain.pkl"):
                brain.train(epochs=10) # Quick boot for demo
            else:
                brain = MetaLearner.load("meta_brain.pkl")
            
            params = brain.predict(dna)
            time.sleep(1.5) # Thinking time
            
        # Display Predicted Params
        st.markdown("#### ✨ AI-Generated Configuration")
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        p_col1.metric("Learning Rate", f"{params['learning_rate']:.5f}")
        p_col2.metric("L2 Regularization", f"{params['weight_decay_l2']:.5f}")
        p_col3.metric("Batch Size", params['batch_size'])
        p_col4.metric("Optimizer", params['optimizer_type'].upper())
        
        # Save params to session state for next step
        st.session_state['params'] = params
        st.session_state['ready_to_train'] = True

    # === PHASE 3: EXECUTION ===
    if st.session_state.get('ready_to_train'):
        st.markdown("---")
        st.header("3. 🚀 Dynamic Bilevel Training")
        
        if st.button("Start Training Engine"):
            # Containers for real-time updates
            chart_col, log_col = st.columns([2, 1])
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Chart placeholders
            with chart_col:
                chart_placeholder = st.empty()
            
            # Real-time data storage
            loss_data = {"epoch": [], "train_loss": [], "val_loss": []}
            
            # Callback function to update UI
            def update_ui(epoch, total_epochs, t_loss, v_loss, metric):
                progress = float(epoch) / total_epochs
                progress_bar.progress(progress)
                status_text.markdown(f"**Epoch {epoch}/{total_epochs}** | Accuracy: **{metric:.2%}**")
                
                # Update Chart
                loss_data["epoch"].append(epoch)
                loss_data["train_loss"].append(t_loss)
                loss_data["val_loss"].append(v_loss)
                
                chart_df = pd.DataFrame(loss_data).set_index("epoch")
                chart_placeholder.line_chart(chart_df)

            # Run Engine
            trainer = DynamicTrainer(
                "temp_data.csv", 
                dna, 
                st.session_state['params'], 
                progress_callback=update_ui
            )
            
            result = trainer.run(epochs=30)
            
            st.success(f"🏆 Training Complete! Final Accuracy: {result['final_metric']:.4f}")
            st.balloons()

            # === NEW: SAVE & DOWNLOAD MODEL ===
            # 1. Save the model locally
            torch.save(trainer.model.state_dict(), "trained_model.pth")
            
            # 2. Create a download button
            with open("trained_model.pth", "rb") as f:
                st.download_button(
                    label="💾 Download Trained Model (.pth)",
                    data=f,
                    file_name="my_metatune_model.pth",
                    mime="application/octet-stream"
                )
