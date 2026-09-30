import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# 1. ตั้งค่าหน้าเพจ Streamlit
st.set_page_config(
    page_title="Dark Mode ML Prediction Lab",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. ปรับแต่ง CSS แก้ไขสีข้อความใน Selectbox และ Button ให้คมชัด
st.markdown("""
<style>
    /* พื้นหลังหลักของเว็บ */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* สไตล์ Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1e293b !important;
        border-right: 1px solid #334155;
    }
    
    /* บังคับให้ Label, Text, Caption ทุกตัวใน Sidebar เป็นสีสว่าง คมชัด */
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: #f1f5f9 !important;
        font-weight: 500;
    }
    
    /* แก้ไขข้อความในช่อง Dropdown (Selectbox) และ เมนูตัวเลือก ให้เป็นสีเข้ม อ่านง่าย */
    div[data-baseweb="select"] * {
        color: #0f172a !important;
        font-weight: 600 !important;
    }
    
    /* FIX: เจาะจงแก้สีข้อความและไอคอนบนปุ่มกด (Button) ทั้งหมด ให้เป็นสีเข้ม อ่านง่ายชัดเจน */
    div[data-testid="stButton"] button,
    div[data-testid="stButton"] button *,
    section[data-testid="stSidebar"] button,
    section[data-testid="stSidebar"] button * {
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    
    /* ปรับแต่งเอฟเฟกต์ตอนเอาเมาส์ไปชี้ปุ่ม (Hover) */
    div[data-testid="stButton"] button:hover,
    div[data-testid="stButton"] button:hover * {
        color: #0284c7 !important;
        border-color: #38bdf8 !important;
    }
    
    /* ข้อความ Caption ใน Sidebar ให้เป็นสีเทาสว่าง */
    section[data-testid="stSidebar"] .stCaption, 
    section[data-testid="stSidebar"] small {
        color: #94a3b8 !important;
    }

    /* ปรับแต่ง Slider ให้เห็นตัวเลขและแถบสีชัดเจน */
    div[data-baseweb="slider"] div {
        color: #38bdf8 !important;
    }
    
    /* การ์ดแสดงผล Prediction Result Card */
    .prediction-card-dark {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
        border: 2px solid #6366f1;
        color: #ffffff;
        padding: 24px;
        border-radius: 18px;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.4);
        margin-bottom: 25px;
    }
    
    .prediction-title-dark {
        font-size: 14px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #cbd5e1 !important;
        margin-bottom: 6px;
    }
    
    .prediction-value-dark {
        font-size: 42px;
        font-weight: 800;
        color: #fbbf24 !important;
        margin: 6px 0;
        text-shadow: 0 0 10px rgba(251, 191, 36, 0.3);
    }
    
    .confidence-badge-dark {
        display: inline-block;
        background-color: rgba(56, 189, 248, 0.2);
        border: 1px solid #38bdf8;
        color: #38bdf8 !important;
        padding: 6px 18px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 700;
    }

    /* Container และ Card บล็อกสไตล์ดาร์กโมด */
    div[data-testid="stForm"], div[data-testid="stVerticalBlock"] > div[data-testid="stBlock"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 16px;
    }

    /* หัวข้อหลักทั้งหมด */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. ฟังก์ชันสำหรับโหลดข้อมูลและสร้างโมเดล
@st.cache_data
def get_iris_model():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"])
    y = iris.target
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X, y)
    return model, X, iris.target_names

@st.cache_data
def get_credit_model():
    np.random.seed(42)
    n_samples = 300
    income = np.random.uniform(15000, 100000, n_samples)
    credit_score = np.random.uniform(300, 850, n_samples)
    debt_ratio = np.random.uniform(0.1, 0.8, n_samples)
    loan_amount = np.random.uniform(50000, 500000, n_samples)
    
    score = (income / 1000) + (credit_score / 10) - (debt_ratio * 100) - (loan_amount / 20000)
    y = np.where(score > 35, 1, 0)
    
    X = pd.DataFrame({
        "Monthly Income ($)": income,
        "Credit Score": credit_score,
        "Debt Ratio (%)": debt_ratio * 100,
        "Loan Amount ($)": loan_amount
    })
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model, X, ["High Risk (Rejected)", "Low Risk (Approved)"]

# ------------------- SIDEBAR: CONTROLS & INPUTS -------------------
with st.sidebar:
    st.title("🌙 Model Settings")
    st.caption("Custom Input & Control Panel")
    
    dataset_mode = st.selectbox(
        "📂 เลือกชุดข้อมูล/โหมดโมเดล (Select Model Task):",
        ["Iris Flower Classification 🌸", "Credit Approval Predictor 💳"]
    )
    
    st.markdown("---")
    st.subheader("🎛️ Adjust Input Parameters")

    # ----- กรณีเลือกโหมด 1: Iris Dataset -----
    if "Iris" in dataset_mode:
        model, X_df, target_names = get_iris_model()
        mean_values = X_df.mean().to_dict()
        
        if "iris_v1" not in st.session_state:
            st.session_state.iris_v1 = 6.2
            st.session_state.iris_v2 = 3.0
            st.session_state.iris_v3 = 4.8
            st.session_state.iris_v4 = 1.8

        if st.button("🎲 สุ่มค่าอัตโนมัติ (Random Sample)", use_container_width=True):
            st.session_state.iris_v1 = float(np.round(np.random.uniform(4.0, 8.0), 1))
            st.session_state.iris_v2 = float(np.round(np.random.uniform(2.0, 4.5), 1))
            st.session_state.iris_v3 = float(np.round(np.random.uniform(1.0, 7.0), 1))
            st.session_state.iris_v4 = float(np.round(np.random.uniform(0.1, 2.5), 1))

        v1 = st.slider("Sepal Length (cm)", 4.0, 8.0, key="iris_v1", step=0.1)
        v2 = st.slider("Sepal Width (cm)", 2.0, 4.5, key="iris_v2", step=0.1)
        v3 = st.slider("Petal Length (cm)", 1.0, 7.0, key="iris_v3", step=0.1)
        v4 = st.slider("Petal Width (cm)", 0.1, 2.5, key="iris_v4", step=0.1)
        
        feature_names = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
        user_input_values = [v1, v2, v3, v4]

    # ----- กรณีเลือกโหมด 2: Credit Approval -----
    else:
        model, X_df, target_names = get_credit_model()
        mean_values = X_df.mean().to_dict()
        
        if "cred_v1" not in st.session_state:
            st.session_state.cred_v1 = 45000.0
            st.session_state.cred_v2 = 680.0
            st.session_state.cred_v3 = 35.0
            st.session_state.cred_v4 = 150000.0

        if st.button("🎲 สุ่มค่าอัตโนมัติ (Random Sample)", use_container_width=True):
            st.session_state.cred_v1 = float(np.round(np.random.uniform(15000, 100000), -2))
            st.session_state.cred_v2 = float(np.round(np.random.uniform(300, 850), 0))
            st.session_state.cred_v3 = float(np.round(np.random.uniform(10, 80), 1))
            st.session_state.cred_v4 = float(np.round(np.random.uniform(50000, 500000), -3))

        v1 = st.slider("Monthly Income ($)", 15000.0, 100000.0, key="cred_v1", step=1000.0)
        v2 = st.slider("Credit Score", 300.0, 850.0, key="cred_v2", step=5.0)
        v3 = st.slider("Debt Ratio (%)", 10.0, 80.0, key="cred_v3", step=0.5)
        v4 = st.slider("Loan Amount ($)", 50000.0, 500000.0, key="cred_v4", step=5000.0)
        
        feature_names = ["Monthly Income ($)", "Credit Score", "Debt Ratio (%)", "Loan Amount ($)"]
        user_input_values = [v1, v2, v3, v4]

    st.markdown("---")
    st.caption("🎨 Theme: **High-Contrast Dark Mode**")

# ------------------- MAIN CONTENT SECTION -------------------
st.title("Interactive ML Lab 🚀")
st.write("ปรับเปลี่ยนพารามิเตอร์แบบเรียลไทม์ โทนสีเข้มถนอมสายตา พร้อมคำนวณเปรียบเทียบกับค่าเฉลี่ยในระบบ")

# คำนวณผลการทำนาย
input_array = np.array([user_input_values])
prediction_idx = model.predict(input_array)[0]
prediction_label = target_names[prediction_idx]
probabilities = model.predict_proba(input_array)[0]
confidence = probabilities[prediction_idx] * 100

st.write("")

# Layout แบ่ง 2 คอลัมน์หลัก
col_left, col_right = st.columns([1, 1], gap="medium")

# --- คอลัมน์ซ้าย: PREDICTION RESULT & PROBABILITIES ---
with col_left:
    with st.container(border=True):
        st.caption("🎯 PREDICTION RESULT")
        
        st.markdown(f"""
        <div class="prediction-card-dark">
            <div class="prediction-title-dark">PREDICTED CLASS</div>
            <div class="prediction-value-dark">{prediction_label}</div>
            <div class="confidence-badge-dark">⚡ Confidence: {confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("MODEL PROBABILITY BY CLASS")
        st.write("ความน่าจะเป็นของแต่ละผลลัพธ์:")
        
        for idx, name in enumerate(target_names):
            prob_percent = probabilities[idx] * 100
            st.write(f"**{name}**: `{prob_percent:.1f}%`")
            st.progress(int(prob_percent))

# --- คอลัมน์ขวา: COMPARISON GRAPH ---
with col_right:
    with st.container(border=True):
        st.caption("📊 USER INPUTS VS DATASET AVERAGE")
        st.write("กราฟเปรียบเทียบข้อมูลที่คุณปรับแต่งเองกับค่าเฉลี่ยของระบบ")
        
        avg_values = [mean_values[feat] for feat in feature_names]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=feature_names,
            y=user_input_values,
            name='Your Custom Values',
            marker_color='#38bdf8',
            width=0.35
        ))

        fig.add_trace(go.Bar(
            x=feature_names,
            y=avg_values,
            name='Dataset Average',
            marker_color='#818cf8',
            width=0.35
        ))

        fig.update_layout(
            barmode='group',
            height=390,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color="#ffffff", size=13)
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#f8fafc", size=12),
            xaxis=dict(gridcolor='#334155', tickfont=dict(color="#f8fafc")),
            yaxis=dict(gridcolor='#334155', tickfont=dict(color="#f8fafc"))
        )

        st.plotly_chart(fig, use_container_width=True)