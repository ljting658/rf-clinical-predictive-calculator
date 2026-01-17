import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ==================== SCI论文级界面配置 ====================
st.set_page_config(
    page_title="Random Forest Predictive Calculator",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 自定义CSS（符合SCI论文视觉规范：简洁、无冗余、专业）
st.markdown("""
    <style>
    /* 整体样式 */
    .main {background-color: #ffffff;}
    /* 标题样式 */
    h1 {color: #1f77b4; font-weight: bold; font-size: 24px;}
    h2 {color: #1f77b4; font-weight: bold; font-size: 20px;}
    /* 输入框样式 */
    .stNumberInput label {font-weight: bold; font-size: 12px; color: #333333;}
    /* 按钮样式 */
    .stButton>button {background-color: #1f77b4; color: white; font-weight: bold;}
    /* 结果卡片样式 */
    .result-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 8px;
        border-left: 5px solid #1f77b4;
        margin-top: 20px;
    }
    /* 说明文本样式 */
    .info-text {font-size: 11px; color: #666666; line-height: 1.5;}
    </style>
""", unsafe_allow_html=True)


# ==================== 加载模型和配置 ====================
@st.cache_resource  # 缓存模型，提升性能
def load_model_and_config():
    # 替换为你的模型路径
    model_path = r"random_forest_model.pkl"
    scaler_path = r"rf_scaler.pkl"
    threshold_path = r"rf_optimal_threshold.txt"
    features_path = r"rf_features.txt"

    # 加载文件
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(threshold_path, "r") as f:
        optimal_threshold = float(f.read().strip())
    with open(features_path, "r") as f:
        features = [line.strip() for line in f.readlines()]

    return model, scaler, optimal_threshold, features


# 加载模型
model, scaler, optimal_threshold, features = load_model_and_config()

# ==================== 界面设计 ====================
# 标题（符合SCI论文命名规范）
st.title("Random Forest Predictive Calculator for NSTE-ACS")
st.markdown("---")

# 1. 变量输入区域（按论文中的特征顺序排列）
st.subheader("Input Variables")
input_data = {}
# 分3列布局，更简洁（SCI界面避免冗余）
col1, col2, col3 = st.columns(3)
for idx, feat in enumerate(features):
    # 按列分配输入框
    with [col1, col2, col3][idx % 3]:
        # 可根据论文补充变量单位/参考范围（SCI关键：提供变量说明）
        input_data[feat] = st.number_input(
            label=f"{feat}",
            value=0.0,
            step=0.01,
            help=f"Reference range: [可补充论文中的参考范围，如0-100]"
        )

# 2. 预测按钮
if st.button("Calculate Prediction"):
    # 构建输入数组（严格按特征顺序）
    input_array = np.array([[input_data[feat] for feat in features]])

    # 标准化（与训练流程一致，SCI核心：保证计算逻辑可复现）
    input_scaled = scaler.transform(input_array)

    # 模型预测（概率+分类）
    pred_prob = model.predict_proba(input_scaled)[0, 1]  # 正类概率
    pred_class = 1 if pred_prob >= optimal_threshold else 0

    # 结果解释（SCI关键：提供临床解读）
    class_interpretation = "High Risk" if pred_class == 1 else "Low Risk"
    prob_interpretation = f"Probability of NSTE-ACS: {pred_prob:.4f}"
    threshold_note = f"Optimal threshold (Youden index): {optimal_threshold:.4f}"

    # 3. 结果展示（SCI级可视化，简洁、信息完整）
    st.markdown("---")
    st.subheader("Prediction Result")
    st.markdown(f"""
    <div class="result-card">
        <p style='font-size:16px; font-weight:bold;'>{class_interpretation}</p>
        <p>{prob_interpretation}</p>
        <p>{threshold_note}</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== 数据下载（SCI关键：支持结果导出） ====================
st.markdown("---")
# 构建输入数据DataFrame
input_df = pd.DataFrame([input_data])
input_df["Prediction_Probability"] = pred_prob if 'pred_prob' in locals() else np.nan
input_df["Prediction_Class"] = pred_class if 'pred_class' in locals() else np.nan

st.download_button(
    label="Download Input & Result (CSV)",
    data=input_df.to_csv(index=False).encode('utf-8'),
    file_name="rf_calculator_result.csv",
    mime="text/csv"
)