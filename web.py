import streamlit as st
import joblib
import pandas as pd


# 网页基础配置
st.set_page_config(page_title="主动脉钙化预测系统", layout="wide")
st.title("基于逻辑回归的主动脉钙化风险预测与SHAP可解释分析")

# 缓存加载训练好的逻辑回归模型
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")
lr_model = load_model()

# 侧边栏录入两个预测变量
with st.sidebar:
    st.header("患者临床指标录入")
    hemoglobin = st.number_input(
        "血红蛋白 (g/L)",
        min_value=50.0, max_value=180.0, value=122.0, step=1.0
    )
    stroke_score = st.number_input(
        "房颤卒中CHA₂DS₂-VA评分",
        min_value=0, max_value=9, value=2, step=1
    )

# 构造模型输入格式
input_data = pd.DataFrame(
    [[hemoglobin, stroke_score]],
    columns=["hemoglobin", "stroke_score"]
)

# 预测触发按钮
if st.button("评估主动脉钙化风险"):
    pred = lr_model.predict(input_data)[0]
    calc_prob = lr_model.predict_proba(input_data)[0][1]

    # 风险结果展示
    if pred == 1:
        st.error(f"⚠️ 预测存在主动脉钙化，钙化发生概率：{calc_prob:.2%}")
    else:
        st.success(f"✅ 预测无主动脉钙化，钙化发生概率：{calc_prob:.2%}")


