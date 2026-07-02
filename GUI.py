import gradio as gr
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载主动脉钙化预测逻辑回归模型
model = joblib.load("model.pkl")

# 预测函数，仅2个自变量
def predict_aorta_calc(Hemoglobin, af_CHADSVA):
    input_df = pd.DataFrame(
        [[Hemoglobin, af_CHADSVA]],
        columns=["Hemoglobin", "af_CHADSVA"]
    )
    # 预测分类+钙化发生概率
    pred_label = model.predict(input_df)[0]
    pred_prob = model.predict_proba(input_df)[0][1]

    # SHAP线性模型特征贡献解释
    explainer = shap.LinearExplainer(model, input_df)
    shap_vals = explainer.shap_values(input_df)

    if pred_label == 1:
        result = f"高度提示存在主动脉钙化，钙化发生概率：{pred_prob:.2%}"
    else:
        result = f"暂未提示主动脉钙化，钙化发生概率：{pred_prob:.2%}"
    return result

# 本地可视化界面
demo = gr.Interface(
    fn=predict_aorta_calc,
    inputs=[
        gr.Number(label="血红蛋白(g/L)", value=125, minimum=50, maximum=180),
        gr.Number(label="房颤卒中CHA₂DS₂-VA评分", value=2, minimum=0, maximum=9)
    ],
    outputs="text",
    title="主动脉钙化风险预测 - 本地GUI工具",
    description="基于逻辑回归模型，纳入血红蛋白、房颤卒中评分双指标预测主动脉钙化发生风险"
)
demo.launch()
