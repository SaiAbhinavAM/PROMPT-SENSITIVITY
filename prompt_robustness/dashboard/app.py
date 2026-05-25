import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Prompt Robustness Evaluation Dashboard", layout="wide")

results_dir = Path(__file__).parent.parent / "results"
csv_path = results_dir / "benchmark.csv"

@st.cache_data
def load_data():
    if not csv_path.exists():
        return pd.DataFrame(), None
    
    df = pd.read_csv(csv_path)
    
    json_files = list(results_dir.glob("run_*.json"))
    run_data = {}
    if json_files:
        latest_run = sorted(json_files)[-1]
        with open(latest_run, 'r', encoding='utf-8') as f:
            run_data = json.load(f)
            
    return df, run_data

df, run_data = load_data()

if df.empty:
    st.warning(f"No benchmark results found at {csv_path}. Please run `python main.py` first.")
    st.stop()

# 1. Layout & Sidebar
st.sidebar.title("Configuration")
models = df["Model"].unique().tolist()
selected_models = st.sidebar.multiselect("Select Models", models, default=models)
selected_metric = st.sidebar.selectbox("Metric to Analyze", ["PRI", "ORI", "IFI", "CS", "HS", "Consistency"])

show_sample_analysis = st.sidebar.toggle("Show per-sample analysis", value=True)
show_flags = st.sidebar.toggle("Show hallucination flags", value=True)

st.title("🛡️ Prompt Robustness Evaluation Dashboard")

# 2. Overview Section
st.header("1. Overview Metrics")

df_filtered = df[df["Model"].isin(selected_models)]
df_grouped = df_filtered # Already aggregated directly generated from benchmark.py

# Generate automatic insights for each model
insights = []
for _, row in df_grouped.iterrows():
    m = row["Model"]
    pri = row.get("PRI", 0)
    consistency = row.get("Consistency", 0)
    hs = row.get("HS", 0)
    cov = row.get("Avg_Cov", 0)
    
    msg = f"**{m}**: "
    human_score = row.get("Human_Score", 0)
    
    if human_score > 0.7:
        msg += "High Quality human-aligned scoring."
    elif human_score > 0.4:
        msg += "Moderate human-aligned scoring."
    else:
        msg += "Poor human-aligned scoring."
        
    insights.append(msg)

if insights:
    st.info("💡 **Summary Insights:**\n" + "\n".join([f"- {i}" for i in insights]))

# Sort models explicitly mapping exactly descending metrics globally:
df_grouped = df_grouped.sort_values(by="Final_Score", ascending=False) if "Final_Score" in df_grouped else df_grouped
df_filtered = df_filtered.sort_values(by="Final_Score", ascending=False) if "Final_Score" in df_filtered else df_filtered

# Metric Cards
cols1 = st.columns(4)
cols2 = st.columns(4)

avg_pri = df_filtered["PRI"].mean() if "PRI" in df_filtered else 0
avg_ori = df_filtered["ORI"].mean() if "ORI" in df_filtered else 0
avg_ifi = df_filtered["IFI"].mean() if "IFI" in df_filtered else 0
avg_cons = df_filtered["Consistency"].mean() if "Consistency" in df_filtered else 0
avg_cs = df_filtered["CS"].mean() if "CS" in df_filtered else 0
avg_hs = df_filtered["HS"].mean() if "HS" in df_filtered else 0
avg_human = df_filtered["Human_Score"].mean() if "Human_Score" in df_filtered else 0
avg_final = df_filtered["Final_Score"].mean() if "Final_Score" in df_filtered else 0

cols1[0].metric("Avg PRI", f"{avg_pri:.3f}")
cols1[1].metric("Avg ORI", f"{avg_ori:.3f}")
cols1[2].metric("Avg IFI", f"{avg_ifi:.3f}")
cols1[3].metric("Avg Consistency", f"{avg_cons:.3f}")

cols2[0].metric("Avg Correctness", f"{avg_cs:.3f}")
cols2[1].metric("Avg Hallucination", f"{avg_hs:.3f}")
cols2[2].metric("Avg Human Score", f"{avg_human:.3f}")
cols2[3].metric("Avg Final Score", f"{avg_final:.3f}")

st.header("1b. Output Tables")

st.subheader("Model Performance Summary")
summary_cols = ["Rank", "Model", "PRI", "ORI", "IFI", "CS", "HS", "Consistency", "Human_Score", "Final_Score"]
if all(c in df_filtered.columns for c in [col for col in summary_cols if col != "Rank"]):
    df_summary = df_filtered.copy()
    if "Final_Score" in df_summary.columns:
        df_summary["Rank"] = df_summary["Final_Score"].rank(ascending=False).astype(int)
        df_summary = df_summary.sort_values("Final_Score", ascending=False)
    
    st.dataframe(df_summary[summary_cols].style.format({c: "{:.3f}" for c in summary_cols if c not in ["Rank", "Model"]}), use_container_width=True)

st.subheader("Advanced Metrics")
adv_cols = ["Model", "SMS_Wasserstein", "TRD_Semantic", "KPIG_Advanced", "USD"]
if all(c in df_filtered.columns for c in adv_cols):
    st.dataframe(df_filtered[adv_cols].style.format({c: "{:.3f}" for c in adv_cols if c != "Model"}), use_container_width=True)

st.subheader("ROUGE Scores")
rouge_cols = ["Model", "ROUGE_1", "ROUGE_2", "ROUGE_L"]
if all(c in df_filtered.columns for c in rouge_cols):
    st.dataframe(df_filtered[rouge_cols].style.format({c: "{:.3f}" for c in rouge_cols if c != "Model"}), use_container_width=True)


# 3. Model Comparison
st.header("2. Model Comparison")
if "Final_Score" in df_grouped.columns:
    fig_final = px.bar(df_grouped, x="Model", y="Final_Score", title="Final Model Ranking (PRI + Human Judge)", color="Model")
    st.plotly_chart(fig_final, use_container_width=True)

metrics_to_plot = ["PRI", "ORI", "IFI", "CS", "HS", "Consistency", "Human_Score", "Final_Score"]
available_metrics = [m for m in metrics_to_plot if m in df_grouped.columns]

if available_metrics:
    # Melt dataframe for grouped bar chart
    melted_df = df_grouped.melt(id_vars=["Model"], value_vars=available_metrics, var_name="Metric", value_name="Score")
    
    fig_comp = px.bar(
        melted_df, x="Model", y="Score", color="Metric", barmode="group",
        title="Comparison across Models",
        text_auto=".2f"
    )
    st.plotly_chart(fig_comp, use_container_width=True)

# 9. Download Button
csv = df_filtered.to_csv(index=False).encode('utf-8')
st.download_button("Download Selected Results CSV", csv, "robustness_results.csv", "text/csv")


# 4 & 5 & 6 & 7. Detailed Sample Analysis
if show_sample_analysis and run_data:
    st.header("3. Detailed Sample Analysis")
    results = run_data.get("results", [])
    
    # Filter by model
    results = [r for r in results if r["model"] in selected_models]
    
    if results:
        # Group by input
        sample_texts = list(set([r["input_text"] for r in results]))
        selected_text = st.selectbox("Select Sample to Analyze", sample_texts)
        
        sample_results = [r for r in results if r["input_text"] == selected_text]
        
        ref = sample_results[0].get("reference_output", "N/A")
        with st.expander("Show Input Text & Reference"):
            st.write("**Input:**")
            st.write(selected_text)
            st.write("**Reference Summary:**")
            st.write(ref)
            
        for res in sample_results:
            pri_val = res.get("pri", 0.0)
            ori_val = res.get("ori_score", 0.0)
            ifi_val = res.get("ifi_score", 0.0)
            cs_val = res.get("cs", 0.0)
            hs_val = res.get("hs_score", 0.0)
            cons_val = res.get("consistency", 0.0)
            conf_val = res.get("confidence", 0.0)
            
            # 7. Styling PRI
            if pri_val > 0.6:
                pri_color = "🟢"
            elif pri_val > 0.3:
                pri_color = "🟡"
            else:
                pri_color = "🔴"
                
            st.subheader(f"{res['model']} {pri_color}")
            
            # Metrics
            mcols = st.columns(8)
            mcols[0].metric("PRI", f"{pri_val:.3f}")
            mcols[1].metric("ORI", f"{ori_val:.3f}")
            mcols[2].metric("IFI", f"{ifi_val:.3f}")
            mcols[3].metric("CS", f"{cs_val:.3f}")
            mcols[4].metric("Consistency", f"{cons_val:.3f}")
            mcols[5].metric("HS", f"{hs_val:.3f}")
            
            human_score = res.get("human_score", 0.0)
            final_score = res.get("final_score", 0.0)
            mcols[6].metric("Human Score", f"{human_score:.3f}")
            mcols[7].metric("Final Score", f"{final_score:.3f}")
            
            adv_cols = st.columns(4)
            sms_w = res.get("sms_wasserstein", 0.0)
            trd_s = res.get("trd_semantic", 0.0)
            kpig_a = res.get("kpig_advanced", 0.0)
            usd_val = res.get("usd", 0.0)
            adv_cols[0].metric("SMS Wasserstein", f"{sms_w:.3f}")
            adv_cols[1].metric("TRD Semantic", f"{trd_s:.3f}")
            adv_cols[2].metric("KPIG Advanced", f"{kpig_a:.3f}")
            adv_cols[3].metric("USD", f"{usd_val:.3f}")
            
            # Logs
            logs = res.get("interpretability_logs", [])
            if logs:
                for log in logs:
                    st.warning(f"⚠️ {log}")
                    
            # Outputs
            st.write("**Outputs across prompt variants:**")
            flags = res.get("flags", [])
            
            for idx, (p, r) in enumerate(zip(res.get("prompts", []), res.get("responses", []))):
                f_list = flags[idx] if idx < len(flags) else []
                # Highlight based on flags
                bg_color = "#f0f2f6"  # default
                text_color = "#000000"
                if "hallucinated" in f_list and show_flags:
                    bg_color = "#ffebee" # red
                    text_color = "#c62828"
                elif "irrelevant" in f_list:
                    bg_color = "#fff3e0" # orange
                    text_color = "#ef6c00"
                elif not f_list:
                    bg_color = "#e8f5e9" # green
                    text_color = "#2e7d32"
                
                st.markdown(
                    f"<div style='background-color: {bg_color}; color: {text_color}; padding: 10px; border-radius: 5px; margin-bottom: 5px;'>"
                    f"<b>Prompt:</b> {p}<br/>"
                    f"<b>Response:</b> {r}"
                    f"</div>",
                    unsafe_allow_html=True
                )
