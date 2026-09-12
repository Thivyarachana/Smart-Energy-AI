
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score
)
import plotly.graph_objects as go
import plotly.express as px

# ==========================================================
# SMARTENERGY AI
# Predictive Electricity Consumption & Energy Waste Analyzer
# ==========================================================

st.set_page_config(
    page_title="SmartEnergy AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- 90% BLUE / 10% WHITE --------------------
st.markdown("""
<style>
:root {
    --navy:#041a3d;
    --deep:#06285f;
    --blue:#0b4dbb;
    --bright:#1976f3;
    --sky:#62a8ff;
    --pale:#eaf4ff;
    --white:#ffffff;
    --ink:#08234d;
    --muted:#667891;
}
.stApp {
    background: linear-gradient(135deg,#041a3d 0%,#06285f 48%,#0b4dbb 100%);
    color:#fff;
}
.block-container {max-width:1450px;padding-top:1.2rem;padding-bottom:3rem;}
[data-testid="stSidebar"] {background:#041a3d;border-right:1px solid rgba(255,255,255,.14);}
[data-testid="stSidebar"] * {color:#fff !important;}
.hero {
    padding:30px 34px;border-radius:26px;margin-bottom:22px;
    background:linear-gradient(135deg,rgba(255,255,255,.13),rgba(255,255,255,.035));
    border:1px solid rgba(255,255,255,.18);
    box-shadow:0 20px 50px rgba(0,0,0,.20);
}
.hero h1 {font-size:44px;margin:0;letter-spacing:-1.2px;}
.hero p {color:#dceaff;font-size:16px;margin:9px 0 0;}
.section {font-size:25px;font-weight:850;margin:24px 0 13px;}
.card {
    background:#fff;color:#08234d;border-radius:20px;padding:21px;
    box-shadow:0 14px 35px rgba(0,0,0,.15);margin-bottom:18px;
}
.card h3,.card h4 {color:#06285f;margin-top:0;}
.metric {
    background:#fff;color:#08234d;border-radius:18px;padding:18px 20px;
    min-height:112px;box-shadow:0 12px 30px rgba(0,0,0,.14);
}
.metric-label {color:#667891;font-size:12px;font-weight:800;text-transform:uppercase;letter-spacing:.7px;}
.metric-value {color:#0b4dbb;font-size:29px;font-weight:900;margin-top:7px;}
.metric-sub {color:#667891;font-size:12px;}
.result {
    background:linear-gradient(135deg,#fff,#eaf4ff);color:#08234d;
    border-radius:24px;padding:26px;text-align:center;
    box-shadow:0 18px 45px rgba(0,0,0,.20);
}
.result .big {font-size:56px;font-weight:950;color:#0b4dbb;line-height:1;}
.result .sub {color:#667891;font-weight:750;margin-top:8px;}
.badge {display:inline-block;margin-top:12px;padding:7px 15px;border-radius:999px;
    background:#eaf4ff;color:#0b4dbb;font-weight:900;font-size:13px;}
.tip {
    padding:15px 18px;margin:9px 0;border-radius:15px;
    background:rgba(255,255,255,.10);border:1px solid rgba(255,255,255,.14);
}
.tip strong{color:#fff}.tip span{color:#dceaff;}
div[data-baseweb="input"] > div,div[data-baseweb="select"] > div {
    background:#fff !important;border-radius:10px !important;
}
label,.stSlider label,.stNumberInput label,.stSelectbox label {color:#fff !important;font-weight:700 !important;}
.stButton > button {
    width:100%;border:0;border-radius:12px;background:#fff;color:#0b4dbb;
    font-weight:900;padding:12px 20px;box-shadow:0 8px 20px rgba(0,0,0,.18);
}
.stButton > button:hover {background:#eaf4ff;color:#06285f;}
.footer{text-align:center;color:#dceaff;padding:24px;font-size:12px;}
</style>
""", unsafe_allow_html=True)

DATA = Path(__file__).parent / "energy_data.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA)

@st.cache_resource
def train_models(df):
    features = [
        "Temperature_C","Humidity_pct","Occupants","Home_Size_sqft",
        "AC_Hours","Fan_Hours","Fridge_Hours","WashingMachine_Hours",
        "TV_Hours","Computer_Hours","WaterHeater_Hours","Previous_Day_kWh",
        "Solar_Available","Weekday","Month"
    ]
    X = df[features]
    y_reg = df["Daily_Consumption_kWh"]
    y_cls = df["Energy_Waste"]

    X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
        X, y_reg, y_cls, test_size=.22, random_state=42, stratify=y_cls
    )

    rf_reg = RandomForestRegressor(
        n_estimators=350, max_depth=12, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    )
    rf_reg.fit(X_train, yr_train)
    reg_pred = rf_reg.predict(X_test)

    linear = Pipeline([
        ("scale", StandardScaler()),
        ("model", LinearRegression())
    ])
    linear.fit(X_train, yr_train)
    linear_pred = linear.predict(X_test)

    rf_cls = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=2,
        random_state=42, class_weight="balanced", n_jobs=-1
    )
    rf_cls.fit(X_train, yc_train)
    cls_pred = rf_cls.predict(X_test)
    cls_prob = rf_cls.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(yc_test, cls_prob)

    results = {
        "reg_mae": mean_absolute_error(yr_test, reg_pred),
        "reg_rmse": np.sqrt(mean_squared_error(yr_test, reg_pred)),
        "reg_r2": r2_score(yr_test, reg_pred),
        "linear_r2": r2_score(yr_test, linear_pred),
        "cls_accuracy": accuracy_score(yc_test, cls_pred),
        "cls_precision": precision_score(yc_test, cls_pred, zero_division=0),
        "cls_recall": recall_score(yc_test, cls_pred, zero_division=0),
        "cls_f1": f1_score(yc_test, cls_pred, zero_division=0),
        "auc": roc_auc_score(yc_test, cls_prob),
        "cm": confusion_matrix(yc_test, cls_pred),
        "fpr": fpr, "tpr": tpr,
        "test_actual": yr_test, "test_pred": reg_pred,
        "features": features,
        "importance": rf_reg.feature_importances_
    }
    return rf_reg, linear, rf_cls, results

df = load_data()
rf_reg, linear, rf_cls, M = train_models(df)

# -------------------- Sidebar --------------------
with st.sidebar:
    st.markdown("## ⚡ SmartEnergy AI")
    st.caption("Predictive electricity intelligence")
    st.divider()
    page = st.radio(
        "Navigation",
        ["🔮 Energy Prediction","📊 ML Analytics","♻️ Energy Waste Analyzer","ℹ️ About"]
    )
    st.divider()
    st.markdown("### Models")
    st.success("Random Forest Regression")
    st.success("Random Forest Classification")
    st.caption(f"Dataset: {len(df):,} records")
    st.caption(f"Consumption R²: {M['reg_r2']:.3f}")

st.markdown("""
<div class="hero">
<h1>⚡ SmartEnergy <span style="color:#62a8ff;">AI</span></h1>
<p>Predict electricity consumption, detect energy waste and turn household usage patterns into practical energy-saving insights.</p>
</div>
""", unsafe_allow_html=True)

# -------------------- Prediction --------------------
if page == "🔮 Energy Prediction":
    st.markdown('<div class="section">🏠 Household Energy Profile</div>', unsafe_allow_html=True)

    a,b,c = st.columns(3)
    with a:
        temp = st.slider("Temperature (°C)",16.0,43.0,30.0,.5)
        humidity = st.slider("Humidity (%)",25,95,65)
        occupants = st.number_input("Number of Occupants",1,10,3)
        home_size = st.number_input("Home Size (sq ft)",400,4000,1200,50)
        solar = st.selectbox("Solar Power Available?",["No","Yes"])
    with b:
        ac = st.slider("AC Usage (hours/day)",0.0,12.0,4.0,.5)
        fan = st.slider("Fan Usage (hours/day)",0.0,18.0,8.0,.5)
        fridge = st.slider("Refrigerator Usage (hours/day)",8.0,24.0,18.0,.5)
        washing = st.slider("Washing Machine (hours/day)",0.0,8.0,1.5,.5)
    with c:
        tv = st.slider("TV Usage (hours/day)",0.0,12.0,4.0,.5)
        computer = st.slider("Computer Usage (hours/day)",0.0,12.0,3.0,.5)
        heater = st.slider("Water Heater (hours/day)",0.0,5.0,1.0,.5)
        previous = st.number_input("Previous Day Consumption (kWh)",4.0,45.0,18.0,.5)
        weekday = st.selectbox("Day Type",["Weekend","Weekday"])
        month = st.selectbox("Month",list(range(1,13)),index=8)

    predict = st.button("⚡ PREDICT ELECTRICITY CONSUMPTION")

    if predict:
        row = pd.DataFrame([{
            "Temperature_C":temp,"Humidity_pct":humidity,"Occupants":occupants,
            "Home_Size_sqft":home_size,"AC_Hours":ac,"Fan_Hours":fan,
            "Fridge_Hours":fridge,"WashingMachine_Hours":washing,
            "TV_Hours":tv,"Computer_Hours":computer,"WaterHeater_Hours":heater,
            "Previous_Day_kWh":previous,"Solar_Available":1 if solar=="Yes" else 0,
            "Weekday":1 if weekday=="Weekday" else 0,"Month":month
        }])

        pred = float(rf_reg.predict(row)[0])
        waste_probability = float(rf_cls.predict_proba(row)[0][1])
        monthly = pred * 30
        annual = pred * 365

        if waste_probability >= .70:
            risk, label = "HIGH WASTE RISK", "Immediate energy optimization recommended"
        elif waste_probability >= .40:
            risk, label = "MODERATE WASTE RISK", "Some usage patterns can be improved"
        else:
            risk, label = "LOW WASTE RISK", "Energy usage looks relatively efficient"

        st.markdown('<div class="section">Prediction Result</div>', unsafe_allow_html=True)
        x,y,z = st.columns(3)
        with x:
            st.markdown(f"""
            <div class="result">
              <div style="font-weight:850">PREDICTED DAILY CONSUMPTION</div>
              <div class="big">{pred:.1f}</div>
              <div class="sub">kilowatt-hours / day</div>
              <div class="badge">{risk}</div>
            </div>
            """, unsafe_allow_html=True)
        with y:
            st.markdown(f"""
            <div class="card">
              <h3>📅 Consumption Forecast</h3>
              <p><b>Daily:</b> {pred:.1f} kWh</p>
              <p><b>30-day estimate:</b> {monthly:.0f} kWh</p>
              <p><b>Annual estimate:</b> {annual/1000:.2f} MWh</p>
              <p><b>Waste probability:</b> {waste_probability*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        with z:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=waste_probability*100,
                number={"suffix":"%","font":{"size":30,"color":"#0b4dbb"}},
                title={"text":"Energy Waste Risk","font":{"color":"#06285f"}},
                gauge={"axis":{"range":[0,100]},"bar":{"color":"#0b4dbb"},
                       "bgcolor":"#eaf4ff","borderwidth":0}
            ))
            gauge.update_layout(height=230,margin=dict(l=20,r=20,t=55,b=5),
                                paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(gauge,use_container_width=True)

        st.markdown('<div class="section">💡 Personalized Energy Actions</div>', unsafe_allow_html=True)

        tips=[]
        if ac >= 5:
            tips.append(("❄️ AC","AC usage is a major energy driver. Increase thermostat efficiency and reduce unnecessary runtime."))
        if heater >= 2:
            tips.append(("🔥 Water Heater","High heater runtime can significantly increase consumption. Use a timer or reduce runtime."))
        if computer >= 6:
            tips.append(("💻 Computer","Long computer usage can add avoidable consumption. Enable sleep mode when idle."))
        if washing >= 3:
            tips.append(("🧺 Washing Machine","Batch full loads and avoid repeated small cycles."))
        if temp >= 34 and ac >= 3:
            tips.append(("🌡️ Hot Weather","High temperature combined with AC usage is increasing predicted demand."))
        if solar == "No":
            tips.append(("☀️ Solar Opportunity","Solar availability is a potential long-term way to offset daytime consumption."))
        if not tips:
            tips.append(("✅ Efficient Pattern","Your entered usage pattern does not show a major obvious waste driver. Continue monitoring daily consumption."))

        for title, text in tips[:5]:
            st.markdown(f'<div class="tip"><strong>{title}</strong><br><span>{text}</span></div>',unsafe_allow_html=True)

# -------------------- ML Analytics --------------------
elif page == "📊 ML Analytics":
    st.markdown('<div class="section">🤖 Machine Learning Performance</div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    cards=[
        (c1,"R² Score",f"{M['reg_r2']:.3f}","Random Forest Regression"),
        (c2,"MAE",f"{M['reg_mae']:.2f} kWh","Lower is better"),
        (c3,"RMSE",f"{M['reg_rmse']:.2f} kWh","Lower is better"),
        (c4,"ROC-AUC",f"{M['auc']:.3f}","Waste classifier")
    ]
    for col,title,value,sub in cards:
        with col:
            st.markdown(f'<div class="metric"><div class="metric-label">{title}</div><div class="metric-value">{value}</div><div class="metric-sub">{sub}</div></div>',unsafe_allow_html=True)

    l,r=st.columns(2)
    with l:
        actual=np.array(M["test_actual"])
        pred=np.array(M["test_pred"])
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=actual,y=pred,mode="markers",name="Predictions",
                                 marker=dict(size=6)))
        line=[actual.min(),actual.max()]
        fig.add_trace(go.Scatter(x=line,y=line,mode="lines",name="Ideal"))
        fig.update_layout(title="Actual vs Predicted Consumption",
                          xaxis_title="Actual kWh",yaxis_title="Predicted kWh",
                          height=430,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="white",
                          font_color="#06285f")
        st.plotly_chart(fig,use_container_width=True)

    with r:
        cm=M["cm"]
        fig=go.Figure(go.Heatmap(
            z=cm,x=["Predicted Efficient","Predicted Waste"],
            y=["Actual Efficient","Actual Waste"],
            text=cm,texttemplate="%{text}",colorscale="Blues"
        ))
        fig.update_layout(title="Energy Waste Confusion Matrix",height=430,
                          paper_bgcolor="rgba(0,0,0,0)",font_color="#06285f")
        st.plotly_chart(fig,use_container_width=True)

    l,r=st.columns(2)
    with l:
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=M["fpr"],y=M["tpr"],mode="lines",
                                 name=f"Random Forest (AUC {M['auc']:.3f})"))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Baseline",
                                 line=dict(dash="dash")))
        fig.update_layout(title="ROC Curve — Waste Detection",
                          xaxis_title="False Positive Rate",yaxis_title="True Positive Rate",
                          height=400,paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="white",font_color="#06285f")
        st.plotly_chart(fig,use_container_width=True)
    with r:
        comparison=pd.DataFrame({
            "Model":["Random Forest Regression","Linear Regression"],
            "R²":[M["reg_r2"],M["linear_r2"]]
        })
        fig=px.bar(comparison,x="Model",y="R²",title="Regression Model Comparison")
        fig.update_layout(height=400,paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="white",font_color="#06285f")
        st.plotly_chart(fig,use_container_width=True)

    imp=pd.DataFrame({"Feature":M["features"],"Importance":M["importance"]}).sort_values("Importance")
    fig=px.bar(imp.tail(10),x="Importance",y="Feature",orientation="h",
               title="Top Factors Influencing Consumption")
    fig.update_layout(height=430,paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="white",font_color="#06285f")
    st.plotly_chart(fig,use_container_width=True)

# -------------------- Waste Analyzer --------------------
elif page == "♻️ Energy Waste Analyzer":
    st.markdown('<div class="section">♻️ Household Energy Waste Intelligence</div>', unsafe_allow_html=True)

    waste_rate=df["Energy_Waste"].mean()*100
    avg=df["Daily_Consumption_kWh"].mean()
    high=df.loc[df["Energy_Waste"]==1,"Daily_Consumption_kWh"].mean()

    a,b,c=st.columns(3)
    for col,title,value,sub in [
        (a,"Average Daily Consumption",f"{avg:.1f} kWh","Across demonstration dataset"),
        (b,"High-Waste Profiles",f"{waste_rate:.1f}%","Model target distribution"),
        (c,"Avg. High-Waste Consumption",f"{high:.1f} kWh","Profiles flagged as waste")
    ]:
        with col:
            st.markdown(f'<div class="metric"><div class="metric-label">{title}</div><div class="metric-value">{value}</div><div class="metric-sub">{sub}</div></div>',unsafe_allow_html=True)

    l,r=st.columns(2)
    with l:
        fig=px.histogram(df,x="Daily_Consumption_kWh",color="Energy_Waste",
                         nbins=35,title="Consumption Distribution")
        fig.update_layout(height=420,paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="white",font_color="#06285f")
        st.plotly_chart(fig,use_container_width=True)
    with r:
        appliance={
            "AC":df["AC_Hours"].mean(),
            "Fan":df["Fan_Hours"].mean(),
            "Fridge":df["Fridge_Hours"].mean(),
            "Washing Machine":df["WashingMachine_Hours"].mean(),
            "TV":df["TV_Hours"].mean(),
            "Computer":df["Computer_Hours"].mean(),
            "Water Heater":df["WaterHeater_Hours"].mean()
        }
        ap=pd.DataFrame({"Appliance":list(appliance),"Average Hours":list(appliance.values())})
        fig=px.bar(ap,x="Appliance",y="Average Hours",title="Average Appliance Usage")
        fig.update_layout(height=420,paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="white",font_color="#06285f",
                          xaxis_tickangle=-35)
        st.plotly_chart(fig,use_container_width=True)

    st.markdown("""
    <div class="card">
      <h3>🧠 How the waste analyzer works</h3>
      <p>The classification model learns patterns associated with unusually high energy use. It uses household size, occupancy, appliance usage, weather, previous consumption and solar availability.</p>
      <p><b>Important:</b> the included dataset is synthetic for educational demonstration. A real energy-management product should be trained and validated on appropriately collected, consented meter data.</p>
    </div>
    """,unsafe_allow_html=True)

# -------------------- About --------------------
else:
    st.markdown("""
    <div class="card">
      <h3>⚡ SmartEnergy AI</h3>
      <p><b>SmartEnergy AI</b> is a predictive machine-learning application that estimates household electricity consumption and identifies potential energy-waste patterns.</p>

      <h4>Machine Learning Pipeline</h4>
      <ol>
        <li>Collect household and appliance usage features</li>
        <li>Prepare and split the dataset into training and testing sets</li>
        <li>Train Random Forest and Linear Regression models for consumption prediction</li>
        <li>Train a Random Forest classifier for energy-waste detection</li>
        <li>Evaluate with MAE, RMSE, R², Accuracy, Precision, Recall, F1 and ROC-AUC</li>
        <li>Visualize predictions and feature importance</li>
        <li>Generate personalized energy-saving recommendations</li>
      </ol>

      <h4>From data to intelligent Energy Decisions</h4>
      <p>An end-to-end supervised learning solution that transforms energy-usage data 
    into actionable insights through predictive modeling, model training and testing, 
    regression, classification, confusion-matrix analysis, and ROC-curve evaluation — 
    all within a practical real-world energy management application.</p>

      <h4>Project Stack</h4>
      <p>Python • Pandas • NumPy • Scikit-learn • Plotly • Streamlit</p>
    </div>
    """,unsafe_allow_html=True)

st.markdown('<div class="footer">SmartEnergy AI • Predict • Detect • Optimize • Built for educational ML demonstration</div>',unsafe_allow_html=True)
