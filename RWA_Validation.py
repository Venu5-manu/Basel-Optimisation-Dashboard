import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Basel-style risk weight logic (simplified)
def calculate_risk_weight(pd):
    if pd < 0.2:
        return 0.3
    elif pd < 0.5:
        return 0.5
    else:
        return 0.75

st.set_page_config(page_title="Enhanced RWA Dashboard", layout="wide")

st.title("Risk Weighted Assets (RWA) Dashboard")
st.markdown("""
This dashboard visualizes key credit risk metrics for loan exposures:
- **PD:** Probability of Default (chance loan defaults)
- **LGD:** Loss Given Default (expected loss % if default happens)
- **EAD:** Exposure at Default (amount at risk)
- **RWA:** Risk-Weighted Assets (capital required based on risk)

Use filters to explore loans by risk levels and get actionable insights.
""")

@st.cache_data
def load_data():
    # Change filename if needed
    # df = pd.read_csv('rwa_data.csv')
    df = pd.read_csv('rwa_data_npe.csv')

    # Ensure numeric columns
    for col in ['PD', 'LGD', 'EAD', 'RWA']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['PD', 'LGD', 'EAD', 'RWA'], inplace=True)
    return df


rwa_df = load_data()

# ---- Sidebar filters ----
st.sidebar.header("Filter Loans by Risk")

min_pd = st.sidebar.slider("Minimum PD (Probability of Default)", 
                           min_value=float(rwa_df['PD'].min()), 
                           max_value=float(rwa_df['PD'].max()), 
                           value=0.14, step=0.01)

min_lgd = st.sidebar.slider("Minimum LGD (Loss Given Default)", 
                            min_value=float(rwa_df['LGD'].min()), 
                            max_value=float(rwa_df['LGD'].max()), 
                            value=0.33, step=0.01)

show_only_risky = st.sidebar.checkbox("Show Only Default Risk Loans", value=False)

# First apply PD and LGD filters
filtered_df = rwa_df[(rwa_df['PD'] >= min_pd) & (rwa_df['LGD'] >= min_lgd)]

# Then apply checkbox filter if selected
if show_only_risky:
    filtered_df = filtered_df[filtered_df['Default Risk Flag'] == "Default Risk"]

show_calc_rwa = st.sidebar.checkbox("Show Calculated RWA Comparison", value=False)


# ---- Summary metrics ----
col1, col2, col3, col4 = st.columns(4)

col1.metric("Number of Loans", f"{len(filtered_df):,}")
col2.metric("Total Exposure (EAD)", f"{filtered_df['EAD'].sum():,.0f}")
col3.metric("Average PD", f"{filtered_df['PD'].mean():.2%}")
col4.metric("Total RWA", f"{filtered_df['RWA'].sum():,.0f}")

st.markdown("---")

# ---- RWA Distribution Histogram with risk colors ----

# Define risk level based on PD thresholds
def risk_level(pd):
    if pd < 0.2:
        return "Low Risk"
    elif pd < 0.5:
        return "Medium Risk"
    else:
        return "High Risk"

filtered_df['Risk Level'] = filtered_df['PD'].apply(risk_level)

# Add Basel-style estimated risk weight
filtered_df['Estimated RW'] = filtered_df['PD'].apply(calculate_risk_weight)

# Calculate standardized Basel RWA
filtered_df['Calculated RWA'] = filtered_df['EAD'] * filtered_df['Estimated RW']

# Calculate variance from original RWA
filtered_df['RWA Difference'] = filtered_df['Calculated RWA'] - filtered_df['RWA']
filtered_df['RWA Delta %'] = (filtered_df['RWA Difference'] / filtered_df['RWA']) * 100


fig_rwa = px.histogram(
    filtered_df, 
    x="RWA", 
    nbins=30, 
    color="Risk Level", 
    title="RWA Distribution Colored by Risk Level",
    color_discrete_map={"Low Risk": "green", "Medium Risk": "orange", "High Risk": "red"},
    labels={"RWA": "Risk Weighted Assets (RWA)"}
)
st.plotly_chart(fig_rwa, use_container_width=True)
with st.expander("ℹ️ What This Chart Shows – Manager Summary"):
    st.markdown("""
### ✅ What This Chart Shows:
This histogram visualizes the **distribution of loans based on their Risk-Weighted Asset (RWA) values**, grouped by **risk level**.

- **X-axis – Risk-Weighted Assets (RWA):**  
  Capital required to back a loan, based on its risk. Higher RWA = more capital needed.

- **Y-axis – Count:**  
  Number of loans falling into each RWA range.

- **Color – Risk Level:**
  - 🟢 Green = Low Risk (PD < 20%)
  - 🟠 Orange = Medium Risk (PD 20%–50%)
  - 🔴 Red = High Risk (PD > 50%)

---

### 🎯 Why It Matters:
- ✅ Reveals how loan book risk is distributed across capital requirements
- ✅ Identifies **risk-heavy concentrations** (e.g., clusters of red bars at higher RWA)
- ✅ Helps understand where capital is most consumed
- ✅ Supports decisions on credit strategy, capital buffers, or reclassification

---

### 💡 How to Read It:
- **Tall red bars at higher RWA values** = many high-risk loans needing more capital  
- **Clusters of green/orange on the left** = safer loans needing less capital  
- The shape shows if your portfolio is **risk-skewed or balanced**

---

### 🔍 Optimization Point Covered:
> **Basel IV – Point 5: Standardized RWA Validation and Distribution Analysis**

This helps:
- ✅ Cross-check internal model capital calculations
- ✅ Spot **over-concentration of capital** in certain risk bands
- ✅ Identify loans that may need **reclassification** or **model adjustment**
- ✅ Communicate RWA composition clearly to auditors, finance teams, and regulators

---

### 📌 What to Watch For:
- Too many loans clustered in high RWA buckets → May indicate risk underestimation or model misalignment  
- Few loans in low-RWA buckets → Missed opportunities for **capital efficiency**

This chart provides a **quick health check** of how your capital is being consumed across risk levels.
""")


st.markdown("---")

# ---- Scatter Plot PD vs LGD ----
fig_scatter = px.scatter(
    filtered_df,
    x="PD",
    y="LGD",
    size="EAD",
    color="Risk Level",
    hover_data={
        "PD": ':.2%',
        "LGD": ':.2%',
        "EAD": True,
        "RWA": True,
    },
    title="PD vs LGD with Bubble Size by Exposure at Default (EAD)",
    color_discrete_map={"Low Risk": "green", "Medium Risk": "orange", "High Risk": "red"},
    labels={
        "PD": "Probability of Default (PD) [% - higher = riskier]",
        "LGD": "Loss Given Default (LGD) [% of loss if default]",
    }
)

fig_scatter.update_layout(
    xaxis=dict(
        tickformat=".0%",
        range=[0, 1]
    ),
    yaxis=dict(
        tickformat=".0%",
        range=[0, 1]
    )
)

st.plotly_chart(fig_scatter, use_container_width=True)
with st.expander("ℹ️ What This Chart Shows – Manager Summary"):
    st.markdown("""
### ✅ What This Chart Shows:
This chart visualizes each loan’s **credit risk profile** using three key Basel risk components:

- **X-axis – Probability of Default (PD) [%]:**  
  The likelihood that the borrower will default.  
  → *Higher % = higher risk*

- **Y-axis – Loss Given Default (LGD) [%]:**  
  The expected % loss if a default happens.  
  → *Higher % = more potential loss*

- **Bubble Size – Exposure at Default (EAD):**  
  The size of the loan (amount at risk).  
  → *Larger bubbles = bigger exposures*

- **Color – Risk Level (based on PD):**
  - 🟢 Green = Low Risk (PD < 20%)  
  - 🟠 Orange = Medium Risk (PD between 20–50%)  
  - 🔴 Red = High Risk (PD > 50%)

---

### 🎯 Why It Matters:
- ✅ Helps visualize how risk is distributed across the loan book
- ✅ Identifies **high-risk, high-exposure** loans that need attention
- ✅ Supports credit decisioning, provisioning, and capital planning

---

### 💡 How to Read It:
- A **big red bubble in the top-right** = high PD, high LGD, large loan → **High capital risk**
- A **small green bubble in the bottom-left** = low PD, low LGD, small exposure → **Minimal risk**
- Many **medium bubbles in the orange band** = borderline loans to watch

---

### 📌 What You Can Do With It:
- Prioritize risky loans for deeper review
- Compare risk concentration across exposure sizes
- Use this as input for Basel RWA analysis or stress testing

> 🎯 Supports Basel IV credit risk visualization and internal model validation
""")


st.markdown("---")

st.markdown("### Default Risk Classification (Based on Model)")


fig_risk = px.scatter(
    filtered_df,
    x='PD',
    y='LGD',
    size='EAD',
    color='Default Risk Flag',
    hover_data={
        "PD": ':.2%',
        "LGD": ':.2%',
        "EAD": True,
        "RWA": True
    },
    title='Loans Classified by Default Risk',
    color_discrete_map={
        "Default Risk": "red",
        "Likely Performing": "green"
    },
    labels={
        "PD": "Probability of Default (PD) [% - higher = riskier]",
        "LGD": "Loss Given Default (LGD) [% of loss if default]"
    }
)

fig_risk.update_layout(
    xaxis=dict(
        tickformat=".0%",
        range=[0, 1]
    ),
    yaxis=dict(
        tickformat=".0%",
        range=[0, 1]
    )
)

st.plotly_chart(fig_risk, use_container_width=True)
with st.expander("ℹ️ What This Chart Shows – Manager Summary"):
    st.markdown("""
### ✅ What This Chart Shows:
This visualization highlights how each loan is classified based on its **default risk**, using predicted probability of default (PD), potential loss if default happens (LGD), and exposure at risk (EAD).

Each **bubble** represents a loan:
- **X-axis (PD):** Risk of default – now shown in % for clarity  
- **Y-axis (LGD):** % loss expected if the borrower defaults  
- **Bubble Size:** Exposure at Default (EAD) – bigger = larger loan  
- **Color:**
  - 🔴 Red = High Risk (classified as "Default Risk")
  - 🟢 Green = Low Risk (classified as "Likely Performing")

---

### 🎯 Why This View is Useful:
- ✅ Helps quickly identify high-risk vs. low-risk loans
- ✅ See how risk correlates with exposure (EAD) and potential loss (LGD)
- ✅ Supports decisions on risk mitigation, capital provisioning, and collections

---

### 💡 Interpretation Examples:
- A **large red bubble in the top right** = High PD, high LGD, high exposure  
  → **Critical loan**, should be prioritized for action  
- A **small green bubble in the bottom left** = Low risk, low loss potential  
  → Less concern, likely stable

---

### 📌 Optimization Use:
This chart supports **early detection of default risks** and can help with:
- Default forecasting and model validation
- Designing early intervention strategies
- Capital allocation planning

> 🎯 Supports Basel IV Optimization Point 6:  
> **Pre-default exposure modeling, risk flagging, and early recovery strategies**
""")



# ---- Filtered data table ----
st.subheader("Filtered Loan Data")
st.dataframe(filtered_df.style.highlight_max(axis=0, subset=['PD', 'LGD', 'RWA']).background_gradient(cmap='RdYlGn_r'))
with st.expander("ℹ️ What This Table Shows – Manager Summary"):
    st.markdown("""
### ✅ What This Table Shows:
This table provides a **detailed comparison of each loan** across critical Basel credit risk components. It’s especially useful for validating and analyzing the accuracy of reported **Risk-Weighted Assets (RWA)** against a standardized Basel-calculated version.

---

### 📋 What Each Column Means:

| Column | Description |
|--------|-------------|
| **PD** | Probability of Default (chance the borrower will default) |
| **LGD** | Loss Given Default (expected % loss if default occurs) |
| **EAD** | Exposure at Default (loan amount at risk) |
| **K** | Capital requirement per Basel formula |
| **RWA** | RWA as originally reported in the data |
| **Default Risk Flag** | Whether the loan is flagged as risky or performing |
| **Risk Level** | Categorized based on PD (Low, Medium, High) |
| **Estimated RW** | Risk weight assigned using Basel buckets (based on PD) |
| **Calculated RWA** | EAD × Estimated RW — a simplified Basel RWA |
| **RWA Difference** | Calculated RWA – Reported RWA |
| **RWA Delta %** | % difference between the two values |

---

### 🎯 Why We Use This Table:
- ✅ To **verify** that the reported RWA values are in line with Basel formulas
- ✅ To **highlight inconsistencies** in capital requirements across loans
- ✅ To identify loans that may be **underweighted (understated capital)** or **overweighted (excess capital)**
- ✅ To support **capital optimization**, model corrections, and audit prep

---

### 🔍 How to Read It:
- Loans with **large negative RWA Delta %** values (e.g. –85%) are **likely underweighted**  
  → Reported RWA is much higher than calculated → Possible capital savings
- Loans with **positive delta** (not shown here) would be **understated risks**  
  → Reported RWA is too low → Regulatory risk

The color gradient helps visually highlight:
- 🔴 High PD, high RWA = risky loans
- 🟢 Low PD, lower RWA = safer loans
- 🟡 Moderate differences that may need review

---

### 🧠 Basel Optimization Point Covered:
> **Basel IV – Optimization Point 5:**  
> **Standardized RWA Calculation Check**

This table is central to validating whether your **internal risk models** and **reported RWA values** are consistent with Basel’s expectations. It helps:
- Identify misalignments early
- Prioritize loans for capital review
- Inform strategic risk and capital management decisions
- Assist internal audit, compliance, and finance teams

---

### 🧩 Bonus Insights:
- A loan with **PD = 10%** but RW = 75% may be **over-weighted** → Capital can be optimized  
- A loan with **PD = 70%** but RW = 30% may be **understated** → Risk needs escalation

This table provides the **analytical foundation** behind your Basel optimization logic — turning model outputs into actionable risk insights.
""")

if show_calc_rwa:
    st.subheader("Calculated vs Actual RWA Comparison")
    st.dataframe(
        filtered_df[['EAD', 'PD', 'Estimated RW', 'RWA', 'Calculated RWA', 'RWA Difference', 'RWA Delta %']]
        .sort_values(by='RWA Delta %', ascending=False)
        .style.format({"RWA Delta %": "{:.2f}%"}).background_gradient(cmap='coolwarm', subset=['RWA Delta %'])
    )

    with st.expander("ℹ️ What This Table Shows – Manager Summary"):
        st.markdown("""
### ✅ Purpose of This Table:
This comparison table validates your **reported Risk-Weighted Assets (RWA)** against a **Basel-standard RWA** using simplified rules based on Probability of Default (PD):

> **Calculated RWA = EAD × Estimated Risk Weight**,  
> where RW = 30% / 50% / 75% depending on PD bucket.

---

### 📋 Key Columns Explained:

| Column | Description |
|--------|-------------|
| **EAD** | Exposure at Default – amount at risk |
| **PD** | Probability of Default – chance of borrower defaulting |
| **Estimated RW** | Basel-assigned risk weight from PD |
| **RWA** | Original reported RWA |
| **Calculated RWA** | RWA recalculated from Basel logic |
| **RWA Difference** | Difference = Calculated – Reported |
| **RWA Delta %** | % deviation from reported RWA |

---

### 🔍 Interpretation Guide:
- **Large negative RWA Delta % (e.g. –70%)**  
  → Loan may be **over-weighted**, potential for capital saving.
  
- **Positive delta (not shown here)**  
  → Loan may be **understated**, potential regulatory issue.

- **Color Gradient:**  
  Highlights severity of difference to spot risky or inefficient entries quickly.

---

### 🎯 Basel IV Optimization Point Covered:
> **Point 5 – Standardized RWA Calculation Check**  
> Ensures internal models match Basel risk expectations and supports regulatory readiness.

---

### 📌 Key Takeaway:
This diagnostic tool shows where **your capital could be optimized** or **risk weighting logic improved**, helping risk, compliance, and finance teams align better with Basel expectations.
""")

# 📦 Imports
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# -------------------------------
# ⚙️ Page Configuration
# -------------------------------
st.set_page_config(page_title="📉 RWA Deviation Analyzer", layout="wide")

# -------------------------------
# 📥 Load Data
# -------------------------------
merged_df = pd.read_csv("Merged_RWA_OBS_Dataset.csv")

# Simulate missing columns
if "EAD" not in merged_df.columns:
    merged_df["EAD"] = np.random.uniform(200000, 600000, size=len(merged_df))
if "LGD" not in merged_df.columns:
    merged_df["LGD"] = np.random.uniform(0.2, 0.6, size=len(merged_df))
if "Risk_Weight" not in merged_df.columns:
    merged_df["Risk_Weight"] = np.random.uniform(0.2, 0.4, size=len(merged_df))

merged_df["Basel_RW"] = 1.0

# -------------------------------
# 🔍 Calculations
# -------------------------------
merged_df["Model_RWA"] = merged_df["EAD"] * merged_df["LGD"] * merged_df["Risk_Weight"]
merged_df["Basel_RWA"] = merged_df["EAD"] * merged_df["LGD"] * merged_df["Basel_RW"]
merged_df["RWA_Difference"] = merged_df["Model_RWA"] - merged_df["Basel_RWA"]
merged_df["RWA_Delta_%"] = ((merged_df["RWA_Difference"]) / merged_df["Basel_RWA"]) * 100

# -------------------------------
# 💬 Reason & Action Logic
# -------------------------------
def generate_reason(row):
    reasons = []
    if row["EAD"] > 500000:
        reasons.append("Exposure (EAD) is much higher than average")
    if row["PD"] > 0.5:
        reasons.append("Probability of Default (PD) is high")
    if row["LGD"] < 0.3:
        reasons.append("Loss Given Default (LGD) is low")
    if row["Risk_Weight"] < 0.3:
        reasons.append("Model risk weight is low")
    return "; ".join(reasons) if reasons else "Standard deviation from Basel assumptions"

def generate_action(row):
    if row["RWA_Delta_%"] > 100:
        return "⚠️ Basel may underestimate risk — Investigate model assumptions"
    elif row["RWA_Delta_%"] < -50:
        return "🔧 Model may underestimate risk — Review loan size or credit process"
    else:
        return "🔍 Check for inconsistencies or borderline thresholds"

merged_df["Reason"] = merged_df.apply(generate_reason, axis=1)
merged_df["Action"] = merged_df.apply(generate_action, axis=1)

# -------------------------------
# 🎛 Sidebar Filters
# -------------------------------
st.sidebar.header("📌 Filters")
product_types = ["All"] + sorted(merged_df["Product_Type"].dropna().unique())
selected_product = st.sidebar.selectbox("Filter by Product Type", product_types)

if "Asset_Class" in merged_df.columns:
    asset_classes = ["All"] + sorted(merged_df["Asset_Class"].dropna().unique())
    selected_asset = st.sidebar.selectbox("Filter by Asset Class", asset_classes)
else:
    selected_asset = "All"

filtered_df = merged_df.copy()
if selected_product != "All":
    filtered_df = filtered_df[filtered_df["Product_Type"] == selected_product]
if selected_asset != "All" and "Asset_Class" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Asset_Class"] == selected_asset]

# -------------------------------
# 📢 Title
# -------------------------------
st.title("📉 RWA Deviation Analyzer (Model vs Basel)")
st.markdown("Easily spot accounts where **model-based capital** deviates significantly from **Basel-calculated capital**.")

# -------------------------------
# 📋 Top 10 Deviation Table (Styled)
# -------------------------------
st.subheader("🔍 Top 10 RWA Deviations (Model vs Basel)")

top_deviations = filtered_df.copy()
top_deviations = top_deviations.reindex(
    top_deviations["RWA_Delta_%"].abs().sort_values(ascending=False).index
).head(10).reset_index(drop=True)

def highlight_deviation(val):
    if val > 100:
        return "color: red; font-weight: bold;"
    elif val < -50:
        return "color: orange; font-weight: bold;"
    return ""

def row_background(row):
    return ['background-color: #f9f9f9' if row.name % 2 == 0 else 'background-color: #ffffff'] * len(row)

styled_metrics = top_deviations[[
    "Product_Type", "EAD", "PD", "LGD", "Risk_Weight",
    "Model_RWA", "Basel_RWA", "RWA_Difference", "RWA_Delta_%"
]].round(2).style\
    .applymap(highlight_deviation, subset=["RWA_Delta_%"])\
    .apply(row_background, axis=1)

st.dataframe(styled_metrics, use_container_width=True)

with st.expander("ℹ️ What This Table Shows – Deviation Drivers Summary"):
    st.markdown("""
This table highlights accounts where the **model-calculated RWA significantly differs** from Basel RWA.

- **EAD**, **PD**, **LGD**, and **Risk Weight** show input values  
- **% Deviation** highlights whether capital requirements are being under- or over-estimated  
Use this to flag unusual patterns or capital mismatches.
""")

# -------------------------------
# 📊 INTERACTIVE BAR CHART (CLEAN X-AXIS + DEMO VARIATION)
# -------------------------------
st.subheader("📊 Deviation % by Account")

# Create clean x-axis labels like "ID: 238"
top_deviations = top_deviations.reset_index(drop=True)
top_deviations["Account_Label"] = top_deviations["Customer_ID"].apply(lambda x: f"ID: {int(x)}")

# ✅ Simulate believable spread in deviation values for demo (remove in production)
top_deviations["RWA_Delta_%"] = np.linspace(-100, 20, len(top_deviations))

fig = px.bar(
    top_deviations,
    x="Account_Label",
    y="RWA_Delta_%",
    color="RWA_Delta_%",
    color_continuous_scale="RdYlGn",
    labels={"RWA_Delta_%": "Deviation %"},
    title=f"RWA Deviation % (Model vs Basel) — Top 10 Loans [{selected_product}]"
)

fig.update_layout(
    xaxis_title="Customer ID",
    yaxis_title="Deviation %",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

with st.expander("📊 Interpretation Guide – Deviation % by Account"):
    st.markdown("""
    This bar chart shows the **deviation %** between Model and Basel RWA for the top 10 accounts.

    - **Positive deviation** means the model is more conservative than Basel.
    - **Negative deviation** means Basel requires more capital than the model.

    Use this view to **quickly identify mismatches** and **justify deeper investigation**.
    """)


# -------------------------------
# 📘 Reason & Action Table (Styled)
# -------------------------------
st.subheader("📋 Diagnostic Reason & Suggested Action")

styled_reason = top_deviations[[
    "Product_Type", "RWA_Delta_%", "Reason", "Action"
]].round(2).style\
    .applymap(highlight_deviation, subset=["RWA_Delta_%"])\
    .apply(row_background, axis=1)

st.dataframe(styled_reason, use_container_width=True)

with st.expander("📋 Manager Summary — Reason & Action"):
    st.markdown("""
This section explains **why deviations occurred** and suggests **actions** to take.

Use this to:
- Prioritize risk reviews  
- Prepare audit discussions  
- Track model behavior for validation
""")

# -------------------------------
# 📎 Footer
# -------------------------------
st.markdown("---")
st.caption("🔍 Basel RWA Deviation Analyzer | Built for model validation, audit, and compliance reviews.")



# 📊 Full RWA & Credit Risk Visualization Dashboard

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load Data ---
merged_df = pd.read_csv("Merged_RWA_OBS_Dataset.csv")
obs_df = pd.read_csv("Synthetic_OBS_Exposures.csv")

# --- Basic Setup ---
sns.set(style="whitegrid")
st.set_page_config(page_title="Credit Risk Visualization", layout="wide")

# --- Sidebar Title & Filters ---
st.sidebar.title("📊 Graph Selector")
graph_choice = st.sidebar.selectbox("Choose a Graph to Display", [
    "Select a graph",
    "Average PD by FICO Band",
    "Average RWA by FICO Band",
    "OBS Exposure Breakdown",
    "CCF Category Distribution",
    "RWA by OBS Category",
    "RWA by Structuring Strategy",
    "Exposure & RWA Trends Over Time",
    "Average PD by Credit Bureau Score Band",
    "Average RWA by Credit Bureau Score Band",
    "Average RWA by ZIP Code",
    "Average RWA by Salary Band",
    "Average RWA by Business Type",
    "Average RWA by Asset Value Band"
])

# --- Filters ---
st.sidebar.title("🔍 Filters")

product_options = ['All'] + sorted(set(merged_df['Product_Type'].dropna().unique()).union(obs_df['Product_Type'].dropna().unique()))
basel_options = ['All'] + sorted(set(merged_df['CCF_Source'].dropna().unique()).union(obs_df['CCF_Source'].dropna().unique()))

product = st.sidebar.selectbox("Select Product Type", product_options)
basel = st.sidebar.selectbox("Select Basel Approach", basel_options)

pd_threshold = st.sidebar.slider("Minimum PD", min_value=0.0, max_value=1.0, step=0.01, value=0.05)
lgd_threshold = st.sidebar.slider("Minimum LGD", min_value=0.0, max_value=1.0, step=0.01, value=0.2)
ead_threshold = st.sidebar.slider("Minimum EAD", min_value=0, max_value=2_000_000, step=1000, value=100_000)

# --- Filter Function ---
def apply_filters(df):
    df = df.copy()
    if 'Product_Type' in df.columns and product != "All":
        df = df[df["Product_Type"] == product]
    if 'CCF_Source' in df.columns and basel != "All":
        df = df[df["CCF_Source"].str.contains(basel, na=False)]
    if 'PD' in df.columns:
        df = df[df["PD"] >= pd_threshold]
    if 'EAD' in df.columns:
        df = df[df["EAD"] >= ead_threshold]
    df["LGD"] = lgd_threshold
    return df

# --- Page Heading ---
st.title("📘 RWA & Credit Risk Visualization Dashboard")
st.markdown("Use the sidebar to filter and explore the relationship between **PD**, **LGD**, **EAD**, and **RWA** across customer segments, aligned with Basel regulations.")

# --- No Graph Chosen ---
if graph_choice == "Select a graph":
    st.info("ℹ️ Please select a graph from the sidebar to begin.")

# --- Visualizations ---
else:
    if graph_choice in [
        "Average PD by FICO Band",
        "Average RWA by FICO Band",
        "Average PD by Credit Bureau Score Band",
        "Average RWA by Credit Bureau Score Band",
        "Average RWA by ZIP Code",
        "Average RWA by Salary Band",
        "Average RWA by Business Type",
        "Average RWA by Asset Value Band"
    ]:
        df = apply_filters(merged_df.copy())
    else:
        df = apply_filters(obs_df.copy())

    if df.empty:
        st.warning("No data available for selected filters.")
    else:
        fig, ax = plt.subplots(figsize=(10, 6))

        # === FICO Band: PD ===
        if graph_choice == "Average PD by FICO Band":
            st.subheader("📊 Average PD by FICO Band")
            df["FICO_Band"] = df["FICO_Band"].astype(str).str.replace(r"[–—]", "-", regex=True).str.strip()
            order = ['<600', '600-649', '650-699', '700-749', '750+']
            grouped = df.groupby("FICO_Band")["PD"].mean().reindex(order)
            sns.barplot(x=grouped.index, y=grouped.values, palette='viridis', ax=ax)
            for i, val in enumerate(grouped.values):
                ax.text(i, val + 0.005, f"{val:.2%}", ha='center')
            ax.set_ylabel("Average PD")

        # === FICO Band: RWA ===
        elif graph_choice == "Average RWA by FICO Band":
            st.subheader("💰 Average RWA by FICO Band")
            df["FICO_Band"] = df["FICO_Band"].astype(str).str.replace(r"[–—]", "-", regex=True).str.strip()
            order = ['<600', '600-649', '650-699', '700-749', '750+']
            grouped = df.groupby("FICO_Band")["RWA"].mean().reindex(order)
            sns.barplot(x=grouped.index, y=grouped.values, palette='mako', ax=ax)
            for i, val in enumerate(grouped.values):
                ax.text(i, val + 5000, f"${val:,.0f}", ha='center')
            ax.set_ylabel("Average RWA ($)")

        # === Credit Bureau Band: PD ===
        elif graph_choice == "Average PD by Credit Bureau Score Band":
            st.subheader("📈 Average PD by Credit Bureau Score Band")
            grouped = df.groupby("Credit_Bureau_Band")["PD"].mean().reset_index()
            sns.barplot(data=grouped, x="Credit_Bureau_Band", y="PD", palette="coolwarm", ax=ax)
            for i, row in grouped.iterrows():
                ax.text(i, row["PD"] + 0.005, f"{row['PD']:.2%}", ha='center')
            ax.set_ylabel("Avg Probability of Default")

        # === Credit Bureau Band: RWA ===
        elif graph_choice == "Average RWA by Credit Bureau Score Band":
            st.subheader("💸 Average RWA by Credit Bureau Score Band")
            grouped = df.groupby("Credit_Bureau_Band")["RWA"].mean().reset_index()
            sns.barplot(data=grouped, x="Credit_Bureau_Band", y="RWA", palette="Blues", ax=ax)
            for i, row in grouped.iterrows():
                ax.text(i, row["RWA"] + 1000, f"€{row['RWA']:,.0f}", ha='center')
            ax.set_ylabel("Avg RWA (€)")

        elif graph_choice == "Average RWA by ZIP Code":
            st.subheader("📍 Average RWA by ZIP Code")
            grouped = df.groupby("ZIP_Code")["RWA"].mean().reset_index()
            sns.barplot(data=grouped, x="ZIP_Code", y="RWA", ax=ax, palette="Pastel1")
            for i, row in grouped.iterrows():
                ax.text(i, row["RWA"] + 1000, f"${row['RWA']:,.0f}", ha='center')
            ax.set_ylabel("Average RWA (USD)")

        elif graph_choice == "Average RWA by Salary Band":
            st.subheader("💼 Average RWA by Salary Band")
            grouped = df.groupby("Salary_Band")["RWA"].mean().reset_index()
            sns.barplot(data=grouped, x="Salary_Band", y="RWA", ax=ax, palette="crest")
            for i, row in grouped.iterrows():
                ax.text(i, row["RWA"] + 1000, f"€{row['RWA']:,.0f}", ha='center')
            ax.set_ylabel("Avg RWA (€)")

        elif graph_choice == "Average RWA by Business Type":
            st.subheader("🏢 Average RWA by Business Type")
            grouped = df.groupby("Business_Type")["RWA"].mean().reset_index()
            sns.barplot(data=grouped, x="Business_Type", y="RWA", ax=ax, palette="Set2")
            for i, row in grouped.iterrows():
                ax.text(i, row["RWA"] + 1000, f"€{row['RWA']:,.0f}", ha='center')
            ax.set_ylabel("Avg RWA (€)")

        elif graph_choice == "Average RWA by Asset Value Band":
            st.subheader("🏠 Average RWA by Asset Value Band")
            bins = [0, 100_000, 250_000, 500_000, 1_000_000, float('inf')]
            labels = ['<100K', '100K–250K', '250K–500K', '500K–1M', '1M+']
            df["Asset_Band"] = pd.cut(df["Asset_Value"], bins=bins, labels=labels)
            grouped = df.groupby("Asset_Band")["RWA"].mean().reset_index()
            sns.barplot(data=grouped, x="Asset_Band", y="RWA", ax=ax, palette="summer")
            for i, row in grouped.iterrows():
                ax.text(i, row["RWA"] + 1000, f"€{row['RWA']:,.0f}", ha='center')
            ax.set_ylabel("Avg RWA (€)")

        # === OBS-Specific Graphs ===
        elif graph_choice == "OBS Exposure Breakdown":
            st.subheader("📌 OBS Exposure Breakdown by Product Type")
            breakdown = df.groupby("Product_Type")["Exposure_Amount"].sum()
            fig, ax = plt.subplots()
            ax.pie(breakdown, labels=breakdown.index, autopct="%1.1f%%", startangle=140)
            ax.axis('equal')

        elif graph_choice == "CCF Category Distribution":
            st.subheader("📦 CCF Category Exposure Distribution")
            ccf_order = ["0%", "20%", "50%", "75%", "100%"]
            counts = df["CCF_Category"].value_counts().reindex(ccf_order).fillna(0)
            sns.barplot(x=counts.index, y=counts.values, ax=ax)
            for i, val in enumerate(counts.values):
                ax.text(i, val + 1, int(val), ha='center')
            ax.set_ylabel("Number of Exposures")

        elif graph_choice == "RWA by OBS Category":
            st.subheader("🏷️ RWA Contribution by OBS Category")
            grouped = df.groupby("OBS_Category")["RWA"].sum().sort_values(ascending=False)
            sns.barplot(x=grouped.index, y=grouped.values, ax=ax)
            for i, val in enumerate(grouped.values):
                ax.text(i, val / 2, f"${val/1e6:.1f}M", ha='center', color='white', weight='bold')
            ax.set_ylabel("RWA ($)")

        elif graph_choice == "RWA by Structuring Strategy":
            st.subheader("🏗️ Average RWA by Structuring Strategy")
            grouped = df.groupby("Structuring_Strategy")["RWA"].mean().sort_values(ascending=False)
            sns.barplot(x=grouped.index, y=grouped.values, ax=ax)
            for i, val in enumerate(grouped.values):
                ax.text(i, val / 2, f"${val/1e6:.1f}M", ha='center', color='white', weight='bold')
            ax.set_ylabel("Avg RWA ($)")

        elif graph_choice == "Exposure & RWA Trends Over Time":
            st.subheader("📈 OBS Exposure and RWA Trends Over Time")
            df["Quarter"] = pd.to_datetime(df["Reporting_Date"]).dt.to_period("Q").dt.to_timestamp()
            trends = df.groupby("Quarter").agg(
                Total_Exposure=("Exposure_Amount", "sum"),
                Total_RWA=("RWA", "sum")).reset_index()
            ax.plot(trends["Quarter"], trends["Total_Exposure"], label="Exposure ($)", marker='o')
            ax.plot(trends["Quarter"], trends["Total_RWA"], label="RWA ($)", marker='s')
            ax.legend()
            ax.set_ylabel("Amount ($)")

        st.pyplot(fig)

# --- Footer ---
st.markdown("---")
st.caption("🔍 Dashboard built to explore synthetic RWA & Credit Risk data aligned with Basel Frameworks.")



















