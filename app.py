


import streamlit as st
import numpy as np
import plotly.graph_objects as go

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="HF-ITR Calculator",
    page_icon="🧠",
    layout="wide"
)

# =====================================================
# TITLE
# =====================================================

st.title("Human-Factor ITR Calculator for SSVEP-BCI")

st.write(
    "This web prototype extends the conventional ITR calculator "
    "by incorporating visual irritation and predicted fatigue."
)

# =====================================================
# SIDEBAR INPUTS
# =====================================================

st.sidebar.header("Conventional ITR Inputs")

N = st.sidebar.number_input(
    "Number of Commands / Classes",
    min_value=2,
    value=2,
    step=1
)



total_trials = st.sidebar.number_input(
    "Total Trials",
    min_value=0,
    value=0,
    step=1
)


correct_trials = st.sidebar.number_input(
    "Correct Trials",
    min_value=0,
    max_value=max(int(total_trials), 0),
    value=0,
    step=1
)

selection_time = st.sidebar.number_input(
    "Selection Time per Trial (s)",
    min_value=0.1,
    value=0.1,
    step=0.1
)

st.sidebar.markdown("---")
st.sidebar.header("Human-Factor Input")

irritation = st.sidebar.slider(
    "Visual Irritation Score",
    min_value=1,
    max_value=5,
    value=4
)

st.sidebar.markdown("---")
st.sidebar.header("HF-ITR Setting")

wI = st.sidebar.slider(
    "Weight of Irritation (wI)",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)

wF = 1 - wI

alpha = st.sidebar.slider(
    "Human-Factor Scaling Factor (α)",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)

# =====================================================
# ACCURACY
# =====================================================

if total_trials == 0:
    accuracy = 0
    accuracy_percent = 0
else:
    accuracy = correct_trials / total_trials
    accuracy_percent = accuracy * 100

# =====================================================
# CONVENTIONAL ITR
# =====================================================

def calculate_itr(N, P, T):
    if P <= 0:
        return 0.0

    if P >= 1:
        bits_per_trial = np.log2(N)
    else:
        bits_per_trial = (
            np.log2(N)
            + P * np.log2(P)
            + (1 - P) * np.log2((1 - P) / (N - 1))
        )

    bits_per_trial = max(bits_per_trial, 0)

    itr = bits_per_trial * (60 / T)

    return itr


itr = calculate_itr(
    N=N,
    P=accuracy,
    T=selection_time
)

# =====================================================
# PREDICTED FATIGUE
# =====================================================

# Prototype assumption:
# Higher irritation leads to higher predicted fatigue.
predicted_fatigue = irritation

# Keep fatigue within 1–5
predicted_fatigue = min(
    max(predicted_fatigue, 1),
    5
)

# =====================================================
# HF-ITR
# =====================================================

I_norm = (irritation - 1) / 4
F_norm = (predicted_fatigue - 1) / 4

penalty = (
    wI * I_norm
    + wF * F_norm
)

hf_itr = itr * (
    1 - alpha * penalty
)

reduction = itr - hf_itr

if itr > 0:
    reduction_percent = (reduction / itr) * 100
else:
    reduction_percent = 0

# =====================================================
# OUTPUT CARDS
# =====================================================

st.subheader("Results")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Accuracy",
        f"{accuracy_percent:.2f}%"
    )

with col2:
    st.metric(
        "Conventional ITR",
        f"{itr:.2f} bits/min"
    )

with col3:
    st.metric(
        "Predicted Fatigue",
        f"{predicted_fatigue:.1f} / 5"
    )

with col4:
    st.metric(
        "HF-ITR",
        f"{hf_itr:.2f} bits/min"
    )

# =====================================================
# INTERPRETATION
# =====================================================

st.subheader("Interpretation")

if accuracy_percent <= (100 / N):
    st.warning(
        f"Accuracy is at or below chance level "
        f"({100/N:.2f}%). Conventional ITR is approximately zero."
    )

if irritation <= 2:
    st.success(
        "Low visual irritation. The stimulus may be suitable for longer use."
    )
elif irritation == 3:
    st.warning(
        "Moderate visual irritation. The stimulus may require monitoring."
    )
else:
    st.error(
        "High visual irritation. The stimulus may reduce practical usability."
    )

st.info(
    f"HF-ITR is reduced by {reduction_percent:.2f}% "
    f"after applying the human-factor penalty."
)

# =====================================================
# ITR CURVE CHART
# =====================================================

st.subheader("ITR Curve")

acc_range = np.linspace(1, 100, 300)
acc_decimal = acc_range / 100

itr_curve = []

for p in acc_decimal:
    itr_value = calculate_itr(
        N=N,
        P=p,
        T=selection_time
    )
    itr_curve.append(itr_value)

# HF-ITR curve using the same human-factor penalty
hf_itr_curve = [
    value * (1 - alpha * penalty)
    for value in itr_curve
]

fig = go.Figure()

# Conventional ITR curve
fig.add_trace(
    go.Scatter(
        x=acc_range,
        y=itr_curve,
        mode="lines",
        name="Conventional ITR",
        line=dict(width=3)
    )
)

# HF-ITR curve
fig.add_trace(
    go.Scatter(
        x=acc_range,
        y=hf_itr_curve,
        mode="lines",
        name="HF-ITR",
        line=dict(width=3, dash="dash")
    )
)

# Current ITR point
fig.add_trace(
    go.Scatter(
        x=[accuracy_percent],
        y=[itr],
        mode="markers+text",
        name="Current ITR",
        text=[f"ITR = {itr:.2f}"],
        textposition="top center",
        marker=dict(size=12)
    )
)

# Current HF-ITR point
fig.add_trace(
    go.Scatter(
        x=[accuracy_percent],
        y=[hf_itr],
        mode="markers+text",
        name="Current HF-ITR",
        text=[f"HF-ITR = {hf_itr:.2f}"],
        textposition="bottom center",
        marker=dict(size=12)
    )
)

fig.update_layout(
    height=420,
    margin=dict(l=20, r=20, t=30, b=20),
    xaxis_title="Accuracy (%)",
    yaxis_title="bits/min",
    plot_bgcolor="white",
    paper_bgcolor="white",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5
    )
)

fig.update_yaxes(
    showgrid=True,
    gridcolor="rgba(0,0,0,0.08)",
    zeroline=False
)

fig.update_xaxes(
    showgrid=False,
    range=[0, 100]
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# SUMMARY TABLE
# =====================================================

st.subheader("Input and Output Summary")

summary_data = {
    "Parameter": [
        "Number of Commands / Classes",
        "Total Trials",
        "Correct Trials",
        "Accuracy",
        "Selection Time",
        "Visual Irritation",
        "Predicted Fatigue",
        "wI",
        "wF",
        "α",
        "Conventional ITR",
        "HF-ITR",
        "Reduction"
    ],
    "Value": [
        N,
        total_trials,
        correct_trials,
        f"{accuracy_percent:.2f}%",
        f"{selection_time:.2f} s",
        f"{irritation} / 5",
        f"{predicted_fatigue:.1f} / 5",
        f"{wI:.2f}",
        f"{wF:.2f}",
        f"{alpha:.2f}",
        f"{itr:.2f} bits/min",
        f"{hf_itr:.2f} bits/min",
        f"{reduction_percent:.2f}%"
    ]
}

st.table(summary_data)

# =====================================================
# FORMULAS
# =====================================================

st.subheader("Formulas")

st.latex(
r"""
B =
\log_2(N)
+
P\log_2(P)
+
(1-P)
\log_2
\left(
\frac{1-P}{N-1}
\right)
"""
)

st.latex(
r"""
ITR = B \times \frac{60}{T}
"""
)

st.latex(
r"""
HF\text{-}ITR =
ITR
\times
\left[
1-
\alpha
\left(
w_I
\frac{I-I_{min}}
     {I_{max}-I_{min}}
+
w_F
\frac{F-F_{min}}
     {F_{max}-F_{min}}
\right)
\right]
"""
)

st.write(
    "In this prototype, predicted fatigue is estimated from visual irritation."
)

st.latex(
r"""
F = I
"""
)