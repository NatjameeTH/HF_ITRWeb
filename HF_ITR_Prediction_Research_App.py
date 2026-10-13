"""HF-ITR Web-Based Research Calculator.



Start:

    python -m pip install -r requirements.txt

    python -m streamlit run HF_ITR_Prediction_Research_App.py



Research prototype for preliminary laboratory estimation only.

The app does not acquire EEG or predict classification accuracy.

"""



from __future__ import annotations



import math



import numpy as np

import pandas as pd

import plotly.graph_objects as go

import streamlit as st



st.set_page_config(

    page_title="HF-ITR | Research Calculator",

    page_icon="🧠",

    layout="wide",

    initial_sidebar_state="expanded",

)



WEIGHT_CASES = (0.0, 0.25, 0.50, 0.75, 1.0)

BLUE = "#526BFF"

RED = "#FF6349"

GREEN = "#00B989"

PURPLE = "#AC63EB"

NAVY = "#24384F"

GRID = "#E7EDF4"



st.markdown(

    """

    <style>

      [data-testid="stAppViewContainer"] {

          background: linear-gradient(180deg, #F5F9FD 0, #FFFFFF 410px);

      }

      .block-container {max-width: 1190px; padding-top: 1.8rem; padding-bottom: 3rem;}

      h1, h2, h3 {color:#24384F; letter-spacing:-.02em;}

      [data-testid="stMetric"] {background:#FFF; border:1px solid #E0E8F0;

          border-radius:13px; padding:14px 16px;}

      [data-testid="stMetricValue"] {color:#24384F;}

      .hero {border-radius:20px; background:linear-gradient(110deg, #16364F, #24657A);

          padding:44px 46px; color:#FFFFFF;}

      .hero h1 {color:#FFFFFF; font-size:clamp(2rem,4vw,3.4rem); margin:.6rem 0;}

      .hero p {color:#E8F5FA; font-size:1.05rem; max-width:780px;}

      .eyebrow {text-transform:uppercase; color:#2D807B; font-size:.78rem;

          font-weight:750; letter-spacing:1.4px; margin-bottom:.65rem;}

      .tag {text-transform:uppercase; color:#B9F9EA; font-size:.78rem;

          font-weight:750; letter-spacing:1.4px;}

      .stButton > button {border-radius:10px;}

      /* Home CTA button: teal, without changing weight-preset buttons */
      .st-key-home_open_button button {
          background-color: #24657A !important;
          border-color: #24657A !important;
          color: #FFFFFF !important;
          border-radius: 10px;
          min-height: 44px;
      }
      .st-key-home_open_button button:hover {
          background-color: #17495B !important;
          border-color: #17495B !important;
          color: #FFFFFF !important;
      }
      .st-key-home_open_button button:focus-visible {
          outline: 3px solid #7CC5CF;
          outline-offset: 2px;
      }

      /* Keep a clear gap between the 3 cards and the CTA */
      .home-button-gap {height: 22px;}


      div[data-testid="stAlert"] {border-radius:12px;}

      /* Equal-height cards on the Home page */
      .how-card {
          box-sizing: border-box;
          min-height: 205px;
          height: 100%;
          padding: 22px 18px;
          border: 1px solid #D8DFE7;
          border-radius: 12px;
          background: #FFFFFF;
          display: flex;
          flex-direction: column;
      }
      .how-card h3 {
          margin: 0 0 18px 0;
          font-size: 1.18rem;
          line-height: 1.35;
          color: #24384F;
      }
      .how-card p {
          margin: 0;
          line-height: 1.6;
          color: #24384F;
      }
      @media(max-width:720px){
          .hero{padding:28px 24px;}
          .how-card{min-height:0;}
      }

    </style>

    """,

    unsafe_allow_html=True,

)





# -----------------------------------------------------------------------------

# Calculation engine: independent of page logic.

# -----------------------------------------------------------------------------

def conventional_itr(n: int, p: float, t: float) -> float:

    """Shannon-style BCI ITR (bits/min), displayed as zero at/below chance.



    ``p`` is 0...1 and ``t`` is seconds per selected unit/window.

    The zero-at/below-chance rule is an explicit app reporting convention.

    """

    if n < 2 or not all(math.isfinite(v) for v in (p, t)) or not (0 <= p <= 1) or t <= 0:

        return float("nan")

    if p <= 1 / n:

        return 0.0

    if p == 1.0:

        bits = math.log2(n)

    else:

        bits = math.log2(n) + p * math.log2(p) + (1 - p) * math.log2((1 - p) / (n - 1))

    return float(max(0.0, bits) * 60 / t)





def predict_fatigue(irritation: float) -> float:

    """Preliminary linear fatigue estimate, clipped to the scale 1...5."""

    if not math.isfinite(irritation) or not 1 <= irritation <= 5:

        raise ValueError("Visual irritation must be between 1 and 5.")

    return float(np.clip(0.50 * irritation + 1.56, 1, 5))





def human_factor_penalty(irritation: float, fatigue: float, w_ir: float) -> float:

    """Normalized weighted factor, where wFt = 1 - wIr."""

    if not all(math.isfinite(v) for v in (irritation, fatigue, w_ir)):

        raise ValueError("Scores and weight must be finite.")

    if not (1 <= irritation <= 5 and 1 <= fatigue <= 5 and 0 <= w_ir <= 1):

        raise ValueError("Scores must be 1–5 and wIr must be 0–1.")

    return float(w_ir * (irritation - 1) / 4 + (1 - w_ir) * (fatigue - 1) / 4)





def compute_hfitr(itr: float, irritation: float, fatigue: float, w_ir: float) -> float:

    return float(itr * (1 - human_factor_penalty(irritation, fatigue, w_ir)))





def weight_sensitivity(itr: float, irritation: float, fatigue: float) -> pd.DataFrame:

    """Compare the five *weights* for a single entered parameter setting."""

    equal = compute_hfitr(itr, irritation, fatigue, 0.5)

    return pd.DataFrame([

        {

            "Irritation weight (wIr)": wi,

            "Fatigue weight (wFt)": 1 - wi,

            "Conventional ITR (bits/min)": itr,

            "HF-ITR (bits/min)": compute_hfitr(itr, irritation, fatigue, wi),

            "Change vs equal weights (%)": (

                100 * (compute_hfitr(itr, irritation, fatigue, wi) / equal - 1)

                if equal > 0 else np.nan

            ),

        }

        for wi in WEIGHT_CASES

    ])





def as_csv_bytes(df: pd.DataFrame) -> bytes:

    return df.to_csv(index=False).encode("utf-8-sig")





# -----------------------------------------------------------------------------

# Plotly figures.

# -----------------------------------------------------------------------------

def decorate_figure(fig: go.Figure, height: int = 430, legend_right: bool = False) -> go.Figure:

    fig.update_layout(

        template="plotly_white",

        height=height,

        font=dict(family="Arial, sans-serif", size=12, color=NAVY),

        margin=dict(l=15, r=160 if legend_right else 15, t=30, b=20),

        legend=(

            dict(orientation="v", x=1.02, xanchor="left", y=1, yanchor="top")

            if legend_right else dict(orientation="h", x=0, y=1.16, xanchor="left")

        ),

    )

    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False)

    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, rangemode="tozero")

    return fig





def itr_accuracy_chart(n: int, t: float, p: float, itr: float, hf_itr: float,

                       penalty: float) -> go.Figure:

    """Conventional ITR solid line, HF-ITR dashed line, 2 live markers."""

    xs = np.linspace(1 / n, 1, 300)

    itr_ys = np.array([conventional_itr(n, float(x), t) for x in xs])

    hf_ys = itr_ys * (1 - penalty)



    fig = go.Figure()

    fig.add_trace(go.Scatter(

        x=xs * 100, y=itr_ys, mode="lines", name="Conventional ITR",

        line=dict(color=BLUE, width=3),

        hovertemplate="Accuracy: %{x:.1f}%<br>ITR: %{y:.3f} bits/min<extra></extra>",

    ))

    fig.add_trace(go.Scatter(

        x=xs * 100, y=hf_ys, mode="lines", name="Estimated HF-ITR",

        line=dict(color=RED, width=3, dash="dash"),

        hovertemplate="Accuracy: %{x:.1f}%<br>HF-ITR: %{y:.3f} bits/min<extra></extra>",

    ))

    fig.add_trace(go.Scatter(

        x=[p * 100], y=[itr], mode="markers", name="Current ITR",

        marker=dict(color=GREEN, size=11, line=dict(color="#FFFFFF", width=1.5)),

        hovertemplate="Current accuracy: %{x:.2f}%<br>ITR: %{y:.3f} bits/min<extra></extra>",

    ))

    fig.add_trace(go.Scatter(

        x=[p * 100], y=[hf_itr], mode="markers", name="Current HF-ITR",

        marker=dict(color=PURPLE, size=11, line=dict(color="#FFFFFF", width=1.5)),

        hovertemplate="Current accuracy: %{x:.2f}%<br>HF-ITR: %{y:.3f} bits/min<extra></extra>",

    ))

    if p > 1 / n:

        fig.add_vline(x=p * 100, line=dict(color="#B9C5D1", dash="dot", width=1))

    decorate_figure(fig, height=480, legend_right=True)

    fig.update_xaxes(title="Classification accuracy (%)", range=[0, 100], dtick=20)

    fig.update_yaxes(title="bits/min")

    return fig





def weight_chart(itr: float, irritation: float, fatigue: float, w_ir: float) -> go.Figure:

    ws = np.linspace(0, 1, 101)

    hfs = [compute_hfitr(itr, irritation, fatigue, float(w)) for w in ws]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=ws, y=hfs, mode="lines", name="HF-ITR",

                             line=dict(color=BLUE, width=3)))

    fig.add_trace(go.Scatter(x=[w_ir], y=[compute_hfitr(itr, irritation, fatigue, w_ir)],

                             mode="markers", name="Selected weight",

                             marker=dict(color=PURPLE, size=11)))

    fig.add_vline(x=0.5, line_dash="dot", line_color="#93A5B8",

                  annotation_text="Equal weights", annotation_position="top left")

    decorate_figure(fig, height=360)

    fig.update_xaxes(title="Irritation weight (wIr)", range=[0, 1], dtick=0.25)

    fig.update_yaxes(title="HF-ITR (bits/min)")

    return fig





# -----------------------------------------------------------------------------

# Shared navigation / research notice.

# -----------------------------------------------------------------------------

def change_page(target: str) -> None:

    st.session_state["page"] = target





def change_weight(value: float) -> None:

    st.session_state["w_ir"] = float(value)





def research_notice() -> None:

    st.warning(

        "**EXPLORATORY RESEARCH PROTOTYPE — PRELIMINARY LABORATORY USE ONLY.** "

        "This calculator is not a clinical tool, a validated individual fatigue predictor, "

        "or a real-time BCI system. Fatigue estimation and HF-ITR are model-based."

    )





if "page" not in st.session_state:

    st.session_state["page"] = "Home"

if "w_ir" not in st.session_state:

    st.session_state["w_ir"] = 0.5



with st.sidebar:

    st.markdown("## 🧠 HF-ITR")

    st.caption("Web-Based Research Calculator")

    page = st.radio("Navigation", ["Home", "Calculator", "Methodology"],

                    key="page", label_visibility="collapsed")

    st.divider()

    st.caption("**Research prototype** · Preliminary laboratory estimation only")





# -----------------------------------------------------------------------------

# PAGE 1: MAIN / HOME

# -----------------------------------------------------------------------------

if page == "Home":

    st.markdown(

        """

        <section class="hero">

          <div class="tag">SSVEP-BCI · Exploratory Research Prototype</div>

          <h1>Human-Factor<br>Information Transfer Rate</h1>

          <p>Calculate and directly Compare conventional ITR and HF-ITR using communication performance and visual irritation inputs.</p>

          <p><strong>One calculator. For human-factor-aware BCI evaluation.</strong></p>

        </section>

        """,

        unsafe_allow_html=True,

    )

    st.write("")

    st.subheader("How the calculator works")

    c1, c2, c3 = st.columns(3, gap="large")

    with c1:
        st.markdown("""
        <div class="how-card">
          <h3>1 · Enter BCI performance</h3>
          <p>Set the number of commands, classification accuracy and time per selection.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="how-card">
          <h3>2 · Enter visual irritation</h3>
          <p>Estimate fatigue from irritation, or use an observed fatigue score.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="how-card">
          <h3>3 · Compare two metrics</h3>
          <p>Adjust wIr/wFt and view conventional ITR and HF-ITR together.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="home-button-gap" aria-hidden="true"></div>',
                unsafe_allow_html=True)

    st.button("Open HF-ITR Calculator →", type="primary", use_container_width=True,
              key="home_open_button", on_click=change_page, args=("Calculator",))

    st.write("")

    research_notice()

    st.info(

        "The tool does not predict EEG classification accuracy. Weight changes adjust the "

        "HF-ITR index by definition; the conventional ITR remains unchanged."

    )





# -----------------------------------------------------------------------------

# PAGE 2: THE GENERIC HF-ITR CALCULATOR

# -----------------------------------------------------------------------------

elif page == "Calculator":

    st.markdown('<div class="eyebrow">Interactive research calculation</div>',

                unsafe_allow_html=True)

    st.title("HF-ITR Calculator")

    st.write("Enter parameters for any BCI configuration. Compare conventional ITR and HF-ITR instantly.")

    research_notice()



    col_left, col_right = st.columns(2, gap="large")

    with col_left:

        with st.container(border=True):

            st.markdown("### 1. Conventional ITR inputs")

            n = int(st.number_input("Number of BCI commands (N)", min_value=2,

                                    max_value=30, value=5, step=1))

            p_mode = st.radio("Accuracy input method", ["Correct / total", "Direct accuracy (%)"],

                              horizontal=True)

            if p_mode == "Correct / total":

                a, b = st.columns(2)

                with a:

                    total = int(st.number_input("Total presentations", min_value=1,

                                                max_value=1000000, value=100, step=1))

                with b:

                    correct = int(st.number_input("Correct selections", min_value=0,

                                                  max_value=total, value=min(total, 80), step=1))

                p = correct / total

                st.caption(f"Calculated accuracy: **{100*p:.2f}%**")

            else:

                p = float(st.number_input("Classification accuracy (%)", min_value=0.0,

                                          max_value=100.0, value=80.0, step=0.5)) / 100

            t = float(st.number_input("Time per selection / detection (T, seconds)",

                                      min_value=0.1, max_value=600.0, value=6.0,

                                      step=0.5, format="%.1f"))

            time_mode = st.radio("T represents", ["Detection window only", "Full selection time"],

                                 horizontal=True)

            if time_mode == "Detection window only":

                st.caption("Window-based theoretical ITR; cue, fixation and rest are excluded.")

            else:

                st.caption("Include all relevant selection overhead in the T value you enter.")



    with col_right:

        with st.container(border=True):

            st.markdown("### 2. Human-factor inputs")

            ir = float(st.slider("Visual irritation (Ir), 1–5", min_value=1.0,

                                 max_value=5.0, value=4.0, step=0.25))

            fatigue_mode = st.radio("Fatigue input method", [

                "Estimate fatigue from irritation (pre-experimental)",

                "Enter observed fatigue score",

            ])

            if fatigue_mode.startswith("Estimate"):

                ft = predict_fatigue(ir)

                st.metric("Regression-estimated fatigue (Ft)", f"{ft:.2f} / 5")

                st.caption("Ft = clip(0.50 × Ir + 1.56, 1, 5). Not independently validated for new users.")

            else:

                ft = float(st.slider("Observed visual fatigue (Ft), 1–5",

                                     min_value=1.0, max_value=5.0, value=3.0, step=0.25))

                st.metric("Observed fatigue (Ft)", f"{ft:.2f} / 5")

                st.caption("Uses the fatigue rating you enter, instead of a regression estimate.")



    with st.container(border=True):

        st.markdown("### 3. Weight adjustment")

        st.write("Select a standard weight pair or adjust wIr freely. wFt updates automatically, so both weights sum to 1.")

        cols = st.columns(5)

        for col, preset in zip(cols, WEIGHT_CASES):

            with col:

                st.button(f"{preset:.2f} / {1-preset:.2f}",

                          key=f"preset_{preset:.2f}", use_container_width=True,

                          type="primary" if abs(st.session_state["w_ir"] - preset) < 1e-9 else "secondary",

                          on_click=change_weight, args=(preset,))

        w_ir = float(st.slider("Irritation weight (wIr)", min_value=0.0, max_value=1.0,

                               step=0.05, key="w_ir"))

        w_ft = 1 - w_ir

        w1, w2, w3 = st.columns(3)

        w1.metric("Irritation weight (wIr)", f"{w_ir:.2f}")

        w2.metric("Fatigue weight (wFt)", f"{w_ft:.2f}")

        w3.metric("Weight sum", f"{w_ir+w_ft:.2f}")

        st.caption("Default: equal weights (0.50 / 0.50). Other weights are sensitivity scenarios, not optimized coefficients.")



    itr = conventional_itr(n, p, t)

    penalty = human_factor_penalty(ir, ft, w_ir)

    hf_itr = compute_hfitr(itr, ir, ft, w_ir)



    st.subheader("Calculation results")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Classification accuracy", f"{p*100:.2f}%")

    m2.metric("Conventional ITR", f"{itr:.3f} bits/min")

    m3.metric("HF-ITR", f"{hf_itr:.3f} bits/min")

    m4.metric("Human-factor penalty", f"{penalty*100:.2f}%")

    if p <= 1 / n:

        st.warning("Accuracy is at or below chance (1/N); the app reports both ITR values as zero by convention.")

    else:

        st.info(

            f"With wIr={w_ir:.2f}, wFt={w_ft:.2f}, and Ft={ft:.2f}, "

            f"HF-ITR is **{itr-hf_itr:.3f} bits/min** lower than conventional ITR "

            f"({penalty*100:.2f}% adjustment)."

        )



    st.markdown("### ITR versus classification accuracy")

    st.plotly_chart(itr_accuracy_chart(n, t, p, itr, hf_itr, penalty),

                    use_container_width=True)

    st.caption(

        "Solid blue: conventional ITR. Dashed red: HF-ITR at the selected human-factor scores "

        "and weights. Green and purple markers: the current calculation. "

        "The curves represent theoretical calculations, not measured changes in accuracy."

    )



    with st.expander("Weight sensitivity (5 standard combinations)", expanded=False):

        ca, cb = st.columns([1.05, 1.0], gap="large")

        with ca:

            st.plotly_chart(weight_chart(itr, ir, ft, w_ir), use_container_width=True)

        with cb:

            table = weight_sensitivity(itr, ir, ft)

            st.dataframe(table, use_container_width=True, hide_index=True,

                         column_config={

                             "Irritation weight (wIr)": st.column_config.NumberColumn(format="%.2f"),

                             "Fatigue weight (wFt)": st.column_config.NumberColumn(format="%.2f"),

                             "Conventional ITR (bits/min)": st.column_config.NumberColumn(format="%.3f"),

                             "HF-ITR (bits/min)": st.column_config.NumberColumn(format="%.3f"),

                             "Change vs equal weights (%)": st.column_config.NumberColumn(format="%+.2f"),

                         })

            st.download_button("Download weight sensitivity (CSV)", as_csv_bytes(table),

                               file_name="hfitr_weight_sensitivity.csv", mime="text/csv",

                               use_container_width=True)

        st.caption("The analysis changes weight assumptions while holding conventional ITR and input scores constant.")



    result = pd.DataFrame([{

        "Commands (N)": n,

        "Accuracy (%)": p * 100,

        "Time T (s)": t,

        "Time basis": time_mode,

        "Visual irritation (Ir)": ir,

        "Fatigue score (Ft)": ft,

        "Fatigue source": fatigue_mode,

        "Irritation weight (wIr)": w_ir,

        "Fatigue weight (wFt)": w_ft,

        "Human-factor penalty (%)": penalty * 100,

        "Conventional ITR (bits/min)": itr,

        "HF-ITR (bits/min)": hf_itr,

    }])

    st.download_button("Download current calculation (CSV)", as_csv_bytes(result),

                       file_name="hfitr_current_calculation.csv", mime="text/csv")



    with st.expander("Show calculation equations", expanded=False):

        st.latex(r"B=\log_2N+P\log_2P+(1-P)\log_2\left(\frac{1-P}{N-1}\right)")

        st.latex(r"ITR=B\frac{60}{T}")

        st.latex(r"\widehat{Ft}=\operatorname{clip}(0.50Ir+1.56,1,5)")

        st.latex(r"HF=w_{Ir}\frac{Ir-1}{4}+(1-w_{Ir})\frac{Ft-1}{4}")

        st.latex(r"HF\text{-}ITR=ITR(1-HF)")





# -----------------------------------------------------------------------------

# PAGE 3: METHODOLOGY AND LIMITATIONS

# -----------------------------------------------------------------------------

else:

    st.markdown('<div class="eyebrow">Formulas and scope</div>', unsafe_allow_html=True)

    st.title("Methodology")

    research_notice()

    with st.container(border=True):

        st.markdown("### Conventional ITR")

        st.latex(r"B=\log_2N+P\log_2P+(1-P)\log_2\left(\frac{1-P}{N-1}\right)")

        st.latex(r"ITR=B\frac{60}{T}")

        st.write("N: number of BCI commands; P: classification accuracy between 0 and 1; T: seconds per selection/detection.")

        st.write("The calculator shows ITR = 0 at or below chance accuracy (P ≤ 1/N), as an explicit reporting convention.")

        st.write("If T covers only detection/stimulation time, ITR excludes cue and rest overhead.")

    with st.container(border=True):

        st.markdown("### Regression-based fatigue estimation")

        st.latex(r"\widehat{Ft}=\operatorname{clip}(0.50Ir+1.56,1,5)")

        st.write("A preliminary fatigue estimate based on an irritation rating; not validated for individual prediction in new users.")

    with st.container(border=True):

        st.markdown("### Human-factor-adjusted ITR")

        st.latex(r"HF=w_{Ir}\frac{Ir-1}{4}+w_{Ft}\frac{Ft-1}{4}")

        st.latex(r"w_{Ir}+w_{Ft}=1")

        st.latex(r"HF\text{-}ITR=ITR(1-HF)")

        st.write("Ir and Ft are five-point ordinal ratings (1 minimum, 5 maximum).")

        st.write("Default weights: wIr = wFt = 0.50. The five predefined weight settings are sensitivity checks, not optimized values.")

    with st.container(border=True):

        st.markdown("### Important limitations")

        st.markdown(

            """- Five-point subjective ratings are ordinal. Linear normalization and weighted arithmetic assume approximately equal intervals.

- The calculator does **not** predict BCI accuracy, acquire EEG or classify stimuli.

- HF-ITR varies with the assumed irritation/fatigue penalty; this is a mathematical adjustment, **not evidence that performance or accuracy declined**.

- The regression is an in-sample exploratory association and is not independently validated as a fatigue prediction model for new users.

- The metric is not validated against independent physiological fatigue or prolonged-use outcomes, and is not intended for clinical decisions.

"""

        )



st.divider()

st.caption("HF-ITR Web-Based Research Calculator · Exploratory laboratory prototype only · Not a clinical or diagnostic tool")
