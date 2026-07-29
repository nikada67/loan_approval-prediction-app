import joblib
import streamlit as st
import pandas as pd
from datetime import datetime

## Load trained model
try:
    model = joblib.load("best_model.pkl")
except FileNotFoundError:
    st.error("⚠️ Model file (best_model.pkl) not found. Please make sure it's in the same folder as this app.")
    st.stop()

## Approximate INR to SGD rate (July 2026) — for reference display only
SGD_RATE = 0.0134

## Streamlit app
st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦", layout="wide")

## Custom styling — unique navy/gold theme, custom font
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background: linear-gradient(180deg, #0B132B 0%, #1C2541 100%);
    }

    h1, h2, h3 {
        color: #F0A500 !important;
        font-weight: 700 !important;
    }

    .stButton>button {
        background-color: #F0A500;
        color: #0B132B;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 0.6em 0;
    }
    .stButton>button:hover {
        background-color: #D4900A;
        color: #0B132B;
    }

    [data-testid="stSidebar"] {
        background-color: #1C2541;
        border-right: 1px solid #F0A500;
    }
    </style>
""", unsafe_allow_html=True)

## Session state — stores this session's tested applications
if 'history' not in st.session_state:
    st.session_state.history = []


def compute_model_features(cibil_score, loan_term, loan_amount, income_annum,
                            residential_assets_value, commercial_assets_value,
                            luxury_assets_value, bank_asset_value):
    """Turn the raw applicant inputs into the 4 engineered features the model expects."""
    total_assets = (residential_assets_value + commercial_assets_value
                     + luxury_assets_value + bank_asset_value)
    loan_to_income_ratio = loan_amount / income_annum
    asset_to_loan_ratio = total_assets / loan_amount

    row = pd.DataFrame([{
        'loan_term': loan_term,
        'cibil_score': cibil_score,
        'Loan_to_Income_Ratio': loan_to_income_ratio,
        'Asset_to_Loan_Ratio': asset_to_loan_ratio,
    }])
    return row.reindex(columns=model.feature_names_in_, fill_value=0)


## Sidebar — branding and context
with st.sidebar:
    st.markdown("### 🏦 LoanSense")
    st.divider()
    st.write("""
    This app predicts whether a loan application will be **Approved** or
    **Rejected**, using a Gradient Boosting model trained on over 4,000
    real loan applications.
    """)
    st.info("💡 **CIBIL Score** is India's credit score system (300–900), similar to a credit score used elsewhere. Higher scores mean stronger creditworthiness and lower default risk.")
    with st.expander("Why these 4 features?"):
        st.write("""
        Feature importance analysis showed that **CIBIL score**, **loan term**,
        **loan-to-income ratio**, and **asset-to-loan ratio** together drive over
        99.8% of the model's predictive power — CIBIL score alone accounts for
        roughly 82%. The two ratios (built from loan amount, income, and total
        assets) turned out to be far more informative than any of the raw rupee
        amounts on their own, so this app collects the raw figures and computes
        the ratios behind the scenes. Features like education and self-employment
        status had essentially zero impact and were dropped.
        """)

## Header
st.title("🏦 Loan Approval Prediction")
st.write("Enter the applicant's details below to predict whether the loan will be approved or rejected.")
st.divider()

## Input form
with st.container(border=True):
    st.subheader("Applicant Details")
    col1, col2 = st.columns(2)

    with col1:
        cibil_score = st.slider("CIBIL Score", min_value=300, max_value=900, value=700,
                                 help="Credit score from 300 (poor) to 900 (excellent)")
        loan_term = st.slider("Loan Term (years)", min_value=2, max_value=20, value=10)
        loan_amount = st.number_input("Loan Amount Requested (₹)", min_value=100000, max_value=50000000,
                                       value=10000000, step=100000)
        st.caption(f"≈ SGD {loan_amount * SGD_RATE:,.0f}")

    with col2:
        income_annum = st.number_input("Applicant's Annual Income (₹)", min_value=100000, max_value=20000000,
                                        value=5000000, step=100000)
        st.caption(f"≈ SGD {income_annum * SGD_RATE:,.0f}")

        loan_to_income_ratio = loan_amount / income_annum
        if loan_to_income_ratio <= 3:
            lti_badge = "🟢 Comfortable"
        elif loan_to_income_ratio <= 5:
            lti_badge = "🟡 Moderate"
        else:
            lti_badge = "🔴 Stretched"
        st.caption(f"Loan-to-income ratio: **{loan_to_income_ratio:.2f}x** — {lti_badge}")

    st.markdown("**Applicant's Assets** — used together to gauge how well the loan is collateral-backed")
    col3, col4, col5, col6 = st.columns(4)
    with col3:
        residential_assets_value = st.number_input("Residential Assets (₹)", min_value=0, max_value=30000000,
                                                     value=3000000, step=100000)
        st.caption(f"≈ SGD {residential_assets_value * SGD_RATE:,.0f}")
    with col4:
        commercial_assets_value = st.number_input("Commercial Assets (₹)", min_value=0, max_value=20000000,
                                                    value=2000000, step=100000)
        st.caption(f"≈ SGD {commercial_assets_value * SGD_RATE:,.0f}")
    with col5:
        luxury_assets_value = st.number_input("Luxury Assets (₹)", min_value=0, max_value=40000000,
                                                value=5000000, step=100000)
        st.caption(f"≈ SGD {luxury_assets_value * SGD_RATE:,.0f}")
    with col6:
        bank_asset_value = st.number_input("Bank Assets (₹)", min_value=0, max_value=15000000,
                                            value=4000000, step=100000)
        st.caption(f"≈ SGD {bank_asset_value * SGD_RATE:,.0f}")

    total_assets = residential_assets_value + commercial_assets_value + luxury_assets_value + bank_asset_value
    asset_to_loan_ratio = total_assets / loan_amount if loan_amount else 0
    if asset_to_loan_ratio >= 1.5:
        atl_badge = "🟢 Well covered"
    elif asset_to_loan_ratio >= 0.7:
        atl_badge = "🟡 Partial coverage"
    else:
        atl_badge = "🔴 Thin coverage"
    st.caption(f"Total assets: ₹{total_assets:,.0f} (≈ SGD {total_assets * SGD_RATE:,.0f}) · "
               f"Asset-to-loan coverage: **{asset_to_loan_ratio:.2f}x** — {atl_badge}")
    st.caption("_Colour bands above are illustrative guides based on this dataset's typical ranges, not the model's exact decision boundary — the model itself weighs all four features together._")

    predict_clicked = st.button("Predict Loan Status", width="stretch", type="primary")

## Live sensitivity readout — how approval probability shifts with CIBIL score,
## holding the other current inputs fixed. No charting library involved,
## just live-updating numbers, to keep the app's dependencies minimal.
st.subheader("How CIBIL Score Affects This Application")

current_row = compute_model_features(cibil_score, loan_term, loan_amount, income_annum,
                                      residential_assets_value, commercial_assets_value,
                                      luxury_assets_value, bank_asset_value)
current_prob = model.predict_proba(current_row)[0][1]

lower_row = compute_model_features(max(300, cibil_score - 50), loan_term, loan_amount, income_annum,
                                    residential_assets_value, commercial_assets_value,
                                    luxury_assets_value, bank_asset_value)
higher_row = compute_model_features(min(900, cibil_score + 50), loan_term, loan_amount, income_annum,
                                     residential_assets_value, commercial_assets_value,
                                     luxury_assets_value, bank_asset_value)
lower_prob = model.predict_proba(lower_row)[0][1]
higher_prob = model.predict_proba(higher_row)[0][1]

col_low, col_mid, col_high = st.columns(3)
col_low.metric(f"CIBIL {max(300, cibil_score - 50)}", f"{lower_prob*100:.1f}%", help="Approval probability 50 points lower")
col_mid.metric(f"CIBIL {cibil_score} (your input)", f"{current_prob*100:.1f}%")
col_high.metric(f"CIBIL {min(900, cibil_score + 50)}", f"{higher_prob*100:.1f}%", help="Approval probability 50 points higher")
st.caption("📌 These three numbers hold your other inputs fixed and only move the CIBIL score, to show how much it alone swings the outcome — it accounts for roughly 82% of the model's feature importance.")

## Predict button
if predict_clicked:
    if loan_amount <= 0 or income_annum <= 0:
        st.error("⚠️ Loan amount and annual income must both be greater than zero.")
    else:
        df_input = compute_model_features(cibil_score, loan_term, loan_amount, income_annum,
                                           residential_assets_value, commercial_assets_value,
                                           luxury_assets_value, bank_asset_value)
        try:
            prediction = model.predict(df_input)[0]
            probability = model.predict_proba(df_input)[0]
            result_label = "Approved" if prediction == 1 else "Rejected"
            confidence = probability[1] if prediction == 1 else probability[0]

            st.divider()
            st.subheader("Result")
            with st.container(border=True):
                if prediction == 1:
                    st.success("Prediction: Loan Approved ✅")
                    st.progress(probability[1], text=f"Confidence: {probability[1]*100:.1f}%")
                    st.balloons()
                else:
                    st.error("Prediction: Loan Rejected ❌")
                    st.progress(probability[0], text=f"Confidence: {probability[0]*100:.1f}%")

            st.session_state.history.append({
                'CIBIL Score': cibil_score,
                'Loan Term (yrs)': loan_term,
                'Loan Amount (₹)': loan_amount,
                'Annual Income (₹)': income_annum,
                'Total Assets (₹)': total_assets,
                'Loan-to-Income': round(loan_amount / income_annum, 2),
                'Asset-to-Loan': round(asset_to_loan_ratio, 2),
                'Prediction': result_label,
                'Confidence (%)': round(confidence * 100, 1)
            })
        except Exception as e:
            st.error(f"⚠️ Something went wrong while making the prediction: {e}")

## Prediction history — this session's tested applications, downloadable
if st.session_state.history:
    st.divider()
    st.subheader("Your Prediction History")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, width="stretch")

    col_a, col_b = st.columns(2)
    with col_a:
        csv = history_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download History as CSV",
            data=csv,
            file_name=f"loan_prediction_history_{datetime.now():%Y%m%d_%H%M}.csv",
            mime="text/csv",
            width="stretch"
        )
    with col_b:
        if st.button("🗑️ Clear History", width="stretch"):
            st.session_state.history = []
            st.rerun()

st.divider()
st.caption("⚠️ This tool produces a statistical prediction based on historical loan data for a course project demo. "
           "It is not financial advice and should not be used for real lending decisions.")
