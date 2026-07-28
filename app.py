import joblib
import streamlit as st
import pandas as pd
import altair as alt

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

## Sidebar — branding and context
with st.sidebar:
    st.markdown("### 🏦 LoanSense")
    st.caption("AI-powered loan approval prediction")
    st.divider()
    st.write("""
    This app predicts whether a loan application will be **Approved** or 
    **Rejected**, using a Gradient Boosting model trained on over 4,000 
    real loan applications.
    """)
    st.info("💡 **CIBIL Score** is India's credit score system (300–900), similar to a credit score used elsewhere. Higher scores mean stronger creditworthiness and lower default risk.")
    with st.expander("Why these 5 features?"):
        st.write("""
        Feature importance analysis showed that cibil_score, loan_term, loan_amount, 
        income_annum, and bank_asset_value together drive nearly all predictive power — 
        features like education and self-employment status had near-zero impact. Using 
        only these 5 features gives a simpler, faster, more interpretable model without 
        sacrificing accuracy.
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
        loan_amount = st.number_input("Loan Amount Requested (₹)", min_value=0, max_value=50000000,
                                       value=10000000, step=100000)
        st.caption(f"≈ SGD {loan_amount * SGD_RATE:,.0f}")

    with col2:
        income_annum = st.number_input("Applicant's Annual Income (₹)", min_value=0, max_value=20000000,
                                        value=5000000, step=100000)
        st.caption(f"≈ SGD {income_annum * SGD_RATE:,.0f}")
        bank_asset_value = st.number_input("Bank Asset Value (₹)", min_value=0, max_value=30000000,
                                            value=4000000, step=100000)
        st.caption(f"≈ SGD {bank_asset_value * SGD_RATE:,.0f}")

    predict_clicked = st.button("Predict Loan Status", use_container_width=True, type="primary")

## Live sensitivity chart — how approval probability shifts with CIBIL score,
## holding the other current inputs fixed, with a marker showing where the
## current application sits.
st.subheader("How CIBIL Score Affects This Application")

cibil_range = list(range(300, 901, 10))
sensitivity_rows = []
for score in cibil_range:
    row = pd.DataFrame({
        'cibil_score': [score],
        'loan_term': [loan_term],
        'loan_amount': [loan_amount],
        'income_annum': [income_annum],
        'bank_asset_value': [bank_asset_value]
    })
    row = row.reindex(columns=model.feature_names_in_, fill_value=0)
    prob_approved = model.predict_proba(row)[0][1]
    sensitivity_rows.append({'CIBIL Score': score, 'Approval Probability': prob_approved})

sensitivity_df = pd.DataFrame(sensitivity_rows)

## Current selection's exact probability, for the marker
current_row = pd.DataFrame({
    'cibil_score': [cibil_score],
    'loan_term': [loan_term],
    'loan_amount': [loan_amount],
    'income_annum': [income_annum],
    'bank_asset_value': [bank_asset_value]
})
current_row = current_row.reindex(columns=model.feature_names_in_, fill_value=0)
current_prob = model.predict_proba(current_row)[0][1]

line = alt.Chart(sensitivity_df).mark_line(color="#F0A500", strokeWidth=3).encode(
    x=alt.X('CIBIL Score:Q', scale=alt.Scale(domain=[300, 900]), title="CIBIL Score"),
    y=alt.Y('Approval Probability:Q', scale=alt.Scale(domain=[0, 1]), title="Predicted Approval Probability")
)

marker_df = pd.DataFrame({'CIBIL Score': [cibil_score]})
marker = alt.Chart(marker_df).mark_rule(color="#FFFFFF", strokeDash=[6, 4], strokeWidth=2).encode(
    x=alt.X('CIBIL Score:Q', scale=alt.Scale(domain=[300, 900]))
)

label_df = pd.DataFrame({
    'CIBIL Score': [cibil_score],
    'Approval Probability': [current_prob],
    'label': [f"Your application: {cibil_score} → {current_prob*100:.1f}%"]
})
point = alt.Chart(label_df).mark_point(color="#FFFFFF", size=100, filled=True).encode(
    x=alt.X('CIBIL Score:Q', scale=alt.Scale(domain=[300, 900])),
    y=alt.Y('Approval Probability:Q', scale=alt.Scale(domain=[0, 1]))
)
text = alt.Chart(label_df).mark_text(align='left', dx=10, dy=-10, color="#FFFFFF", fontSize=13).encode(
    x=alt.X('CIBIL Score:Q', scale=alt.Scale(domain=[300, 900])),
    y=alt.Y('Approval Probability:Q', scale=alt.Scale(domain=[0, 1])),
    text='label:N'
)

chart = (line + marker + point + text).properties(height=350).configure_view(
    fill="#1C2541"
).configure_axis(
    labelColor="#F5F5F5",
    titleColor="#F5F5F5",
    gridColor="#2E3B55"
)

st.altair_chart(chart, use_container_width=True)
st.caption("The dashed line marks your current CIBIL score selection and its predicted approval probability, holding your other inputs fixed.")
st.caption("📌 Notice how sharply approval probability shifts around a CIBIL score of ~545 for this applicant profile — this reflects how dominant credit score is in the model's decision-making, consistent with our feature importance findings (88%).")

## Predict button
if predict_clicked:
    df_input = pd.DataFrame({
        'cibil_score': [cibil_score],
        'loan_term': [loan_term],
        'loan_amount': [loan_amount],
        'income_annum': [income_annum],
        'bank_asset_value': [bank_asset_value]
    })
    df_input = df_input.reindex(columns=model.feature_names_in_, fill_value=0)

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
            'Bank Assets (₹)': bank_asset_value,
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
    st.dataframe(history_df, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        csv = history_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download History as CSV",
            data=csv,
            file_name="loan_prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col_b:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()