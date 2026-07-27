import joblib
import streamlit as st
import pandas as pd

## Load trained model
try:
    model = joblib.load("best_model.pkl")
except FileNotFoundError:
    st.error("⚠️ Model file (best_model.pkl) not found. Please make sure it's in the same folder as this app.")
    st.stop()

## Streamlit app
st.set_page_config(page_title="Loan Approval Prediction", page_icon="💰")
st.title("Loan Approval Prediction")
st.write("Enter the applicant's details below to predict whether the loan will be approved or rejected.")

## User inputs — matches the 5 top features used by the final model
col1, col2 = st.columns(2)

with col1:
    cibil_score = st.slider("CIBIL Score", min_value=300, max_value=900, value=700,
                             help="Credit score from 300 (poor) to 900 (excellent)")
    loan_term = st.slider("Loan Term (years)", min_value=2, max_value=20, value=10)
    loan_amount = st.number_input("Loan Amount Requested ($)", min_value=0, max_value=50000000,
                                   value=10000000, step=100000)

with col2:
    income_annum = st.number_input("Applicant's Annual Income ($)", min_value=0, max_value=20000000,
                                    value=5000000, step=100000)
    bank_asset_value = st.number_input("Bank Asset Value ($)", min_value=0, max_value=30000000,
                                        value=4000000, step=100000)

with st.expander("Why these 5 features?"):
    st.write("""
    Our model was originally trained on all applicant features, but feature importance 
    analysis showed that cibil_score, loan_term, loan_amount, income_annum, and 
    bank_asset_value together drive nearly all predictive power — features like 
    education and self-employment status had near-zero impact. Using only these 5 
    features gave us a simpler, faster, and more interpretable model without 
    sacrificing accuracy.
    """)

## Predict button
if st.button("Predict Loan Status"):

    ## Convert input data to a DataFrame in the same column order the model was trained on
    df_input = pd.DataFrame({
        'cibil_score': [cibil_score],
        'loan_term': [loan_term],
        'loan_amount': [loan_amount],
        'income_annum': [income_annum],
        'bank_asset_value': [bank_asset_value]
    })

    ## Reindex to be safe, in case column order differs
    df_input = df_input.reindex(columns=model.feature_names_in_, fill_value=0)

    try:
        ## Predict
        prediction = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0]

        if prediction == 1:
            st.success(f"Prediction: Loan Approved ✅ (Confidence: {probability[1]*100:.1f}%)")
        else:
            st.error(f"Prediction: Loan Rejected ❌ (Confidence: {probability[0]*100:.1f}%)")
    except Exception as e:
        st.error(f"⚠️ Something went wrong while making the prediction: {e}")