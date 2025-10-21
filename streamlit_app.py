import json

import numpy as np
import pandas as pd
import streamlit as st
from pipelines.deployment_pipeline import prediction_service_loader
from run_deployment import run_deployment


def main():
    st.title("Customer Satisfaction Prediction")
    
    st.markdown("""
    ### Predict customer satisfaction for e-commerce orders
    Enter product and payment details below to get a satisfaction score (0-5).
    Built with ZenML and MLflow for production ML pipelines.
    """)
    
    st.divider()
    
    st.subheader("Product & Payment Information")
    
    payment_sequential = st.sidebar.slider("Payment Sequential")
    payment_installments = st.sidebar.slider("Payment Installments")
    payment_value = st.number_input("Payment Value")
    price = st.number_input("Price")
    freight_value = st.number_input("freight_value")
    product_name_length = st.number_input("Product name length")
    product_description_length = st.number_input("Product Description length")
    product_photos_qty = st.number_input("Product photos Quantity ")
    product_weight_g = st.number_input("Product weight measured in grams")
    product_length_cm = st.number_input("Product length (CMs)")
    product_height_cm = st.number_input("Product height (CMs)")
    product_width_cm = st.number_input("Product width (CMs)")

    if st.button("Predict"):
        try:
            service = prediction_service_loader(
                pipeline_name="continuous_deployment_pipeline",
                pipeline_step_name="mlflow_model_deployer_step",
                running=False,
            )
        except RuntimeError:
            st.write("No service found. Running deployment pipeline...")
            run_deployment()
            return

        df = pd.DataFrame(
            {
                "payment_sequential": [payment_sequential],
                "payment_installments": [payment_installments],
                "payment_value": [payment_value],
                "price": [price],
                "freight_value": [freight_value],
                "product_name_lenght": [product_name_length],
                "product_description_lenght": [product_description_length],
                "product_photos_qty": [product_photos_qty],
                "product_weight_g": [product_weight_g],
                "product_length_cm": [product_length_cm],
                "product_height_cm": [product_height_cm],
                "product_width_cm": [product_width_cm],
            }
        )
        json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
        data = np.array(json_list)
        pred = service.predict(data)
        st.success(f"Customer Satisfaction Score (0-5): {pred}")
    if st.button("Show Model Results"):
        st.subheader("Model Performance Comparison")
        st.write("Performance metrics of different models tested on the dataset:")
        
        df = pd.DataFrame(
            {
                "Models": ["LinearRegression", "LightGBM", "Xgboost"],
                "MSE": [1.864, 1.804, 1.781],
                "RMSE": [1.365, 1.343, 1.335],
                "R2": [0.018, 0.050, 0.062],
            }
        )
        st.dataframe(df)
        
        st.info("Current deployed model: LinearRegression (R2: 0.018)")


if __name__ == "__main__":
    main()
