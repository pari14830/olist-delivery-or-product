import streamlit as st 
import pandas as pd 
import joblib 

st.set_page_config(page_title="Olist Review Risk", page_icon="📦")

# 1. load the saved model (pipeline)
model = joblib.load('model.pkl')
df = pd.read_csv('olist_dashboard.zip')
states = sorted(df['customer_state'].unique())
categories = sorted(df['product_category_name_english'].unique())
#2. title
st.title('Olist: Negative Review Risk')
st.write("Enter the details of an order to see the risk of a negative review.")

#3. inputs: delivery
col1, col2 = st.columns(2)

with col1:
    st.subheader("Delivery")
    promised_days = st.slider("Promised delivery time (days)", 2, 60, 23)
    delivery_days = st.slider("Real delivery time (days)", 0, 60, 12)
    customer_state = st.selectbox("Customer state", states)
    seller_state = st.selectbox("Seller state", states)


# 4. features the app calculates itself
late_day = delivery_days - promised_days
is_late = 1 if late_day > 0 else 0


is_different_state = 1 if customer_state != seller_state else 0
with col2:
    st.subheader("Order")
    category = st.selectbox("Product category", categories)
    price = st.number_input("Price (BRL)", min_value=0.0, value=100.0)
    freight_value = st.number_input("Freight value (BRL)", min_value=0.0, value=20.0)


input_data = pd.DataFrame([{
    'late_day': late_day,
    'price': price,
    'freight_value': freight_value,
    'is_late': is_late,
    'is_different_state': is_different_state,
    'promised_days': promised_days,
    'delivery_days': delivery_days,
    'customer_state': customer_state,
    'seller_id': 'other',
    'seller_state': seller_state,
    'product_category_name_english': category
}])

if st.button("Predict"):
    proba = model.predict_proba(input_data)[0, 1]
    score = round(proba * 100, 1)

    if proba >= 0.5:
        st.error(f"High risk ({score}%): send an apology email or a coupon.")
    else:
        st.success(f"Low risk ({score}%): no action needed.")


   
    
st.divider()
st.header("Batch: score many orders")

uploaded_file = st.file_uploader("Upload a CSV file with orders", type="csv")

if uploaded_file is not None:
    orders = pd.read_csv(uploaded_file)
    # features the app calculates itself (same as the single order)
    orders['late_day'] = orders['delivery_days'] - orders['promised_days']
    orders['is_late'] = (orders['late_day'] > 0).astype(int)
    orders['is_different_state'] = (orders['customer_state'] != orders['seller_state']).astype(int)

    # risk score for every order
    orders['risk_score'] = model.predict_proba(orders)[:, 1]

    st.write("Orders scored:", len(orders))

        # only high-risk orders, highest score first
    high_risk = orders[orders['risk_score'] >= 0.5].sort_values('risk_score', ascending=False)

    st.metric("High-risk orders", f"{len(high_risk)} of {len(orders)}")

    st.dataframe(high_risk[['seller_id','order_id', 'risk_score', 'late_day',
                            'customer_state', 'product_category_name_english']])

    st.download_button("Download high-risk list", high_risk.to_csv(index=False),
                       file_name="high_risk_orders.csv")