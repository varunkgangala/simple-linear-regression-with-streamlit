import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
#tabname
st.set_page_config(page_title="Linear Regression", layout="centered")
def load_csv(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_csv("style.css")
#Title
st.markdown("""
            <div class="card">
            <h1>Linear Regression Model</h1>
            <p>Predict<b> Tip Amount </b> from <b> Total Bill</b> using Linear Regression</p>
            </div>
            """, unsafe_allow_html=True)
#load data
@st.cache_data
def load_data():
    return sns.load_dataset('tips')
df=load_data()
#dataset preview
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html=True)
#prepare data
x,y=df[['total_bill']],df['tip']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
#train model
model=LinearRegression()
model.fit(x_train_scaled,y_train)
#predict
y_pred=model.predict(x_test_scaled)
#evaluation metrics
mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
rmse=np.sqrt(mse)
adj_r2=1-(1-r2)*(len(y_test)-1)/(len(y_test)-2)
#display evaluation
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Total Bill vs Tip Amount")
fig,ax=plt.subplots()
ax.scatter(df['total_bill'],df['tip'],alpha=0.6)
x_sorted = np.sort(df['total_bill'].values.reshape(-1, 1))
x_sorted_scaled = scaler.transform(x_sorted)
y_line = model.predict(x_sorted_scaled)
ax.plot(x_sorted, y_line, color='red')
ax.set_xlabel("Total Bill")
ax.set_ylabel("Tip Amount")
st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

#Performance Metrics
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Performance Metrics")
c1,c2=st.columns(2)
c1.metric("MAE",f"{mae:.2f}")
c2.metric("RMSE",f"{rmse:.2f}")
c3,c4=st.columns(2)
c3.metric("R² Score",f"{r2:.3f}")
c4.metric("Adjusted R²",f"{adj_r2:.3f}")
st.markdown('</div>', unsafe_allow_html=True)

# model coefficients
st.markdown(f"""
            <div class="card">
            <h2>Model Coefficients</h2>
            <p>{model.coef_}</p>
            <h2>Intercept</h2>
            <p>{model.intercept_}</p>
            </div>
            """, unsafe_allow_html=True)

# prediction
st.markdown('<div class="card>', unsafe_allow_html=True)
st.subheader("predict tip amount")
bill=st.slider("Total Bill",df['total_bill'].min(),df['total_bill'].max())
tip=model.predict(scaler.transform([[bill]]))[0]
st.markdown(f'<div class="prediction-box">Predicted Tip Amount: <b>{tip:.2f}</b></div>', unsafe_allow_html=True)
