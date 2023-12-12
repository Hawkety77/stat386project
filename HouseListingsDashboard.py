import streamlit as st
import pandas as pd
import re
import plotly.express as px
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import numpy as np

df_orig = pd.read_csv('zillow_orig.csv')
df = df_orig[['price', 'bathrooms', 'bedrooms', 'city', 'homeType', 'livingArea', 'zipcode', 'priceReduction', 'daysOnZillow', 'latitude', 'longitude']]
df = df[(df['homeType'] != 'LOT') & (df['homeType'] != 'MANUFACTURED')]
df = df[~df['homeType'].isin(['LOT', 'MANUFACTURED', 'MULTI_FAMILY'])]

with open('current_rate.txt', "r") as file:
    todays_30_year_rate = file.read()

rate = float(re.search(r'\d{1,2}\.\d+', todays_30_year_rate).group())/100

def calculate_30_year_payment(price):
    monthly_rate = rate/12
    M = price * (monthly_rate * (1 + monthly_rate)**360)/(((1 + monthly_rate)**360) - 1)
    return M

df['30_year_mortgage'] = df['price'].apply(calculate_30_year_payment)

fig1 = px.scatter_mapbox(
    df,
    lat='latitude',
    lon='longitude',
    size = 'price', 
    color = 'homeType', 
    title = '500+ Most Recent Listings in Utah County'
)
fig1.update_layout(mapbox_style="carto-positron")

df_homeType = df.value_counts('homeType').reset_index()

fig2 = px.pie(
    df_homeType,
    names='homeType',
    values=0, 
    title = 'Distribution of Home Types'
)

fig3 = px.scatter(df[df['homeType'].isin(['SINGLE_FAMILY', 'TOWNHOUSE', 'CONDO'])], 
                 'livingArea', 
                 'price', 
                 color = 'homeType', 
                 trendline= 'lowess', 
                 title = 'Home Price vs Living Space')
fig3.update_xaxes(range = [0, 4000])
fig3.update_yaxes(range = [0, 1000000])
fig3.update_traces(
    line=dict(width=3), 
    selector=dict(type='scatter', mode='lines')
)

fig4 = px.histogram(df[df['price'] < 2000000], x = 'price', 
                    title = 'Home Price Distribution')

df_avg_price_per_city = df.groupby('city').median(numeric_only = True).reset_index().sort_values('price', ascending = False)
fig5 = px.bar(df_avg_price_per_city, x = 'city', y = 'price', 
              title = 'Average Listing Price in Each City')

one_hot_encoded1 = pd.get_dummies(df['homeType'])
one_hot_encoded1.drop(columns = ['SINGLE_FAMILY'], inplace = True)
one_hot_encoded = pd.get_dummies(df['city'])
one_hot_encoded.drop(columns = ['Saratoga Springs'], inplace = True)
df_prepared = pd.concat([df, one_hot_encoded1], axis=1)
df_prepared = pd.concat([df_prepared, one_hot_encoded], axis=1)

df_prepared = df_prepared.drop(columns = ['city', 
                                          'homeType', 
                                          'priceReduction', 
                                          'zipcode', 
                                          'latitude', 
                                          'longitude', 
                                          '30_year_mortgage', 
                                          'daysOnZillow'])
df_prepared = df_prepared[-df_prepared['livingArea'].isna()]
df_prepared = df_prepared[df_prepared['price'] < 1500000]

params = {
    'learning_rate': 0.1,
    'max_depth': 3,
    'n_estimators': 100
}

model = xgb.XGBRegressor(**params)

model.fit(df_prepared.drop(columns = ['price']), df_prepared['price'])

cities = ['Alpine', 'American Fork', 'Cedar Hills', 'Draper', 'Eagle Mountain', 'Elk Ridge', 'Genola',
          'Goshen', 'Heber City', 'Highland', 'Lehi', 'Lindon', 'Mapleton', 'Orem', 'Payson', 'Pleasant Grove',
          'Provo', 'Salem', 'Santaquin', 'Spanish Fork', 'Springville', 'Vineyard', 'Woodland Hills']

def make_house_prediction(bathrooms, bedrooms, livingArea, homeType, city_user):
    df_personal_prediction = pd.DataFrame({
        'bathrooms': [bathrooms],
        'bedrooms': [bedrooms],
        'livingArea': [livingArea], 
        'CONDO': np.where(homeType == 'CONDO', 1, 0), 
        'TOWNHOUSE': np.where(homeType == 'TOWNHOUSE', 1, 0)
    })

    for city in cities:
        df_personal_prediction[city] = np.where(city_user == city, 1, 0)

    estimate = model.predict(df_personal_prediction)[0]

    return estimate

##### Streamlit ######

st.set_page_config(
    page_title="Your App Title",
    page_icon="📊",  # You can set an icon if you like
    layout="wide",    # Use "wide" layout for a wider page
)

st.markdown(f"<h1 style='text-align: center;'>Estimate the cost of a house in Utah County</h1>", unsafe_allow_html= True)
st.markdown(f"<p style='text-align: center;'>*This prediction is not accurate for obscure combinations (i.e. 500 Sqft, 10 bedrooms, 1 bathroom)</p>", unsafe_allow_html= True)

col111, col222, col333, col444, col555 = st.columns(5)

bathrooms = col111.slider('Bathrooms:', 1, 5, 2)
bedrooms = col222.slider('Bedrooms:', 1, 10, 4)
livingArea = col333.slider('Square Footage:', 500, 6000, 2000)
homeType = col444.selectbox('Home Type:', ['CONDO', 'TOWNHOUSE', 'SINGLE_FAMILY'])
city_user = col555.selectbox('City:', cities)

estimate = make_house_prediction(bathrooms, bedrooms, livingArea, homeType, city_user)

col1111, col2222 = st.columns(2)

col1111.markdown(f"<h2 style='text-align: center;'>Predicted Cost: <br> ${int(estimate)}</h2>", unsafe_allow_html= True)
col2222.markdown(f"<h2 style='text-align: center;'>Estimated 30-year mortgage payment*: <br>${int(calculate_30_year_payment(estimate))}/month</h2>", unsafe_allow_html= True)
col2222.text(f'* This estimate is made using the current average 30 year lending rate: {todays_30_year_rate}')

st.markdown(f"<h1 style='text-align: center;'>Utah County Housing Market*</h1>", unsafe_allow_html= True)

col1, col2 = st.columns(2)

col1.plotly_chart(fig1, use_container_width=True)

col2.plotly_chart(fig4, use_container_width=True)

col1.plotly_chart(fig2, use_container_width=True)

col2.plotly_chart(fig3, use_container_width=True)

st.plotly_chart(fig5, use_container_width=True)

st.text('*This is not comparing the price of an identical home in each city (i.e. Houses in Woodland Hills are much larger than houses in Provo)')
