import os
import math
import folium
import geopandas
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
import statsmodels as sm
from datetime import datetime
from folium.plugins import MarkerCluster
from plotly.subplots import make_subplots
from streamlit_folium import folium_static

from text_file import introduction, target_analysis, dictionary

pd.set_option('display.float_format', lambda x: '%.2f' % x)

st.set_page_config( layout='wide')


def get_data( path ):
    data = pd.read_csv(path)

    return data

@st.cache( allow_output_mutation=True)
def get_geofile( url ):
    geofile = geopandas.read_file( url )
    return geofile

def set_title():
    st.title("Real Estate Market Exploration in Seattle")

    st.text_area("Introduction", introduction, height=350)


def set_feature( data ):

    data['price_m2'] = data['price'] / (data['sqft_lot'] * 0.092903)

    return data

def overview_data( data ):

    #===============
    # DATA OVERVIEW
    #===============

    st.title( 'DATA OVERVIEW' )

    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    numerics_columns = data.select_dtypes(include=numerics)
    no_numerics_columns = data.select_dtypes(exclude=numerics)

    st.dataframe( data.head() )

    c1, c2 = st.columns( (1,1) )

    num_attributes = data.select_dtypes( include=['int64', 'float64'] )
    media = pd.DataFrame( num_attributes.apply( np.mean ) )
    mediana = pd.DataFrame( num_attributes.apply( np.median ) )
    std = pd.DataFrame( num_attributes.apply( np.std ) )

    max_ = pd.DataFrame( num_attributes.apply( np.max ) )
    min_ = pd.DataFrame( num_attributes.apply( np.min ) )

    df1 = pd.concat( [max_, min_, media, mediana, std], axis=1).reset_index()
    df1.columns = ['ATTRIBUTES', 'MAX', 'MIN', 'MEAN', 'MEDIAN', 'STD']

    c1.header( 'Descriptive Analysis' )
    c1.dataframe( df1, height=660 )

    text = f"""{dictionary}
Overview:
        1- In this dataset we have {data.shape[0]} houses and {data.shape[1]} attributes. With {numerics_columns.shape[1]} numerics attributes and {no_numerics_columns.shape[1]} non-numeric attribute;
        2- In this sample, we have {len(data["zipcode"].unique())} zipcodes available to analysis; 
    """

    c2.text_area("Overview", text, height=700)

    return None

def univariate_analysis(data):

    st.header( 'Univariate Analysis' )
    c1, c2 = st.columns(2)
    fig = px.box(data, y='price', title='Boxplot of Price')
    c1.plotly_chart(fig, use_container_width=True)


    mediana_rent = data.price.median()

    mediana_rent_format = (
        "$ {:,.2f}".format(mediana_rent)
        .replace(",", "v")
        .replace(".", ",")
        .replace("v", ".")
    )

    graph = [go.Histogram(x=data.price, 
                        nbinsx=50, 
                        marker=dict(color='blue'))]

    line = [go.Scatter(x=[mediana_rent, mediana_rent], 
                    y=[0, 8200], 
                    mode='lines',
                    line=dict(color='lightblue', dash='dash'), 
                    showlegend=True,
                    name=f"Median = {mediana_rent_format}")]

    fig = go.Figure(data=graph+line)

    fig.update_layout(title_text='Price Histogram', 
                    xaxis_title='Price',
                    yaxis_title='Count',
                    autosize=False, 
                    width=900, 
                    height=500)
    
    c2.plotly_chart(fig, use_container_width=True)

    st.dataframe( data[["price"]].describe().T)

    st.text_area("Target Analysis", target_analysis, height=100)

    num_plots = len(data.drop(["id", "date", "lat", "long"], axis=1).columns)
    num_rows = math.ceil(num_plots / 3)
    fig = make_subplots(rows=num_rows, cols=3, subplot_titles=data.drop(["id", "date", "lat", "long"], axis=1).columns)

    for i, column in enumerate(data.drop(["id", "date", "lat", "long"], axis=1).columns):
        row = math.ceil((i + 1) / 3)
        col = (i % 3) + 1
        histogram = go.Histogram(x=data[column], name=column)
        fig.add_trace(histogram, row=row, col=col)

    fig.update_layout(
        title_text="Attributes distribution",
        showlegend=False,
        height=2000,
        width=1500
    )
    st.plotly_chart(fig)

    return None

def bivariate_analysis(data):
    st.header( 'Bivariate Analysis' )


    num_plots = len(data.drop(["id", "date", "lat", "long"], axis=1).columns)
    num_rows = math.ceil(num_plots / 3)
    fig = make_subplots(rows=num_rows, cols=3, subplot_titles=data.drop(["id", "date", "lat", "long"], axis=1).columns)

    for i, column in enumerate(data.drop(["id", "date", "lat", "long"], axis=1).columns):
        row = math.ceil((i + 1) / 3)
        col = (i % 3) + 1
        
        scatter_fig = px.scatter(data, x=f"{column}", y="price", trendline="ols", size="price")
        scatter_trace = scatter_fig['data'][0]
        fig.add_trace(scatter_trace, row=row, col=col)

    fig.update_layout(
        title_text="Attributes distribution",
        showlegend=False,
        height=2000,
        width=1500
    )

    # Exibir gráfico no Streamlit
    st.plotly_chart(fig)

    return None

def portifolio_density( data ):
    #=======================
    # Density of Portifolio
    #=======================

    st.title( 'REGION OVERVIEW' )

    c1, c2 = st.columns( (1,1) )
    c1.header( 'Portifolio Density' )

    data=data.copy()

    #Base Map - Folium
    density_map = folium.Map( location=[data['lat'].mean(), data['long'].mean()], default_zoom_start=15)

    marker_cluster = MarkerCluster().add_to(density_map)

    for name, row in data.iterrows():
        folium.Marker( [row['lat'], row['long'] ], popup='Price R${} on: {}. Features: SQFT {}. BEDROOMS {}. BATHROOMS {}. YEAR BUILT {}.'.format(row['price'], row['date'], row['sqft_living'], row['bedrooms'], row['bathrooms'], row['yr_built'])).add_to(marker_cluster)

    with c1:
        folium_static( density_map )


    #Region Price Map
    c2.header('Price Density')
    data=data.copy()
    data = data[['price', 'zipcode']].groupby( 'zipcode' ).mean().reset_index().rename(columns={'zipcode': 'ZIP', 'price': 'PRICE'})
    #data.columns=['ZIP', 'PRICE']
    #geofile = geofile[geofile['ZIP'].isin( data['ZIP'].tolist()) ]


    region_price_map = folium.Map( location=[data['lat'].mean(), data['long'].mean()], default_zoom_start=15)

    #region_price_map.choropleth( data = data, columns=['ZIP', 'PRICE'], key_on='feature.properties.ZIP', fill_color='YlOrRd', fill_opacity = 0.7, line_opacity = 0.2, legend_name='AVG PRICE')

    with c2:
        folium_static( region_price_map )

    return None

def commercial_distribution( data ):
    # ====================================================
    # Distribuição dos imóveis por categorias comerciais
    # ====================================================

    st.sidebar.title( 'Commercial Options' )
    st.title( 'COMMERCIAL ATTRIBUTES' )

    # ------- Average Price per Year
    data['date'] = pd.to_datetime( data['date']).dt.strftime('%Y-%m-%d')

    min_year_built = int( data['yr_built'].min() )
    max_year_built = int( data['yr_built'].max() )

    st.sidebar.subheader( 'Select Max Year Built' )
    #f_year_built = st.sidebar.slider( 'Year Built', min_year_built, max_year_built, max_year_built )
    st.header( 'Average Price per Year Built' )

    #data = data.loc[data['yr_built'] < f_year_built]


    data = data[['yr_built', 'price']].groupby('yr_built').mean().reset_index()
    fig = px.line(data, x='yr_built', y='price')
    st.plotly_chart(fig, use_container_width=True)

    # ------- Average Price per Day

    #filtros
    min_date = datetime.strptime( data['date'].min(), '%Y-%m-%d' ) 
    max_date = datetime.strptime( data['date'].max(), '%Y-%m-%d' )

    #criando o filtro
    #st.sidebar.subheader( 'Select Max Day' )
    #f_date = st.sidebar.slider( 'Day', min_date, max_date, max_date )
    #st.header( 'Average Price per Day' )

    #filtering dataset
    data['date'] = pd.to_datetime( data['date'])
    #data = data.loc[data['date'] < f_date]

    #plot
    data = data[['date', 'price']].groupby('date').mean().reset_index()
    fig = px.line(data, x='date', y='price')
    st.plotly_chart(fig, use_container_width=True)

    return None

def histogram_graph( data ):

    # ----------------------- Histogram


    # ---- Price
    st.header('Price Distribution')
    st.sidebar.subheader('Select Max Price')

    #filter
    min_price = int( data['price'].min() )
    max_price = int( data['price'].max() )
    avg_price = int( data['price'].mean() )

    f_price = st.sidebar.slider('Price', min_price, max_price, avg_price)

    data = data.loc[data['price'] < f_price]

    fig = px.histogram(data, x='price', nbins=50)
    st.plotly_chart( fig, use_container_width=True)

    return None


def attributes_distribution( data ):

    # ===============================================
    # Distribuição dos imoveis por categoria fisicas
    # ===============================================

    st.sidebar.title( 'Attributes Options' )
    st.title( 'House Attributes' )

    #House per bedrooms
    #filter
    c1, c2 = st.columns(2)
    f_bedrooms = st.sidebar.selectbox('Max Number of bedrooms', sorted( set( data['bedrooms'].unique() ) ) ) 
    data = data.loc[data['bedrooms'] < f_bedrooms]
    #Plot
    fig = px.histogram( data, x='bedrooms', nbins=19)
    c1.header('Houses per Bedrooms')
    c1.plotly_chart( fig, use_container_width=True)

    #House per bathrooms
    f_bathrooms = st.sidebar.selectbox('Max Number of bathrooms', sorted( set( data['bathrooms'].unique() ) ) )
    data = data.loc[data['bathrooms'] < f_bathrooms]
    fig = px.histogram( data, x='bathrooms', nbins=19)
    c2.header('Houses per Bathrooms')
    c2.plotly_chart( fig, use_container_width=True)

    #House per floors
    c1, c2 = st.columns(2)
    f_floors = st.sidebar.selectbox('Max Number of Floors', sorted( set( data['floors'].unique() ) ) )
    data = data.loc[data['floors'] < f_floors]
    c1.header('Houses per Floors')
    fig = px.histogram( data, x='floors', nbins=19)
    c1.plotly_chart( fig, use_container_width=True)

    #House per waterview
    c2.header('Waterview')
    f_waterfront = st.sidebar.checkbox('Only Houses with Water View')
    if f_waterfront:
        data = data[data['waterfront'] == 1]
    else:
        data = data.copy()

    fig = px.histogram( data, x='waterfront', nbins=10)
    c2.plotly_chart( fig, use_container_width=True)

    return None


if __name__ == '__main__':
    #ETL

    #Data Extration
    #get data
    dir_name = os.path.abspath(os.path.dirname(__file__))
    location = os.path.join(dir_name, '/data/raw/data_raw.csv')
    data = pd.read_csv(location)
    #path = '/data/raw/data_raw.csv'
    #data = get_data(path)

    # get geofile
    #url = 'http://data-seattlecitygis.opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'
    #geofile = get_geofile( url )
    
    set_title()
    #Transformation
    data = set_feature( data )

    #Gráficos de Overview
    overview_data( data )

    univariate_analysis(data)

    bivariate_analysis(data)

    #portifolio_density( data )

    #commercial_distribution( data )

    #histogram_graph( data )

    #attributes_distribution( data )
