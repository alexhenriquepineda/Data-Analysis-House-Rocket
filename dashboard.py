from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import folium
import geopandas
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster

import plotly.express as px

from streamlit_folium import folium_static
pd.set_option('display.float_format', lambda x: '%.2f' % x)

st.set_page_config( layout='wide')

@st.cache( allow_output_mutation=True)
def get_data( path ):
    data = pd.read_csv(path)

    return data

@st.cache( allow_output_mutation=True)
def get_geofile( url ):
    geofile = geopandas.read_file( url )
    return geofile

def set_feature( data ):
    #add new features

    #convert to m2
    #data['sqft_lot'] = data['sqft_lot'].apply(lambda x: x / 10,76)

    #add feature price/m2
    data['price_m2'] = data['price'] / data['sqft_lot']

    return data

def overview_data( data ):

    #===============
    # DATA OVERVIEW
    #===============

    f_attributes = st.sidebar.multiselect( 'Enter COLUMNS', data.columns)
    f_zipcode = st.sidebar.multiselect('Enter ZIPCODE', data['zipcode'].unique())

    st.title( 'DATA OVERVIEW' )

    if ( f_zipcode != [] ) & ( f_attributes != [] ):
        data = data.loc[data['zipcode'].isin( f_zipcode ), f_attributes]
    elif ( f_zipcode != [] ) & (f_attributes == [] ):
        data = data.loc[data['zipcode'].isin(f_zipcode), :]  
    elif (f_zipcode == [] ) & ( f_attributes != []):
        data = data.loc[:, f_attributes]
    else:
        data = data.copy()

    st.dataframe( data )

    c1, c2 = st.columns( (1,1) )
    #Average Metrics
    df1 = data[['id', 'zipcode']].groupby( 'zipcode' ).count().reset_index()
    df2 = data[['price', 'zipcode']].groupby( 'zipcode' ).mean().reset_index()
    df3 = data[['sqft_living', 'zipcode']].groupby( 'zipcode' ).mean().reset_index()
    df4 = data[['price_m2', 'zipcode']].groupby( 'zipcode' ).mean().reset_index()

    #merge
    m1 = pd.merge(df1, df2, how='inner', on='zipcode')
    m2 = pd.merge(m1, df3, how='inner', on='zipcode')
    df = pd.merge(m2, df4, how='inner', on='zipcode')

    df.columns = ['ZIPCODE', 'TOTAL HOUSES', 'PRICE', 'SQRT LIVING', 'PRICE/M2']

    c1.header( ' Average Metrics' )
    c1.dataframe( df, height=600 )

    #Statistic Descriptive
    num_attributes = data.select_dtypes( include=['int64', 'float64'] )
    media = pd.DataFrame( num_attributes.apply( np.mean ) )
    mediana = pd.DataFrame( num_attributes.apply( np.median ) )
    std = pd.DataFrame( num_attributes.apply( np.std ) )

    max_ = pd.DataFrame( num_attributes.apply( np.max ) )
    min_ = pd.DataFrame( num_attributes.apply( np.min ) )

    df1 = pd.concat( [max_, min_, media, mediana, std], axis=1).reset_index()
    df1.columns = ['ATTRIBUTES', 'MAX', 'MIN', 'MEAN', 'MEDIAN', 'STD']

    c2.header( 'Descriptive Analysis' )
    c2.dataframe( df1, height=660 )

    return None

def portifolio_density( data, geofile ):
    #=======================
    # Density of Portifolio
    #=======================

    st.title( 'REGION OVERVIEW' )

    c1, c2 = st.columns( (1,1) )
    c1.header( 'Portifolio Density' )

    df=data.copy()

    #Base Map - Folium
    density_map = folium.Map( location=[data['lat'].mean(), data['long'].mean()], default_zoom_start=15)

    marker_cluster = MarkerCluster().add_to(density_map)

    for name, row in df.iterrows():
        folium.Marker( [row['lat'], row['long'] ], popup='Price R${} on: {}. Features: SQFT {}. BEDROOMS {}. BATHROOMS {}. YEAR BUILT {}.'.format(row['price'], row['date'], row['sqft_living'], row['bedrooms'], row['bathrooms'], row['yr_built'])).add_to(marker_cluster)

    with c1:
        folium_static( density_map )


    #Region Price Map
    c2.header('Price Density')
    df=data.copy()
    df = data[['price', 'zipcode']].groupby( 'zipcode' ).mean().reset_index().rename(columns={'zipcode': 'ZIP', 'price': 'PRICE'})
    #df.columns=['ZIP', 'PRICE']
    geofile = geofile[geofile['ZIP'].isin( df['ZIP'].tolist()) ]


    region_price_map = folium.Map( location=[data['lat'].mean(), data['long'].mean()], default_zoom_start=15)

    region_price_map.choropleth( data = df, geo_data=geofile, columns=['ZIP', 'PRICE'], key_on='feature.properties.ZIP', fill_color='YlOrRd', fill_opacity = 0.7, line_opacity = 0.2, legend_name='AVG PRICE')

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
    f_year_built = st.sidebar.slider( 'Year Built', min_year_built, max_year_built, max_year_built )
    st.header( 'Average Price per Year Built' )

    df = data.loc[data['yr_built'] < f_year_built]


    df = df[['yr_built', 'price']].groupby('yr_built').mean().reset_index()
    fig = px.line(df, x='yr_built', y='price')
    st.plotly_chart(fig, use_container_width=True)

    # ------- Average Price per Day

    #filtros
    min_date = datetime.strptime( data['date'].min(), '%Y-%m-%d' ) 
    max_date = datetime.strptime( data['date'].max(), '%Y-%m-%d' )

    #criando o filtro
    st.sidebar.subheader( 'Select Max Day' )
    f_date = st.sidebar.slider( 'Day', min_date, max_date, max_date )
    st.header( 'Average Price per Day' )

    #filtering dataset
    data['date'] = pd.to_datetime( data['date'])
    df = data.loc[data['date'] < f_date]

    #plot
    df = df[['date', 'price']].groupby('date').mean().reset_index()
    fig = px.line(df, x='date', y='price')
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

    df = data.loc[data['price'] < f_price]

    fig = px.histogram(df, x='price', nbins=50)
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
    df = data.loc[data['bedrooms'] < f_bedrooms]
    #Plot
    fig = px.histogram( df, x='bedrooms', nbins=19)
    c1.header('Houses per Bedrooms')
    c1.plotly_chart( fig, use_container_width=True)

    #House per bathrooms
    f_bathrooms = st.sidebar.selectbox('Max Number of bathrooms', sorted( set( data['bathrooms'].unique() ) ) )
    df = data.loc[data['bathrooms'] < f_bathrooms]
    fig = px.histogram( df, x='bathrooms', nbins=19)
    c2.header('Houses per Bathrooms')
    c2.plotly_chart( fig, use_container_width=True)

    #House per floors
    c1, c2 = st.columns(2)
    f_floors = st.sidebar.selectbox('Max Number of Floors', sorted( set( data['floors'].unique() ) ) )
    df = data.loc[data['floors'] < f_floors]
    c1.header('Houses per Floors')
    fig = px.histogram( df, x='floors', nbins=19)
    c1.plotly_chart( fig, use_container_width=True)

    #House per waterview
    c2.header('Waterview')
    f_waterfront = st.sidebar.checkbox('Only Houses with Water View')
    if f_waterfront:
        df = data[data['waterfront'] == 1]
    else:
        df = data.copy()

    fig = px.histogram( df, x='waterfront', nbins=10)
    c2.plotly_chart( fig, use_container_width=True)

    return None


if __name__ == '__main__':
    #ETL

    #Data Extration
    #get data
    path = 'kc_house_data.csv'
    data = get_data(path)

    # get geofile
    url = 'http://data-seattlecitygis.opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'
    geofile = get_geofile( url )
    

    #Transformation
    data = set_feature( data )

    #Gráficos de Overview
    overview_data( data )

    portifolio_density( data, geofile )

    commercial_distribution( data )

    histogram_graph( data )

    attributes_distribution( data )


    #loading








































































































































