import math
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import folium
import geopandas
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster

import plotly.express as px
import plotly.graph_objs as go
from matplotlib import pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots


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

def set_title():
    st.title("Real Estate Market Exploration in Seattle")

    text = """
    In this study, we aim to deepen our analysis of the existing correlations between the physical characteristics of a property, such as its size and number of rooms, 
    with its price and location. The main focus is to understand the underlying factors influencing the property values in Seattle, one of the most dynamic and challenging 
    cities in the United States. The data used were obtained from the Kaggle platform, well-known for its Machine Learning competitions.

    The presented analysis discusses the mentioned variables and examines how they impact both the rental cost and the total value of the property, considering additional 
    charges not specified on Kaggle. We intend to comprehend, for example, how the location in a particular neighborhood can influence the rental value. Additionally, 
    we will investigate the significance of other physical characteristics of the property, such as the square footage, the number of bathrooms and bedrooms, among others, 
    in determining the total sales or rental value. We will also highlight the most expensive and affordable neighborhoods in the city.

    It is important to emphasize that this study was conducted strictly for educational purposes, as there is uncertainty regarding the impartiality or possible gaps and 
    errors in the dataset, as it has not undergone any validation process.

    So, welcome to our exploratory journey into the real estate market of Seattle!
    """

    st.text_area("Introduction", text, height=350)


def set_feature( data ):
    #add new features

    #convert to m2
    #data['sqft_lot'] = data['sqft_lot'].apply(lambda x: x / 10,76)

    #add feature price/m2
    data['price_m2'] = data['price'] / (data['sqft_lot'] * 0.092903)

    return data

def overview_data( data ):

    #===============
    # DATA OVERVIEW
    #===============

    #f_attributes = st.sidebar.multiselect( 'Enter COLUMNS', data.columns)
    #f_zipcode = st.sidebar.multiselect('Enter ZIPCODE', data['zipcode'].unique())

    st.title( 'DATA OVERVIEW' )

    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    numerics_columns = data.select_dtypes(include=numerics)
    no_numerics_columns = data.select_dtypes(exclude=numerics)



    #if ( f_zipcode != [] ) & ( f_attributes != [] ):
    #    data = data.loc[data['zipcode'].isin( f_zipcode ), f_attributes]
    #elif ( f_zipcode != [] ) & (f_attributes == [] ):
    #    data = data.loc[data['zipcode'].isin(f_zipcode), :]  
    #elif (f_zipcode == [] ) & ( f_attributes != []):
    #    data = data.loc[:, f_attributes]
    #else:
    #    data = data.copy()

    st.dataframe( data.head() )

    c1, c2 = st.columns( (1,1) )

    #Statistic Descriptive
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

    text = f"""
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

    text = """
    In these graphs, it is evident that numerous outliers exist in the price attribute. Additionally, there is an asymmetric distribution to the left, characterized by a high concentration of data falling within the range of $300,000 to $600,000
    """

    st.text_area("Target Analysis", text, height=100)

    # Criação de subplots
    num_plots = len(data.drop(["id", "date", "lat", "long"], axis=1).columns)
    num_rows = math.ceil(num_plots / 3)  # Calcular o número de linhas necessárias
    fig = make_subplots(rows=num_rows, cols=3, subplot_titles=data.drop(["id", "date", "lat", "long"], axis=1).columns)

    # Adição de histogramas aos subplots
    for i, column in enumerate(data.drop(["id", "date", "lat", "long"], axis=1).columns):
        row = math.ceil((i + 1) / 3)  # Calcular o número da linha para o subplot atual
        col = (i % 3) + 1  # Calcular o número da coluna para o subplot atual
        histogram = go.Histogram(x=data[column], name=column)
        fig.add_trace(histogram, row=row, col=col)


    # Ajuste do tamanho dos gráficos
    fig.update_layout(
        title_text="Attributes distribution",
        showlegend=False,
        height=2000,  # Ajuste a altura desejada
        width=1500   # Ajuste a largura desejada
    )
    st.plotly_chart(fig)


    return None

def bivariate_analysis(data):
    st.header( 'Bivariate Analysis' )

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
    path = '/data/raw/data_raw.csv'
    data = get_data(path)

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
