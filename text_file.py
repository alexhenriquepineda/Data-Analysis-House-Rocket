introduction = """
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


dictionary = """
Data Dictionary:
        id: Unique identifier for each property
        price: Price of the property in dollars
        bedrooms: Number of bedrooms in the property
        bathrooms: Number of bathrooms in the property
        sqft_living: Total living space area in square feet
        sqft_lot: Total lot area in square feet
        floors: Number of floors in the property
        waterfront: Binary indicator (0 or 1) for waterfront property
        view: Rating of the property's view (typically from 0 to 4)
        condition: Overall condition rating of the property (typically from 1 to 5)
        grade: Overall grade given to the housing unit, based on King County grading system
        sqft_above: Area of the property above ground level in square feet
        sqft_basement: Area of the property's basement in square feet
        yr_built: Year the property was built
        yr_renovated: Year of the last renovation
        zipcode: Zip code of the property location
        lat: Latitude coordinate of the property
        long: Longitude coordinate of the property
        """


target_analysis = """
    In these graphs, it is evident that numerous outliers exist in the price attribute. Additionally, there is an asymmetric distribution to the left, characterized by a high concentration of data falling within the range of $300,000 to $600,000
    """