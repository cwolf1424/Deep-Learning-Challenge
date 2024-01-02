# Deep-Learning-Challenge
Challenge Assignment for Neural Networks

////////////////////////////////////////////
Sources for Code
////////////////////////////////////////////

Layout for assignment ipynb file came from starter file.

The folowing data sources were used according to sample_data readme:

    "california_housing_data*.csv is California housing data from the 1990 US Census; more information is available at: https://developers.google.com/machine-learning/crash-course/california-housing-data-description

    mnist_*.csv is a small sample of the MNIST database, which is described at: http://yann.lecun.com/exdb/mnist/

    anscombe.json contains a copy of Anscombe's quartet; it was originally described in

    Anscombe, F. J. (1973). 'Graphs in Statistical Analysis'. American Statistician. 27 (1): 17-21. JSTOR 2682899.

    and our copy was prepared by the vega_datasets library."

Specific sections directly using sources listed below:

--------------------------------------------------
Setup 
--------------------------------------------------

The following was provided in starter files:

    # Import our dependencies
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    import tensorflow as tf

    #  Import and read the charity_data.csv.
    import pandas as pd 
    application_df = pd.read_csv("https://static.bc-edx.com/data/dl-1-2/m21/lms/starter/charity_data.csv")
    application_df.head()