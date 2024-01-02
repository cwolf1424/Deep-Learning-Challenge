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

and:

    # Replace in dataframe
    for app in application_types_to_replace:
        application_df['APPLICATION_TYPE'] = application_df['APPLICATION_TYPE'].replace(app,"Other")

    # Check to make sure binning was successful
    application_df['APPLICATION_TYPE'].value_counts()

and:

    # Replace in dataframe
    for cls in classifications_to_replace:
        application_df['CLASSIFICATION'] = application_df['CLASSIFICATION'].replace(cls,"Other")

    # Check to make sure binning was successful
    application_df['CLASSIFICATION'].value_counts()

and: 

    # Create a StandardScaler instances
    scaler = StandardScaler()

    # Fit the StandardScaler
    X_scaler = scaler.fit(X_train)

    # Scale the data
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

--------------------------------------------------
Compile, Train and Evaluate the Model
--------------------------------------------------

The following was provided in starter files:

    nn = tf.keras.models.Sequential()

and:

    # Evaluate the model using the test data
    model_loss, model_accuracy = nn.evaluate(X_test_scaled,y_test,verbose=2)
    print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")