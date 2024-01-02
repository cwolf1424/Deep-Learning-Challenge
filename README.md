# Deep-Learning-Challenge
Challenge Assignment for Neural Networks

////////////////////////////////////////////
Nural Network Model Report
////////////////////////////////////////////

Overview:
--------------------------------------------
Overview of the analysis: Explain the purpose of this analysis.

Results:
--------------------------------------------
Results: Using bulleted lists and images to support your answers, address the following questions:

Data Preprocessing

    What variable(s) are the target(s) for your model?
    What variable(s) are the features for your model?
    What variable(s) should be removed from the input data because they are neither targets nor features?

Compiling, Training, and Evaluating the Model

    How many neurons, layers, and activation functions did you select for your neural network model, and why?
    Were you able to achieve the target model performance?
    What steps did you take in your attempts to increase model performance?

Summary:
--------------------------------------------
Summary: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.


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
Deep_Leaning_Code.ipynb
--------------------------------------------------

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

The following code:

    # Compile the model
    nn.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuaracy"])

and:

    # Train the model
    it_nn = nn.fit(X_train_scaled, y_train, epochs=100)

Were from W21_Class2_Activities 01-Ins-Over_the_Moon_NN.ipynb

The method of saving the model:

    nn.save("applicatn_sucess_model1.HDF5")

Was from:

    https://machinelearningmastery.com/save-load-keras-deep-learning-models/


--------------------------------------------------
Deep Learning Optimization.ipynb
--------------------------------------------------

Used a copy of the Deep_Leaning_Code.ipynb as the base for this file so as not to repeat myself.

Please see section on Deep_Leaning_Code.ipynb for referenced code.

Any addtionally referenced code listed below:

