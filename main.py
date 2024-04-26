# Simple machine learning model to convert Celsius to Fahrenheit
# Equation: T(F) = T(C) * 9/5 + 32

# The objective is to predict Y based on X
# X - Independent variable
# Y - Dependent variable

# This is an example of Regression

# A single neuron is the sum of each input that are multiplied by the weight of their transformation plus a bias
# I normalize weights of neuron transformations in a range from -1 to 1
# In this context, the bias is 9/5 since we multiply 9/5 by any Celsius input

# An activation function is applied to this sum which results in our desired output
# The output is the calculation of the input multiply the weight of the transformation plus the bias
# In this context, the bias is 32 since we add it to any product of the weight (9/5) and the Celsius input

# This is an example of supervised learning where we label both input and output data
# It is not learning alone i.e. we know the desired data

# The model learns from experience i.e. repeated examples

# If the models' predicted output does not match my desired output, I calculate an error signal
# In the next iteration, the model considers the error signal and over time makes their predictions more accurate

# In the context of this model, I have a bunch of data (temperature) points corresponding to Celsius and Fahrenheit
# I will show this data to the neural network repeatedly until the error is 0

# A dense network is when every input is connected to every hidden neuron, and every hidden neuron is connected
# to every output

# Regularization helps to ensure I do not over-train the model
# If I train the model too intensely using the training data only, when I try to test the model with images
# it has never seen before, it will fail spectacularly.

import tensorflow as tf  # contains neural network API
import tensorflow.keras as keras  # for model-training functions
import seaborn as sns  # for plotting images/statistics
import pandas as pd  # for data frame manipulation
import numpy as np  # for numerical analysis (matricies, multiplication, etc)
import matplotlib.pyplot as plt  # for data visualization

# 1. Import the dataset
temp_df = pd.read_csv('Celsius+to+Fahrenheit.csv')
# 'describe()' can be used to give a summary of all the statistical info in the dataset
print(temp_df.describe())
# 'info()' can be used to give a summary of the data types in the dataset
print(temp_df.info())

# 2. Visualize the dataset
# Create a scatterplot from the data frame with the x-axis and y-axis corresponding to the columns in the data frame
sns.scatterplot(data=temp_df, x=temp_df['Celsius'], y=temp_df['Fahrenheit'])
# Show the scatterplot
plt.show()

# 3.Create testing and training dataset
# I must make sure the testing dataset is never seen by the model during testing
# I test both the independent variable X and the dependent variable Y
# 'scikitlearn' library can be used to divide the data into training and testing data
# X_train, Y_train, X_test, Y_test are conventionally used for scikitlearn
# In this case, I am just going to use the whole column
X_train = temp_df['Celsius']
Y_train = temp_df['Fahrenheit']
# shape() gives me the number of samples
print(X_train.shape)
print(Y_train.shape)

# 4. Build and train the model
# 'keras' is the standard API to train models
# Sequential() builds my model in a sequential fashion layer by layer [input layer -> hidden layer(s) -> output layer]
model = keras.Sequential()
# We can add Dense layers (every input is linked by weight to all the neurons in the next hidden layer, and so on)
# units - how many neurons we want
# input_shape - tells the network how many inputs each neuron should expect
model.add(keras.layers.Dense(units=1, input_shape=[1]))
# What does the model look like at this point?
# 2 parameters - weight and bias
model.summary()
# Use an optimizer to get the values of the parameters, weight, and bias
# Optimizer argument 1 - learning rate (how fast we want our network to update the weights)
# Optimizer argument 2 - loss (what am I trying to optimize/minimize?)
# loss = mean_squared_error - subtracting ground truth minus network predictions. Take error, square it, get mean.
# Legacy optimizer is recommended by interpreter for running Adam on M1 Mac
model.compile(optimizer=keras.optimizers.legacy.Adam(0.8), loss='mean_squared_error')
# Fit the training data to the model
# Save it in 'epochs_hist' to see the performance of the network as it trains across all epochs,
epochs_hist = model.fit(X_train, Y_train, epochs=300)

# 5. Evaluate the model
# Show me the history of the model
# I can get my history keys like this
print(epochs_hist.history.keys())
# Set up graph
# loss - key we want to visualize
plt.plot(epochs_hist.history['loss'])
plt.title('Model Loss Progress During Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Training Loss'])
# Show the loss history as the model progresses
plt.show()


# Now that the model is trained, it should have the best values of weight & bias that can map the input and output data
# What are the values of the weights?
# The weight is 1.8, which makes sense since in the formula, we multiply the independent variable by 9/5
# The bias is 32, which makes sense since in the formula, we add 32 to the output
print(model.get_weights())

# To deploy the network in practice, you give it an input and ask it to predict the output
# predict() - takes the input, feeds it to the model (feed forward pass), and predicts the output
Temp_C = 10
Temp_F = model.predict([Temp_C])
print("Temperature in Fahrenheit using Trained ANN:", Temp_F)

Real_Temp_F = Temp_C * (9/5) + 32
print("Real temperature from equation:", Real_Temp_F)