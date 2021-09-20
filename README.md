# Neural Network Charity Analysis 

## Overview of the Analysis 
Alphabet Soup is a non-profit foundation. They are a philanthropic foundation dedicated to helping organizations that protect the environment, improve people's well-being, and unify the world. With my knowledge of machine learning and neural entworks, I used the features in the dataset to create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup. The dataset contains more than 34,000 organizations that have received funding from Alphabet Soup over the years. Below are the steps taken for this analysis: 

1. Preprocessing the data for the neural network
2. Compile, Train and Evaluate the Model
3. Optimizing the model


## Results
Within this dataset are a number of columns that capture metadata about each organization, such as the following:

-EIN and NAME—Identification columns
-APPLICATION_TYPE—Alphabet Soup application type
-AFFILIATION—Affiliated sector of industry
-CLASSIFICATION—Government organization classification
-USE_CASE—Use case for funding
-ORGANIZATION—Organization type
-STATUS—Active status
-INCOME_AMT—Income classification
-SPECIAL_CONSIDERATIONS—Special consideration for application
-ASK_AMT—Funding amount requested
-IS_SUCCESSFUL—Was the money used effectively

### Data Preprocessing
1. What variable(s) are considered the target(s) for your model?
The variable I considered the target of my model is the "IS_SUCCESSFUL" columns, which measures if the money used effectively. 

2. What variable(s) are considered to be the features for your model?
The following variables are considered to be the features of my model: 
-APPLICATION_TYPE—Alphabet Soup application type
-AFFILIATION—Affiliated sector of industry
-CLASSIFICATION—Government organization classification
-USE_CASE—Use case for funding
-ORGANIZATION—Organization type
-STATUS—Active status
-INCOME_AMT—Income classification
-SPECIAL_CONSIDERATIONS—Special consideration for application
-ASK_AMT—Funding amount requested

3. What variable(s) are neither targets nor features, and should be removed from the input data?
The variables "EIN" and "NAME" columns were neither targets nor features of my model. I dropped these two variables from the input data. 

### Compiling, Training, and Evaluating the Model
1. How many neurons, layers, and activation functions did you select for your neural network model, and why?
For my neural network model I had 2 hidden layers. My first layer had 80 neurons and the second had 30 neurons. There is also an output layer. The first and second hidden layer have the "relu" activation function and the activation function for the output layer is "sigmoid."

```
# Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.
number_input_features = len(X_train[0])
hidden_nodes_layer1 = 80
hidden_nodes_layer2 = 30

nn = tf.keras.models.Sequential()

# First hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu"))

# Second hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="relu"))

# Output layer
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# Check the structure of the model
nn.summary()
```

2. Were you able to achieve the target model performance?
My neural network model was not able to reach the target model performance of 75%. The accuracy for my model was only 71%. 

```
# Evaluate the model using the test data
model_loss, model_accuracy = nn.evaluate(X_test_scaled,y_test,verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

268/268 - 1s - loss: 1.0334 - accuracy: 0.7102
Loss: 1.033350944519043, Accuracy: 0.7102040648460388
```

3. What steps did you take to try and increase model performance?

#### Attempt 1: Removed Additional Feature
For my first attempt to increase my model performance, I removed the "USE_CASE" column. My model accuracy decreased from 71% to 53%. 

```
# Drop the non-beneficial ID columns, 'EIN' and 'NAME' and additional feature.
application_df.drop(['EIN', 'NAME', 'USE_CASE'], axis=1, inplace=True)
```

```
# Evaluate the model using the test data
model_loss, model_accuracy = nn.evaluate(X_test_scaled,y_test,verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

268/268 - 1s - loss: 0.9388 - accuracy: 0.5332
Loss: 0.9388410449028015, Accuracy: 0.5331778526306152
```

#### Attempt 2: Added Additional Hidden Layers and Additional Neurons to Hidden Layers 
My next attempt, I added more hidden layers and added more neurons to a hidden layer. This decreased by model accuracy from my initial attempt to 53%.  

```
# Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.

number_input_features = len(X_train[0])
hidden_nodes_layer1 = 100
hidden_nodes_layer2 = 50
hidden_nodes_layer3 = 20

nn = tf.keras.models.Sequential()

# First hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu"))

# Second hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="relu"))

# Third hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer3, activation="relu"))


# Output layer
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# Check the structure of the model
nn.summary()
```

```
# Evaluate the model using the test data
model_loss, model_accuracy = nn.evaluate(X_test_scaled,y_test,verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

268/268 - 1s - loss: 0.7036 - accuracy: 0.5327
Loss: 0.7035847902297974, Accuracy: 0.532711386680603
```

#### Attempt 3: Changed Activation Function of Output Layer. 
For my last attempt, I changed the activation function of output layer from "sigmoid" to "tanh." With this third attempt, the accuracy of the model went down even more to 46%.

```
# Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.

number_input_features = len(X_train[0])
hidden_nodes_layer1 = 100
hidden_nodes_layer2 = 50
hidden_nodes_layer3 = 20

nn = tf.keras.models.Sequential()

# First hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu"))

# Second hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="relu"))

# Third hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer3, activation="relu"))


# Output layer
nn.add(tf.keras.layers.Dense(units=1, activation="tanh"))

# Check the structure of the model
nn.summary()
```

```
# Evaluate the model using the test data
model_loss, model_accuracy = nn.evaluate(X_test_scaled,y_test,verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

268/268 - 1s - loss: 3.6382 - accuracy: 0.4619
Loss: 3.638166666030884, Accuracy: 0.4619241952896118
```


## Summary 
Please see the following models and the accuracy percentage: 

1. Initial Model - 71% 
2. Attempt 1 Model - 53%
3. Attempt 2 Model - 53%
4. Attempt 3 Model - 46%

The higest accuracy percentage was 71% and still under the target model performace of 75%. The loss in accuracy may be explained because the model was overfitted. My recommendation to further improve the model would be to remove more features or to add more data to the dataset to increase accuracy. Another suggestion could be to use the Random Forest classifiers. The random forest is a robust and accurate model due to their sufficient number of estimators and tree depth. Lastly, the random forest models also has a faster performance than neural networks and could potentially help avoid the data from being overfitted.

