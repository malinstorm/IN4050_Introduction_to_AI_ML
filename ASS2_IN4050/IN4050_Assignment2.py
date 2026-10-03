# IN3050/IN4050 Mandatory Assignment 2, 2024: Supervised Learning
# Converted from the original Jupyter notebook.
# Code cells are preserved in their original order; Markdown cells are comments.


# %% [markdown] Cell 1
# ## IN3050/IN4050 Mandatory Assignment 2, 2024: Supervised Learning

# %% [markdown] Cell 2
# ### Rules
#
# Before you begin the exercise, review the rules at this website: https://www.uio.no/english/studies/examinations/compulsory-activities/mn-ifi-mandatory.html , in particular the paragraph on cooperation. This is an individual assignment. You are not allowed to deliver together or copy/share source-code/answers with others. Read also the "Routines for handling suspicion of cheating and attempted cheating at the University of Oslo": https://www.uio.no/english/studies/examinations/cheating/index.html By submitting this assignment, you confirm that you are familiar with the rules and the consequences of breaking them.
#
# ### Delivery
#
# **Deadline**: Friday, March 22, 2024, 23:59
#
# Your submission should be delivered in Devilry. You may redeliver in Devilry before the deadline, but include all files in the last delivery, as only the last delivery will be read. You are recommended to upload preliminary versions hours (or days) before the final deadline.
#
# ### What to deliver?
#
# You are recommended to solve the exercise in a Jupyter notebook, but you might solve it in a regular Python script if you prefer.
#
# #### Alternative 1
# If you prefer not to use notebooks, you should deliver the code, your run results, and a PDF report where you answer all the questions and explain your work.
#
# #### Alternative 2
# If you choose Jupyter, you should deliver the notebook. You should answer all questions and explain what you are doing in Markdown. Still, the code should be properly commented. The notebook should contain results of your runs. In addition, you should make a pdf of your solution which shows the results of the runs. (If you can't export: notebook -> latex -> pdf on your own machine, you may do this on the IFI linux machines.)
#
# Here is a list of *absolutely necessary* (but not sufficient) conditions to get the assignment marked as passed:
#
# - You must deliver your code (Python script or Jupyter notebook) you used to solve the assignment.
# - The code used for making the output and plots must be included in the assignment. 
# - You must include example runs that clearly shows how to run all implemented functions and methods.
# - All the code (in notebook cells or python main-blocks) must run. If you have unfinished code that crashes, please comment it out and document what you think causes it to crash. 
# - You must also deliver a pdf of the code, outputs, comments and plots as explained above.
#
# Your report/notebook should contain your name and username.
#
# Deliver one single compressed folder (.zip, .tgz or .tar.gz) which contains your complete solution.
#
# Important: if you weren’t able to finish the assignment, use the PDF report/Markdown to elaborate on what you’ve tried and what problems you encountered. Students who have made an effort and attempted all parts of the assignment will get a second chance even if they fail initially. This exercise will be graded PASS/FAIL.

# %% [markdown] Cell 3
# ### Goals of the assignment
# The goal of this assignment is to get a better understanding of supervised learning with gradient descent. It will, in particular, consider the similarities and differences between linear classifiers and multi-layer feed forward neural networks (multi-layer perceptrons, MLP) and the differences and similarities between binary and multi-class classification. A significant part is dedicated to implementing and understanding the backpropagation algorithm. 
#
# ### Tools
# The aim of the exercises is to give you a look inside the learning algorithms. You may freely use code from the weekly exercises and the published solutions. You should not use machine learning libraries like Scikit-Learn or PyTorch, because the point of this assignment is for you to implement things from scratch. You, however, are encouraged to use tools like NumPy and Pandas, which are not ML-specific.
#
# The given precode uses NumPy. You are recommended to use NumPy since it results in more compact code, but feel free to use pure Python if you prefer. 
#
# ### Beware
# This is a revised assignment compared to earlier years. If anything is unclear, do not hesitate to ask. Also, if you think some assumptions are missing, make your own and explain them!

# %% [markdown] Cell 4
# ### Initialization

# %% Cell 5
import numpy as np
import matplotlib.pyplot as plt
import sklearn # This is only to generate a dataset

# %% [markdown] Cell 6
# ## Datasets
#
# We start by making a synthetic dataset of 2000 instances and five classes, with 400 instances in each class. (See https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html regarding how the data are generated.) We choose to use a synthetic dataset---and not a set of natural occuring data---because we are mostly interested in properties of the various learning algorithms, in particular the differences between linear classifiers and multi-layer neural networks together with the difference between binary and multi-class data. In addition, we would like a dataset with instances represented with only two numerical features, so that it is easy to visualize the data. It would be rather difficult (although not impossible) to find a real-world dataset of the same nature. Anyway, you surely can use the code in this assignment for training machine learning models on real-world datasets.
#
# When we are doing experiments in supervised learning, and the data are not already split into training and test sets, we should start by splitting the data. Sometimes there are natural ways to split the data, say training on data from one year and testing on data from a later year, but if that is not the case, we should shuffle the data randomly before splitting. (OK, that is not necessary with this particular synthetic data set, since it is already shuffled by default by Scikit-Learn, but that will not be the case with real-world data) We should split the data so that we keep the alignment between X (features) and t (class labels), which may be achieved by shuffling the indices. We split into 50% for training, 25% for validation, and 25% for final testing. The set for final testing *must not be used* till the end of the assignment in part 3.
#
# We fix the seed both for data set generation and for shuffling, so that we work on the same datasets when we rerun the experiments. This is done by the `random_state` argument and the `rng = np.random.RandomState(2024)`.

# %% Cell 7
# Generating the dataset
from sklearn.datasets import make_blobs
X, t_multi = make_blobs(n_samples=[400, 400, 400, 400, 400], centers=[[0,1],[4,2],[8,1],[2,0],[6,0]], 
                  n_features=2, random_state=2024, cluster_std=[1.0, 2.0, 1.0, 0.5, 0.5])

# %% Cell 8
# Shuffling the dataset
indices = np.arange(X.shape[0])
rng = np.random.RandomState(2024) # setter samme randomgenererte utgangspunkt, slik at testdatene er like hver gang
rng.shuffle(indices) # shuffler dataene

# %% Cell 9
# Splitting into train, dev and test
X_train = X[indices[:1000],:]
X_val = X[indices[1000:1500],:]
X_test = X[indices[1500:],:] 
t_multi_train = t_multi[indices[:1000]]
t_multi_val = t_multi[indices[1000:1500]]
t_multi_test = t_multi[indices[1500:]] # not supposed to be touched until the end of part 3 in the assignment
print(len(X_train))

# %% [markdown] Cell 10
# Next, we will  make a second dataset with only two classes by merging the existing labels in (X,t), so that `0`, `1` and `2` become the new `0` and `3` and `4` become the new `1`. Let's call the new set (X, t2). This will be a binary set.
# We now have two datasets:
#
# - Binary set: `(X, t2)`
# - Multi-class set: `(X, t_multi)`

# %% Cell 11
print(type(t_multi_train))
t2_train = t_multi_train >= 3
t2_train = t2_train.astype('int') # casting
t2_val = (t_multi_val >= 3).astype('int')
t2_test = (t_multi_test >= 3).astype('int') # casting

# %% Cell 12
# scaling input data, trainig, validation and testing 
class MMScaler():
    
    def fit(self, X):
        self.max_val = np.max(X, axis=0)
        self.min_val = np.min(X, axis=0)
    
    def transform(self, X):
        return (X - self.min_val)/(self.max_val - self.min_val)

sc = MMScaler()

sc.fit(X_train)
X_train_scaled = sc.transform(X_train)

sc.fit(X_val)
X_val_scaled = sc.transform(X_val)

sc.fit(X_test)
X_test_scaled = sc.transform(X_test)

# %% [markdown] Cell 13
# We can plot the two traning sets.

# %% Cell 14
plt.figure(figsize=(8,6)) # You may adjust the size
plt.scatter(X_train[:, 0], X_train[:, 1], c=t_multi_train, s=10.0)
plt.title("Multi-class set")

# %% Cell 15
plt.figure(figsize=(8,6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=t2_train, s=10.0)
plt.title('Binary data set')

# %% [markdown] Cell 16
# # Part 1: Linear classifiers
# ### Linear regression

# %% [markdown] Cell 17
# We see that even the binary set (X, t2) is far from linearly separable, and we will explore how various classifiers are able to handle this. We start with linear regression with the Mean Squared Error (MSE) loss, although it is not the most widely used approach for classification tasks: but we are interested. You may make your own implementation from scratch or start with the solution to the weekly exercise set 7. 
# We include it here with a little added flexibility.

# %% Cell 18
def add_bias(X, bias):
    """X is a NxM matrix: N datapoints, M features
    bias is a bias term, -1 or 1, or any other scalar. Use 0 for no bias
    Return a Nx(M+1) matrix with added bias in position zero
    """
    N = X.shape[0]
    biases = np.ones((N, 1)) * bias # Make a N*1 matrix of biases
    # Concatenate the column of biases in front of the columns of X.
    return np.concatenate((biases, X), axis  = 1) 

# %% Cell 19
class NumpyClassifier():
    """Common methods to all Numpy classifiers --- if any"""
    
    # implement linear regression model here :)
    

# %% Cell 20
class NumpyLinRegClass(NumpyClassifier):

    def __init__(self, bias=-1):
        self.bias=bias
    
    
    def fit(self, X_train, t_train, lr = 0.7, epochs=400): # LR, (init=0.1,epochs=10) (unscaled (0.06,500)), (scaled (0.7,400))
        """X_train is a Nxm matrix, N data points, M features
        t_train is avector of length N,
        the targets values for the training data"""
        
        if self.bias:
            X_train = add_bias(X_train, self.bias)
            
        (N, M) = X_train.shape
        
        #X_train_bias = add_bias(X_train,self.bias)
        
        self.weights = weights = np.zeros(M)
        self.error = error = [] # list to store the MSE/loss
        self.accuracy_calc = accuracy_calc = [] # list to store accuracy per epoch
        
        for e in range(epochs):
            
            #loss = np.mean((X_train @ weights - t_train)**2)
            error.append(np.mean((X_train @ weights - t_train)**2))
            # calculating accuracy
            accuracy_calc.append(accuracy(X_train@weights > 0.5, t_train)) # plottes
            # calculating weights
            weights -= lr / N *  X_train.T @ (X_train @ weights - t_train)
                
    def predict(self, X, threshold=0.5):
        """X is a Kxm matrix for some K>=1
        predict the value for each point in X"""
        if self.bias:
            X = add_bias(X, self.bias)
        ys = X @ self.weights
        return ys > threshold

# %% Cell 21
def mse(t_train, pred):
    sum_errors = 0.
    for i in range(0,len(t_train)):
        sum_errors += (t_train[i] - pred[i])**2
    mean_squared_error = sum_errors/len(t_train)
    return mean_squared_error

# %% [markdown] Cell 22
# We can train and test a first classifier.

# %% Cell 23
def accuracy(predicted, gold):
    return np.mean(predicted == gold)

# %% Cell 24
def plot_decision_regions(X, t, clf=[], size=(8,6)):
    """Plot the data set (X,t) together with the decision boundary of the classifier clf"""
    # The region of the plane to consider determined by X
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # Make a prediction of the whole region
    
    h = 0.02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Classify each meshpoint.
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=size) # You may adjust this

    # Put the result into a color plot
    plt.contourf(xx, yy, Z, alpha=0.2, cmap = 'Paired')

    plt.scatter(X[:,0], X[:,1], c=t, s=10.0, cmap='Paired')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Decision regions")
    plt.xlabel("x0")
    plt.ylabel("x1")

# %% Cell 25
# tuning
# loop to plot different lr and epochs

lr_var = [0.03,0.03,0.04,0.05,0.06,0.07,0.06,0.07,0.09,0.09,0.09,0.05,0.02,0.06]
epoch_var = [300,200,800,800,800,800,900,900,900,1000,2000,3000,5000,7000]

print("UNSCALED DATA:")
for i in range(len(lr_var)):
    lr = lr_var[i]
    epochs = epoch_var[i]
    cl.fit(X_train, t2_train,lr, epochs)

    
    print("Accuracy with lr = ",lr," epoch = ",epochs,":",accuracy(cl.predict(X_train), t2_train))

lr_var = [0.3,0.3,0.4,0.5,0.6,0.7,0.6,0.7,0.9,0.9,0.9,0.5,0.2,0.6]
print("\nSCALED DATA:")
for i in range(len(lr_var)):
    lr = lr_var[i]
    epochs = epoch_var[i]
    cl_scaled.fit(X_train_scaled,t2_train,lr,epochs)
    
    
# see from the result below that eta = 0.06 and epoch = 500 can be used for tuning.
# lowest MSE
    print("Accuracy with lr = ",lr," epoch = ",epochs,":",accuracy(cl_scaled.predict(X_train_scaled), t2_train))
    

# %% Cell 26
# testing unscaled dataset
cl = NumpyLinRegClass()
cl.fit(X_train, t2_train, lr = 0.04,epochs = 800) # picked out the best values from the tuning
accuracy(cl.predict(X_train), t2_train)


plt.figure(figsize=(8,6))
plt.plot(cl.error,label='Loss')
plt.plot(cl.accuracy_calc,label='Accuracy',color='seagreen')
plt.xlabel('Epochs')
plt.ylabel('Values')
plt.legend()
plt.title('Error and accuracy, unscaled')

# testing the scaled dataset
cl_scaled = NumpyLinRegClass()
cl_scaled.fit(X_train_scaled,t2_train,lr = 0.6, epochs = 800) # picked out the best values from the tuning
#accuracy(cl_scaled.predict(X_val_scaled), t2_val)
accuracy(cl_scaled.predict(X_train_scaled), t2_train)

plt.figure(figsize=(8,6))
plt.plot(cl_scaled.error,label='Loss')
plt.plot(cl_scaled.accuracy_calc,label='Accuracy',color='seagreen')
plt.xlabel('Epochs')
plt.ylabel('Values')
plt.legend()
plt.title('Error and accuracy, scaled')

# %% [markdown] Cell 27
# The following is a small procedure which plots the data set together with the decision boundaries. 
# You may modify the colors and the rest of the graphics as you like.
# The procedure will also work for multi-class classifiers

# %% Cell 28
plot_decision_regions(X_train, t2_train, cl) # unscaled

# %% Cell 29
plot_decision_regions(X_train_scaled, t2_train, cl_scaled) # scaled

# %% [markdown] Cell 30
# ### Task: Tuning
#
# The result is far from impressive. 
# Remember that a classifier which always chooses the majority class will have an accuracy of 0.6 on this data set.
#
# Your task is to try various settings for the two training hyper-parameters, learning rate and the number of epochs, to get the best accuracy on the validation set. 
#
# Report how the accuracy varies with the hyper-parameter settings. It it not sufficient to give the final hyperparameters. You must also show how you found then and results for alternative values you tried aout.
#
# When you are satisfied with the result, you may plot the decision boundaries, as above.

# %% [markdown] Cell 31
# ### Task: Scaling

# %% [markdown] Cell 32
# We have seen in the lectures that scaling the data may improve training speed and sometimes the performance. 
#
# - Implement a scaler, at least the standard scaler (normalizer), but you can also try other techniques
# - Scale the data
# - Train the model on the scaled data
# - Experiment with hyper-parameter settings and see whether you can speed  up  the training.
# - Report final hyper-parameter settings and show how you found them.

# %% [markdown] Cell 33
# ## Logistic regression
# a) You should now implement a logistic regression classifier similarly to the classifier based on linear regression.
# You may use the code from the solution to weekly exercise set week07.
#
# b) In addition to the method `predict()` which predicts a class for the data, include a method `predict_probability()` which predict the probability of the data belonging to the positive class.
#
# c) So far, we have not calculated the loss explicitly in the code. Extend the code to calculate the loss on the training set for each epoch and to store the losses such that the losses can be inspected after training. The prefered loss for logistic regression is binary cross-entropy, but you can also try mean squared error. The most important is that your implementation of the loss corresponds to your implementation of the gradient descent.
# Also, calculate and store accuracies after each epoch.
#
# d) In addition, extend the `fit()` method with optional arguments for a validation set (X_val, t_val). If a validation set is included in the call to `fit()`, calculate the loss and the accuracy for the validation set after each epoch. 
#
# e) The training runs for a number of epochs. We cannot know beforehand for how many epochs it is reasonable to run the training. One possibility is to run the training until the learning does not improve much. Extend the `fit()` method with two keyword arguments, `tol` (tolerance) and `n_epochs_no_update` and stop training when the loss has not improved with more than `tol` after `n_epochs_no_update`. A possible default value for `n_epochs_no_update` is 5. Also, add an attribute to the classifier which tells us after fitting how many epochs it was trained for.
#
# f) Train classifiers with various learning rates, and with varying values for `tol` for finding the optimal values. Also consider the effect of scaling the data.
#
# g) After a succesful training, for your best model, plot both training loss and validation loss as functions of the number of epochs in one figure, and both training and validation accuracies as functions of the number of epochs in another figure. Comment on what you see. Are the curves monotone? Is this as expected?

# %% Cell 34
class NumpyLogReg(NumpyClassifier):
    
    def __init__(self, bias=-1):
        self.bias=bias
        self.validation_data = None  
    def fit(self, X_train, t_train,lr, epochs,tol,n_epochs_no_update = 10,X_valid=None,t_valid=None):
    
        """X_train is a Nxm matrix, N data points, M features
        t_train is avector of length N,
        the targets values for the training data"""
        
        (N, M) = X_train.shape
        
        # shuffling the dataset, tip from chat GPT
        shuffle = np.random.permutation(len(X_train))
        X_train_shuffle = X_train[shuffle]
        t_train_shuffle = t_train[shuffle]
       
            
        # generatin g a random initialization of weights, tip from chat GPT    
        #self.weights = weights = np.random.randn(M+1)
        self.weights = weights = np.zeros(M+1)
        self.error_train = error_train = [] # cross entropy
        self.accuracy_train = accuracy_train = []
        self.accuracy_val = accuracy_val = []
        self.error_val = error_val = []
        self.error_tol = error_tol = 0
        self.epoch_count = epoch_count = 0 # var to use to accumulate accuracy_tol for n_epochs
        
        validation_data = not(X_valid is None and t_valid is None)
        
        if self.bias:
           
            X_train = add_bias(X_train, self.bias)
            if validation_data:
                X_valid = add_bias(X_valid, self.bias)
        
        
        for e in range(epochs):
            
           
            # calculating loss
            error_train.append(((self.loss(self.forward(X_train),t_train_shuffle))))
            # calculating accuracy
            accuracy_train.append(accuracy(X_train@weights > 0.5, t_train_shuffle)) # accuracy on training set
            
            if validation_data:
                error_val.append(((self.loss(self.forward(X_valid),t_valid)))) # loss on val set
                accuracy_val.append(accuracy(X_valid@weights > 0.5, t_valid)) # accuracy on val set
            
    
                
            if e > 0: 
                if error_tol < tol:
                    error_tol = error_train[-1] - error_train[-2] #the difference in loss, last e of array- (last e -1)
                    if abs(error_tol < tol): # doesn't care about the change in the direction of the loss
                            epoch_count += 1
                    else: epoch_count = 0
            
            if  epoch_count >= n_epochs_no_update:
                print("Ran ",e,"number of rounds") # if less, print number of rounds it ran
                break          
                    
            # calculating weights             
            weights -= (lr / N) *  X_train.T @ (self.forward(X_train) - t_train)
           
    # cross entropy loss
    def loss(self,pred,true):
        limit = 1e-15
        pred = np.clip(pred, limit,(1-limit))
        return -np.mean(true*np.log(pred) + ((1-true) * (np.log(1 - (pred))))) 
    # sigmoid function       
    def logistic(self,x):      
        return 1/(1+np.exp(-x))
    # activation function
    def forward(self, X): # 
        return self.logistic(X @ self.weights)
    
    def predict(self, x, threshold=0.5): # denne skal bare returnere 1 eller 0
        """X is a Kxm matrix for some K>=1
        predict the value for each point in X"""
        z = add_bias(x,self.bias)
        return (self.forward(z)> threshold).astype('int') # returns bool True/False
    

# %% Cell 35
# tuning
# loop to plot different lr and epochs
log_reg = NumpyLogReg()
log_reg_scaled = NumpyLogReg()
lr_var = [0.3,0.3,0.4,0.5,0.6,0.7,0.6,0.7,0.9,0.9,0.9,0.5,0.2,0.6]
epoch_var = [300,200,800,800,800,800,900,900,900,1000,2000,3000,5000,7000]
number_of_epochs = 10
tol = 1e-5
print("UNSCALED DATA:")
for i in range(len(lr_var)):
    lr = 0.08 #lr_var[i]
    epochs = 200#epoch_var[i]
    log_reg.fit(X_train, t2_train,lr,epochs, tol = 1e-5,n_epochs_no_update=number_of_epochs,X_valid=X_val,t_valid=t2_val)
    
    print("Accuracy with lr = ",lr,"tolerance: ", tol," epoch = ",epochs,":",accuracy(log_reg.predict(X_train),t2_train))

tol = 1e-7
lr_var = [0.03,0.03,0.04,0.05,0.06,0.07,0.06,0.07,0.09,0.09,0.09,0.05,0.02,0.06]
print("\nSCALED DATA:")
for i in range(len(lr_var)):
    lr = 1
    epochs = 2000#epoch_var[i]
    log_reg_scaled.fit(X_train_scaled,t2_train,lr,epochs, tol = 1e-7,n_epochs_no_update=number_of_epochs,X_valid=X_val_scaled,t_valid=t2_val)
    
    print("Accuracy with lr = ",lr,"tolerance: ", tol," epoch = ",epochs,":", accuracy(log_reg_scaled.predict(X_train_scaled),t2_train))
    
    

# %% Cell 36
log_reg = NumpyLogReg()
log_reg_scaled = NumpyLogReg()


number_of_epochs = 10
# testing without scaled data

log_reg.fit(X_train, t2_train,lr=0.08,epochs=200, tol = 1e-5,n_epochs_no_update=number_of_epochs,X_valid=X_val,t_valid=t2_val)

plt.figure(figsize=(8,6))
plt.plot(log_reg.error_train,label='Loss_CE_train')
plt.plot(log_reg.error_val,label='Loss_CE_val')
plt.xlabel('Epochs')
plt.ylabel('Values')
plt.legend()
plt.title('Error Unscaled Data')

plt.figure(figsize=(8,6))
plt.plot(log_reg.accuracy_train,label='Accuracy_train',color='seagreen')
plt.plot(log_reg.accuracy_val,label='Accuracy_val',color='r')
plt.xlabel('Epochs')
plt.ylabel('Values')
plt.legend()
plt.title('Accuracy Unscaled Data')

plot_decision_regions(X_train, t2_train, log_reg)


# testing with Scaled data
log_reg_scaled.fit(X_train_scaled,t2_train,lr = 1,epochs=2000, tol = 1e-7,n_epochs_no_update=number_of_epochs,X_valid=X_val_scaled,t_valid=t2_val)
    
plt.figure(figsize=(8,6))
plt.plot(log_reg_scaled.error_train,label='Loss_CE_train')
plt.plot(log_reg_scaled.error_val,label='Loss_CE_val')
plt.xlabel('Epochs')
plt.ylabel('Values')
plt.legend()
plt.title('Error Scaled Data')

plt.figure(figsize=(8,6))
plt.plot(log_reg_scaled.accuracy_train,label='Accuracy_train',color='seagreen')
plt.plot(log_reg_scaled.accuracy_val,label='Accuracy_val',color='r')
plt.xlabel('Epochs')
plt.ylabel('Values')
plt.legend()
plt.title('Accuracy Scaled Data')

plot_decision_regions(X_train_scaled, t2_train, log_reg_scaled)

# %% [markdown] Cell 37
# ## Multi-class classifiers
# We turn to the task of classifying when there are more than two classes, and the task is to ascribe one class to each input. We will now use the set (X, t_multi).

# %% [markdown] Cell 38
# ### "One-vs-rest" with logistic regression
# We saw in the lectures how a logistic regression classifier can be turned into a multi-class classifier using the one-vs-rest approach. We train one logistic regression classifier for each class. To predict the class of an item, we run all the binary classifiers and collect the probability score from each of them. We assign the class which ascribes the highest probability.
#
# Build such a classifier. Train the resulting classifier on (X_train, t_multi_train), test it on (X_val, t_multi_val), tune the hyper-parameters and report the accuracy.
#
# Also plot the decision boundaries for your best classifier similarly to the plots for the binary case.

# %% Cell 39
class NumpyLogReg_OneVsRest(NumpyClassifier):
    
    def __init__(self, bias=-1):
        self.bias=bias
        self.validation_data = None
    def fit(self, X_train, t_train,lr, epochs,tol,n_epochs_no_update = 10,X_valid=None,t_valid=None):
    
        """X_train is a Nxm matrix, N data points, M features
        t_train is avector of length N,
        the targets values for the training data"""
        
        (N, M) = X_train.shape
        
        validation_data = not(X_valid is None and t_valid is None)
           
        if self.bias:           
            X_train = add_bias(X_train, self.bias)
            if validation_data:
                X_valid = add_bias(X_valid, self.bias)
            
        # generatin g a random initialization of weights, tip from chat GPT    
        #self.weights = weights = np.random.randn(M+1)
        self.weights = weights = np.zeros(M+1)
        self.error_train = error_train = [] # cross entropy
        self.accuracy_train = accuracy_train = []
        self.accuracy_val = accuracy_val = []
        self.error_val = error_val = []
        self.error_tol = error_tol = 0
        self.epoch_count = epoch_count = 0 # var to use to accumulate accuracy_tol for n_epochs
         
        for e in range(epochs):
                   
            # calculating loss 
            error_train.append(((self.loss(self.forward(X_train),t_train))))
            # calculating accuracy
            accuracy_train.append(accuracy_3(self.predict_bin(X_train) > 0.5, t_train)) # accuracy on training set
           
               
            if validation_data:
                error_val.append(((self.loss(self.forward(X_valid),t_valid)))) # loss on val set
                accuracy_val.append(accuracy_3(self.predict_bin(X_valid) > 0.5, t_valid)) # accuracy on val set
            
        
            if e > 0:
                if error_tol < tol:
                    error_tol = error_train[-1] - error_train[-2] # the differnce in loss, last element of array -
                    # (last element -1)
                    if abs(error_tol) < tol: # doesn't care about the change in the direction of the loss
                        epoch_count += 1 # makes sure that the loss is monitored
                    else:
                        epoch_count = 0
                if epoch_count >= n_epochs_no_update:
                    print("Ran",e,"number of rounds") # if less, print number of rounds it ran
                    break          
                        
            # calculating weights             
            weights -= (lr / N) *  X_train.T @ (self.forward(X_train) - t_train)    
        
    def loss(self,pred,true):
        limit = 1e-15
        pred = np.clip(pred, limit,(1-limit))
        return -np.mean(true*np.log(pred) + ((1-true) * (np.log(1 - (pred))))) # cross entropy loss
            
    def logistic(self,x):      
        return 1/(1+np.exp(-x))

    def forward(self, X):
        return self.logistic(X @ self.weights)
    
    def predict(self, X): # returning the highest prob node
        threshold = 0.5
        """X is a Kxm matrix for some K>=1
        predict the value for each point in X"""
        z = add_bias(X,self.bias)
        return (self.forward(z)> threshold).astype('int') # returns true if self.forward(z) is higher then threshold
          
        #output = self.forward(X)
        #output = output.reshape(-1, 1) # reshaping
        
        #return np.argmax(output,axis=1)#output.argmax(axis=1) # return the highest probability
    
    def predict_bin(self, X, threshold=0.5):
        """X is a Kxm matrix for some K>=1
        predict the value for each point in X"""
        #z = add_bias(X,self.bias)
        
        return (self.forward(X)> threshold).astype('int') # returns true if self.forward(z) is higher then threshold
          
    

# %% Cell 40
def accuracy_3(predicted, gold):
    predicted = predicted.astype(int)
    
    return np.mean(predicted == gold)

# %% Cell 41
def plot_decision_regions_OneVsRest(X, t, clf_list=[], size=(8,6)):
        
        # The region of the plane to consider determined by X
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
        # Make a prediction of the whole region
        
        h = 0.02  # step size in the mesh\n",
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        plt.figure(figsize=size) # You may adjust this\n",
        
    # iterate over the whole list, tip from chatGPT to print all dec bounds in the same plot
        for clf in clf_list:
            
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            # Classify each meshpoint.
            Z = Z.reshape(xx.shape)
            # Put the result into a color plot
            plt.contourf(xx, yy, Z, alpha=0.2, cmap = 'Paired')
            
        # Plot data points - solution from chatGPT
        #for i, clf in enumerate(clf_list):
            # Filter data points and targets for class i
        for i in range(len(set(t))):
            X_class = X[t == i]
            plt.scatter(X[:,0], X[:,1], c=t, s=10.0, cmap='Paired')
    
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("Decision regions")
        plt.xlabel("x0")
        plt.ylabel("x1")
   

# %% Cell 42
# Implement 1 vs rest

class1 = NumpyLogReg_OneVsRest()
class2 = NumpyLogReg_OneVsRest()
class3 = NumpyLogReg_OneVsRest()
class4 = NumpyLogReg_OneVsRest()
class5 = NumpyLogReg_OneVsRest()

One_vs_Rest = [] # for storing all classes after training

epochs_var = [1,50,100,200,500,1000,2000,4000,6000,8000,10000]
learning_rate = [0.00001,0.0001,0.001,0.01,0.1,0.00005,0.0005,0.005,0.05,0.5,0.00009,0.0009,0.009,0.09,0.9,1.2,1.4,1.6,1.8,2,3]
for unique_target in range(len(set(t_multi_train))):
# splitting the data, setting class1 to zero, class2 (the rest) to 1 for, both train and validation
    
    t_train_split = t_multi_train == unique_target
    t_train_split = t_train_split.astype('int')
    t_val_split = t_multi_val == unique_target
    t_val_split = t_val_split.astype('int')
    
    
    if unique_target == 0:      
        lr = 1 #learning_rate[i]
        epochs = 50#epochs_var[i]
        tol = 1e-5 #tol_var[i]
            
        class1.fit(X_train,t_train_split,lr,epochs,tol,n_epochs_no_update=10,X_valid=X_val,t_valid=t_val_split)
        print("Accuracy with lr = ",lr,"tolerance: ", tol," epoch = ",epochs,":", 
            accuracy_3(class1.predict(X_train),t_train_split), "val:", accuracy_3(class1.predict(X_val),t_val_split))
        One_vs_Rest.append(class1)
    if unique_target == 1:
        lr = 1 #learning_rate[i]
        epochs = 50# epochs_var[i]
        tol = 1e-5 #tol_var[i]
    
        class2.fit(X_train,t_train_split,lr,epochs,tol,n_epochs_no_update=10,X_valid=X_val,t_valid=t_val_split)
        print("Accuracy with lr = ",lr,"tolerance: ", tol," epoch = ",epochs,":", 
            accuracy_3(class2.predict(X_train),t_train_split), "val:", accuracy_3(class2.predict(X_val),t_val_split))
        One_vs_Rest.append(class2)
    if unique_target == 2:
        lr = 1 #learning_rate[i]
        epochs = 1000 # epochs_var[i]
        tol = 1e-5 #tol_var[i]
        
        class3.fit(X_train,t_train_split,lr,epochs,tol,n_epochs_no_update=10,X_valid=X_val,t_valid=t_val_split)
        print("Accuracy with lr = ",lr,"tolerance: ", tol," epoch = ",epochs,":", 
                accuracy_3(class3.predict(X_train),t_train_split), "val:", accuracy_3(class3.predict(X_val),t_val_split))
        One_vs_Rest.append(class3)
    
    if unique_target == 3:      
        lr = 0.61 #learning_rate[i]
        epochs = 200 #epochs_var[i]
        tol = 1e-5 #tol_var[i]
            
        class4.fit(X_train,t_train_split,lr,epochs,tol,n_epochs_no_update=10,X_valid=X_val,t_valid=t_val_split)
        print("Accuracy with lr = ",lr,"tolerance: ", tol," epoch = ",epochs,":", 
            accuracy_3(class4.predict(X_train),t_train_split), "val:", accuracy_3(class4.predict(X_val),t_val_split))
        One_vs_Rest.append(class4)
    
    if unique_target == 4:
        lr = 0.6 #learning_rate[i]
        epochs = 400#epochs_var[i]
        tol = 1e-5 #tol_var[i]
            
        class5.fit(X_train,t_train_split,lr,epochs,tol,n_epochs_no_update=10,X_valid=X_val,t_valid=t_val_split)
        print("Accuracy with lr = ",lr,"tolerance: ", tol," epoch = ",epochs,":", 
            accuracy_3(class4.predict(X_train),t_train_split), "val:", accuracy_3(class5.predict(X_val),t_val_split))
        One_vs_Rest.append(class5)

        t_tr = t_multi_train[(t_multi_train == 0) & (t_multi_train == 3)] = 1
        t_va = t_val_split[(t_multi_val == 0) & (t_multi_val == 3)] = 1
       
        
#plot_decision_regions(X_train, t_multi_train, class1) # unscaled
#plot_decision_regions(X_train, t_multi_train, class2) # unscaled
plot_decision_regions(X_train, t_multi_train, class3) # unscaled
#(X_train, t_multi_train, class4) # unscaled
#plot_decision_regions(X_train, t_multi_train, class5) # unscaled

plot_decision_regions_OneVsRest(X_train,t_multi_train,One_vs_Rest)

# %% Cell 43
# tuning
# loop to plot different lr and epochs


# %% [markdown] Cell 44
# ### For IN4050 students: Multinomial logistic regression
# The following part is only mandatory for IN4050 students. IN3050 students are also welcome to make it a try. Everybody has to do the part 2 on multi-layer neural networks. 
#
# In the lectures, we contrasted the one-vs-rest approach with the multinomial logistic regression, also called softmax classifier. Implement also this classifier, tune the parameters, and compare the results to the one-vs-rest classifier. (Don't expect a large difference on a simple task like this.)
#
# Remember that this classifier uses softmax in the forward phase. For loss, it uses categorical cross-entropy loss. The loss has a somewhat simpler form than in the binary case. To calculate the gradient is a little more complicated. The actual gradient and update rule is simple, however, as long as you have calculated the forward values correctly.

# %% Cell 45
class NumpyLogReg_MultiNom(NumpyClassifier):

    def __init__(self, bias=-1):
        self.bias=bias
        self.validation_data = None
    
    def fit(self, X_train, t_train,lr, epochs,tol,n_epochs_no_update=10,n_class=None,X_valid=None,t_valid=None):
        """X_train is a Nxm matrix, N data points, M features\n
        t_train is avector of length N,the targets values for the training data\"\"\"\n"""
        
        (N, M) = X_train.shape
    
        self.weights = weights = np.random.randn(M+1,n_class) # tip from Chat GPT, random generate init of weights
        #self.weights = weights = np.zeros((M+1,n_class))
        
        #arrays for storing\n"
        self.error_train = error_train = []
        self.accuracy_train = accuracy_train = []
        self.accuracy_val = accuracy_val = []
        self.error_val = error_val = []
        self.error_tol = error_tol = 0
        self.epoch_count = epoch_count = 0

        validation_data = not(X_valid is None and t_valid is None) # checks if there are val-data 
    
        if self.bias:
            X_train = add_bias(X_train, self.bias)
            if validation_data:
                X_valid = add_bias(X_valid, self.bias)
                    
            t_train_encode = self.labels_arange(t_train,n_class) # change the encoding to [0,1,0,0,0] etc.
            t_train_encode = t_train_encode.astype('i')
            
            if validation_data:
                t_val_encode = self.labels_arange(t_valid,n_class) # change the encoding to [0,1,0,0,0] etc.
        

        for e in range(epochs):    
  
            # calculating loss
            error_train.append(((self.loss(self.forward(X_train),t_train_encode))))
            # calculating accuracy
            accuracy_train.append(accuracy(self.predict(X_train), t_train)) # accuracy on training set
        
            if validation_data:
                error_val.append(((self.loss(self.forward(X_valid),t_val_encode)))) # loss on val set
                accuracy_val.append(accuracy(self.predict(X_valid), t_valid)) # accuracy on val set
            # to check if value is below tolerance after n rounds
            if e > 0:
                if error_tol < tol:
                    error_tol = error_train[-1] - error_train[-2] # the differnce in loss, last element of array -
                    # (last element -1)
                    if abs(error_tol) < tol: # doesn't care about the change in the direction of the loss
                        epoch_count += 1 # makes sure that the loss is monitored
                    else:
                        epoch_count = 0
                if epoch_count >= n_epochs_no_update:
                    print("Ran",e,"number of rounds") # if less, print number of rounds it ran
                    break          
                        

                # updating the weights
                weights -= (lr/N) *  X_train.T @ (self.forward(X_train) - t_train_encode) + 2*0.01*weights
                
                X_train[0][0] -= (lr) * np.mean((self.forward(X_train) - t_train_encode)) # update the bias
                #print(X_train[0][0])
                #print((lr) * np.mean((self.forward(X_train) - t_train_encode)))
                
        
    def loss(self,pred,true): #categorical cross entropy loss
        limit = 1e-15
        pred = np.clip(pred, limit,(1-limit))
        return -np.mean(true * np.log(pred))    
    
    def forward(self, X): # SOFT MAX
        #print(X[0])
        probabilities = X @ self.weights # compute raw scores for the samples in the X input data\n"
        exp_probabilities = np.exp(probabilities) # exponential calculated on the raw scores\n"
        # activation/soft max function, return values are normalized probabilites\n"
        return exp_probabilities/np.sum(exp_probabilities, axis = 1, keepdims = True) # normalizing the exponential of the raw scores\n"
    def predict(self, X):
        """X is a Kxm matrix for some K
        predict the value for each point in X"""
        z = self.forward(X)
        z_pred = np.argmax(z, axis=1) # returns the arg max of z, the highest probable value 0-4
        z_ar = self.labels_arange(z_pred,5)
        z_ar = z_ar.astype('i') # convert to int32
        return z_ar # returns the one hot encoded array\n"
    def labels_arange(self,samples,n_class):
        labels_ar = np.zeros((len(samples),n_class)) # make an array containing zeros adjusted for class-size
        labels_ar[np.arange(len(samples)),samples] = 1 # sets the corresponding element to 1
        return labels_ar
    def predict_external(self, X):
        if self.bias:
            X = add_bias(X, self.bias)
        z = self.forward(X)
        z_pred = np.argmax(z, axis=1)
        return z_pred # returns the arg max of z, the highest probability

# %% Cell 46
multinom = NumpyLogReg_MultiNom()

multinom.fit(X_train, t_multi_train,lr=1e-3, epochs=200,tol=1e-7,n_epochs_no_update=10,n_class=5,X_valid=X_val,t_valid=t_multi_val)


# %% [markdown] Cell 47
# # Part 2: Multi-layer neural networks

# %% [markdown] Cell 48
# ## A first non-linear classifier

# %% [markdown] Cell 49
# The following code is a simple implementation of a multi-layer perceptron or feed-forward neural network.
# For now, it is quite restricted.
# There is only one hidden layer.
# It can only handle binary classification.
# In addition, it uses a simple final layer similar to the linear regression classifier above.
# One way to look at it is what happens when we add a hidden layer to the linear regression classifier.

# %% [markdown] Cell 50
# The MLP class below misses the implementation of the `forward()` function. Your first task is to implement it. 
#
# Remember that in the forward pass, we "feed" the input to the model, the model processes it and produces the output. The function should make use of the logistic activation function and bias.

# %% Cell 51
# First, we define the logistic function and its derivative:
def logistic(x):
    return 1/(1+np.exp(-x))

def logistic_diff(y):
    return y * (1 - y)

# %% Cell 52
class MLPBinaryLinRegClass(NumpyClassifier):
    """A multi-layer neural network with one hidden layer"""
    
    def __init__(self, bias=-1, dim_hidden = 6):
        """Intialize the hyperparameters"""
        self.bias = bias
        # Dimensionality of the hidden layer
        self.dim_hidden = dim_hidden
        
        self.activ = logistic
        
        self.activ_diff = logistic_diff
        
        self.vaidation_data = None
    def forward(self, X):
        """TODO: 
        Perform one forward step. 
        Return a pair consisting of the outputs of the hidden_layer
        and the outputs on the final layer"""
        
        hidden_activations = self.activ(X@self.weights1) # activation function given the input vector@weights1
        hidden_outs = add_bias(hidden_activations, self.bias) # adding bias
        outputs = hidden_outs@self.weights2 # output from hidden layer
        return hidden_outs, outputs
        
        raise NotImplementedError
        # return hidden_outs, outputs
    
    def fit(self, X_train, t_train, lr=0.001, epochs = 100,tol=1e-5):
        """Intialize the weights. Train *epochs* many epochs.
        
        X_train is a NxM matrix, N data points, M features
        t_train is a vector of length N of targets values for the training data, 
        where the values are 0 or 1.
        lr is the learning rate
        """
        self.lr = lr
        
        # Turn t_train into a column vector, a N*1 matrix:
        T_train = t_train.reshape(-1,1)
            
        dim_in = X_train.shape[1] 
        dim_out = T_train.shape[1]
        
        # Initialize the weights
        self.weights1 = (np.random.rand(
            dim_in + 1, 
            self.dim_hidden) * 2 - 1)/np.sqrt(dim_in)
        self.weights2 = (np.random.rand(
            self.dim_hidden+1, 
            dim_out) * 2 - 1)/np.sqrt(self.dim_hidden)
        X_train_bias = add_bias(X_train, self.bias)
        
        for e in range(epochs):
            # One epoch
            # The forward step:
            hidden_outs, outputs = self.forward(X_train_bias)
            # The delta term on the output node:
            out_deltas = (outputs - T_train)
            # The delta terms at the output of the hidden layer:
            hiddenout_diffs = out_deltas @ self.weights2.T
            # The deltas at the input to the hidden layer:
            hiddenact_deltas = (hiddenout_diffs[:, 1:] * 
                                self.activ_diff(hidden_outs[:, 1:]))  

            # Update the weights:
            self.weights2 -= self.lr * hidden_outs.T @ out_deltas
            self.weights1 -= self.lr * X_train_bias.T @ hiddenact_deltas 
            
    
    def predict(self, X):
        """Predict the class for the members of X"""
        Z = add_bias(X, self.bias)
        forw = self.forward(Z)[1]
        score= forw[:, 0]
        return (score > 0.5)

# %% Cell 53
# tuning
# loop to plot different lr and epochs

MLP_unscaled = MLPBinaryLinRegClass()
MLP_scaled = MLPBinaryLinRegClass()


lr_var = [0.001,0.1,0.2,0.5,0.6,0.7,0.6,0.7,0.9,0.9,0.9,0.5,0.2,0.6]
epoch_var = [50,100,500,700,1000,3000,5000,8000,10000,20000,40000,50000,65000,150000]
number_of_epochs = 10
tol = 1e-5
print("UNSCALED DATA:")
for i in range(len(lr_var)):
    lr = 0.0001 #lr_var[i]
    epochs = 150000 #epoch_var[i]
    MLP_unscaled.fit(X_train, t2_train,lr,epochs, tol = 1e-5)
    
    print("Accuracy with lr = ",lr,"tolerance: ", tol," epoch = ",epochs,":",accuracy(MLP_unscaled.predict(X_train),t2_train))

tol = 1e-7
lr_var = [0.03,0.03,0.04,0.05,0.06,0.07,0.06,0.07,0.09,0.09,0.09,0.05,0.02,0.06]
print("\nSCALED DATA:")
for i in range(len(lr_var)):
    lr = 0.001 #lr_var
    epochs = 65000 #epoch_var[i]
    MLP_scaled.fit(X_train_scaled,t2_train,lr,epochs, tol = 1e-7)
    
    print("Accuracy with lr = ",lr,"tolerance: ", tol," epoch = ",epochs,":", accuracy(MLP_scaled.predict(X_train_scaled),t2_train))
    
plot_decision_regions(X_train, t2_train, MLP_unscaled)
plot_decision_regions(X_train_scaled, t2_train, MLP_scaled)

# %% Cell 54
class MLPBinaryLinRegClass_ext(NumpyClassifier):
    """A multi-layer neural network with one hidden layer"""
    
    def __init__(self, bias=-1, dim_hidden = 8):
        """Intialize the hyperparameters"""
        self.bias = bias
        # Dimensionality of the hidden layer
        self.dim_hidden = dim_hidden
        
        self.activ = logistic
        
        self.activ_diff = logistic_diff
        
        self.validation_data = validation_data = None
        
    def forward(self, X):
        """TODO: 
        Perform one forward step. 
        Return a pair consisting of the outputs of the hidden_layer
        and the outputs on the final layer"""
        
        hidden_activations = self.activ(X@self.weights1) # activation function given the input vector@weights1
        hidden_outs = add_bias(hidden_activations, self.bias) # adding bias
        outputs = hidden_outs@self.weights2 # output from hidden layer
        return hidden_outs, outputs
        
        raise NotImplementedError
        # return hidden_outs, outputs
    
    def fit(self, X_train, t_train, lr=0.001, epochs = 5500,tol=1e-7,n_epochs_no_update=10,X_valid=None,t_valid=None):
        """Intialize the weights. Train *epochs* many epochs.
        
        X_train is a NxM matrix, N data points, M features
        t_train is a vector of length N of targets values for the training data, 
        where the values are 0 or 1.
        lr is the learning rate
        """
        self.validation_data = not(X_valid is None and t_valid is None)
        self.lr = lr
        
        self.error_train = error_train = []
        self.accuracy_train = accuracy_train = []
        self.accuracy_val = accuracy_val = []
        self.error_val = error_val = []
        self.error_tol = error_tol = 0
        self.epoch_count = epoch_count = 0
        
        # Turn t_train into a column vector, a N*1 matrix:
        T_train = t_train.reshape(-1,1)
        if self.validation_data:
            T_val = t_valid.reshape(-1,1)
      
        dim_in = X_train.shape[1] 
        dim_out = T_train.shape[1]
        
        # Initialize the weights
        self.weights1 = (np.random.rand(dim_in + 1, self.dim_hidden) * 2 - 1)/np.sqrt(dim_in)
        self.weights2 = (np.random.rand(self.dim_hidden+1, dim_out) * 2 - 1)/np.sqrt(self.dim_hidden)
        
        # add bias
        X_train_bias = add_bias(X_train, self.bias) 
        
        if self.validation_data:
             X_val_bias = add_bias(X_val, self.bias) 

        
        for e in range(epochs):
            # One epoch
            # The forward step:
            hidden_outs, outputs = self.forward(X_train_bias)
            hidden_outs_val,outputs_val = self.forward(X_val_bias)
            
            # The delta term on the output node:
            out_deltas = (outputs - T_train)
            # The delta terms at the output of the hidden layer:
            hiddenout_diffs = out_deltas @ self.weights2.T
            # The deltas at the input to the hidden layer:
            hiddenact_deltas = (hiddenout_diffs[:, 1:] * self.activ_diff(hidden_outs[:, 1:]))  

            # calc loss
            error_train.append(self.loss(outputs,T_train))
            # calc acc
            accuracy_train.append(accuracy(outputs > 0.5,T_train))
            
            if self.validation_data:
                error_val.append(self.loss(outputs_val,T_val))
                accuracy_val.append(accuracy(outputs_val > 0.5,T_val))
            
            # to check if value is below tolerance after n rounds
            if e > 0:
                if error_tol < tol:
                    error_tol = error_train[-1] - error_train[-2] # the differnce in loss, last element of array -
                    # (last element -1)
                    if abs(error_tol) < tol: # doesn't care about the change in the direction of the loss
                        epoch_count += 1 # makes sure that the loss is monitored
                    else:
                        epoch_count = 0
                if epoch_count >= n_epochs_no_update:
                    print("Ran",e,"number of rounds") # if less, print number of rounds it ran
                    break          
                        
            # Update the weights:
            self.weights2 -= self.lr * hidden_outs.T @ out_deltas
            self.weights1 -= self.lr * X_train_bias.T @ hiddenact_deltas 
    
    def loss(self,pred,true): # cross entropy loss
        limit = 1e-15
        pred = np.clip(pred, limit,(1-limit))
        return -np.mean(true * np.log(pred))    
    
    def predict(self, X):
        """Predict the class for the members of X"""
        Z = add_bias(X, self.bias)
        forw = self.forward(Z)[1]
        score= forw[:, 0]
        return (score > 0.5)

# %% Cell 55
MLP_ext_unscaled = MLPBinaryLinRegClass_ext()
MLP_ext_scaled = MLPBinaryLinRegClass_ext()

lr_var = [0.001,0.1,0.2,0.5,0.6,0.7,0.6,0.7,0.9,0.9,0.9,0.5,0.2,0.6]
epoch_var = [50,100,500,700,1000,3000,5000,8000,10000,20000,40000,50000,65000,150000]
number_of_epochs = 10
tol = 1e-5
 
print("UNSCALED DATA:")
for i in range(len(lr_var)):
    lr = 0.001 #lr_var[i]
    epochs = 5500 #epoch_var[i]
    MLP_ext_unscaled.fit(X_train, t2_train,lr,epochs, tol = 1e-7,n_epochs_no_update=10,X_valid=X_val,t_valid=t2_val)
    
    #print("Accuracy with lr = ",lr,"tolerance: ", tol," epoch = ",epochs,":",accuracy(MLP_unscaled.predict(X_train),t2_train))

tol = 1e-7
lr_var = [0.03,0.03,0.04,0.05,0.06,0.07,0.06,0.07,0.09,0.09,0.09,0.05,0.02,0.06]
print("\nSCALED DATA:")
for i in range(len(lr_var)):
    lr = 0.001 #lr_var
    epochs = 5500 #epoch_var[i]
    MLP_ext_scaled.fit(X_train_scaled,t2_train,lr,epochs, tol = 1e-7,n_epochs_no_update=10,X_valid=X_val,t_valid=t2_val)
    
    #print("Accuracy with lr = ",lr,"tolerance: ", tol," epoch = ",epochs,":", accuracy(MLP_scaled.predict(X_train_scaled),t2_train))
  

plot_decision_regions(X_train, t2_train, MLP_ext_unscaled)
plot_decision_regions(X_train_scaled, t2_train, MLP_ext_scaled)

# %% Cell 56

MLP_ext_unscaled = MLPBinaryLinRegClass_ext()
MLP_ext_scaled = MLPBinaryLinRegClass_ext()


number_of_epochs = 10
# testing without scaled data

MLP_ext_unscaled.fit(X_train, t2_train,lr=0.001,epochs=8000, tol = 1e-5,n_epochs_no_update=10,X_valid=X_val,t_valid=t2_val)

plt.figure(figsize=(8,6))
plt.plot(MLP_ext_unscaled.error_train,label='Loss_CE_train')
plt.plot(MLP_ext_unscaled.error_val,label='Loss_CE_val')
plt.xlabel('Epochs')
plt.ylabel('Values')
plt.legend()
plt.title('Error Unscaled Data')

plt.figure(figsize=(8,6))
plt.plot(MLP_ext_unscaled.accuracy_train,label='Accuracy_train',color='seagreen')
plt.plot(MLP_ext_unscaled.accuracy_val,label='Accuracy_val',color='r')
plt.xlabel('Epochs')
plt.ylabel('Values')
plt.legend()
plt.title('Accuracy Unscaled Data')

plot_decision_regions(X_train, t2_train, MLP_ext_unscaled)


# testing with Scaled data
MLP_ext_scaled.fit(X_train_scaled,t2_train,lr = 0.001,epochs=5500, tol = 1e-7,n_epochs_no_update=10,X_valid=X_val,t_valid=t2_val)
    
plt.figure(figsize=(8,6))
plt.plot(MLP_ext_scaled.error_train,label='Loss_CE_train')
plt.plot(MLP_ext_scaled.error_val,label='Loss_CE_val')
plt.xlabel('Epochs')
plt.ylabel('Values')
plt.legend()
plt.title('Error Scaled Data')

plt.figure(figsize=(8,6))
plt.plot(MLP_ext_scaled.accuracy_train,label='Accuracy_train',color='seagreen')
plt.plot(MLP_ext_scaled.accuracy_val,label='Accuracy_val',color='r')
plt.xlabel('Epochs')
plt.ylabel('Values')
plt.legend()
plt.title('Accuracy Scaled Data')

plot_decision_regions(X_train_scaled, t2_train, MLP_ext_scaled)

# %% [markdown] Cell 57
# When implemented, this model can be used to make a non-linear classifier for the set `(X, t2)`. Experiment with settings for learning rate and epochs and see how good results you can get. 
# Report results for various settings. Be prepared to train for a long time (but you can control it via the number of epochs and hidden size). 
#
# Plot the training set together with the decision regions as in Part I.

# %% Cell 58
MLP_ext_unscaled = MLPBinaryLinRegClass_ext()
MLP_ext_scaled = MLPBinaryLinRegClass_ext()

lr = 0.001
tol = 1e-7
epochs = 5500
dim = MLP_ext_unscaled.dim_hidden

MLP_tr,MLP_va,MLP_tr_sc,MLP_va_sc = [],[],[],[]

print("Unscaled:")
print("Mean and std over 10 runs", "lr:",lr,"epochs:",epochs,"tol:",tol, "Dim:",dim)
for i in range(10):
    lr = 0.001
    epoch = 8000
    
    MLP_ext_unscaled.fit(X_train, t2_train,lr,epochs, tol = 1e-7,n_epochs_no_update=10,X_valid=X_val,t_valid=t2_val)
    
    MLP_tr.append(accuracy( MLP_ext_unscaled.predict(X_train),t2_train))
    MLP_va.append(accuracy( MLP_ext_unscaled.predict(X_val),t2_val))

MLP_mean_tr = np.mean(MLP_tr)
MLP_mean_va = np.mean(MLP_va)
MLP_STD_tr = np.std(MLP_tr)
MLP_STD_va = np.std(MLP_va)

print("Mean train:",MLP_mean_tr.round(4),"Mean val:",MLP_mean_va.round(4),"STD train:",MLP_STD_tr.round(4),"STD val:",MLP_STD_va.round(4))



print("\nScaled:")

print("Mean and std over 10 runs", "lr:",lr,"epochs:",epochs,"tol:",tol, "Dim:",dim)
for i in range(10):
    lr = 0.001
    epoch = 55000
    
    MLP_ext_scaled.fit(X_train_scaled, t2_train,lr,epochs, tol = 1e-7,n_epochs_no_update=10,X_valid=X_val_scaled,t_valid=t2_val)
    
    MLP_tr.append(accuracy( MLP_ext_scaled.predict(X_train_scaled),t2_train))
    MLP_va.append(accuracy( MLP_ext_scaled.predict(X_val_scaled),t2_val))

MLP_mean_tr = np.mean(MLP_tr)
MLP_mean_va = np.mean(MLP_va)
MLP_STD_tr = np.std(MLP_tr)
MLP_STD_va = np.std(MLP_va)

print("Mean train:",MLP_mean_tr.round(4),"Mean val:",MLP_mean_va.round(4),"STD train:",MLP_STD_tr.round(4),"STD val:",MLP_STD_va.round(4))



# %% [markdown] Cell 59
# # Improving the MLP classifier
# You should now make changes to the classifier similarly to what you did with the logistic regression classifier in part 1.
#
# a) In addition to the `predict()` method, which predicts a class for the data, include the `predict_probability()` method which predict the probability of the data belonging to the positive class. The training should be based on these values, as with logistic regression.
#
# b) Calculate the loss and the accuracy after each epoch and store them for inspection after training.
#
# c) Extend the `fit()` method with optional arguments for a validation set `(X_val, t_val)`. If a validation set is included in the call to `fit()`, calculate the loss and the accuracy for the validation set after each epoch.
#
# d) Extend the `fit()` method with two keyword arguments, `tol` (tolerance) and `n_epochs_no_update` and stop training when the loss has not improved for more than `tol` after `n_epochs_no_update`. A possible default value for `n_epochs_no_update` is 5. Add an attribute to the classifier which tells us after fitting how many epochs it was trained on.
#
# e) Tune the hyper-parameters: `lr`, `tol` and `dim-hidden` (size of the hidden layer).
# Also, consider the effect of scaling the data.
#
# f) After a succesful training with the best setting for the hyper-parameters, plot both training loss and validation loss as functions of the number of epochs in one figure, and both training and validation accuracies as functions of the number of epochs in another figure. Comment on what you see.
#
# g) The MLP algorithm contains an element of non-determinism. Hence, train the classifier 10 times with the optimal hyper-parameters and report the mean and standard deviation of the accuracies over the 10 runs.

# %% [markdown] Cell 60
# ## For IN4050-students: Multi-class neural network

# %% [markdown] Cell 61
# The following part is only mandatory for IN4050 students. IN3050 students are also welcome to make it a try. This is the most fun part of the set :) )
#
# The goal is to use a feed-forward neural network for non-linear multi-class classfication and apply it to the set `(X, t_multi)`.
#
# Modify the network to become a multi-class classifier. As a sanity check of your implementation, you may apply it to `(X, t_2)` and see whether you get similar results as above.
#
# Train the resulting classifier on `(X_train, t_multi_train)`, test it on `(X_val, t_multi_val)`, tune the hyper-parameters and report the accuracy.
#
# Plot the decision boundaries for your best classifier.

# %% Cell 62
class MLP_multi_class(NumpyClassifier):
    """A multi-layer neural network with one hidden layer"""
    
    def __init__(self, bias=-1, dim_hidden = 8):
        """Intialize the hyperparameters"""
        self.bias = bias
        # Dimensionality of the hidden layer
        self.dim_hidden = dim_hidden
        
        self.activ = logistic
        
        self.activ_diff = logistic_diff
        
        self.validation_data = validation_data = None
        
    def forward(self, X):
        """TODO: 
        Perform one forward step. 
        Return a pair consisting of the outputs of the hidden_layer
        and the outputs on the final layer"""
        
        hidden_activations = self.activ(X@self.weights1) # activation function given the input vector@weights1
        hidden_outs = add_bias(hidden_activations, self.bias) # adding bias
        outputs = hidden_outs@self.weights2 # output from hidden layer
        outputs = np.exp(outputs) # exponential calculated on the raw scores\n" to calc soft max on the outputs
        return hidden_outs, outputs/np.sum(outputs, axis = 1, keepdims = True)
        
        raise NotImplementedError
        # return hidden_outs, outputs
    
    def fit(self, X_train, t_train, lr=0.001, epochs = 5500,tol=1e-7,n_class= 5,n_epochs_no_update=10,X_valid=None,t_valid=None):
        """Intialize the weights. Train *epochs* many epochs.
        
        X_train is a NxM matrix, N data points, M features
        t_train is a vector of length N of targets values for the training data, 
        where the values are 0 or 1.
        lr is the learning rate
        """
        self.validation_data = not(X_valid is None and t_valid is None)
        self.lr = lr
        
        self.error_train = error_train = []
        self.accuracy_train = accuracy_train = []
        self.accuracy_val = accuracy_val = []
        self.error_val = error_val = []
        self.error_tol = error_tol = 0
        self.epoch_count = epoch_count = 0
        
        # Turn t_train into a column vector, a N*1 matrix:
        T_train = t_train.reshape(-1,1)
        if self.validation_data:
            T_val = t_valid.reshape(-1,1)
      
        dim_in = X_train.shape[1] 
        dim_out = T_train.shape[1]
        
        # Initialize the weights
        self.weights1 = (np.random.rand(dim_in + 1, self.dim_hidden) * 2 - 1)/np.sqrt(dim_in)
        self.weights2 = (np.random.rand(self.dim_hidden+1, dim_out) * 2 - 1)/np.sqrt(self.dim_hidden)
        
        # add bias
        X_train_bias = add_bias(X_train, self.bias) 
        
        if self.validation_data:
             X_val_bias = add_bias(X_val, self.bias) 
        
        T_train_encode = self.labels_arange(T_train,n_class) # change the encoding to [0,1,0,0,0] etc.
            
        if validation_data:
            T_val_encode = self.labels_arange(T_val,n_class) # change the encoding to [0,1,0,0,0] etc.
        
        for e in range(epochs):
            # One epoch
            # The forward step:
            hidden_outs, outputs = self.forward(X_train_bias)
            hidden_outs_val,outputs_val = self.forward(X_val_bias)
            
            # The delta term on the output node:
            out_deltas = (outputs - T_train_encode)
            # The delta terms at the output of the hidden layer:
            hiddenout_diffs = out_deltas @ self.weights2.T
            # The deltas at the input to the hidden layer:
            hiddenact_deltas = (hiddenout_diffs[:, 1:] * self.activ_diff(hidden_outs[:, 1:]))  

            # calc loss
            error_train.append(self.loss(outputs,T_train_encode))
            # calc acc
            accuracy_train.append(accuracy(outputs > 0.5,T_train_encode))
            
            if self.validation_data:
                error_val.append(self.loss(outputs_val,T_val_encode))
                accuracy_val.append(accuracy(outputs_val > 0.5,T_val_encode))
            
            # to check if value is below tolerance after n rounds
            if e > 0:
                if error_tol < tol:
                    error_tol = error_train[-1] - error_train[-2] # the differnce in loss, last element of array -
                    # (last element -1)
                    if abs(error_tol) < tol: # doesn't care about the change in the direction of the loss
                        epoch_count += 1 # makes sure that the loss is monitored
                    else:
                        epoch_count = 0
                if epoch_count >= n_epochs_no_update:
                    print("Ran",e,"number of rounds") # if less, print number of rounds it ran
                    break          
                        
            # Update the weights:
            self.weights2 -= self.lr * hidden_outs.T @ out_deltas 
            self.weights1 -= self.lr * X_train_bias.T @ hiddenact_deltas 
            X_train[0][0] -= self.lr * np.mean((out_deltas - t_train_encode)) # update the bias
           
                
    def loss(self,pred,true): # cross entropy loss
        limit = 1e-15
        pred = np.clip(pred, limit,(1-limit))
        return -np.mean(true * np.log(pred))    
    
    def predict(self, X):
        """Predict the class for the members of X"""
        Z = add_bias(X, self.bias)
        forw = self.forward(Z)[1]
        z_pred = np.argmax(forw, axis=1) # returns the arg max of z, the highest probable value 0-4
        z_ar = self.labels_arange(z_pred,5)
        z_ar = z_ar.astype('i') # convert to int32
        return z_ar
    
   
    def predict(self, X):
        """X is a Kxm matrix for some K
        predict the value for each point in X"""
        z = add_bias(X, self.bias)
        z = self.forward(z)
        z_pred = np.argmax(z, axis=1) # returns the arg max of z, the highest probable value 0-4
        z_ar = self.labels_arange(z_pred,5)
        z_ar = z_ar.astype('i') # convert to int32
        return z_ar # returns the one hot encoded array\n"
    def labels_arange(self,samples,n_class):
        labels_ar = np.zeros((len(samples),n_class)) # make an array containing zeros adjusted for class-size
        labels_ar[np.arange(len(samples)),samples] = 1 # sets the corresponding element to 1
        return labels_ar
   

# %% [markdown] Cell 63
# # Part III: Final testing
# We can now perform a final testing on the held-out test set we created in the beginning.
#
# ## Binary task (X, t2)
# Consider the linear regression classifier, the logistic regression classifier and the multi-layer network with the best settings you found. Train each of them on the training set and evaluate on the held-out test set, but also on the validation set and the training set. Report the performance in a 3 by 3 table.
#
# Comment on what you see. How do the three different algorithms compare? Also, compare the results between the different dataset splits. In cases like these, one might expect slightly inferior results on the held-out test data compared to the validation and training data. Is this the case? 
#
# Also report precision and recall for class 1.

# %% Cell 64
# tuning
# loop to plot different lr and epochs

number_of_epochs = 10
lr = 0.6
epochs = 800

print("LIN REG:")
lin_reg = NumpyLinRegClass()
lin_reg_val = NumpyLinRegClass()
lin_reg_test = NumpyLinRegClass()

lin_reg.fit(X_train_scaled, t2_train,lr,epochs)
lin_reg_test.fit(X_val_scaled, t2_val,lr,epochs)
lin_reg_val.fit(X_test_scaled, t2_test,lr,epochs)   

print("Accuracy train with lr = ",lr,"tolerance: ", tol," epoch = ",epochs,":",accuracy(lin_reg.predict(X_train_scaled),t2_train))
print("Accuracy val with lr = ",lr,"tolerance: ", tol," epoch = ",epochs,":",accuracy(lin_reg_val.predict(X_val_scaled),t2_val))
print("Accuracy test with lr = ",lr,"tolerance: ", tol," epoch = ",epochs,":",accuracy(lin_reg_test.predict(X_test_scaled),t2_test))

print("LOG REG:")

log_reg = NumpyLogReg()
log_reg_val = NumpyLogReg()
log_reg_test = NumpyLogReg()


number_of_epochs = 10
epochs = 2000#epoch_var[i]

log_reg.fit(X_train_scaled, t2_train,lr,epochs, tol = 1e-7,n_epochs_no_update=number_of_epochs)
log_reg_test.fit(X_train_scaled, t2_train,lr,epochs, tol = 1e-7,n_epochs_no_update=number_of_epochs,X_valid=X_test_scaled,t_valid=t2_test)
log_reg_val.fit(X_train_scaled, t2_train,lr,epochs, tol = 1e-7,n_epochs_no_update=number_of_epochs,X_valid=X_val_scaled,t_valid=t2_val)   

print("Accuracy train with lr = ",lr,"tolerance: ", tol," epoch = ",epochs,":",accuracy(log_reg.predict(X_train_scaled),t2_train))
print("Accuracy val with lr = ",lr,"tolerance: ", tol," epoch = ",epochs,":",accuracy(log_reg_val.predict(X_val_scaled),t2_val))
print("Accuracy test with lr = ",lr,"tolerance: ", tol," epoch = ",epochs,":",accuracy(log_reg_test.predict(X_test_scaled),t2_test))


print("\nMLP:")

lr = 0.001
epochs = 5500

MLP_ext = MLPBinaryLinRegClass_ext()
MLP_ext_val = MLPBinaryLinRegClass_ext()
MLP_ext_test = MLPBinaryLinRegClass_ext()

MLP_ext.fit(X_train_scaled, t2_train,lr,epochs, tol = 1e-7,n_epochs_no_update=number_of_epochs)
MLP_ext_val.fit(X_train_scaled, t2_train,lr,epochs, tol = 1e-7,n_epochs_no_update=number_of_epochs,X_valid=X_test_scaled,t_valid=t2_test)
MLP_ext_test.fit(X_train_scaled, t2_train,lr,epochs, tol = 1e-7,n_epochs_no_update=number_of_epochs,X_valid=X_val_scaled,t_valid=t2_val)   

print("Accuracy train with lr = ",lr,"tolerance: ", tol," epoch = ",epochs,":",accuracy(MLP_ext.predict(X_train_scaled),t2_train))
print("Accuracy val with lr = ",lr,"tolerance: ", tol," epoch = ",epochs,":",accuracy(MLP_ext_val.predict(X_val_scaled),t2_val))
print("Accuracy test with lr = ",lr,"tolerance: ", tol," epoch = ",epochs,":",accuracy(MLP_ext_test.predict(X_test_scaled),t2_test))


# %% [markdown] Cell 65
