"""
    1. Supervised: ML linear regession
    2. Supervised: K nearest neighbour
        ├── ML on breast cancer
        ├── own custom function
        |   ├── test custom function using small data sample
        |   ├── custom function on breast cancer
        ├── K nearest model on breast cancer data::: compare result ML vs custom function
    3. Supervised: Support Vector Machine (SVM)
        ├── ML on breast cancer
        ├── own custom class
        |   ├── find the best fit
        |   ├── predict using optimised w & b
        |   ├── visualise using optimised w & b
        ├── test custom class using small data sample
    4. Unsupervised: Clustering 
        ├── flat
        |   ├── ML on small example
        |   ├── ML on Titanic data. Classify if passenger survive or not
        |   |   ├── custom function to handle non-numerical data
        |   ├── Custom flat clustering class
        |   |   ├── find best fit
        |   |   ├── predict 
        |   ├── test custom class using small data set 
        |   ├── test custom class using Titanic data
        ├── hierarchical
        |   ├── ML on Titanic data. 
        |   ├── Custom hierarchical clustering class
        |   |   ├── find best fit
        |   |   ├── predict 
        |   ├── test custom class using small data set 
"""
# print(__doc__)  # to print anything inside the triple quotation mark on top of the code

import pandas as pd 
import quandl
import math, datetime, random, time
import numpy as np
from sklearn import preprocessing, model_selection , svm, neighbors
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans, MeanShift
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
from collections import Counter
import warnings

""" 
INTRODUCTION LINEAR REGRESSION 
regression = trying to find the best fit line between 2 data sets
y = a*x + b ::: where x = independent variable = feature. y = dependent variable = label

alternative source, in case quandl data no longer available https://www.alphavantage.co/documentation/#
api for alpha advantage = 1K6ZO0UR9SUDGV8H
quandl.ApiConfig.api_key = "69prqnuU7Hv5YrX6zcJ5"
"""
def ML_LinearRegression_StockPrice():
    # get data from quandl and insert into dataframe variable, since output of the get function is a dataframe result
    df = quandl.get('WIKI/GOOGL')   # despite un-searchable, still be abel to get data with that keyword
    # print ( df.head() )
    # print ( len(df) )

    # create new dataframe from dataframe, selective columns only
    df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

    # calculate high/close % change, close/open % change. This will add new column to the dataframe automatically
    df ['prcnt_highclose'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close']
    df ['prcnt_closeopen'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open']

    # create new dataframe from dataframe, selective columns only
    df = df[['Adj. Close','prcnt_highclose','prcnt_closeopen','Adj. Volume']]


    # fill NA with subtaintially large data so that it becomes outliers
    df.fillna(-99999, inplace=True)     # set inplace to True so that don't have to reassign df

    # len returns total number of dataframe. Percentage forecast = proportion of the dataset that we want to forecast
    ### then scale up the number and convert to integer
    prctge_forecast = 0.01   
    lag_days = int(math.ceil(prctge_forecast * len(df)))
    # print (lag_days)

    # set forecast column
    forecast_column = 'Adj. Close'

    # create column label from forecast column set above = shift up data of the chosen column
    df['label'] = df[forecast_column].shift(-lag_days)  # negative number in shift ~ shift up
    # print ( df.iloc[[lag_days]])    # the row that will be shift up to first row
    # print ( df[[forecast_column,'label']])  # forecast column & shifted forecast column 

    # drop column label from dataframe and convert it to array. Then assign it as feature
    X = np.array(df.drop(['label'],1))

    # normalise or standardise dataset to eliminate skewness
    X = preprocessing.scale(X)

    # assign to X_recent X value from the lag day to the most recent data. X_recent wouldn't have the y value (known result, dependent value)
    X_recent = X[-lag_days:]

    # shorten X to from the beginning up to the lag day
    X = X[:-lag_days]

    # there will be NA in new column label so drop NA 
    df.dropna(inplace=True)
    # print ( df[[forecast_column,'label']].tail())   # checking if forecast sounds correct in the most recent data or not. i.e. values in forecast column & shifted forecast column are not too far away from each other
    
    # convert column label to array. Then assign it as label
    y = np.array(df['label'])

    # split X & y between training and testing. This will shuffle the data while still keeping x and y connected, and separate 20% of the data size for testing purpose
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)

    """ comment this out after save classifer to pickle to avoid training data every single running code """ """
    # choose regression model, assign it to classifier variable 
    clf = LinearRegression(n_jobs=-1)

    # fit training data
    clf.fit(X_train, y_train)

    # save the classifier to pickle to avoid the training step. The purpose is to avoid the training step, save time and memory especially when having huge dataset
    ### good idea to train it again once a month
    ### open pickle file with intention to write binary. Name variable as f
    with open (r'Datamart\\linear_regression.pickle', 'wb') as f:
        # dump classifier to f variable
        pickle.dump(clf, f)
    """

    # open pickle file in read binary mode, read then assign to variable
    pickle_in = open(r'Datamart\linear_regression.pickle', 'rb')

    # load pickle to classifier 
    clf = pickle.load(pickle_in)

    # find test score of testing data
    accuracy = clf.score(X_test, y_test)
    # print('linear regression::: lag days =', lag_days, ', accuracy score =', accuracy)

    """
    # choose another regression model. Support Vectore Machine
    clf = svm.SVR()

    # fit training data
    clf.fit(X_train, y_train)

    # find test score of testing data
    accuracy = clf.score(X_test, y_test)
    print('Epsilon Vector regression::: lag days =', lag_days, ', accuracy score =', accuracy)
    """

    # predict y value for x_recent
    forecast_y = clf.predict(X_recent)
    # for item in forecast_y: print( item )
    # print('linear regression::: lag days =', lag_days, ', accuracy score =', accuracy)

    # draw graph
    style.use('ggplot')

    # create a column forecast, fill with nan value
    df['forecast'] = np.nan

    # find the last date in given dataset. Use 'name' to print the index value
    last_date = df.iloc[-1].name

    # convert to unix date
    last_unix = last_date.timestamp()

    # convert 1 day to unix timestamp
    oneday = 86400

    # find the next date in unix timestamp
    next_unix = last_unix + oneday

    # iterrate throught the forecast_y list, assign date for each forecast value = 1 day after the previous day
    for i in forecast_y:
        # convert next date value from unix date to proper timestamp
        next_date = datetime.datetime.fromtimestamp(next_unix)

        # append new row into df dataframe, with index value = next day, value for other column = nan, value forecast column = value from forecast_y
        ### for whatever number of columns in dataframe df up to the second last column, fill it with nan. Fill the last column with value from forecast_y list [i]
        df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

        # add one day to next unix 
        next_unix += oneday
    # print( df ) # print out to visualise the new dataframe df

    # plot adjust close column
    df['Adj. Close'].plot()

    # plot forecast column
    df['forecast'].plot()

    # set legend to bottom right of the graph
    plt.legend(loc=4)

    # set x-axis label to date
    plt.xlabel('Date')

    # set y-axis label to price
    plt.ylabel('Price')

    # show graph
    plt.show()


"""
K NEAREST NEIGHBOUR
Classify train data into group. Find the shortest distance from test data to train data. New data would belong to group that has the shortest distance

source: https://archive.ics.uci.edu/ml/datasets.php
"""
def ML_Knearest_BreastCancer():
    # read data and insert it to dataframe
    df = pd.read_csv(r'Datamart/breast-cancer-wisconsin.data')

    # replace missing data by outlier number
    df.replace('?', -99999, inplace=True)

    # drop id column, as it's useless data
    df.drop(['id'], axis=1, inplace=True)

    # use all columns apart from column class as feature, independent value
    X = np.array(df.drop(['class'],1))

    # use column 'class' as label, dependent value
    y = np.array(df['class'])

    # split X & y between training and testing. This will shuffle the data while still keeping x and y connected, and separate 20% of the data size for testing purpose
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    # select classification model
    clf = neighbors.KNeighborsClassifier(n_jobs=1)

    # train this classification
    clf.fit(X_train, y_train)

    # test this classification
    accuracy = clf.score(X_test, y_test)
    # print('Accuracy =', accuracy)

    """ hypothetical exxample """
    # make up a dummy data equivalent to data for all columns in the dataframe (without id column). Make sure search the make up data to see if it's already exist
    example_measure = np.array([[4,2,1,5,4,2,1,2,3],[4,2,1,5,10,2,3,2,3],[4,5,1,5,4,5,0,2,3]])

    # reshape example data. Value in bracket = (sample count, reduce sample count by). E.g. sample count =20, reduce to sample count =1 ~~ reshape(20,-19)
    example_measure = example_measure.reshape(len(example_measure),-1) 

    # predict class result from make up data
    prediction = clf.predict(example_measure)
    # print('Prediction =', prediction)

    return accuracy


"""
create own function to classify new data into group. Default value for k is 3
data_dict is dictionary type. predict is list type
"""
def k_nearest_neighbour(data_dict, predict, k=3):
    # warn user if number of group (key) in data is more than k
    if len(data_dict) >= k:
        warnings.warn('K is less than total voting group!!')

    # define distance as empty list
    list_distances = []

    # loop through each group in known data dictionary
    for group in data_dict:
        # loop through each feature in group
        for features in data_dict[group]:
            # calculate euclidean distance, using numpy formula to find magnitude (or norm) of vector
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))

            # append euclidean distance with its group to distances list, so that we can sort to find the shortest distance later on
            list_distances.append([euclidean_distance, group])

    # print ( list_distances )    # unsorted list distances
    # print ( sorted(list_distances))     # sorted list distances ascending
    # print ( sorted(list_distances)[:k])     # retrieve the smallest distance, up to the k-th smallest distance
    # sort distances list ascending and take up to the k-th smallest value. 
    ### loop through all k items in the sorted distance list
    # for i in sorted(lists_distances[:k]):
    #     # each i is a list that has 2 values: distance and its group name. Assign to votes list the group name for each i
    #     ### E.g. for k=5, votes = [group1, group1, group4, group2, group5]
    #     votes = i[1]
    # advance version of the code above
    votes = [i[1] for i in sorted(list_distances)[:k]]

    # find the most common group name in the votes list.
    ### E.g. for k=5, votes = [group1, group1, group4, group2, group5]. Most common group name = group1
    # print ( Counter(votes).most_common())   # count of each group name in the votes list
    # print ( Counter(votes).most_common(1))  # most common group name and its count, in a list
    # print ( Counter(votes).most_common(1)[0]) # most common group name and its count, in a tuple
    # final vote = most common group name
    vote_result = Counter(votes).most_common(1)[0][0]

    # calculate confidence of accuracy
    confidence = Counter(votes).most_common(1)[0][1]/k

    return vote_result, confidence


""" 
test custom k nearest function 
"""
def test_custom_Knearest_function():
    # create hypothetical dataset in dictionary style. Set k and set r
    dataset = {'k':[[1,2],[3,4],[3,3]], 'r':[[7,4],[7,7],[10,9]]}

    # new data that we want to test
    new_data = [4,4]

    # result from our own custom function. Test with hypothetical data
    result , confidence = k_nearest_neighbour(dataset, new_data, k=3)
    print (result)

    # loop for every set in dataset dictionary. Then loop for every item in each set. Plot each item's location into scatter plot
    # for i in dataset:
    #     for ii in dataset[i]:
    #         [ plt.scatter(ii[0], ii[1], s=100, color=i) ]
    # advance version of the code above
    [ [ [ plt.scatter(ii[0], ii[1], s=100, color=i) ] for ii in dataset[i] ] for i in dataset ]

    # plot new data. Assign colour to new data as whatever group colour it belongs
    plt.scatter(new_data[0], new_data[1], color=result)

    # display 
    plt.show()


"""
test custom k nearet function with breast cancer data    
"""
def test_custom_Knearest_function_BreastCancer():
    # # read data and insert it to dataframe
    # df = pd.read_csv(r'Datamart/breast-cancer-wisconsin.data', na_values='?', header=0).fillna(-99999).drop(['id'], axis=1)

    # # replace missing data by outlier number
    # df.replace('?', -99999, inplace=True)

    #  # drop id column, as it's useless data
    # df.drop(['id'], 1, inplace=True)
    # short version of the above 3 lines
    df = pd.read_csv(r'Datamart/breast-cancer-wisconsin.data', na_values='?', header=0).fillna(-99999).drop(['id'], axis=1)
        
    # convert all data to float, to clean data. Then convert dataframe value to list
    full_data = df.astype(float).values.tolist()

    # shuffle the data
    random.shuffle(full_data)

    # define test size
    test_size = 0.2

    # assign train and test data, into a list
    train_data = full_data[:-int(test_size * len(full_data))]   # = full data from beginning of list to (1-test_size)-th of full data
    test_data = full_data[-int(test_size * len(full_data)):]    # = full data from (1-test_size)-th of full data until the end

    # assign train set and test set, into dictionary
    train_set = {2:[] , 4:[]}
    test_set = {2:[] , 4:[]}

    # loop for each list in list train data
    for i in train_data:
        # append to train set dictionary. 
        ### i[-1] = the class column of the data (classify either 2 or 4) --> if i[-1] == 2 then append to group 2, else to group 4
        ### append each list in train data, where not appending the last column because i[:-1]
        train_set[ i[-1] ].append( i[:-1] )

    # loop for each list in list test data
    for i in test_data:
        # append to test set dictionary. 
        ### i[-1] = the class column of the data (classify either 2 or 4) --> if i[-1] == 2 then append to group 2, else to group 4
        ### append each list in train data, where not appending the last column because i[:-1]
        test_set[ i[-1] ].append( i[:-1] )

    # pre-define default variable 
    correct = 0
    total = 0

    # loop for each group in test_set. 2 and then 4
    for group in test_set:
        # loop for each list in group
        for item in test_set[group]:
            # use custome build function to test for each list against train set
            result, confidence = k_nearest_neighbour(train_set, item, k=5)  # k=5 because default value in scikit learn is also 5. We try to compare the same apple to apple here
            # check if result from our custom built function return correct value or not. By comparing to group name (2 or 4) in the most outer loop
            if result == group:
                # plus 1 to correct
                correct += 1
            # plus 1 to total to count total number of item for all group has been tested
            total += 1

    # print accuracy
    # print ('Accuracy:', correct/total)
    accuracy = correct/total

    return accuracy


""" 
run ML breast cancer n times. Then run custom test same amount of n times. Then compare the 2 results
"""
def compare_ML_vs_custom_Knearest():
    for n in range(25):     # 25 times
        ML_accuracy = []

        # run ML test. Then append to accuracy list
        starttime = time.time()
        ML_accuracy.append(ML_Knearest_BreastCancer())
        print('ML','run',n+1,':::', time.time() - starttime, 'seconds')

        Custom_accuracy = []
        # run custom test. Then append to accuracy list
        starttime = time.time()
        Custom_accuracy.append(test_custom_Knearest_function_BreastCancer())
        print('Custom','run',n+1,':::', time.time() - starttime, 'second')

    print ('Compare overall accuracy between 2 tests:')
    # calculate overall accuracy after n runs
    print('ML:::',sum(ML_accuracy)/ len(ML_accuracy))

    # calculate overall accuracy after n runs
    print('Custom:::',sum(Custom_accuracy)/ len(Custom_accuracy))


"""
INTRODUCTION SUPPORT VECTOR MACHINE (SVM)
find the best separating hyperplanes / decision boundary that separating different data group
The distance between the hyperplanes and the associated data is the largest

Example: in 2 dimensions and 2 data groups: (-) negative and (+) positive
find Support Vector for each group, that each Vector touchs the most outside of it data group. 
Their equations are w.x+b=1 for positive Supporting Vector & w.x+b=-1 for negative Supporting Vector
where w = vector that perpendicular to the decision boundary line
the Best Separating Vector of 2 groups is constrained between these 2 Supporting Vectors.
in fact, hyperplan equation of the Best Separating Vector is wx+b=0
introducing variable Yi that Yi=-1 for every negative x and Yi=1 for every positive x. 
multiply Yi to (w.x + b) for each negative and positive class, yield that Yi * (w.x + b) - 1 >= 0 for both negative and positive classes
==> that function becomes constraint number 1 for negative and positive classes
for x(-) being on the negative Support Vector, and x(+) being on the positive Support Vector
we can find the width of 2 Support Vectors by taking dot product of [(the Norm unit w) and (difference of the 2 vector above)]
so that width of 2 Support Vectors = [x(+) - x(-)] * w / ||w|| 
plugging Yi * (w.x + b) - 1 = 0 to equation above, we have width of 2 Support Vectors = 2 / ||w||
==> maximising width of 2 Supporting Vectors = minimising ||w|| or minimising 1/2 * ||w|| ^ 2 ==> minimising magnitude of vector w ==> this becomes constraint number 2 
Combining constraint number 1 and 2 together to a function: L (w, b)
we want to minimise w and maximise b
use Langarian (differentiation) in respect to ||w|| and b to find the result
We can use trial and error to find the optimised solution for constraint number 1:  Yi * (w.x + b) - 1 >= 0 
first we start with a value for w, [w1, w2]. Then we plug that and b into Yi * (w.x + b) - 1 to see if it matches criteria. Append all b into a list and then use largest b
we also have to test very every combination of negative and positive w1, w2. [-w1, w2],[w1,-w2],[-w1,-w2] because the magnitude of these combinations are same.
we reduce the value of w, because we want to minimize w. Then again, plug that and b into Yi * (w.x + b) - 1 to see if it matches criteria. Compare any found b with the previous b, and take the higher b value, because we want to maximise b

Soft margin purpose ::: C value ::: lower C ~ margin error is less matter. And vice versa.

Multiple class ::: choose between OVR (One versus Rest) and OVO (One versus One) ::: how algorithm treat relationship between different classification

Kernel ::: to convert to different dimensional space. 

cupcake https://www.youtube.com/watch?v=N1vOgolbjSc. Good understanding of what normalise data do as well
MIT https://www.youtube.com/watch?v=_PwhiWxHK8o
Additional explaination: https://www.svm-tutorial.com/2014/11/svm-understanding-math-part-1/ (read all three parts)   

break down of each parameter: https://youtu.be/93AjE1YY5II
"""
def ML_svm_BreastCancer():
    # read data and insert it to dataframe. Replace na value with extremly large number, drop column id. 
    df = pd.read_csv(r'Datamart/breast-cancer-wisconsin.data', na_values='?', header=0).fillna('-99999').drop(['id'], axis=1)

    # use all columns apart from column class as feature, independent value
    X = np.array(df.drop(['class'], axis=1))

    # use column 'class' as label, dependent value
    y = np.array(df['class'])

    # split X & y between training and testing. This will shuffle the data while still keeping x and y connected, and separate 20% of the data size for testing purpose
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    # select classification model
    clf = svm.SVC()

    # train this classification
    clf.fit(X_train, y_train)

    # test this classification
    accuracy = clf.score(X_test, y_test)
    print('Accuracy =', accuracy)

    """ hypothetical exxample """
    # make up a dummy data equivalent to data for all columns in the dataframe (without id column). Make sure search the make up data to see if it's already exist
    example_measure = np.array([[4,2,1,5,4,2,1,2,3],[4,2,1,5,10,2,3,2,3],[4,5,1,5,4,5,0,2,3]])

    # reshape example data. Value in bracket = (sample count, reduce sample count by). E.g. sample count =20, reduce to sample count =1 ~~ reshape(20,-19)
    example_measure = example_measure.reshape(len(example_measure),-1) 

    # predict class result from make up data
    prediction = clf.predict(example_measure)
    print('Prediction =', prediction)


"""
Create our own SVM class to be precise
Mainly based on constraint Yi * (w.x + b) - 1 >= 0, then use trial and error to gradually reduce value for w. 
Then for each value of w, find maximum value of b. As long as there is w and b that met the criteria Yi * (w.x + b) - 1 >= 0 
"""
class Support_Vector_Machine:
    # when call a class, only the init method will run, no other method
    def __init__(self, visualisation=True):
        # in case the user doesn't specify anything, set the visualisation to default value True
        self.visualisation = visualisation

        # set color: 1 as red, -1 as blue. This will store property to the class. Create property if not yet exist
        self.colors = {1:'r', -1:'b'}

        if self.visualisation:
            # fig = figure in the whole window
            self.fig = plt.figure()
            # grid 1x1
            self.ax = self.fig.add_subplot(1,1,1)

    # train data
    def fit(self, data):
        # insert data to class's data
        self.data = data
        # dictionary of optimised w and b { ||w|| : [w,b] ...}
        dict_optimised = {}

        # transformation dictionary to check all possible negative & positive points
        transforms = [[1,1], [-1,1], [1,-1], [-1,-1]]

        # create all data list, append feature to list. Then find max, min in all data list
        all_data=[]
        # loop for each classification class
        for class_y in self.data:
            # loop for each set in the group
            for featureset in self.data[class_y]:
                # loop for each value in the set
                for feature in featureset:
                    # append each value to all data list. I.e. all data list becomes [1, 7, 2, 8, 3, 8, 5, 1, 6, -1, 7, 3]
                    all_data.append(feature)
        
        # find max and min value from all data list. Store max and min feature to class
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)

        # clear everything in all data list
        all_data = None

        # define step size. Take big step first, then gradually smaller step.
        step_size = [self.max_feature_value * 0.1,
                    self.max_feature_value * 0.01,
                    self.max_feature_value * 0.001] 
        
        # determine the stepping range for b ::: (-1) * (range multiple) * (max feature value) =< b =< (1) * (range multiple) * (max feature value)
        # b doesn't need to be as precise as w, and it costs more memory + calculation time so make it small
        b_range_multiple = 5

        # determine the step size for b ::: (step size of w) * (b step size multiple)
        # again, step of b doesn't have to be as small as w
        b_multiple = 5

        # assumption value for w = max feature value * 10. Since we want to minimize w, we start from the largest possible point, and start reducing values for w
        latest_optimum = self.max_feature_value * 10    # use 10 to minimize processing time. Can choose any number, as long as larger than 1

        # begin each w stepping
        for step in step_size:
            # assuming vector w value = [max feature value * 10, max feature value * 10]. 
            w = np.array([latest_optimum,latest_optimum])
            
            # set default value being we have not yet optimised. optimised = false
            bolean_w_minimized = False

            # loop until optimised
            while not bolean_w_minimized:
                # iterate through b. Begin each b stepping. Going from negative to positive value to ensure that b is maximised for each w step
                for b in np.arange(-1 * (self.max_feature_value * b_range_multiple) ,   # starting point = (-1) * (range multiple) * (max feature value)
                                    self.max_feature_value * b_range_multiple,          # stop point = (1) * (range multiple) * (max feature value)
                                    step * b_multiple):                                 # how big the step is = (step size of w) * (b step size multiple)
                    # loop through transforms list
                    for transform in transforms:
                        # multiply each w to its transform negative and positive value::: w_t = w transformed
                        w_t = w * transform

                        # # finish assigning w and b for each check
                        # print ('w, b combination =', w_t, ':::', b ) 

                        # set default found option to true, assuming that our contraint yi*(w.xi+b) -1 >= 0 has been met. 
                        # if doesn't satisfy our constraint yi*(w.xi+b) -1 >= 0 , will change to false
                        boolean_b_found = True

                        # (weakest link in the SVM fundamental) ???????????
                        # loop for each class in given data
                        for class_y in self.data:
                            # loop for each set in each class
                            for xi in self.data[class_y]:
                                # label yi = class name = class_y
                                yi = class_y

                                # check if match our constraint yi*(w.xi+b) -1 >= 0
                                ### (class name: 1 or -1) * [ dotproduct(w:transformed max feature value array & xi:set array in group) + b ] -1 >= 0
                                if not yi*(np.dot(w_t,xi) + b) -1 >= 0:
                                    # if not statisfied with our constraint, then it's not optimised. 
                                    boolean_b_found = False
                                    # get out of loop xi, since this w_t value is no longer optimised 
                                    break

                            # get out of loop class_y, since this w_t value is no longer optimised 
                            if not boolean_b_found:
                                break
                                    
                        # if option found boolean remain true ~ satisfy our constraint
                        if boolean_b_found:
                            # append to dictionary optimised { ||w|| : [w,b] } ::: magnitude of vector ||w|| and b to the key ||w||
                            dict_optimised[np.linalg.norm(w_t)] = [w_t, b]      # np.linalg.norm() returns magnitude of the vector

                # once finished with every b option. Go to next reduced w by its step 
                # check if value of w vector is negative. If already negative, don't need to go any lower 
                # Because using transformation above, we have already checked for each negative value in vector w
                if w[0] < 0:    
                    bolean_w_minimized = True
                    print('optimised one w step.')
                else:
                    # take one step lower
                    w = w - step

            # once finished with every b step. Sort key ||w|| ascending.
            optimised_vector = sorted([n for n in dict_optimised])
            
            # take the lowest key. I.e. lowest magnitude of ||w||
            optimised_choice = dict_optimised[optimised_vector[0]]

            # w = first element in the { ||w|| : [w,b] } where ||w|| found above was the lowest. Store w in class as an array, since w is a vector
            self.w = optimised_choice[0]

            # b = second element in the { ||w|| : [w,b] } where ||w|| found above was the lowest. Store b in class as a value
            self.b = optimised_choice[1]

            # assign new value for latest optimum = value w we will start with from the top. And start to reduce w gradually by each step from this value
            latest_optimum = optimised_choice[0][0] + step*2

                                
    # predict, using self data (w and b) 
    def predict(self, features):
        # find sign of w.x + b. Numpy.sign() return 1 if > 0, -1 if < 0, 0 if = 0. 
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)

        # visualise in graph
        if classification != 0 and self.visualisation:
            # draw in scatter. Color = either 1 or -1
            self.ax.scatter(features[0], features[1], s=200, marker='*', color=self.colors[classification])

        return classification

    # visualise
    def visualise(self):
        # plot each data point as scatter. Define scatter size and color for each feature data set
        # for i in self.data:
        #     for x in self.data[i]:
        #         self.ax.scatter(x[0], x[1], s=100, color=self.colors[i])
        # advance version of 3 lines above
        [ [self.ax.scatter(x[0], x[1], s=100, color=self.colors[i]) for x in self.data[i]] for i in self.data]

        # definition of hyperplane w.x + b . Give use the hyperplane we seek, given value v
        ### v = w.x + b. When v = 1 (positive support vector); v = -1 (negative support vector); 0 = decision boundry
        def hyperplane(x, w, b, v):
            return (-w[0] * x - b + v) / w[1]

        # extend out the data range a little so that lines fully cover all data
        # datarange = (self.min_feature_value * 0.9, self.max_feature_value * 1.1)
        # hyp_x_min = datarange[0]
        # hyp_x_max = datarange[1]
        # simplify those 3 lines above
        hyp_x_min = self.min_feature_value * 0.9
        hyp_x_max = self.max_feature_value * 1.1

        # positive support vector::: w.x + b = 1 
        psv1 = hyperplane(hyp_x_min , self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max , self.w, self.b, 1)

        # plot positive support vector
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], color='k')

        # negative support vector::: w.x + b = -1 
        nsv1 = hyperplane(hyp_x_min , self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max , self.w, self.b, -1)

        # plot negative support vector
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], color='k')

        # decision boundary::: w.x + b = 0 
        dcb1 = hyperplane(hyp_x_min , self.w, self.b, 0)
        dcb2 = hyperplane(hyp_x_max , self.w, self.b, 0)

        # plot decision boundary
        self.ax.plot([hyp_x_min, hyp_x_max], [dcb1, dcb2], color='y', dashes=(5,1)) # color = yellow + dash line 

        # show
        plt.show()


"""
create our own data for training and for testing
"""
def test_custom_Support_Vector_Machine():
    # Classification group = -1 or 1. Not too sure why have to use numpy array
    example_dict_data = {-1: [[1,7], [2,8], [3,8]],
                        1: [[5,1],[6,-1],[7,3]]}

    starttime = time.time()
    # initiate __init__ 
    custom_svm = Support_Vector_Machine()   

    # find optimised w and b
    custom_svm.fit(data = example_dict_data)
    print ( time.time() - starttime,"seconds")

    # choose style
    style.use('ggplot')

    # # graph known data 
    # custom_svm.visualise()


    # create our own data to predict their classification 
    example_predict_data = [[10,0] , [1,3] , [3,4] , [3,5] , [5,5] , [5,6] , [6,-5] , [5,8] ]

    # loop through each set to predict and plot them into graph
    for each_set in example_predict_data:
        custom_svm.predict(each_set)

    # graph known data and prediction
    custom_svm.visualise()


"""
INTRODUCTION UNSUPPERVISED MACHINE LEARNING: CLUSTERING
flat::: we tell machine how many class V.S. hierarchical::: machine decides how many class

Machine will find the centroids points. Number of centroids depends on number of classification. Calculate distances between centroids and data around it
Keeps on finding the centroid until the distances between raw data points are roughly the same i.e centroid is exactly in the centre of data points
"""
def ML_clustering_flat_smallExample():
    # create dummy data as feature. Machine's task is to assign each of these features into labels
    X = np.array([[1,2], [1.5,1.8], [5,8], [8,8], [1,0.6], [9,11]])

    style.use('ggplot')
    # display in graph without centroid
    # plt.scatter(X[:,0], X[:,1], s=150)
    # plt.show()

    # choose regression, classification model. n_cluster = # of classificaiton 
    clf = KMeans(n_clusters=2)

    # train machine using dummy data. 
    ### NOTICE: here, we don't have any label. In ML Knearest and SVM, we separated data to X train, X test, Y train, Y test.
    clf.fit(X)

    # use attribute of clf object: centroids
    centroids = clf.cluster_centers_

    # labels = array of labels of features. 
    labels = clf.labels_

    # set colour. Period (.) means it will be a dot
    colors = ["g.","r.", "c.", "b.", "k.", "o."]

    # loop through each data set
    for i in range(len(X)):
        # plot value of each set and its color will be determined by its label value and retrieved from the color list
        plt.plot (X[i][0], X[i][1], colors[labels[i]], markersize=25)

    # graph centroid as scatter
    plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=150, linewidths=5)
    plt.show()


"""
Machine learning flat clustering unsupervised classification, 2 clusters. Titanic survivor example
"""
def ML_clustering_flat_Titanic():
    # data description for Titanic xlsx file:
        # Pclass = Passenger Class (1 = 1st = top of the ship; 2 = 2nd; 3 = 3rd = bottom of the ship where flooded first)
        # survived = Survival (0 = No; 1 = Yes)
        # name = Name
        # sex = Sex
        # age = Age
        # sibsp = Number of Siblings/Spouses Aboard
        # parch = Number of Parents/Children Aboard
        # ticket = Ticket Number
        # fare = Passenger Fare (British pound)
        # cabin = Cabin
        # embarked = Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
        # boat = Lifeboat
        # body = Body Identification Number
        # home.dest = Home/Destination

    # insert excel data to dataframe
    df = pd.read_excel(r'Datamart\titanic.xlsx')
    # print (df.head())

    # drop unused columns 
    df.drop(['body','name'], axis=1, inplace=True)

    # convert all columns that looks like numeric to numeric
    df = df.convert_objects(convert_numeric=True)

    # fillna with 0
    df.fillna(0, inplace=True)
    # print(df.head())

    # function to convert string to numeric
        # insert data of the column we want to convert into a set. Since a set doesn't have duplication, this will give us a unique values of the column
        # index number of the set will replace original values in the columns
    def handle_nonNumeric_data(df):
        # very column names
        columns = df.columns.values

        # loop through every columns
        for column in columns:
            # create empty dictionary. This dictionary will be renewed everytime go to another column 
            text_digit_values = {}
            
            # function to convert to integer. Return number that equivalent to text value, stored in text digit values dictionary
            def convert_to_int(value):
                return text_digit_values[value]
            
            # check datatype of column NOT integer or float
            if df[column].dtype != np.int64 and df[column].dtype != np.float64:
                # insert all column values to list
                full_contents = df[column].values.tolist()

                # insert full content list to set
                unique_contents = set(full_contents)

                # set starting value of x
                x = 0

                # loop through each value in unique content set
                for unique in unique_contents:
                    # check if value is already in dictionary created above
                    if unique not in text_digit_values:
                        # insert unique value to dictionary. So that dictionary becomes {'a':0, 'b':1, ...}
                        text_digit_values[unique] = x
                        # plus 1 to x
                        x += 1

                # map result of the convert_to_int finction with every value of the column. I.e. replacing every value of the column with number in dictionary
                ### map function works like VLookup, Match
                df[column] = list(map(convert_to_int, df[column]))

        return df

    # call the function above
    df = handle_nonNumeric_data(df)
    # print(df.head())

    # drop further column, if desired. Gotta use intuitive and manual selection here, until achieve higher accuracy rate
    df.drop(['pclass'], axis=1, inplace=True)

    # feature = everything in dataframe, except some unused columns, which have been dropped above. Also this time drop survived column because it originally label
    ### turn it to array so that every row in dataframe becomes a list. And X becomes list of lists
    X = np.array(df.drop(['survived'], axis=1).astype(float))   # also convert all data to float if not yet converted

    # scale X
    X = preprocessing.scale(X)

    # actual label data. y becomes a list of label
    y = np.array(df['survived'])
    # print(X)
    # print(y)

    # choose classification. n_cluster = The number of clusters to form as well as the number of centroids to generate.
    clf = KMeans(n_clusters=2)  # since this this FLAT clustering, we tell machine how many class we want to separate

    # train data
    clf.fit(X)

    # set starting correct number 
    int_correct = 0

    # loop through every row in feature. I.e. every row in dataframe
    for i in range(len(X)):
        # convert each row to float (again) and convert to array
        predict_me = np.array(X[i].astype(float))
        # reshape ~ transpose from (2x1) shape to (1x2) shape
        predict_me = predict_me.reshape(-1, len(predict_me))
        # output label of machine
        y_label = clf.predict(predict_me)

        # check if output result machine generate matches actual data. Since the output result is as a list. Use first item in the list [0]
        if y_label[0] == y[i]:
            # plus 1 to correct variable
            int_correct += 1

    # end of loop. Print result
    accuracy = int_correct / len(y)

    # use higher of 1- accuracy or accuracy to always get consistent result
    print('Correct probability =', max(1- accuracy, accuracy))


"""
create our own custom clustering class. Minimize distances between centroid and data surround centroid
"""
class clustering_flat_Kmeans():
    # initialisation. Clustering k = number of classes = k. Margin of error = tolerance. Max number of iteration = max number of attempt to find the most accurate distance
    def __init__(self, k=2, tolerance=0.01, max_iter=300):
        self.k = k
        self.tolerance = tolerance
        self.max_iter = max_iter

    # train machine, will store result into self
    def fit(self, train_data):
        # define centroid as empty. Append new centroid to this dictionary everytime found a new centroid
        self.dict_centroids = {}

        # loop through k number of clusters
        for i in range(self.k):
            # pick the first k-th data as the starting centroid
            self.dict_centroids[i] = train_data[i]

        # start iteration until max iteration
        for i in range(self.max_iter):
            # define classification: contains centroid index and feature (train) data that is closest to each centroid
            self.dict_class = {}

            # loop through k number of clusters, to create k numbers of empty class in dictionary
            for n in range(self.k):
                self.dict_class[n] = []

            # loop through each feature
            for feature in train_data:
                # calculate the distance between each feature data and each centroid
                ### there are 2 clusters. So that distance[0] = distance between feature to centroid[0]; distance[1] = distance between feature to centroid[1]
                # for centroid in self.dic_centroids:
                #     # use numpy's norm() to calculate magnitude
                #     distances = np.linalg.norm(feature - self.dict_centroids[centroid])
                # advance version of the loop above
                distances = [np.linalg.norm(feature - self.dict_centroids[centroid]) for centroid in self.dict_centroids]
                # print(distances)

                # categorise feature into classification = Either 0 or 1, depends on shortest distances between feature vs centroid 0 and centroid 1
                classification = distances.index(min(distances))
                # print('data :::', feature, ' belongs to centroid:::',classification)

                # append feature to class that it's closest = this particular feature is closest to this class
                self.dict_class[classification].append(feature)

            # store current checked centroid to another dict variable for comparison
            prev_centroids = dict(self.dict_centroids)

            # print('current centroid =',self.dict_centroids)
            # print('features that closest to current centroid :::')
            # loop through each class in class dictionary
            for classification in self.dict_class:
                # print(self.dict_class[classification])
                # redefine new location for each centroid = average of location of all data that are currently stored in each class. That's why it's call Kmean
                self.dict_centroids[classification] = np.average(self.dict_class[classification], axis=0)
            # print('next centroid to check =', self.dict_centroids)
            
            # assume that we have found optimised
            bool_optimised = True

            # loop for each new centroid
            for c in self.dict_centroids:
                # calculate the different distance between the new centroid and the previous centroid
                ### if the movemoment (distance) of current centroid from the previous centroid is more than the tolerance amount, set above. 
                ### take absolute value to capture case training data is negative
                if abs(np.sum((self.dict_centroids[c] - prev_centroids[c])/prev_centroids[c] * 100.0)) > self.tolerance:
                    # Then not yet optimised
                    bool_optimised = False
                    # print('\n','check again :::')
            
            # if already optimised, get out of current iteration. I.e. done
            ### else optimised = false, go to the next iteration. I.e. check using the new centroid
            if bool_optimised:
                break

    # use stored result in self to predict new data
    def predict(self, data_to_predict):
        # calculate distance between each centroid (already optimised) and data to predict
        distances = [np.linalg.norm(data_to_predict - self.dict_centroids[centroid]) for centroid in self.dict_centroids]

        # predicted cluster = Either 0 or 1, depends on shortest distances between feature vs centroid 0 and centroid 1
        closest_cluster = distances.index(min(distances))

        return closest_cluster


"""
test custom flat clustering function using small data sample
"""
def test_custom_clustering_flat_Kmeans():
    # create dummy data as feature. Our custom function is to assign each of these features into labels
    X = np.array([[1,2], [1.5,1.8], [5,8], [8,8], [1,0.6], [9,11]])

    style.use('ggplot')

    # define colour. b=blue, g=gren, r=red, c=cyan, m=magenta, y=yellow, k=black, w=white
    colors = ["g","r", "c", "b", "k", "w"]

    # train data using custom flat clustering function
    clf = clustering_flat_Kmeans()
    clf.fit (X)

    # graph each centroid
    for centroid in clf.dict_centroids:
        plt.scatter(clf.dict_centroids[centroid][0], clf.dict_centroids[centroid][1], marker='o', color='k', linewidths=5)

    # loop for each class
    for cla in clf.dict_class:
        # set color 
        color = colors[cla]
        # graph for each train data in class
        for trainData in clf.dict_class[cla]:
            plt.scatter(trainData[0], trainData[1], marker='x', color=color, s=150, linewidths=5)

    # create new data for prediction
    y = np.array([[1,3], [8,9], [0,3], [5,4], [7,7], [4.5,5]])

    # predict new data. Loop for each data need to predict
    for unknown in y:
        predicted_class = clf.predict(unknown)
        # graph prediction data and their class
        plt.scatter(unknown[0], unknown[1], marker='*', color=colors[predicted_class], s=150, linewidths=5)

    plt.show()


"""
test custom flat clustering function using Titanic data
"""
def test_custom_clustering_flat_Kmeans_Titanic():
    # insert excel data to dataframe
    df = pd.read_excel(r'Datamart\titanic.xlsx')
    # print (df.head())

    # drop unused columns 
    df.drop(['body','name'], axis=1, inplace=True)

    # convert all columns that looks like numeric to numeric
    df = df.convert_objects(convert_numeric=True)

    # fillna with 0
    df.fillna(0, inplace=True)
    # print(df.head())

    # function to convert string to numeric
        # insert data of the column we want to convert into a set. Since a set doesn't have duplication, this will give us a unique values of the column
        # index number of the set will replace original values in the columns
    def handle_nonNumeric_data(df):
        # very column names
        columns = df.columns.values

        # loop through every columns
        for column in columns:
            # create empty dictionary. This dictionary will be renewed everytime go to another column 
            text_digit_values = {}
            
            # function to convert to integer. Return number that equivalent to text value, stored in text digit values dictionary
            def convert_to_int(value):
                return text_digit_values[value]
            
            # check datatype of column NOT integer or float
            if df[column].dtype != np.int64 and df[column].dtype != np.float64:
                # insert all column values to list
                full_contents = df[column].values.tolist()

                # insert full content list to set
                unique_contents = set(full_contents)

                # set starting value of x
                x = 0

                # loop through each value in unique content set
                for unique in unique_contents:
                    # check if value is already in dictionary created above
                    if unique not in text_digit_values:
                        # insert unique value to dictionary. So that dictionary becomes {'a':0, 'b':1, ...}
                        text_digit_values[unique] = x
                        # plus 1 to x
                        x += 1

                # map result of the convert_to_int finction with every value of the column. I.e. replacing every value of the column with number in dictionary
                ### map function works like VLookup, Match
                df[column] = list(map(convert_to_int, df[column]))

        return df

    # call the function above
    df = handle_nonNumeric_data(df)
    # print(df.head())

    # drop further column, if desired. Gotta use intuitive and manual selection here, until achieve higher accuracy rate
    df.drop(['pclass'], axis=1, inplace=True)

    # feature = everything in dataframe, except some unused columns, which have been dropped above. Also this time drop survived column because it originally label
    ### turn it to array so that every row in dataframe becomes a list. And X becomes list of lists
    X = np.array(df.drop(['survived'], axis=1).astype(float))   # also convert all data to float if not yet converted

    # scale X
    X = preprocessing.scale(X)

    # actual label data. y becomes a list of label
    y = np.array(df['survived'])
    # print(X)
    # print(y)

    # choose classification: our custom class clustering flat
    clf = clustering_flat_Kmeans()  

    # train data
    clf.fit(X)

    # set starting correct number 
    int_correct = 0

    # loop through every row in feature. I.e. every row in dataframe
    for i in range(len(X)):
        # convert each row to float (again) and convert to array
        predict_me = np.array(X[i].astype(float))
        # reshape ~ transpose from (2x1) shape to (1x2) shape
        predict_me = predict_me.reshape(-1, len(predict_me))
        # output label of machine
        y_label = clf.predict(predict_me)

        # check if output result machine generate matches actual data. Since the output result is as a list. Use first item in the list [0]
        if y_label == y[i]:
            # plus 1 to correct variable
            int_correct += 1

    # end of loop. Print result
    accuracy = int_correct / len(y)

    # use higher of 1- accuracy or accuracy to always get consistent result
    print('Correct probability =', max(1- accuracy, accuracy))


"""
Machine learning hierarchical clustering unsupervised classification, 2 clusters. Titanic survivor example
"""
def ML_clustering_hierarchical_Titanic():
    # insert excel data to dataframe
    df = pd.read_excel(r'Datamart\titanic.xlsx')
    # print (df.head())

    # create copy of the orginal dataframe, before modifying the data (convert to text to numeric, drop columns etc...)
    original_df = pd.DataFrame.copy(df)     # cannot do original df = df because if we modify df, it will also modify original df

    # drop unused columns 
    df.drop(['body','name'], axis=1, inplace=True)

    # convert all columns that looks like numeric to numeric
    df = df.convert_objects(convert_numeric=True)

    # fillna with 0
    df.fillna(0, inplace=True)
    # print(df.head())

    # function to convert string to numeric
        # insert data of the column we want to convert into a set. Since a set doesn't have duplication, this will give us a unique values of the column
        # index number of the set will replace original values in the columns
    def handle_nonNumeric_data(df):
        # very column names
        columns = df.columns.values

        # loop through every columns
        for column in columns:
            # create empty dictionary. This dictionary will be renewed everytime go to another column 
            text_digit_values = {}
            
            # function to convert to integer. Return number that equivalent to text value, stored in text digit values dictionary
            def convert_to_int(value):
                return text_digit_values[value]
            
            # check datatype of column NOT integer or float
            if df[column].dtype != np.int64 and df[column].dtype != np.float64:
                # insert all column values to list
                full_contents = df[column].values.tolist()

                # insert full content list to set
                unique_contents = set(full_contents)

                # set starting value of x
                x = 0

                # loop through each value in unique content set
                for unique in unique_contents:
                    # check if value is already in dictionary created above
                    if unique not in text_digit_values:
                        # insert unique value to dictionary. So that dictionary becomes {'a':0, 'b':1, ...}
                        text_digit_values[unique] = x
                        # plus 1 to x
                        x += 1

                # map result of the convert_to_int finction with every value of the column. I.e. replacing every value of the column with number in dictionary
                ### map function works like VLookup, Match
                df[column] = list(map(convert_to_int, df[column]))

        return df

    # call the function above
    df = handle_nonNumeric_data(df)
    # print(df.head())

    # drop further column, if desired. Gotta use intuitive and manual selection here, until achieve higher accuracy rate
    df.drop(['pclass'], axis=1, inplace=True)

    # feature = everything in dataframe, except some unused columns, which have been dropped above. Also this time drop survived column because it originally label
    ### turn it to array so that every row in dataframe becomes a list. And X becomes list of lists
    X = np.array(df.drop(['survived'], axis=1).astype(float))   # also convert all data to float if not yet converted

    # scale X
    X = preprocessing.scale(X)

    # actual label data. y becomes a list of label
    y = np.array(df['survived'])
    # print(X)
    # print(y)

    # choose classification
    clf = MeanShift()  

    # train data
    clf.fit(X)

    # label and cluster result from machine learning 
    labels_y = clf.labels_
    cluster_centers = clf.cluster_centers_

    # # create new empty column in original dataframe
    # original_df['class_result'] = np.nan

    # # loop through each row in dataframe
    # for i in range(len(X)):
    #     # insert label result from machine learning to the new column created in original dataframe
    #     original_df['class_result'].iloc[i] = labels_y[i]
    # advance version of the code above
    original_df['class_result'] = pd.Series(labels_y)

    """ futher analysis """
    # unique number of cluster
    n_clusters_ = len(cluster_centers)      #or  len(np.unique(labels_y))
    
    # dictionary to store each class and survival rate for each class
    dict_survival_rates = {}

    # loop through each cluster result
    for i in range(n_clusters_):
        # temporary df = all rows that the column result being equals to the current cluster
        current_cluster_df = original_df[ (original_df['class_result'] == float(i)) ]

        # filter out within this cluster, how many people survived
        survival_number = current_cluster_df [ (current_cluster_df['survived'] == 1) ]

        # calculate survival percentage = survival number / total number of people in current cluster
        survival_percent = len(survival_number) / len(current_cluster_df)

        # append result to survival rates dictionary
        dict_survival_rates[i] = survival_percent
    
    # print survival rates dictionary = listing all of clusters machine learning found and survival percentage for each cluster
    print (dict_survival_rates)

    # print all original data where machine learning assign it to cluster 0
    print ( original_df[ (original_df['class_result'] == 0)])

    # describe data where machine learning assign it to cluster 0
    print ( original_df[ (original_df['class_result'] == 0)].describe() )


"""
create our own custom clustering class. 

Drawback: if we set radius to a constant number, the code run fast. But we would not have that luxury in a real world where data set is big, or in case of multi dimension
--> not realistic in real world to fix radius
Solution: use a large radius. But everytime we check distance between data point and the current centroid, we penalise when the data is further away from the centroid
"""
class clustering_hierarchical_MeanShift():
    # initialisation. radius = radius around each centroid.
    def __init__(self, radius=None, radius_norm_step=100):                                                       # solution: introduce radius norm step
        self.radius = radius
        self.radius_norm_step = radius_norm_step                                                                 # solution: introduce radius norm step

    # train data. Take constant variable from self. Store result into self
    def fit(self, train_data):
        # case radius is not specified by user                                                                   # solution: introduce radius norm step
        if self.radius == None:
            # take average of all data dimension ~ consider it as centroid of all data
            all_data_centroid = np.average(abs(train_data), axis=0)
            # calculate norm of all data centroid
            all_data_norm = np.linalg.norm(all_data_centroid)
            # set radius to something other than None = norm of all data centroid / radius norm step
            self.radius = all_data_norm / self.radius_norm_step

        # define weight. Reverse order so that the list starts from 99, 98, ... 0                               # solution: introduce radius norm step
        list_weights = [i for i in range(self.radius_norm_step)][::-1]

        # emtpy dictionary to store centroids
        dict_centroids = {}

        # loop through all data
        for i in range(len(train_data)):
            # assigning centroid to all of data = assuming each data is a centroid
            dict_centroids[i] = train_data[i]

        # loop until incorrect. We could have tolerance here like in Kmeans function but scikit learn doesn't have tolerance so are don't have either
        while True:
            # create empty list to store new centroid
            list_new_centroids = []

            # loop through each current centroid. 
            for i in dict_centroids:
                # i = key in centroid dictionary. Store value of key i into centroid variable
                centroid = dict_centroids[i]

                # empty list to store any feature that is inside this centroid
                list_within_radius = []

                # empty list to store weight of each feature, relative to this centroid                         # solution: introduce radius norm step
                list_weight_feature = []

                # loop through each feature to check if any feature is within radius of current centroid
                for each_feature in train_data:
                    """ old solution ::: not realistic --> discard """ """
                    # check if any feature is within radius of current centroid
                    if np.linalg.norm(each_feature - centroid) < self.radius:
                        # append that feature into list 'in radius'
                        list_within_radius.append(each_feature)
                    """ """ old solution ::: not realistic --> discard """
                                                                                                                # solution: introduce radius norm step
                    # calculate distance
                    distance = np.linalg.norm(each_feature - centroid)

                    # check for special circumstance where each centroid is exactly each feature data. i.e. at the very first iteration
                    if distance == 0:
                        # redefine distance to be a number slightly more than 0
                        distance = 0.00001

                    # ratio distance / radius. To see how far away the feature data is from the centroid in question, compare to the radius calculated above
                    weight_index = int(distance / self.radius) 

                    # if the ratio is more than norm step. Reduce it to radius norm step value i.e. cap the weight index to 100
                    if weight_index > self.radius_norm_step - 1:
                        weight_index = self.radius_norm_step - 1

                    # create a list: duplicate each feature data by its weight amount 
                    ### When distance is large, weight index is large, equivalent value in list weights is small --> weight of data when further away from centroid is less
                    # to_add = (list_weights[weight_index]**2)*[each_feature]
                    # alternatively: append to list 'weight each feature data'
                    list_weight_feature.append(list_weights[weight_index]**2)       # why having to square? 

                    # Add value to list 'within radius'
                    # list_within_radius += to_add
                    # alternatively: append to list 'within radius' like old solution
                    list_within_radius.append(each_feature)
                
                # after checking all data in train data. Check if list weight each feature data is empty
                if not list_weight_feature:
                    # once looping through all feature data, calculate new centroid = average ( vector of every data that is within radius of current centroid)    
                    new_centroid = np.average(list_within_radius, axis=0)   # calculate average without using list weight each feature data
                else:
                    # calculate average without using list weight each feature data                             # solution: introduce radius norm step
                    ### more info about numpy average with weight https://stackoverflow.com/questions/38241174/weighted-average-using-numpy-average
                    new_centroid = np.average(list_within_radius, axis=0, weights=np.array(list_weight_feature))

                # np.average returns a numpy array. So we convert that into a tuple. Then append it to new centroid list 
                list_new_centroids.append(tuple(new_centroid))

            # after looping all elements in dict centroid. Unique element of new centroid list.
            ### we need to find the unique because once we loop through every feature data, we will find new centroid that are exactly identical to each other
            uniques = sorted(list(set(list_new_centroids)))
            # print(uniques)

            # iterate through unique list to find any centroid that too close to each other                     # solution: introduce radius norm step
            to_remove = []

            # loop uniques list
            for i in uniques:
                # again, loop through uniques list
                for ii in uniques:
                    # if 2 items are the same, pass
                    if i == ii:
                        pass

                    # if 2 centroids' distances are smaller than the current radius
                    ### excluding when item i is already in the remove list. And excluding when item ii is already in the remove list --> avoid duplicate in to be remove items
                    elif np.linalg.norm(np.array(i) - np.array(ii)) <= self.radius and i not in to_remove and ii not in to_remove:
                        # append it to remove list
                        to_remove.append(ii)
                        # since already found the item ii that is too close to item i. Get out of loop ii, check the next i
                        break
            
            # loop list to remove                                                                               # solution: introduce radius norm step
            for i in to_remove:
                uniques.remove(i)

            # make a copy version of the current centroid that we have just complete checking
            prev_centroids = dict(dict_centroids)

            # create a new empty centroid dictionary again
            dict_centroids = {}

            # append values in unique list of new centroid found above, into centroid dictionary
            for i in range(len(uniques)):
                dict_centroids[i] = np.array(uniques[i])

            # assume that we have optimised
            bool_optimised = True

            # loop every item in either previous centroid or new centroid dictionary
            for i in dict_centroids:
                # check if each item in previous centroid dictionary is the same as new centroid dictionary.
                ### we can do this checking because we have sorted both dictionary
                if not np.array_equal( dict_centroids[i], prev_centroids[i] ):
                    # we have not optimised
                    ### we only optimised if previous centroid and new centroid dictionary are identical ~ the centroid would not be move 
                    bool_optimised = False
                    
                    # since we have not optimised, no need to check through the rest of previous centroid vs current centroid
                    break
            
            # since we have optimised, get out of infinite while true loop. 
            ### Else continue checking using the new centroid, stored in centroid dictionary
            if bool_optimised: 
                break

        # Outside of the infinite 'while true' loop. Since we have optimised. Store optimised centroid into self
        self.centroids = dict_centroids

        # create new dictionary in self. To store for each found centroid, the feature data the belongs to each centroid
        self.dict_class = {}
        
        # loop through self's centroid
        for i in range(len(self.centroids)):
            # create the same amount of key in class dictionary
            self.dict_class[i] = []
        
        # loop through each feature data
        for each_feature in train_data:
            # calculate the distance between the feature and each centroid. Store them in a list
            distance = [ np.linalg.norm(each_feature - self.centroids[i]) for i in self.centroids ]

            # the shortest distance shall be the distance of the feature set to its closest centroid. 
            ### classification of the feature set = index of where that min number is in the self centroid list ~ distances list
            classification = distance.index(min(distance))

            # append feature data to class dictionary of the centroid key it belongs to 
            self.dict_class[classification].append(each_feature)

    def predict(self, data_to_predict):
        # calculate the distance between the feature and each centroid. Store them in a list
        distance = [ np.linalg.norm(data_to_predict - self.centroids[i]) for i in self.centroids ]

        # the shortest distance shall be the distance of the feature set to its closest centroid. 
        ### classification of the feature set = index of where that min number is in the self centroid list ~ distances list
        classification = distance.index[min(distance)]

        return classification


"""
test custom hierarchical clustering function using small data sample. 

Result: not entirely 100% accurate. However, the guid also has similar inaccurate outcome and problem 
Suspicions: training data sample size is small
"""
def test_custom_clustering_hierarchical():
    # create dummy data as feature. Our custom function is to assign each of these features into labels
    X = np.array([[1,2], [1.5,1.8], [5,8], [8,8], [1,0.6], [9,11], [8,2], [10,2], [9,3]])

    # alternatively, create a random bunch of centers
    count_choosen_centers = random.randrange(2,8)
    print('# of centers to create =', count_choosen_centers)
    # then make a bunch of data around our choosen centers. 
    ### centers count = number of groups. n_features = number of column each data has = number of dimension each data has.
    X, y = make_blobs(n_samples=100, centers=count_choosen_centers, n_features=2)     # make blob() function returns X and y as default

    style.use('ggplot')

    # define colour. b=blue, g=gren, r=red, c=cyan, m=magenta, y=yellow, k=black, w=white
    colors = 10*["g", "r", "c", "b", "k", "w"]

    # # scatter train data
    # plt.scatter(X[:,0], X[:,1], s=150)

    # train data using custom flat clustering function
    clf = clustering_hierarchical_MeanShift()
    clf.fit (X)

    # print result = how many group/cluster our custom class found
    print('# of centers found = ',len(clf.dict_class))

    # loop through each class in self's class dictionary 
    for each_class in clf.dict_class:
        # assign color for each class
        color = colors[each_class]

        # loop through each feature data in current class
        for each_feature in clf.dict_class[each_class]:
            # scatter train data
            plt.scatter(each_feature[0], each_feature[1], marker='x', color=color, s=150, linewidths=5)

    # scatter centroid
    for output_centroids in clf.centroids:
        plt.scatter(clf.centroids[output_centroids][0], clf.centroids[output_centroids][1], marker='*', color='k', s=150)

    plt.show()

