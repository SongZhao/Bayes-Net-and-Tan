import numpy as np
import scipy.io.arff as sparff

import sys

def loadData(data_fname):
    data, metadata = sparff.loadarff(data_fname)
    num_features = len(metadata.names())-1
    num_instances = len(data)
    feature, class_res = [], []
    for m in range(num_instances):
        if(data[m][-1] == "malign_lymph"):
        	class_res.append(1)
        else:
        	class_res.append(0);
        featureVector = []
        f1 = []
        for n in range(num_features):
            featureVector.append(metadata[metadata.names()[n]][1].index(data[m][n])) # 0 for no and 1 for yes
        feature.append(featureVector)
    feature_range = []
    for n in range(len(metadata.names())):
        feature_range.append(len(metadata[metadata.names()[n]][1]))
    return np.array(feature), np.array(class_res), metadata, feature_range


def printNB(metadata):
    num_features = len(metadata.names()) - 1
    for n in range(num_features):
        if metadata[metadata.names()[n]][0] == 'nominal':
            print('%s %s' % (metadata.names()[n], metadata.names()[-1]))
    print

def buildNaiveBayesNet(X, Y, numVals):
    # compute the base rate of Y & P(X = X_i|Y) for all i
    P_Y = basicP(Y, 2)
    P_XgY = computeP_XgY(X, Y, numVals)
    return P_Y, P_XgY

def basicP(target, possible_value):
    counts = np.zeros(possible_value, )
    for value in target: 
        counts[value] += 1
    dist = []
    counts += 1;
    for value in counts:
        distribution = (float(value)) / (np.sum(counts))
        dist.append(distribution)
    return dist

def computeP_XgY(X, Y, numVals):
    '''
    compute P(X = X_i | Y = Y_j), for all i and j
    '''
    P_XgY = [[],[]]
    # loop over all values of Y (binary)
    for val in range(0,2):
        #X_sub, Y_sub = getDataSubset(X, Y, np.shape(X)[1], val)
        resize = Y == val
        X_resize = X[resize, :]
        print X_resize
        for n in range(np.shape(X)[1]):
            P_XgY[val].append(basicP(X_resize[:,n], numVals[n]))
    return P_XgY

def getDataSubset(X, Y, idx_feature, val_feature):
    idx_target = Y == val_feature

    X_reduced = X[idx_target,:]
    Y_reduced = Y[idx_target]

    return X_reduced, Y_reduced

def computePredictions_NaiveBayes(X_test, P_Y, P_XgY):
    Y_hat, Y_prob = np.zeros(np.shape(X_test)[0]), np.zeros(np.shape(X_test)[0])
    # loop over rows, make prediction
    for m in range(np.shape(X_test)[0]):
        probs = np.zeros(2)
        for y_val in range(2):
            probs[y_val] = P_Y[y_val]
            for n in range(len(X_test[m,:])):
                temp = P_XgY[y_val][n]
                probs[y_val] *= temp[X_test[m,:][n]]
        # get the idx and the value of the max
        prediction_distribution = np.divide(probs, np.sum(probs))
        predictedClass = np.argmax(prediction_distribution)
        predictedProbability = prediction_distribution[predictedClass]
        Y_hat[m] = np.argmax(prediction_distribution)
        Y_prob[m] = prediction_distribution[predictedClass]
    return Y_hat, Y_prob

def printTestResults(Y_hat, Y_prob, Y_test, metadata):
    y_range = metadata[metadata.names()[-1]][1]
    for m in range(len(Y_test)):
        prediction = y_range[int(Y_hat[m])]
        truth = y_range[Y_test[m]]
        print('%s %s %.12f' % (prediction.strip('"\''), truth.strip('"\''), Y_prob[m]))
    hits = np.sum(np.around(Y_hat) == Y_test)
    print ('\n%d' % hits)

args = [arg for arg in sys.argv]
feature, class_res, metadata, feature_range = loadData(args[1])
test_f, test_c, test_m, test_r = loadData(args[2])
printNB(metadata)

py, p_xny = buildNaiveBayesNet(feature, class_res, feature_range)
y_h, y_p = computePredictions_NaiveBayes(test_f, py, p_xny)
printTestResults(y_h, y_p, test_c, metadata)

