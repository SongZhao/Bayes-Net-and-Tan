import sys
import scipy.io.arff as sia
import numpy as np
import random
import math
import pylab
import re



def sortSet(data, Ys):
    subtrain = []
    for i in Ys:
        subtrain.append(data[data[:,-1]==i,:])
    return subtrain

def getPxy(Xs, nY, Ys, train):
    subtrain = sortSet(train, Ys)
    Pxy = []
    for k  in range(len(Xs)):
        sub1 = []
        for j in Xs[k]:
            n = 0;
            sub2 = []
            for i in range(nY):
                sub2.append((np.sum(subtrain[i][:,k]==j)+1)/float(subtrain[i].shape[0]+len(Xs[k])))
                n = n+1
            sub1.append(sub2)
        Pxy.append(sub1)
    return Pxy

def getPy(nY,Ys,train):
    (N,D) = train.shape
    subtrain = sortSet(train, Ys)
    Py = []
    for i in range(nY):
        Py.append(float(subtrain[i].shape[0]+1)/(N+nY))
    return Py
   
def trainNB(classes, train):
    Xs = classes[:-1]
    Ys = classes[-1]   
    nY = len(Ys)
    subtrain = sortSet(train, Ys)
    Py = getPy(nY, Ys, train) 
    Pxy = getPxy(Xs, nY, Ys, train)

    return Pxy, Py

def calculateP(res):
    res1 = np.exp(res)
    res1 = np.divide(res1, np.sum(res1,axis = 1).reshape(-1,1))
    return res1
    
def testNB(Pxy, Py, test, classes, feats):
    '''
    estimate postieria probability P(Yi|X)
    '''
    Y_htis = []
    Y_p = []
    logres = np.zeros((test.shape[0],len(classes[-1])))
    for i in range(test.shape[0]):
        for j in range(len(classes[-1])):
            sample = test[i,:]
            temp = 0
            n = 0
            for feat_id in range(test.shape[1]-1):
                class_id = classes[feat_id].index(sample[feat_id])
                temp += math.log(Pxy[feat_id][class_id][j])
            logres[i,j] = temp+ math.log(Py[j])
    res = calculateP(logres)
    nCorrect = 0
    for i in range(len(test)):
        truth = np.argmax(res[i])
        if classes[-1][truth] == test[i][-1]:
            nCorrect += 1
        x = float(res[i,truth])
        print classes[-1][truth]," ", test[i][-1]," ", res[i,truth]
    print "\n%d" % nCorrect 
    return res


def interateAll(nVertex, Xs, nY, subtrain, Pxy, Py):
    W=np.random.rand(nVertex, nVertex)
    for i in range(nVertex):
        W[i][i] = -1.0
        for j in range(0,i):
            cpxxy = calcX_XY(Xs, nY, subtrain, i, j)
            pxxy = calcXXY(Xs, nY, subtrain, i, j)
            temp = 0
            for ii in range(len(Xs[i])):
                for jj in range(len(Xs[j])):
                    temp1 = 0
                    for k in range(nY):
                        x = cpxxy[k][jj][ii]/(Pxy[i][ii][k]*Pxy[j][jj][k])
                        x = math.log(x)
                        y = math.log(2)
                        tmp1 = x/y
                        tmp1 *= pxxy[k][jj][ii]
                        temp += tmp1
            W[i][j] = W[j][i] = temp
    return W

def calcX_XY(Xs, nY, subtrain, i, j):
    cpxxy = []
    for k  in range(nY):
        sub1 = []
        for d in Xs[j]:
            n = 0;
            sub2 = []
            for c in Xs[i]:
                sub2.append((1.0+np.sum(np.logical_and(subtrain[k][:,i]==c, subtrain[k][:,j]==d)))/ \
            (subtrain[k].shape[0] + len(Xs[i])*len(Xs[j])))
                n = n+1
            sub1.append(sub2)
        cpxxy.append(sub1)
    return cpxxy

def calcXXY(Xs, nY, subtrain, i, j):
    cpxxy = []
    for k  in range(nY):
        sub1 = []
        for d in Xs[j]:
            n = 0;
            sub2 = []
            for c in Xs[i]:
                sub2.append((1.0+np.sum(np.logical_and(subtrain[k][:,i]==c, subtrain[k][:,j]==d)))/ \
            (train.shape[0] + len(Xs[i])*len(Xs[j])*nY))
                n = n+1
            sub1.append(sub2)
        cpxxy.append(sub1)
    return cpxxy

def compWeight(classes, train):
    Xs = classes[:-1]
    Ys = classes[-1] 
    nY = len(Ys)
    
    subtrain = sortSet(train, Ys)
    nVertex = len(classes)-1
    Pxy = getPxy(Xs, nY, Ys, train)
    Py = getPy(nY, Ys, train)
    w = interateAll(nVertex, Xs, nY, subtrain, Pxy, Py)
    return w

def findEdge(i, j, best_edge,W,max_w):
    t = W[i][j] - max_w
    if t > 0:
        t = 1
    if t == 0:
        best_edge, W, max_w = zero(i, j, best_edge, W, max_w)
    if t == 1:
        best_edge, W, max_w = one(i, j, best_edge, W, max_w)
    return best_edge, W, max_w

def zero(i, j, best_edge, W, max_w):
    if i<best_edge[0] or (i==best_edge[0] and j<best_edge[1]):
        max_w = W[i][j]
        best_edge = [i,j]
    return best_edge, W, max_w

def one(i, j, best_edge, W, max_w):
    max_w = W[i][j]
    best_edge = [i,j]
    return best_edge, W, max_w
          
def findMST(W):
    N = W.shape[0]
    y = [N]
    Vnew = {0:y}
    Vleft = range(1,N)
    while len(Vnew)<N:
        max_w = -10
        best_edge = []
        for i in Vnew:
            for j in Vleft:
                best_edge, W, max_w = findEdge(i, j, best_edge, W, max_w)
        Vnew[best_edge[1]] = [best_edge[0]]+y
        Vleft.remove(best_edge[1])
    for i in range(len(feats)-1):
        parent = []
        for j in Vnew[i]:
            parent.append(feats[j])
        if len(parent) > 1:
        	print ("%s %s %s" % (feats[i], parent[0],parent[1]))
        else:
            print ("%s %s" % (feats[i], parent[0]))
    print     
    return Vnew

def reduceRange(train, classes, x, i):
    subset = train[:,x]==classes[x][i]
    sub = train[subset,:]
    return sub  

 
def compProb(classes, train, x, CV):
    count = dict() 
    nCV = len(CV)
    if train.shape[0] == 0:
        if nCV > 0:  
            cv = CV[0]   
            for i in range(len(classes[cv])):      
                count[i] = compProb(classes, train, x, CV[1:]) 
            res = [cv, train.shape[0], count]
        else:
            for i in range(len(classes[x])):
                count[i] = float(1)/len(classes[x])
            res = [x, train.shape[0], count]        
    else:
        if nCV > 0:  
            cv = CV[0]   
            for i in range(len(classes[cv])): 
                subTrain = reduceRange(train, classes, cv, i)
                count[i] = compProb(classes, subTrain, x, CV[1:]) 
            res = [cv, train.shape[0], count]
        else:
            for i in range(len(classes[x])):
                subTrain = reduceRange(train,classes, x, i)      
                count[i] = float(1+subTrain.shape[0])/(train.shape[0]+len(classes[x]))
            res = [x, train.shape[0], count]
    return res

def trainTAN(classes, train, Vnew):
    CPT = dict()
    for i in Vnew:
        CPT[i] = compProb(classes, train, i, Vnew[i])
    return CPT

def lookupCPT(CPT, i, sample):
    x = CPT[i][0]
    temp = CPT[i]
    while 1:
        class_i = classes[x].index(sample[x])       
        temp = temp[-1][class_i]
        if x == i:
            return temp
        x = temp[0]
            
    
def testTAN(CPT, test, classes, display=True):
    [Nt, D] = test.shape
    nY = len(classes[-1])
    logres = np.ones((Nt,nY))
    
    for i in range(Nt):          
        for j in range(nY):
            sample_cp = list(test[i])   
            sample_cp[-1] = classes[-1][j]
            temp = 0
            for k in range(D):
                temp += math.log(lookupCPT(CPT, k, sample_cp))
            logres[i,j] = temp
            
    res = np.exp(logres)
    res = res/np.sum(res,1).reshape(-1,1)

    nCorrect = 0
    for i in range(len(test)):
        truth = np.argmax(res[i])
        if classes[-1][truth] == test[i][-1]:
            nCorrect += 1
        x = float(res[i,truth])
        print ("%s %s %.12f" % (classes[-1][truth], test[i][-1], x))
    print
    print nCorrect

    return res

        
def drawLearningCurves(classes, train, test, m, name, Nlist=[25,50,100], T=4):
    
    #random.seed(1216)
    
    if train.shape[0] < max(Nlist):
        print "Error: the number of samples cannot be larger than the number of training data"
        return
    
    acc = []  
    if m == 'n':
        title = "Naive Bayes"
        for n in Nlist:
            temp = 0
            for t in xrange(T):
                subtrain = train[ random.sample(range(train.shape[0]), n) ]
                
                ## estimate the prior probabilities P(Y) and conditional probabilities P(Xi|Y)
                Pxy, Py = trainNB(classes, subtrain)

                Pyx, nCorrect = testNB(Pxy, Py, test, classes, feats)
             
                print "[%d]nCorrect = %d" %(n,nCorrect)
                temp += nCorrect
            
            temp = float(temp)/(T*test.shape[0])
            acc.append(temp)
         
            
    elif m == 't': 
        title = "TAN"      
        for n in Nlist:
            temp = 0
            for t in xrange(T):
                subtrain = train[ random.sample(range(train.shape[0]), n) ]
                W = compWeight(classes, subtrain)
                Vnew = findMST(W)
                Vnew[len(Vnew)] = [] 
                CPT = trainTAN(classes, subtrain, Vnew)
                (Pyx, nCorrect) = testTAN(CPT, test, classes, False)
                
                print "[%d]nCorrect = %d" %(n,nCorrect)
                temp += nCorrect
                
            temp = float(temp)/(T*test.shape[0])
            acc.append(temp)
    else:
        print "Error: the input m should be 'n' or 't'."
        return
         
    # draw the learning curve                
    pylab.figure(1)
    pylab.plot(Nlist, acc, 'rx')
    pylab.plot(Nlist, acc, 'r', label = name) 
    pylab.title("Learning Curve of "+title +"\n n=[25,50,100]")
    pylab.xlabel("# of training samples")
    pylab.ylabel("average test-set accuracy")
    #pylab.legend(["minimum", "average", "maximum"], loc = "lower right")
    pylab.legend(loc = 'lower right')
    pylab.savefig('_'.join(["lc", name, m])+".jpg")
    pylab.show() 
    
    
    
args = [arg for arg in sys.argv]



trainFile = args[1]
testFile = args[2] 

name = re.sub("[^A-Za-z']+", ' ', trainFile)
name = name[:name.find(' ')]

m = args[3]   # 'n' stands for 'naive bayes', 't' stands for 'TAN'

## load training and test data
trainData = sia.loadarff(trainFile)
testData = sia.loadarff(testFile)  

## reshape the datasets
train = np.array([[i for i in trainData[0][j]] for j in range(trainData[0].shape[0])])
test = np.array([[i for i in testData[0][j]] for j in range(testData[0].shape[0])])

## get the feature names and the class names
feats = trainData[1].names()

temp = []
for feat in trainData[1].names():
    temp.append(trainData[1][feat])
classes = []
for line in temp:
    classes.append(line[-1])

#classes = [ line[-1] for line in temp]
labels = classes[-1]


if m == 'n':
	'''
	Naive Bayes
	'''
	## estimate the prior probabilities P(Y) and conditional probabilities P(Xi|Y)
	Pxy, Py = trainNB(classes, train)

	## output the structure of Naive Bayes Net
	for i in xrange(len(feats)-1):
		print feats[i], ' ', feats[-1]
	print
		
	## test with Bayes Rule
	Pyx = testNB(Pxy, Py, test, classes, feats)

	## draw learning curves
	#drawLearningCurves(classes, train, test, m, name, Nlist=[25,50,100], T=4)


elif m == 't': 
	W = compWeight(classes, train)
	Vnew= findMST(W)
	Vnew[len(Vnew)] = [] 
	CPT = trainTAN(classes, train, Vnew)


	## test with Bayes Rule
	Pyx = testTAN(CPT, test, classes)

## draw learning curves
#drawLearningCurves(classes, train, test, m, name, Nlist=[25,50,100], T=4)

else:
    errMsg()


