from random import randint
#------------------------------------------------------------------------
# main.py
# 
# Created by: Fabian von Feilitzsch
#
# Project for CS 4710, Artificial Intelligence
# 
# Takes as input a test set and a training set. Extracts two features
# from each case in each data set. The first feature is the number of
# letters in the title of the book. The second is the number of letters
# in the longest word in the title. Two classifiers are generated for
# each feature set, one using Naive Bayes and the other k Nearest 
# Neighbors. It then classifies the test set, printing the success rate 
# for each classifier on each feature vector. 
# 
# Then, four K-fold cross-validation tasks are performed, one for each 
# classifier on each feature vector, printing the success rate, mean, 
# variance and k-value.
#
#------------------------------------------------------------------------

# Driver for program
def main(dataFile):
    n = 0
    dataSet = list()
    with open (dataFile, 'r') as f:
        line = f.readline()
        line = line.split('\t')
        n = int(line[0].split('=')[1])
        for line in f:
            line = line.split('\t')
            dataSet.append(line) 
    assert n == len(dataSet), "The datafile is corrupt ("+str(n)+" != " + str(len(dataSet)) + ")"
    # Generate feature vectors for training and test sets
    # Each feature vector is a vector of (feature, answer) tuples
    titleLengths = getTitleLengths(dataSet)
    wordLengths = getWordLengths(dataSet)
 
    k = 10
    kfoldCV('bayes', titleLengths, k)
    kfoldCV('bayes', wordLengths, k)


# Runs k-fold cross-validation on the data set, returning the list of success scores,
# the mean, the variance and the k-value used
def kfoldCV(algorithm, features, k):
    partitions = partition(features, k)
    errors = list()
        
    # Run the algorithm k times, record error each time
    for i in range(k):
        trainingSet = partitions[:k-1].extend(partitions[k+1:])
        testSet = partitions[k]
        
        error = learnAndClassify(algorithm, trainingSet, testSet)
        
        errors.append(error)
        
    # Compute statistics
    mean = sum(errors)/k
    variance = sum([(error - mean)**2 for error in errors])/(k-1) 

    return (errors, mean, variance, k)

# Trains a classifier, then runs it on the test data set
def learnAndClassify(algorithm, trainingSet, testSet):
    if algorithm == 'bayes':
        classifier = trainNaiveBayes(trainingSet)
    elif algorithm == 'neighbors':
        classifier = trainKNeighbors(trainingSet)

    error = classify(classifier, testSet)
    return error
 
# Divides data set into k partitions
def partition(dataSet, k):
    size = len(dataSet)//k
    print(size)
    partitions = [[] for i in range(k)]
    j = 0
    
    for entry in dataSet:
        x = assign(partitions, k) #TODO: Not sure this assignation method is good
        partitions[x].append(entry)

    for partition in partitions:
        print len(partition)
    return partitions

# Assigns each entry to a non-full partition
def assign(partitions, k):
    x = randint(0,k-1)
    if len(partitions[x]) < 27:
        return x
    else:
        return assign(partitions, k)


# Gets number of characters in length of title for each entry in the dataset
def getTitleLengths(dataSet):
    titles = list()
    for entry in dataSet:
        titles.append((len(entry[2].strip()), entry[0].strip()))
    return titles

# Gets the length of the longest word in the title for each entry in the dataset
def getWordLengths(dataSet):
    wordLengths = list()
    for entry in dataSet:
        words = entry[2].split(" ")
        words.sort(lambda x,y: -1*cmp(len(x), len(y)))
        wordLengths.append(words)
    return wordLengths

# Takes a training set, returns a linear support vector classifier
def trainLinearSVC(trainingSet):
    return False

# Takes a training set, returns a naive bayes classifier
def trainNaiveBayes(trainingSet):
    return False #TODO

# Takes a training set, returns a k nearest neighbors classifier
def trainKNeighbors(trainingSet):
    return False #TODO

# Runs a classifier on the input, outputs the success rate
def classify(classifier, dataSet):
    error = 0
    return error #TODO

   

main('DataSet.txt')
