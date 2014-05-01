from random import randint
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
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
    # Read in file
    n = 0
    genres = set()
    dataSet = list()
    with open (dataFile, 'r') as f:
        line = f.readline()
        line = line.split('\t')
        n = int(line[0].split('=')[1])
        for line in f:
            line = line.split('\t')
            genres.add(line[0])
            dataSet.append(line) 
    genres = list(genres)
    # Replaces genres in data set with numbers, for easier classification
    for entry in dataSet:
        entry[0] = genres.index(entry[0])
    
    assert n == len(dataSet), "The datafile is corrupt ("+str(n)+" != " + str(len(dataSet)) + ")"
    # Generate feature vectors for training and test sets
    # Each feature vector is a vector of (feature, answer) tuples
    titleLengths = getTitleLengths(dataSet)
    wordLengths = getWordLengths(dataSet)
    
    k = 10
    print("Using SVC with:")
    print("\tlength of title:")
    kfoldCV(LinearSVC(dual=False), titleLengths, k)
    print("\tlength of longest word in title:")
    kfoldCV(LinearSVC(dual=False), wordLengths, k)
    
    print("Using Naive Bayes with:")
    print("\tlength of title:")
    kfoldCV(MultinomialNB(), titleLengths, k)
    print("\tlength of longest word in title:")
    kfoldCV(MultinomialNB(), wordLengths, k)

    print("Using K-Nearest Neighbors with:")
    print("\tlength of title:")
    kfoldCV(KNeighborsClassifier(n_neighbors=4), titleLengths, k)
    print("\tlength of longest word in title:")
    kfoldCV(KNeighborsClassifier(n_neighbors=4), wordLengths, k)


# Takes a classifier and k value, runs k-fold cross-validation on the data set, 
# returns the list of success scores, the mean, the variance and the k-value used
def kfoldCV(classifier, features, k):
    partitions = partition(features, k)
    
    errors = list()
        
    # Run the algorithm k times, record error each time
    for i in range(k):
        trainingSet = list()
        for j in range(k):
            if j != i:
                trainingSet.append(partitions[j])
        trainingSet = [item for entry in trainingSet for item in entry]
        testSet = partitions[i]

        error = learnAndClassify(classifier, trainingSet, testSet)
        
        errors.append(error)
        
    # Compute statistics
    mean = sum(errors)/k
    variance = sum([(error - mean)**2 for error in errors])/(k) 
    print("\t\tMean = " + str(mean) + "\n\t\tVariance = " + str(variance)+"\n") 
    return (errors, mean, variance, k)

# Trains a classifier, then runs it on the test data set
def learnAndClassify(classifier, trainingSet, testSet):

    classifier = train(classifier, trainingSet)    
    error = classify(classifier, testSet)

    return error


# Divides data set into k partitions
def partition(dataSet, k):
    size = len(dataSet)//k
    partitions = [[] for i in range(k)]
    j = 0
    
    for entry in dataSet:
        x = assign(partitions, k, size) #TODO: Not sure this assignation method is good
        partitions[x].append(entry)

    return partitions


# Assigns each entry to a non-full partition
def assign(partitions, k, size):
    x = randint(0,k-1)
    while(len(partitions[x]) >= size):
        x = randint(0,k-1)
    return x


# Gets number of characters in length of title for each entry in the dataset
def getTitleLengths(dataSet):
    titles = list()
    for entry in dataSet:
        titles.append((len(entry[2].strip()), entry[0]))
    return titles

# Gets the length of the longest word in the title for each entry in the dataset
def getWordLengths(dataSet):
    wordLengths = list()
    for entry in dataSet:
        words = entry[2].split(" ")
        words.sort(lambda x,y: -1*cmp(len(x), len(y)))
        wordLengths.append((len(words[0]), entry[0]))

    return wordLengths

# trains a model
def train(classifier, trainingSet):
    X = [[entry[0]] for entry in trainingSet]
    y = [entry[1] for entry in trainingSet]
    
    return classifier.fit(X,y)

# Runs a classifier on the input, outputs the success rate
def classify(classifier, dataSet):
    X = [[entry[0]] for entry in dataSet]
    y = [entry[1] for entry in dataSet]

    return classifier.score(X,y)

main('DataSet.txt')
