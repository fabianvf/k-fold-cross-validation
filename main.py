import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
import math
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import sys

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
OUTPUT = False

# Driver for program
def main(dataFile):
    # Read in file
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

    # Replaces genres in data set with numbers, for use with scikit-learn algorithms
    for entry in dataSet:
        entry[0] = genres.index(entry[0])
    
    assert n == len(dataSet), "The datafile is corrupt ("+str(n)+" != " + str(len(dataSet)) + ")"

    
    features = extractFeatures(dataSet)
    mySeed = random.randint(0, sys.maxint)
    
    k = 10
    _output("Using feature vector of descriptive measures:")
#    _output("\tUsing SVC:")
#    kfoldCV(LinearSVC(dual=False), features[0], k, seed=mySeed)
    
    _output("\tUsing Naive Bayes:")
    kfoldCV(MultinomialNB(), features[0], k, seed=mySeed)

    _output("\tUsing K-Nearest Neighbors:")
    kfoldCV(KNeighborsClassifier(n_neighbors=4), features[0], k, seed=mySeed)
    
    k = 10 
    _output("\nUsing feature vector of words in title:")
 #   _output("\tUsing SVC:")
 #   kfoldCV(LinearSVC(dual=False), features[1], k, seed=mySeed)
        
    _output("\tUsing Naive Bayes:")
    kfoldCV(MultinomialNB(), features[1], k, seed=mySeed)

    _output("\tUsing K-Nearest Neighbors:")
    kfoldCV(KNeighborsClassifier(metric='jaccard'), features[1], k, seed=mySeed)
    

# Takes a classifier, feature vector and k value, runs k-fold cross-validation on the data set, 
#     returns the list of success scores, mean, variance, confidence interval, and the k-value used
# Feature vector should be of the form {X1, C1} where X is a vector of feature values and C1 is the target value
# Optional parameter seed allows you to force partitioning to be the same across multiple runs
def kfoldCV(classifier, features, k, seed=None):
    partitions = partition(features, k, seed)
    errors = list()
        
    # Run the algorithm k times, record error each time
    for i in range(k):
        trainingSet = list()
        for j in range(k):
            if j != i:
                trainingSet.append(partitions[j])

        # flatten training set
        trainingSet = [item for entry in trainingSet for item in entry]
        testSet = partitions[i]
        
        # Train and classify model
        trainedClassifier = train(classifier, trainingSet)
        errors.append(classify(classifier, testSet))
        
    # Compute statistics
    mean = sum(errors)/k
    variance = sum([(error - mean)**2 for error in errors])/(k)
    standardDeviation = variance**.5
    confidenceInterval = (mean - 1.96*standardDeviation, mean + 1.96*standardDeviation)
 
    _output("\t\tMean = {0:.2f} \n\t\tVariance = {1:.4f} \n\t\tStandard Devation = {2:.3f} \n\t\t95% Confidence interval: [{3:.2f}, {4:.2f}]"\
            .format(mean, variance, standardDeviation, confidenceInterval[0], confidenceInterval[1]))

    return (errors, mean, variance, confidenceInterval, k)

# trains a model
def train(classifier, trainingSet):
    X = [entry[0] for entry in trainingSet]
    y = [entry[1] for entry in trainingSet]
    return classifier.fit(X,y)

# Runs a classifier on the input, outputs the success rate
def classify(classifier, dataSet):
    X = [entry[0] for entry in dataSet]
    y = [entry[1] for entry in dataSet]

    return classifier.score(X,y)

# Divides data set into k partitions
def partition(dataSet, k, seed=None):
    size = math.ceil(len(dataSet)/float(k))
    partitions = [[] for i in range(k)]
    j = 0
    
    for entry in dataSet:
        x = assign(partitions, k, size, seed) 
        partitions[x].append(entry)

    return partitions


# Assigns each entry to a non-full partition
def assign(partitions, k, size, seed=None):
    if seed is not None:
        random.Random(seed)
    x = random.randint(0,k-1)
    while(len(partitions[x]) >= size):
        x = random.randint(0,k-1)
    return x

# Extracts two feature sets, one with descriptive statistics, the other a vector of the words in the title
def extractFeatures(dataSet):
    vector1, vector2 = list(), list()
    
    stemmer = PorterStemmer()
    # Produces list of all unique word stems in the titles in the dataset
    wordBag = list({stemmer.stem(word) for entry in dataSet for word in entry[2].strip().split(" ") if not word in stopwords.words('english')})


    for entry in dataSet:
        genre, isbn, title, authors = entry[0], entry[1].strip(), entry[2].strip(), entry[3].strip()

        wordList, authorList = [word for word in title.split(" ")], [author.strip() for author in authors.split(";")]
        sortedWords = sorted(wordList, key = lambda x: -1*len(x))
        nonStopWords = [word for word in sortedWords if not word in stopwords.words('english')]
        stemmedWords = [stemmer.stem(word) for word in nonStopWords]

        # Quantitative data about the title
        shortestWord = len(nonStopWords[-1])
        longestWord = len(nonStopWords[0])
        meanWord = sum([len(word) for word in nonStopWords])/len(nonStopWords)
        wordSD = (sum([(len(word)-meanWord)**2 for word in nonStopWords])/len(nonStopWords))**.5

        vector1.append([(len(authorList), len(wordList), longestWord, shortestWord, meanWord, wordSD), genre])
        
        # Creates a vector storing whether a word in a dataset occurred in the title
        occurrences = tuple(1 if word in stemmedWords else 0 for word in wordBag)
        
        vector2.append([occurrences, genre])

    return (vector1,vector2)

# Helper print function to allow disabling of normal output for debugging/use in other libraries
def _output(x):
    if OUTPUT:
        print(x)

if __name__ == "__main__":
    OUTPUT = True
    main('DataSet.txt') 
