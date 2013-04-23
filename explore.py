from formatting import *
from helpers import MAP,formatData
from time import time,sleep
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC,SVC
import numpy as np

trainNum = 250
testNum = 1000
authors = processAuthors()
journals = processJournals()
conferences = processConferences()
papers = processPapers()
papers,authors = processPaperAuthor(papers,authors)
aIds,corrects,wrongs = processTrain()

trainIds = aIds[:trainNum]
trainCorrects = corrects[:trainNum]
trainWrongs = wrongs[:trainNum]

testIds = aIds[-testNum:]
testCorrects = corrects[-testNum:]
testWrongs = wrongs[-testNum:]

trainX,trainY,trainIds = formatData(trainIds,trainCorrects,trainWrongs,authors,papers,journals,conferences)
testX,testY,testIds = formatData(testIds,testCorrects,testWrongs,authors,papers,journals,conferences)
print len(testX),len(testIds)
testIds = np.array(testIds)

vect = TfidfVectorizer(max_features=2000)
trainX = vect.fit_transform(trainX)
testX = vect.transform(testX)
print trainX.shape

# clf = RandomForestClassifier(verbose=2)
t0 = time()
clf = SVC(probability=True)
clf.fit(trainX,trainY)
print 'time to train',time()-t0

t0 = time()
predictions = clf.predict_proba(testX)
print 'time to predict',time()-t0
# clf.fit(trainX.toarray(),trainY)
# predictions = clf.predict_proba(testX.toarray())

percisions = []
for i in xrange(np.max(testIds)):
	predictionProbs = predictions[:,0][(testIds == i)]
	# print 'prediction probs'
	# print predictionProbs
	sortIdxs = np.argsort(predictionProbs)
	# print 'sorted'
	# print sortIdxs
	possible = []
	possible.extend(testCorrects[i])
	possible.extend(testWrongs[i])
	possible = np.array(possible)
	# print 'possiblePapers'
	# print possible
	guesses = possible[sortIdxs]
	# print 'guessed order'
	# print guesses
	percision = MAP(testCorrects[i],guesses)
	# print percision
	percisions.append(percision)
print 'MAP',np.mean(percisions)