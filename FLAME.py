from formatting import *
from helpers import MAP,formatData,formatX,angularDistance
from time import time,sleep
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
import numpy as np
import sys, traceback
from collections import defaultdict
from time import time
import cPickle

start = time()
authors = processAuthors()
journals = processJournals()
conferences = processConferences()
papers = processPapers()
papers,authors = processPaperAuthor(papers,authors)

print 'time to load all data',time()-start

totalPapers = 10000

t0 = time()
pEdges = defaultdict(list)
paperVecs = ['' for i in xrange(len(papers))]
pEdges = [[] for i in xrange(len(papers))]
# paperVecs  = {}
# pEdges = {}
for i,pId in enumerate(papers):
	if i < totalPapers:
		try:
			[]
			# [pEdges[pId].append(aPId) for aId in papers[pId]['Authors'] for aPId in authors[aId]['Papers']]
			# for aId in papers[pId]['Authors']:
			# 	for aPId in authors[aId]['Papers']:
			# 		if aPId != pId:
			# 			pEdges[int(pId)].append(int(aPId))
			[pEdges[int(pId)].append(int(aPId)) for aId in papers[pId]['Authors'] for aPId in authors[aId]['Papers'] if aPId != pId]
			paperVecs[int(pId)] = formatX(papers[pId],authors,papers,journals,conferences)
			# paperVecs[pId] = formatX(papers[pId],authors,papers,journals,conferences)
		except:
			print 'error at',i

print 'time to format into strings and relations',time()-t0

print pEdges[1039313]
print [int(aPId) for aId in papers['1039313']['Authors'] for aPId in authors[aId]['Papers'] if aPId != pId]

# t0 = time()
# keys = sorted([int(key) for key in paperVecs])
# idsToIndex = cPickle.load(open('data/idsToIndex.pkl','rb'))
# idsToIndex = {}
# for i,key in enumerate(keys):
# 	idsToIndex[key] = i

# print 'time to make keys and indexes',time()-t0

# t0 = time()
# paperVecsList = [paperVecs[str(key)] for key in keys]
# pEdgesList = []
# for key in keys:
# 	edges = []
# 	for pId in pEdges[str(key)]:
# 		try:
# 			edges.append(idsToIndex[int(pId)])
# 		except:
# 			pass
# 	pEdgesList.append(edges)
# pEdgesList2 = [pEdges[str(key)] for key in keys]

# print 'time to flip into list index space',time()-t0

# print paperVecsList[0]
# print pEdgesList2[0]
# print pEdgesList[0]

t0 = time()
vect = joblib.load('data/tfidfvect.skl')
print 'time to load TFIDF',time()-t0

t0 = time()
paperVecsList = vect.transform(paperVecsList)
print 'time to transform TFIDF',time()-t0
print paperVecsList.shape

t0 = time()
edgeWeights = {}
pDensities = []
for i,paper in enumerate(paperVecsList):
	print i
	weights = []
	nPaperIndexes = pEdgesList[i]
	for idx in list(set(nPaperIndexes)):
		neighborPaper = paperVecsList[idx]
		key = tuple(sorted([i,idx]))
		if key in edgeWeights:
			weight = edgeWeights[key]
		else:
			nRelations = nPaperIndexes.count(idx)
			similarity = cosine(paper.todense(),neighborPaper.todense())
			weight = similarity*nRelations
			edgeWeights[key] = weight
		weights.append(weight)
	pDensities.append(sum(weights))
pDensities = np.array(pDensities)

print 'time to calc densities',time()-t0

t0 = time()
clusters = 0
isolated = 0
for i,paper in enumerate(paperVecsList):
	# if i % 1000 == 0: print i
	pWeight = pDensities[i]
	nPaperIndexes = pEdgesList[i]
	nWeights = pDensities[list(set(nPaperIndexes))]
	if len(nWeights) > 0:
		# print pWeight
		# print nWeights
		# print nWeights.max()
		if pWeight > nWeights.max():
			clusters += 1
	else:
		isolated += 1

print clusters/float(totalPapers)
print isolated/float(totalPapers)

print 'time to find clusters',time()-t0

print 'time to run',time()-start