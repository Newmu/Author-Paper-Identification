from formatting import *
from helpers import MAP,formatData,formatX
from time import time,sleep
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
import numpy as np
import sys, traceback
from collections import defaultdict
from time import time

authors = processAuthors()
journals = processJournals()
conferences = processConferences()
papers = processPapers()
papers,authors = processPaperAuthor(papers,authors)

pEdges = defaultdict(list)
paperVecs = {}
for i,pId in enumerate(papers):
	try:
		for aId in papers[pId]['Authors']:
			for aPId in authors[aId]['Papers']:
				if aPId != pId:
					pEdges[pId].append(aPId)
		paperVecs[pId] = formatX(papers[pId],authors,papers,journals,conferences)
	except:
		print 'error at',i

keys = sorted([int(key) for key in paperVecs])
paperVecsList = [paperVecs[str(key)] for key in keys]
pEdgesList = [pEdges[str(key)] for key in keys]

print paperVecsList[0]
print pEdgesList[0]

t0 = time()
vect = TfidfVectorizer(max_features=50000)
vect.fit(paperVecsList)
print 'time to fit TFIDF',time()-t0

joblib.dump(vect,'data/tfidfvect.skl',compress=3)

paperVecsList = vect.transform(paperVecsList)