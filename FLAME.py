from formatting import *
from helpers import MAP,formatData,formatX
from time import time,sleep
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import sys, traceback
from collections import defaultdict

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
			pEdges[pId].extend(authors[aId]['Papers'])
		paperVecs[pId] = formatX(papers[pId],authors,papers,journals,conferences)
	except:
		print 'error at',i
		# traceback.print_exc(file=sys.stdout)

print paperVecs['1']
print pEdges['1']

vect = TfidfVectorizer(max_features=10000)
paperVecs = vect.fit_transform(paperVecs)