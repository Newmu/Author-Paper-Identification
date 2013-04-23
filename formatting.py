import csv
from time import time
from re import split,findall
from cPickle import dump
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from itertools import combinations
from scipy.spatial.distance import pdist,cdist,cosine
import matplotlib.pyplot as plt

def processAuthors():
	f = open('data/Author.csv')
	reader = csv.reader(f,delimiter=',', skipinitialspace=True)
	authors = {}
	for i,data in enumerate(reader):
		if i == 0:
			fields = data
			print 'author fields',fields
		else:
			authors[data[0]] = dict(zip(fields[1:],data[1:]))
	print 'num authors',len(authors)
	return authors

def processJournals():
	f = open('data/Journal.csv')
	reader = csv.reader(f,delimiter=',', skipinitialspace=True)
	journals = {}
	for i,data in enumerate(reader):
		if i == 0:
			fields = data
			print 'journal fields',fields
		else:
			journals[data[0]] = dict(zip(fields[1:],data[1:]))
	print 'num journals',len(journals)
	return journals

def processConferences():
	f = open('data/Conference.csv')
	reader = csv.reader(f,delimiter=',', skipinitialspace=True)
	conferences = {}
	for i,data in enumerate(reader):
		if i == 0:
			fields = data
			print 'conference fields',fields
		else:
			conferences[data[0]] = dict(zip(fields[1:],data[1:]))
	print 'num conferences',len(conferences)
	return conferences

def processPapers():
	f = open('data/Paper.csv')
	reader = csv.reader(f,delimiter=',', skipinitialspace=True)
	papers = {}
	# nYears = 0
	# nKeywords = 0
	# nConferences = 0
	# nJournals = 0
	# nTitles = 0
	years = []
	for i,data in enumerate(reader):
		# if i % 100000 == 0: print i
		if i == 0:
			fields = data
			print 'paper fields',fields
		else:
			# if data[1] != '': nTitles += 1
			# if 1500 < int(data[2]) < 2013: nYears += 1
			# if int(data[3]) != 0: nConferences += 1
			# if int(data[4]) != 0: nJournals += 1
			# if data[5] != '': nKeywords += 1
			data[-1] = data[-1].lower().replace('keywords:','').replace('keywords','').replace('key words','')
			data[1] = data[1].lower()
			# data[-1] = findall(r"[\w']+", data[-1])
			# data[-1] = split('[?.,;:|]', data[-1])
			# data[-1] = filter(None, data[-1])
			papers[data[0]] = dict(zip(fields[1:],data[1:]))
	# print 'prob year',nYears/float(i)
	# print 'prob keywords',nKeywords/float(i)
	# print 'prob conference',nConferences/float(i)
	# print 'prob journal',nJournals/float(i)
	# print 'prob title',nTitles/float(i)
	print 'num papers',len(papers)
	return papers

def processPaperAuthor(papers,authors):
	f = open('data/PaperAuthor.csv')
	reader = csv.reader(f,delimiter=',', skipinitialspace=True)
	for i,data in enumerate(reader):
		if i == 0:
			fields = data
			print 'paperAuthor fields',fields
		else:
			# if i < 2257250:
			pId = data[0]
			aId = data[1]
			if pId in papers:
				paper = papers[pId]
				if 'Authors' in paper:
					paper['Authors'].append(aId)
				else:
					paper['Authors'] = [aId]
			else:
				paper = {}
				paper['Year'] = '0'
				paper['ConferenceId'] = '0'
				paper['JournalId'] = '0'
				paper['Keyword'] = ''
				paper['Title'] = ''
				paper['Authors'] = [aId]
				papers[pId] = paper
			if aId in authors:
				author = authors[aId]
				if 'Papers' in author:
					author['Papers'].append(pId)
				else:
					author['Papers'] = [pId]
			else:
				author = {}
				author['Name'] = data[2]
				author['Affiliation'] = data[3]
				author['Papers'] = [pId]
				authors[aId] = author
	print 'num papers',len(papers),'num authors',len(authors)
	return papers,authors

def getDistances(vecs):
	dists = []
	for u,v in combinations(vecs,2):
		dist = cosine(u.todense(),v.todense())
		dists.append(dist)
	return dists

def getStats(pIds):
	titles = []
	years = []
	for pId in pIds:
		paper = papers[pId]
		title = paper['Title']
		year = int(paper['Year'])
		if title != '':
			titles.append(title)
		if 1500 < year < 2014: 
			years.append(year)
	return titles,year

def MAP(confirmed,guesses):
	cSet = set(confirmed)
	scores = []
	for paper in cSet:
		i = guesses.index(paper)
		gSet = set(guesses[:i+1])
		score = len(cSet.intersection(gSet))/float(len(gSet))
		scores.append(score)
	return sum(scores)/len(scores)

# def processTrain(vect):
def processTrain():
	f = open('data/Train.csv')
	reader = csv.reader(f,delimiter=',', skipinitialspace=True)
	corrects = []
	wrongs = []
	aIds = []
	for i,data in enumerate(reader):
		if i == 0:
			fields = data
			print fields
		else:
			aIds.append(data[0])
			corrects.append(data[1].split(' '))
			wrongs.append(data[2].split(' '))
			# confirmed = [paperId for paperId in data[1].split(' ')]
			# wrong = [paperId for paperId in data[2].split(' ')]
			# cTitles,cYears = getStats(confirmed)
			# wTitles,wYears = getStats(wrong)
			# print np.mean(cYears)
			# print np.mean(wYears)
			# cTitles = vect.transform(cTitles)
			# wTitles = vect.transform(wTitles)
			# print np.mean(getDistances(cTitles))
			# print np.mean(getDistances(wTitles))
	# plt.subplot(1,2,1)
	# plt.hist(correct,bins=100)
	# plt.subplot(1,2,2)
	# plt.hist(wrong,bins=100)
	# plt.show()
	return aIds,corrects,wrongs