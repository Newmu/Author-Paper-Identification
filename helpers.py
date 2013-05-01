import numpy as np
from scipy.spatial.distance import cosine

def angularDistance(u,v):
	cosDist = cosine(u.todense(),v.todense())
	angDist = 1-((2*np.arccos(cosDist))/np.pi)
	return angDist

def MAP(confirmed,guesses):
	guesses = list(guesses)
	cSet = set(confirmed)
	scores = []
	for paper in cSet:
		i = guesses.index(paper)
		gSet = set(guesses[:i+1])
		score = len(cSet.intersection(gSet))/float(len(gSet))
		scores.append(score)
	return sum(scores)/len(scores)

# def formatX(paper,author,authors,papers,journals,conferences):
def formatX(paper,authors,papers,journals,conferences):
	data = []
	if paper['ConferenceId'] != '0' and paper['ConferenceId'] != '-1':
		try:
			data.extend(conferences[paper['ConferenceId']].values())
		except:
			pass
	if paper['JournalId'] != '0' and paper['JournalId'] != '-1':
		try:
			data.extend(journals[paper['JournalId']].values())
		except:
			pass
	# for author in paper['Authors']:
	# 	data.extend(authors[author])
	data.append(paper['Title'])
	data.append(paper['Keyword'])
	data.append(paper['Year'])
	x = ' '.join(data)
	return x

def formatData(aIds,corrects,wrongs,authors,papers,journals,conferences):
	X = []
	Y = []
	ids = []
	for i,aId in enumerate(aIds):
		author = authors[aId]
		correct = corrects[i]
		wrong = wrongs[i]
		for paper in correct:
			paper = papers[paper]
			x = formatX(paper,author,authors,papers,journals,conferences)
			X.append(x)
			Y.append(1)
			ids.append(i)
		for paper in wrong:
			paper = papers[paper]
			x = formatX(paper,author,authors,papers,journals,conferences)
			X.append(x)
			Y.append(0)
			ids.append(i)
	return X,Y,ids