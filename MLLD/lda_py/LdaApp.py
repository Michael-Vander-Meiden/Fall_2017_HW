from org.petuum.jbosen import PsApplication, PsTableGroup
from org.petuum.jbosen.table import IntTable
from org.apache.commons.math3.special import *
from DataLoader import DataLoader
import random
import time
import os
import sys
import random


TOPIC_TABLE = 0
WORD_TOPIC_TABLE = 1

class LdaApp(PsApplication):
	def __init__(self, dataFile, outputDir, numWords, numTopics,
				alpha, beta, numIterations, numClocksPerIteration, staleness):
		self.outputDir = outputDir
		self.numWords = numWords
		self.numTopics = numTopics
		self.alpha = alpha
		self.beta = beta
		self.numIterations = numIterations
		self.numClocksPerIteration = numClocksPerIteration
		self.staleness = staleness
		self.dataLoader = DataLoader(dataFile)
		
	def logDirichlet_vector(self, alpha):
		sumLogGamma = 0.0
		logSumGamma = 0.0
		for value in alpha:
			sumLogGamma += Gamma.logGamma(value)
			logSumGamma += value
		
		return sumLogGamma - Gamma.logGamma(logSumGamma)

	def logDirichlet_const(self, alpha, k):
		return k * Gamma.logGamma(alpha) - Gamma.logGamma(k*alpha)
	
	def getColumn(self, matrix, columnId):
		# TO DO: 
		# Get the column of wordTopicTable according to columnId
		column_as_list = [matrix.get(rowId, columnId) for rowId in range(self.numWords)]
		# ... fill me out ...
		return column_as_list
		
		
	def getRow(self, matrix, rowId):
		# TO DO: 
		# Get the row of docTopicTable according to rowId
		row_as_list = matrix[rowId][:]
		# ... fill me out ...
		#
		return row_as_list

	def getLogLikelihood(self, wordTopicTable, docTopicTable):
		lik = 0.0
		for k in range(self.numTopics):
			temp = self.getColumn(wordTopicTable, k)
			for w in range(self.numWords):
				 temp[w] += self.beta
		  
			lik += self.logDirichlet_vector(temp)
			lik -= self.logDirichlet_const(self.beta, self.numWords)
	  
		for d in range(len(docTopicTable)):
			temp = self.getRow(docTopicTable, d)
			for k in range(self.numTopics):
				temp[k] += self.alpha
			
			lik += self.logDirichlet_vector(temp)
			lik -= self.logDirichlet_const(self.alpha, self.numTopics)
		return lik
  
	# TO DO: 
	# Sample function
	def sample(self,k):
		return random.random_int(0,k-1)
	# ... fill me out ...
	#

		
	def initialize(self):
		# Create global topic count table. self table only has one row, which
		# contains counts for all topics.
		PsTableGroup.createDenseIntTable(TOPIC_TABLE, self.staleness, self.numTopics)
		# Create global word-topic table. self table contains numWords rows, each
		# of which has numTopics columns.
		PsTableGroup.createDenseIntTable(WORD_TOPIC_TABLE, self.staleness, self.numTopics)
  
	def runWorkerThread(self, threadId):
		clientId = PsTableGroup.getClientId()

		# Load data for this thread
		print("Client %d thread %d loading data..." % (clientId, threadId))
		part = PsTableGroup.getNumLocalWorkerThreads() * clientId + threadId
		numParts = PsTableGroup.getNumTotalWorkerThreads()
		w = self.dataLoader.load(part, numParts)

		# Get global tables
		topicTable = PsTableGroup.getIntTable(TOPIC_TABLE)
		wordTopicTable = PsTableGroup.getIntTable(WORD_TOPIC_TABLE)

		print self.numWords
		print self.numTopics
		# Initialize LDA variables

		print("Client %d thread %d initializeing variables..." % (clientId, threadId))
		docTopicTable = [[0] * self.numTopics for _ in range(len(w))]

		# TO DO: 
		# Initialize Sampling
		self.z = [[-1 for j in xrange(len(w[d]))] for d in xrange(len(w))]
		
		for d in range(len(w)):
			for i in range(len(w[d])):
				word = w[d][i]
				k = random.randint(0,self.numTopics-1)
				self.z[d][i] = k
				#am i incrementing the right column and row in doctopic?
				docTopicTable[d][k]+=1
				wordTopicTable.inc(word,k,1)
				topicTable.inc(0,k,1)
		# ... fill me out ...
		#


		# Do LDA Gibbs sampling
		print("Client %d thread %d starting gibbs sampling..." % (clientId, threadId))
		llh = [0.0] * self.numIterations
		sec = [0.0] * self.numIterations
		totalSec = 0.0
		for	iterId in range(self.numIterations):
			startTime = time.time()
			# Each iteration consists of a number of batches, and we clock
			# between each to communicate parameters according to SSP
			for batch in range(self.numClocksPerIteration):
				begin = len(w) * batch / self.numClocksPerIteration
				end = len(w) * (batch + 1) / self.numClocksPerIteration
				# TO DO:
				# Loop through each document in the current batch
				#
				# ... fill me out ...
				#

			# Calculate likelihood and elapsed time
			totalSec += (time.time() - startTime)
			sec[iterId] = totalSec
			llh[iterId] = self.getLogLikelihood(wordTopicTable, docTopicTable)
			print("Client %d thread %d completed iteration %d" % (clientId, threadId, iterId+1))
			print("    Elapsed seconds: %f" % (sec[iterId]))
			print("    Log-likelihood: %.15e" % (llh[iterId]))

		PsTableGroup.globalBarrier()

		# Output likelihood
		print("Client %d thread %d writing likelihood to file..." % (clientId, threadId))

		try:
			with open(os.path.join(self.outputDir, "likelihood_%d-%d.csv" % (clientId, threadId)), 'w') as writer:
				for i in range(self.numIterations):
					writer.write("%d,%f,%.15e\n" % (i+1, sec[i], llh[i]))
		except Exception as detail:
			print(detail)
			sys.exit(1)

		PsTableGroup.globalBarrier()

		# Output tables
		if clientId == 0 and threadId == 0:
			print("Client %d thread %d writing word-topic table to file..." % (clientId, threadId))

			try:
				with open(os.path.join(self.outputDir, "word-topic.csv"), 'w') as writer:
					for i in range(self.numWords):
						counter = map(lambda k : str(wordTopicTable.get(i, k)), range(self.numTopics))
						writer.write(','.join(counter) + '\n')
			except Exception as detail:
				print(detail)
				sys.exit(1)

		PsTableGroup.globalBarrier()

		print("Client %d thread %d exited." % (clientId, threadId))






	
if __name__ == "__main__":
	lda = LdaApp()
	config = PsConfig()
	lda.run(config)
	