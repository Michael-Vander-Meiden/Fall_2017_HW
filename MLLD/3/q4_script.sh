#!/bin/bash


for f in {'/afs/cs.cmu.edu/project/bigML/dbpedia_17fall_hw3/abstract.tiny.train','/afs/cs.cmu.edu/project/bigML/dbpedia_17fall_hw3/abstract.smaller.train','/afs/cs.cmu.edu/project/bigML/dbpedia_17fall_hw3/abstract.small.train','/afs/cs.cmu.edu/project/bigML/dbpedia_17fall_hw3/abstract.medium.train','/afs/cs.cmu.edu/project/bigML/dbpedia_17fall_hw3/abstract.large.train','/afs/cs.cmu.edu/project/bigML/dbpedia_17fall_hw3/abstract.full.train'}
do
	for((i=1;i<=1;i++));
	do 
		shuf $f; 
	done | python q4.py 10000 0.5 .001 20 2009 abstract.tiny.test >> q4.txt
done