#!/bin/bash


for mu in {0.0,0.00001,0.0001,0.001,0.01,0.1,0.2,0.5,1}
do
	for((i=1;i<=5;i++));
	do 
		shuf abstract.tiny.train; 
	done | python q2.py 10000 0.5 $mu 20 2009 abstract.tiny.test >> q2.txt
done