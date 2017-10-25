#!/bin/bash


for D in {10,100,1000,10000,100000}
do
	for((i=1;i<=20;i++));
	do 
		shuf abstract.small.train; 
	done | python q2.py $D 0.5 .001 20 2009 abstract.small.test >> q3.txt
done