import math
from guineapig import *

# supporting routines can go here
def tokenizeDoc(cur_doc):
	return re.findall('\\w+', cur_doc)



#always subclass Planner
class NB(Planner):
	# params is a dictionary of params given on the command line. 
	# e.g. trainFile = params['trainFile']
	params = GPig.getArgvParams()
	trainFile = params['trainFile']
	print trainFile

# always end like this
if __name__ == "__main__":
    NB().main(sys.argv)

# supporting routines can go here
