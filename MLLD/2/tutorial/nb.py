import math
from guineapig import *
import sys
import pdb
import re

# supporting routines can go here
def tokenizeDoc(cur_doc):
    return re.findall('\\w+', cur_doc)

def split_line(line):
    splits = re.split(r'\t+', line)
    labels = tokenizeDoc(splits[1])
    words = tokenizeDoc(splits[2])
    return labels, words

def split_test(line):
    splits = re.split(r'\t+', line)
    ids = splits[0]
    words = tokenizeDoc(splits[2])
    return ids, words

def get_lw_pairs(line):
    labels, words = split_line(line)
    for word in words:
        for label in labels:
            yield (label,word)

def get_iw_pairs(line):
    ids, words = split_test(line)
    for word in words:
        yield (ids,word)

def get_labels(line):
    labels,words = split_line(line)
    for label in labels:
        yield(label)

def build_hash(wordcounts, model):
    model[wordcounts[0][0]] = wordcounts[1]
    return model

def build_lnhash(x,lnhash):
    lnhash[x[0]]=lnhash.get(x[0],0)+x[1]
    return lnhash

def build_lxhash(x,lxhash):
    lxhash[x[0]] = x[1]
    return lxhash

def label_priors(mydict):
    total = sum(mydict.values())
    mydict = {k: math.log((float(v)+1.0)/(float(total)+1.0*17)) for k, v in mydict.iteritems()}
    return mydict

def unpack_i_w_w_h(a,b):
    return((a[0],a[1],b[1]))

def word_prob_maker(iden, w, nhash, tnhash, v):
    prob_dict = {}
    for k in tnhash:
        num_k = nhash.get(k,0.0)+1
        den_k = tnhash.get(k,0.0)+1.0+float(v)
        prob_dict[k] = math.log(num_k/den_k)
    return (iden,w,prob_dict)

def sum_probs(id_hash, a):
    iden, w, hashbrown = a
    for k in hashbrown:
        id_hash[k] = id_hash.get(k,0.0)+hashbrown[k]
    return id_hash

def add_priors(a, h1, h2):
    for k in h2:
        h1[k] = h1[k]+h2[k]
    return (a,h1)

def predict(a,b):
    maximum = max(b, key=b.get)
    return (a,maximum,b[maximum])


#always subclass Planner
class NB(Planner):
    # params is a dictionary of params given on the command line. 
    # e.g. trainFile = params['trainFile']
    #first we want to get each word and each label with the word
    params = GPig.getArgvParams()
    trainFile = params['trainFile']
    
    #count words using the wordcount example given
    wordcounts = ReadLines(trainFile) | Flatten(by=get_lw_pairs) | Group(by=lambda x:x, reducingTo=ReduceToCount(), combiningTo=ReduceToCount())
    
    #get unique words and count them
    unique_words = Map(wordcounts, by=lambda ((label, word), n):word) | Distinct()
    vocab_size = Group(unique_words, by=lambda x: 'null', reducingTo=ReduceToCount(), combiningTo=ReduceToCount()) | Map(by=lambda (a,b):b)

    #now we will aggregate these counts into a series of hash tables. Each table has the label as a key and the count as the value
    count_dict = Group(wordcounts, by=lambda ((label, word), n):word, reducingTo=ReduceTo(dict, by=lambda model, x:build_hash(x,model)))
    
    #build priors
    #first result will give us (label, count) for each word
    label_wcounts = Map(wordcounts, by=lambda ((label, word), n):(label,n))
    #This will combines all identical counts into dictionary
    lw_dict = Group(label_wcounts, by=lambda x:'null', reducingTo=ReduceTo(dict, by=lambda lnhash, x:build_lnhash(x,lnhash))) | Map(by=lambda (null, dictionary):dictionary)
    #now we need to get a count of the number of each label
    #TODO: this is wrong for some reason
    label_xcounts = ReadLines(trainFile) | Flatten(by=get_labels)|Group(by=lambda x:x, reducingTo=ReduceToCount(), combiningTo=ReduceToCount())
    #turn these cuonts into a hash table
    lx_dict =  Group(label_xcounts, by=lambda x:'null', reducingTo=ReduceTo(dict, by=lambda lxhash, x:build_lxhash(x,lxhash))) | Map(by=lambda (null, dictionary):dictionary)
    #TODO compute the priors
    lpriors = Map(lx_dict, by=lambda mydict:label_priors(mydict))

    #training is done, time for testing
    testFile = params['testFile']

    # id + word tuples
    test_word_list = ReadLines(testFile) | Flatten(by=get_iw_pairs)


    i_w_w_h = Join(Jin(test_word_list, by=lambda (i,w):w), Jin(count_dict, by=lambda (w, h):w)) \

    #get this view into a nicer format
    id_word_hash = Map(i_w_w_h, by=lambda (a,b):unpack_i_w_w_h(a,b))

    #now we add all the data we need to calculate the prob for a word
    data_mashup0 = Augment(id_word_hash, sideview=lw_dict,loadedBy=lambda v:GPig.onlyRowOf(v)) | Map(by=lambda ((a,b,c),d):(a,b,c,d))

    #add vocab to this data
    data_mashup1 = Augment(data_mashup0, sideview=vocab_size, loadedBy=lambda v:GPig.onlyRowOf(v)) | Map(by=lambda ((a,b,c,d),e):(a,b,c,d,e))
    
    #return ID, word, probhash
    word_probs = Map(data_mashup1, by=lambda (a,b,c,d,e):word_prob_maker(a,b,c,d,e))
    
    #Group by ID, adding all the dicts together
    id_prob_hash = Group(word_probs, by=lambda (a, b, hashbrowns): a,reducingTo=ReduceTo(dict, by=lambda id_hash, a: sum_probs(id_hash,a)))
    #add priors
    id_total_probs = Augment(id_prob_hash, sideview=lpriors, loadedBy=lambda v:GPig.onlyRowOf(v)) \
                    | Map(by=lambda ((a,prob_hash),prior_hash):add_priors(a,prob_hash,prior_hash))

    #make the output!
    output = Map(id_total_probs, by=lambda (a,b):predict(a,b))

# always end like this
if __name__ == "__main__":
    NB().main(sys.argv)

# supporting routines can go here
