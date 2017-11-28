import argparse
import sys
from pyspark import SparkContext, SparkConf
from operator import add
import re

def create_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--iterations', type=int, default=2,
                      help='Number of iterations of label propagation')
  parser.add_argument('--edges_file', default=None,
                      help='Input file of edges')
  parser.add_argument('--seeds_file', default=None,
                      help='File that contains labels for seed nodes')
  parser.add_argument('--eval_file', default=None,
                      help='File that contains labels of nodes to be evaluated')
  parser.add_argument('--number_of_excutors', type=int, default=8,
                      help='Number of iterations of label propagation')
  return parser


def split_properties(junk):
  properties = junk[1][0]
  output = list()
  for prop in properties:
    output.append((junk[0],(prop,dict(),junk[1][1])))
  return output


class LabelPropagation:
    def __init__(self, graph_file, seed_file, eval_file, iterations, number_of_excutors):
        conf = SparkConf().setAppName("LabelPropagation")
        conf = conf.setMaster('local[%d]'% number_of_excutors)\
                 .set('spark.executor.memory', '3G')\
                 .set('spark.driver.memory', '3G')\
                 .set('spark.driver.maxResultSize', '3G')
        self.spark = SparkContext(conf=conf)
        self.graph_file = graph_file
        self.seed_file = seed_file
        self.eval_file = eval_file
        self.n_iterations = iterations
        self.n_partitions = number_of_excutors * 2

    def run(self):
        lines = self.spark.textFile(self.graph_file, self.n_partitions)
        # [TODO]
        udg = lines.flatMap(lambda r: ((r.split("\t")[0],(r.split("\t")[1],r.split("\t")[-1], dict())),(r.split("\t")[1],(r.split("\t")[0],r.split("\t")[-1], dict()))))


        print udg.collect()


        lines = self.spark.textFile(self.seed_file, self.n_partitions)
        
        # [TODO]

        for t in range(self.n_iterations):
            # [TODO]
            pass
    def eval(self):
        lines = self.spark.textFile(self.eval_file, self.n_partitions)

        
        # [TODO]



if __name__ == "__main__":
    args = create_parser().parse_args()
    lp = LabelPropagation(args.edges_file, args.seeds_file, args.eval_file, args.iterations, args.number_of_excutors)
    lp.run()
    lp.eval()
    lp.spark.stop()
