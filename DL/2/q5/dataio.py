import numpy as np


def txt_to_array(file):
    output = list()
    with open(file) as f:
        for line in f:
            #output.append([float(x) for x in line.strip().split(',')])

            values = list()
            parsed_line = line.strip().split(',')
            for value in parsed_line:
            	values.append(float(value))
            output.append((values))
    output = np.array(output)
    return output[:, :-1]