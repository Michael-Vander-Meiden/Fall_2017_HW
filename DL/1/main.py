import nn
import helper
import dataio

training_data = dataio.data_from_doc()
validation_data = dataio.data_from_doc(docname = 'digitsvalid.txt')
a = nn.FCN()
a.fit(training_data, validation_data)