import nn
import helper
import dataio
import numpy as np
import pdb
import matplotlib.pyplot as plt

training_data = dataio.training_from_doc()

validation_data = dataio.validation_from_doc()
learning_rates = [.1,.01,.2,.5]

a = nn.FCN(rand_seed=1)

#pdb.set_trace()
a.fit(training_data, validation_data)

#pdb.set_trace()
vce=np.mean([a.v_ce_array, b.v_ce_array, c.v_ce_array, d.v_ce_array, e.v_ce_array], axis=0)
tce=np.mean([a.t_ce_array, b.t_ce_array, c.t_ce_array, d.t_ce_array, e.t_ce_array], axis=0)
va= np.mean([a.v_ac_array, b.v_ac_array, c.v_ac_array, d.v_ac_array, e.v_ac_array], axis=0)
ta= np.mean([a.t_ac_array, b.t_ac_array, c.t_ac_array, d.t_ac_array, e.t_ac_array], axis=0)

plt.plot(a.epoch_nums,vce, 'r--', tce, 'b--')
plt.savefig('q6a.png')
plt.show()

plt.plot(a.epoch_nums,va, 'r--', ta, 'b--')
plt.savefig('q6b.png')
plt.show()


best_results = a.v_ac_array[-1]
best_network = a
for net in [a,b,c,d,e]:
	if net.v_ac_array[-1] >best_results:
		best_network = net

np.save('best_weights',best_network.weights)

print a.v_ce_array
print vce