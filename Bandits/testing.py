

# Cell
import numpy as np

a = np.array([1, 2, 3])

exps = np.exp(a) 
softmax = exps / exps.sum()

print(softmax)

np.random.choice(range(len(a)) ,size=1, p=softmax)[0]

