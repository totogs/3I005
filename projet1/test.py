import numpy as np
listeX = [12, 54, 85, 5, 6, 4, 8, 2, 0, 16, 5 ,5]
listeY = [np.random.normal(0, 0.5, size=2) for x in listeX]

for l in listeY:
    print l
