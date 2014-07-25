from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
import math
import Image
import numpy as np
from datetime import datetime

start = datetime.now()

for i in (100, 1000, 2000):
	print(i)
	
	
end = datetime.now()
print(end - start)

S=2
ws = np.ones(S)
ws /= S
print(ws)
