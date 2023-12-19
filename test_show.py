import os
from matplotlib import pyplot as plt
import numpy as np

fname = os.path.join("jena_climate_2009_2016.csv")
  
with open(fname) as f:
    data = f.read()
  
lines = data.split("\n")
print(lines[0])

header = lines[0].split(",")
print(header)

lines = lines[1:] 
print(len(lines))

temperature = np.zeros((len(lines),))
raw_data = np.zeros((len(lines), len(header) - 1)) 

for i, line in enumerate(lines):
    values = [float(x) for x in line.split(",")[1:]]
    temperature[i] = values[1]                       
    raw_data[i, :] = values[:]     

# plt.plot(range(len(temperature)), temperature)
plt.plot(range(1440), temperature[:1440])
plt.show()

print(temperature)