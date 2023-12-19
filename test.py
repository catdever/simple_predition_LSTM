import numpy as np 
from tensorflow import keras
int_sequence = np.arange(10)                                
dummy_dataset = keras.utils.timeseries_dataset_from_array(
    data=int_sequence[:-3],                                 
    targets=int_sequence[3:],                               
    sequence_length=3,                                      
    batch_size=2,                                           
)
 
for inputs, targets in dummy_dataset:
    for i in range(inputs.shape[0]):
        print([int(x) for x in inputs[i]], int(targets[i]))