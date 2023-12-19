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
    print(inputs, " : ", targets)
    # for i in range(inputs.shape[0]):
    #     print([int(x) for x in inputs[i]], int(targets[i]))

# outputs 1
# tf.Tensor([[0 1 2] [1 2 3]], shape=(2, 3), dtype=int32)  :  tf.Tensor([3 4], shape=(2,), dtype=int32)
# tf.Tensor([[2 3 4] [3 4 5]], shape=(2, 3), dtype=int32)  :  tf.Tensor([5 6], shape=(2,), dtype=int32)
# tf.Tensor([[4 5 6]], shape=(1, 3), dtype=int32)  :  tf.Tensor([7], shape=(1,), dtype=int32)
    
# outputs 2
# [0, 1, 2] 3
# [1, 2, 3] 4
# [2, 3, 4] 5
# [3, 4, 5] 6
# [4, 5, 6] 7