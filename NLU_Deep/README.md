# Testing the model with NLU
The name of the folder describes the type of implementation.</br>
* model.h5 : The model weights for corresponding deep learning model
* nlu.h5 : weights for nlu model
* DQN1.py : Replace with required model with this.</br>
This describes how to train the model and test the model with differet reward functions

## Running Dialouges
1. How to run an see a sample run with dialouges
```
python test_NLU_DDQN.py -m model.h5 -n nlu.h5
```
### Sample Run

```
tulika@nlp-server:~/ECML$ python test_NLU-DDQN.py -m model.h5 -n nlu.h5
/home1/tulika/anaconda2/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.
in memeory
/home1/tulika/ECML/NLU_model.py:41: UserWarning: Update your `Conv1D` call to the Keras 2 API: `Conv1D(64, 5, padding="same", activation="relu")`
  model.add(Convolution1D(64, 5, border_mode='same', activation='relu'))
Loading the NLU model
2018-04-11 23:14:07.677989: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2018-04-11 23:14:07.707082: E tensorflow/stream_executor/cuda/cuda_driver.cc:406] failed call to cuInit: CUDA_ERROR_NO_DEVICE
2018-04-11 23:14:07.707142: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:158] retrieving CUDA diagnostic information for host: nlp-server
2018-04-11 23:14:07.707157: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:165] hostname: nlp-server
2018-04-11 23:14:07.707228: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:189] libcuda reported version is: 384.111.0
2018-04-11 23:14:07.707264: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:369] driver version file contents: """NVRM version: NVIDIA UNIX x86_64 Kernel Module  384.111  Tue Dec 19 23:51:45 PST 2017
GCC version:  gcc version 5.4.0 20160609 (Ubuntu 5.4.0-6ubuntu1~16.04.6) 
"""
2018-04-11 23:14:07.707290: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:193] kernel reported version is: 384.111.0
2018-04-11 23:14:07.707300: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:300] kernel version seems to match DSO: 384.111.0
('model.h5', 'nlu.h5')
in init
in build_model
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_2 (Dense)              (None, 75)                450       
_________________________________________________________________
activation_1 (Activation)    (None, 75)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 13)                988       
_________________________________________________________________
activation_2 (Activation)    (None, 13)                0         
=================================================================
Total params: 1,438
Trainable params: 1,438
Non-trainable params: 0
_________________________________________________________________
in build_model
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_4 (Dense)              (None, 75)                450       
_________________________________________________________________
activation_3 (Activation)    (None, 75)                0         
_________________________________________________________________
dense_5 (Dense)              (None, 13)                988       
_________________________________________________________________
activation_4 (Activation)    (None, 13)                0         
=================================================================
Total params: 1,438
Trainable params: 1,438
Non-trainable params: 0
_________________________________________________________________
in load
Loading the modeal model.h5 
Loading FIle model.h5
in load
Loading the model model.h5 
State: [0. 0. 0. 0. 0.]
[126 126  48 126  78 126  27  94  44 126  33]
['flights', 'from', 'washington', 'to', 'baltimore', 'on', 'twenty', 'eight', 'december', 'at', 'evening']
['O', 'O', 'B-fromloc.city_name', 'O', 'B-toloc.city_name', 'O', 'B-depart_date.day_number', 'I-depart_date.day_number', 'B-flight_stop', 'O', 'B-depart_time.period_of_day']
[126 126  48 126  78 126  27  94  44 126  33]
1.00000002769
[0.9999716, 0.99998164, 0.99999714, 0.99998355, 0.9999895, 0.9990212, 0.9999006, 0.99992335, 0.4656351, 0.99985075, 0.99743336]
['$NULL$', '$NULL$', '$DCITY$', '$NULL$', '$ACITY$', '$NULL$', '$DATE$', '$DATE$', '$NULL$', '$NULL$', '$TIME$']


Agent >Hello How may I help you?
User  >flights from washington to baltimore on twenty eight december at evening


#####The action 4
State: [[1.    1.    1.    0.997 0.   ]]
[18]
['business']
['B-class_type']
[18]
1.00000037807
[0.99937326]
['$CLASS$']


Agent >Please specify the class of flight?
User  >business


#####The action 12
State: [[1.    1.    1.    0.997 0.999]]


Agent >Here is you itenary
You are travelling from washington to baltimore on twenty eight at evening via business.
Thanks for using the flight attendant
User  >['Thanks']


#####The action 12
```

