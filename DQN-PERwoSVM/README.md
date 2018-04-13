# How to replicate for the given model
The name of the folder describes the type of implementation.</br>
* SVM : uses SVM
* woSVM : doesn't use SVM
* PER : Uses Priority Experience Replay
* DDQN : Works on the Double DQN algorithm
* DQN : Works on the DQN algorithm
This describes how to train the model and test the model with differet reward functions

## Training
1. Training with old reward of +-1 
```
python train_modified.py
```

2. Training with the new reward
```
python train_modified1.py
```

The models of each are stored in ./save/ folder along with the time_date and model architecture information


## Testing

The model can be tested to see the actions taken using the  following script

```
python test_policy.py ./save/<model_name>
```


