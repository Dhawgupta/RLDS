# Reinforcement Learning Dialouge System
The name and folder are as : 
1. DDQN - Double DQN
2. PER - Priority Experience Replay
3. SVM - with SVM 
4. woSVM - without SVM

So ,
```DDQN-PERwoSVM``` : is Double DQN Priority Experience Replay without SVM     
```NLU_Deep``` : contains the integration of real NLU with the Deep RL Model
```NLU_Table``` : contains the integration of the NLU with the traditional Q_table method

To run individual model its **README** is present in its respective folder.
The folder SVM contains the code to run the SVM model with its dummy .This  model gives recommendation to the DRL method



| Algorithm | Average Episodic Reward | Average Dialouge Length | Training Time (in hrs) |
| ------ | ------ | ------ | ------ | 
| DQN with SVM | -313.25 ± 308.63 | 367.52 ± 315.03 | 40.71 |
| DDQN with SVM  | -273.52 ± 271.97 | 330.52 ± 278.67 | 54.2 |
| DQN-PER with SVM | -131.80 ± 181.13 | 183.2 ± 182.07 | 18.47 |
| DQN-PER | -569.85 ± 469.48 | 589.03 ± 479.09 | 16.83 |
| DDQN-PER with SVM | 57.20 ± 7.99 | 7.67 ± 0.53 | 20.82 |
| DDQN-PER | 50.07 ± 8.11 | 8.09 ± 1.06 | 17.74 |