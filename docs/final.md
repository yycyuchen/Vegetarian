---
layout: default
title: Final Report
---

### Video Summary

### Project Summary

### Approach
#### Environment/ Minecraft Map
length: 20  
width: 50  
stained glass wall: 3  
<br />  

#### Reward System
Carrot: +5  
Cooked_mutton: -1  
Mutton: -2  
<br />

#### Actions of agent
1. Action 0: Move forward for 1 block.
2. Action 1: Turn 1 which is 90 degrees to the right
3. Action 2: Turn -1 which is 90 degrees to the left
4. Action 3: Jump
<br />

#### Machine Learning Algorithms
##### ***1. Convert data***
<div style="text-align:left;">
<img src="./image/final_tr.png" height="70%" width="90%" />
</div>
We changed Jackson's observation space. Rather than Jackson simply observing the grass that has been generated and does not move, we changed the observation equation. Jackson can directly observe carrots and muttons during the learning process. We solved the problem of converting the observation API into a grid location around the agent Jackson. The above image shows the conversion formula we used. Get_observation function uses the observation API to get items around the agent(5 * 5) and returns the values of x, y, z. Since it is the same plane, the y value is the same. We successfully converted each x, z coordinate into a corresponding index. In our formular, the upper case "X" and "Z" represent item location and lower case 'x' and 'z'represent agent location. Here is our code below. 

```math
        index = self.obs_size * self.obs_size // 2 + (int)(item['x'] - agent['x']) + (int)(item['z'] - agent['z']) * self.obs_size
```

For example, our agent Jackson is now at location of our map (12, 31), and item is at location of our map (13, 32), his observation 5 * 5 = 25. Firstly, we consider Jackson at location (0,0) in our observation map(show left above). We need to get half of the floor of array.size, that is, the integer obtained by dividing the square of our observation size by 2. We can get Index = 5 * 5 //2 + (13 - 12) + (32 - 31) * 5 = 18. Finally, it will be saved as an array.
<br />

##### ***2. Set Carrot Path***  
The picture below is the "carrot path" we set randomly. We randomly generate carrots in the forward grid, left or right grids after setting the first carrot. In order to make Jackson rotate less frequently, I generate 2-4 carrots continuously at the position where the next carrot is randomly generated. Since our map is long and narrow, we will make the carrots in the forward grid more than left or right grids when randomly generated. As usual, we will do a border check to prevent randomly generated carrots from being outside the map.

<div style="text-align:left;">
<img src="./image/final_map1.png" height="30%" width="40%" />
</div>
<br />

```math
        while(len(carrot_list) < total_carrot and (carrot_list[-1] // self.width) < forward_bound):
            next_step = np.random.choice([1, -1, 20], replace=False)
            if((carrot_list[-1] + next_step) not in carrot_list and (carrot_list[-1] % self.width) + next_step > left_bound and
                 ((carrot_list[-1] % self.width) + next_step < right_bound or next_step == self.width)):
                
                times = 2   #left or right for two step
                if(next_step == self.width):
                    times = np.random.randint(3, 4) #forward 3 to 4 step

                for n in range(times):
                    carrot_list.append(carrot_list[-1] + next_step)

        #add the carrot and grass on the map

        for coor in carrot_list:
            print(coor % self.width, coor // self.width)
            carrot_xml += "<DrawItem x='{}' y ='2' z ='{}' type ='carrot' />".format(coor % self.width, coor // self.width)
            grass_xml += "<DrawBlock x='{}' y='1' z='{}' type='grass' />".format(coor % self.width, coor // self.width)
```  
<br />

##### ***3. Mutton Distribution***  
In order to give Jackson a penalty, we set up a mutton next to the "carrot road". If Jackson, a vegetarian, finds meat, he will deduct points. Because the venue we set up is 20* 50. In order to ensure that mutton and carrot do not appear on the same grid, we use an isolation algorithm. As you can see in the picture below. We take mutton as the center and confirm that no carrots will be placed on the eight grids around it which indices are -21, -20, -19, -1, +1, +19, +20, +21. This prevents mutton and carrot from appearing on the same grid. In addition, we will also pay attention to the ratio of mutton to carrot. We make sure that mutton will not be too much and the agent will lose a lot of points. We will also ensure that there are too few muttons so that the agent has no chance to encounter mutton and cannot learn. Therefore, when we set the ratio of the number of muttons to the number of carrots to 3:4, the distribution of muttons is the best.

<div style="text-align:left;">
<img src="./image/final_mul.png" height="30%" width="20%" />
</div>
Here is our code for setting the mutton.
```
        # Mutton Distribution
        
        muttons_map = []
        mutton_total = total_carrot * 0.8

        while(len(muttons_map) < mutton_total):
            valid_coor = True
            coor = np.random.randint((self.start_x + 1) * self.width, self.width * self.length)
            for z in range(-1, 2):
                for x in range(-1, 2):
                    if (coor + z * self.width + x) in carrot_list:
                        valid_coor = False
                        break
            if(valid_coor == True):
                muttons_map.append(coor)

        #add the mutton on the map
        for coor in muttons_map:
            mutton_type = 'mutton'
            if(np.random.randint(0,2) == 0):
                mutton_type = 'cooked_mutton'
            mutton_xml += "<DrawItem x='{}' y ='2' z ='{}' type ='{}' />".format(coor % self.width, coor // self.width, mutton_type)
            bedrock_xml += "<DrawBlock x='{}' y='1' z='{}' type='bedrock' />".format(coor % self.width, coor // self.width)
```
<br />

##### ***4. Q-learning***  

<div style="text-align:left;">
<img src="./image/final_q_alg.png" height="70%" width="70%" />
</div>
(source: image refer from our lecture 8 notes)

S: current state <br>
A: current action <br>
Q(S, A): old values <br>
$$\alpha:$$ learning rate <br>
R: rewards <br>
$$\gamma:$$ discount factor <br>
$$\max_a Q(S,a):$$ slightly estimate of optimal future value <br>

<br/>
We have tried using Q-learning, but it is not very stable from Jackson's results. We found that the Q-learning algorithm did not greatly improve Jackson's score. Through a lot of time learning, Jackson still has no obvious improvement after 100,000 steps. In observing Jackson's learning process, we found that it sometimes pauses when making decisions, which may also waste its learning time. After discussion with Kolby, we decided to use PPO (Proximal Policy Optimization) with rllib and ray to train the agent in a random environment.
<br />

##### ***5. Proximal Policy Optimization (PPO)***  
PPO trains a random strategy in a strategy-based manner, which can be updated in small batches in multiple training steps, and then the best strategy can be selected through the strategy. This means that it will explore through sampling operations based on the latest version of its random strategy. PPO is the built-in trainer of RLlib, which solves the problem of difficult to determine the step length. The PPO used in our project is a strategy-based algorithm that can only be trained using the data generated by the currently optimized strategy. When Jackson uses a piece of data (status, action, reward, new status), after updating the parameters of the strategy network, the "optimization" strategy will be changed immediately. We all know that the randomness of action selection depends on initial conditions and training procedures. During the training process, PPO usually becomes less and less random as the updated rules encourage the strategy to take advantage of discovered rewards. This may cause the strategy to fall into a local optimal state. Below we provide the Pseudocode of PPO.

<div style="text-align:left;">
<img src="./image/final_ppo_alg.png" height="70%" width="70%" />
</div>
<br />

Compared with Q-learning, PPO provides more stable results, but requires more training steps. We did a comparison between using continuous movement and discrete movement and found that in some cases, when the agent should stop moving for one second, it still moves. Therefore, we decided to switch to discrete motion. Discrete exercise can significantly increase our training speed, because we have to train more episodes at the same time.

### Evaluation

***Qualitative:***

<br />

***Quantitative:***

<br />


### Resources Used

- [Malmo XML Schema Documentation](https://microsoft.github.io/malmo/0.14.0/Schemas/Mission.html)
- [Malmo XML template](https://canvas.eee.uci.edu/courses/34142/quizzes/144375)
- [RL — Proximal Policy Optimization (PPO) Explained](https://jonathan-hui.medium.com/rl-proximal-policy-optimization-ppo-explained-77f014ec3f12)
- [PPO Pseudocode](https://spinningup.openai.com/en/latest/algorithms/ppo.html#proximal-policy-optimization)
- [RLlib Algorithms](https://docs.ray.io/en/master/rllib-algorithms.html#proximal-policy-optimization-ppo)
- [Q-Learning Wiki](https://en.wikipedia.org/wiki/Q-learning)
- [Simple Reinforcement Learning:Q-learning](https://towardsdatascience.com/simple-reinforcement-learning-q-learning-fcddc4b6fe56)
- [Q-Learning Algorithm](https://towardsdatascience.com/a-beginners-guide-to-q-learning-c3e2a30a653c)
