---
layout: page
title: Proposal
---



<br />

## Summary of the Project

Our project is to simulate a set of reward policies in Minecraft. We will create platforms with different sizes and shapes to allow the agent to walk different paths from the starting point（emerald_block）to the end point（redstone_block）. We will place different foods in some diamond_blocks. Each different food will represent a reward or punishment score（negative rewards）. The agent can move in four different directions and each step is 1 block. Each movement consumes a certain score value. Our goal is to let the agent find the optimal path through the learning of the environment, which means that the agent needs to get the highest score in a limited number of steps as much as possible.

***Input:***
The state of the agent, such as, the coordinates or the current rewards. 

***Output:***
An agent need to take actions, such as, moving directions, picking up foods, or changing rewards.




<br />

## AI/ML Algorithms 
 
- Dynamic Programming
- Deep Q-Learning (Reinforcement Learning)
- Greedy Algorithm
- More on later

Currently, our project plan to use Dynamic Programming and Deep Q-Learning (Reinforcement Learning) to record agent scores after taking different actions. We plan to use Greedy Algorithm to optimize the agent. Definitively, Malmos API’s depth map functionality will be used when process the image. We may add more algorithms in the implementation process, which will be updated in our future discussions.


<br />

## Evaluation Plan
    
    
***Qualitative:***

In the process of reinforcement learning, the evaluation metrics would be the reward scores that the agent obtains. The reward scores can be positive or even negative. The baseline is the agent must walk from the starting point (emerald_block) to the end point (redstone_block) without falling. The agent will obtain reward scores by taking different reasonable paths. The length of the path taken by the agent is positively correlated with the consumption, which means that the more blocks the agent walks, the more reward scores will be deducted. Different blocks will have food representing different reward scores.

<br />

***Quantitative:***

The initial sanity check is that the agent can find a reasonable route including the correct starting point (emerald_block) to the end point (redstone_block) and will not fall to death. He can get a low score but must survive. As the agent continuously learns the environment, it can find the optimal solution brilliantly. The agent needs to be equipped with the ability that it can achieve as much reward scores as it can within a relatively short path.




<br />

## Appointment with the Instructor


Wednesday, January 20, 2021, 04:15pm (Pacific Time - US & Canada)



<br />

## Weekly Meeting

We meet twice a week from 09:00pm - 10:00pm (Pacific Time - US & Canada) on Wednesday and Sunday through zoom.  
On Sundays, we assign each one what we need to do next week and on Wednesdays we discuss and help each other.

