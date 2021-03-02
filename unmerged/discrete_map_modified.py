# Rllib docs: https://docs.ray.io/en/latest/rllib.html

try:
    from malmo import MalmoPython
except:
    import MalmoPython

import sys
import time
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
from numpy.random import randint
import random

import gym, ray
from gym.spaces import Discrete, Box
from ray.rllib.agents import ppo
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Vegetarian(gym.Env):

    def __init__(self, env_config):  
        # Static Parameters
        self.length = 50
        self.width = 20
        self.start_x = 0
        self.start_z = 0
        self.size = 50
        self.reward_density = .1
        self.penalty_density = 0.02
        self.obs_size = 5
        self.max_episode_steps = 100
        self.log_frequency = 10
        self.target_step = 200000
        self.action_dict = {
            0: 'move 1',  # Move one block forward
            1: 'turn 1',  # Turn 90 degrees to the right
            2: 'turn -1',  # Turn 90 degrees to the left
            #3: 'jump 1',  # Jump 
        }

        # Rllib Parameters
        self.action_space = Discrete(len(self.action_dict))
        self.observation_space = Box(-1, 1, shape=(self.obs_size * self.obs_size, ), dtype=np.float32)

        # Malmo Parameters
        self.agent_host = MalmoPython.AgentHost()
        try:
            self.agent_host.parse( sys.argv )
        except RuntimeError as e:
            print('ERROR:', e)
            print(self.agent_host.getUsage())
            exit(1)

        # Vegetarian Parameters
        self.obs = None
        self.allow_break_action = False
        self.episode_step = 0
        self.episode_return = 0
        self.returns = []
        self.steps = []
        self.num_carrot = 0
        self.num_cooked_mutton = 0
        self.num_mutton = 0
        self.return_carrot = []
        self.return_mutton = []
        self.return_cooked_mutton = []

        self.total_items = {'carrot': 0, 'mutton': 0, 'cooked_mutton': 0}

    def reset(self):
        """
        Resets the environment for the next episode.

        Returns
            observation: <np.array> flattened initial obseravtion
        """
        
        # Reset Malmo
        world_state = self.init_malmo()

        # Reset Variables
        self.returns.append(self.episode_return)
        current_step = self.steps[-1] if len(self.steps) > 0 else 0
        self.steps.append(current_step + self.episode_step)
        self.episode_return = 0
        self.episode_step = 0

        #reset number of items
        self.return_carrot.append(self.num_carrot)
        self.return_mutton.append(self.num_mutton)
        self.return_cooked_mutton.append(self.num_cooked_mutton)
        self.total_items['carrot'] = self.total_items['carrot'] + self.num_carrot
        self.total_items['mutton'] = self.total_items['mutton'] + self.num_mutton
        self.total_items['cooked_mutton'] = self.total_items['cooked_mutton'] + self.num_cooked_mutton
        self.num_carrot = 0
        self.num_cooked_mutton = 0
        self.num_mutton = 0

        # Log
        # print("RESET... "+ str(self.returns))
        
        if len(self.returns) > self.log_frequency + 1 and \
            len(self.returns) % self.log_frequency == 0:
            self.log_returns()

        # Get Observation
        self.obs, self.allow_break_action = self.get_observation(world_state)
        
        return self.obs

    def step(self, action):
        """
        Take an action in the environment and return the results.

        Args
            action: <int> index of the action to take

        Returns
            observation: <np.array> flattened array of obseravtion
            reward: <int> reward from taking action
            done: <bool> indicates terminal state
            info: <dict> dictionary of extra information
        """
        
        command = self.action_dict[action]
        if command != 'attack 1' or self.allow_break_action:
            self.agent_host.sendCommand(command)
            time.sleep(.2)
            self.episode_step += 1
        # Get Observation
        world_state = self.agent_host.getWorldState()
        
        for error in world_state.errors:
            print("Error:", error.text)
        self.obs, self.allow_break_action = self.get_observation(world_state) 

        # Get Done
        done = not world_state.is_mission_running 

        # Get Reward
        reward = 0
        for r in world_state.rewards:
            reward += r.getValue()
        self.episode_return += reward

        print("REWARD: " + str(self.episode_return))
        
        #Get number of food
        #num_carrot = 0
        #num_cooked_mutton = 0
        #num_mutton = 0
        #for f in world_state.rewards:
        #    num_carrot += f.getValue()
        #    num_cooked_mutton += f.getValue()
        #    num_mutton += f.getValue()
        return self.obs, reward, done, dict()
        #return self.obs, reward, num_carrot,num_cooked_mutton,num_mutton, done, dict()

    def get_mission_xml(self):
        wall_block = ""
        carrot_xml = ""
        grass_xml = ""
        mutton_xml = ""
        bedrock_xml = ""

        for num in range(self.start_z - 1, self.length + self.start_z + 1):
            for y in range(2,5):
                wall_block += "<DrawBlock x='{}' y='{}' z='{}' type='stained_glass' colour='PINK' />".format(self.start_z - 1,y, num)
                wall_block += "<DrawBlock x='{}' y='{}' z='{}' type='stained_glass' colour='PINK' />".format(self.width + 1,y, num)
            

        for num in range(self.start_x - 1, self.width + self.start_x + 1):
            for y in range(2,5):
                wall_block += "<DrawBlock x='{}' y='{}' z='{}' type='stained_glass' colour='PINK' />".format(num,y, self.start_z - 1)
                wall_block += "<DrawBlock x='{}' y='{}' z='{}' type='stained_glass' colour='PINK' />".format(num,y, self.length + 1)

        total_carrot = 100
        carrot_list = [np.random.randint(65, 75)]   #start at z = 3,
        left_bound = self.start_x + 2
        right_bound = self.width - 2
        forward_bound = self.length - 4

        # carrot_location = [10,1]
        # for i in range(0, total_carrot):
        #     rand = random.randint(0,2)
        #     if rand == 0:
        #         carrot_location[0] = carrot_location[0] - 1
        #         carrot_location[1] = carrot_location[1] + 1
        #     elif rand == 1:
        #         carrot_location[0] = carrot_location[0]
        #         carrot_location[1] = carrot_location[1] + 1
        #     else:
        #         carrot_location[0] = carrot_location[0] + 1
        #         carrot_location[1] = carrot_location[1] + 1
            
        #     if carrot_location[0] > 20:
        #         carrot_location[0] = 20
        #     carrot_list.append(carrot_location[0] + carrot_location[1] * self.width)

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

        #mutton 
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

        return '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
                <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
                    <About>
                        <Summary>Carrot Collector</Summary>
                    </About>
                    <ServerSection>
                        <ServerInitialConditions>
                            <Time>
                                <StartTime>12000</StartTime>
                                <AllowPassageOfTime>false</AllowPassageOfTime>
                            </Time>
                            <Weather>clear</Weather>
                        </ServerInitialConditions>
                        <ServerHandlers>
                            <FlatWorldGenerator generatorString="3;7,2;1;"/>
                            <DrawingDecorator>''' + \
                                "<DrawCuboid x1='{}' x2='{}' y1='2' y2='2' z1='{}' z2='{}' type='air'/>".format(self.start_x, self.width, self.start_z, self.length) + \
                                "<DrawCuboid x1='{}' x2='{}' y1='1' y2='1' z1='{}' z2='{}' type='stone'/>".format(self.start_x, self.width, self.start_z, self.length) + \
                                wall_block + \
                                carrot_xml + \
                                grass_xml + \
                                mutton_xml + \
                                bedrock_xml + \
                                '''
                                <DrawBlock x='10'  y='1' z='0' type='redstone_block' />
                            </DrawingDecorator>
                            <ServerQuitWhenAnyAgentFinishes/>
                        </ServerHandlers>
                    </ServerSection>
                    <AgentSection mode="Survival">
                        <Name>CarrotCollector</Name>
                        <AgentStart>
                            <Placement x="10.5" y="2" z="0.5" pitch="45" yaw="0"/>
                            <Inventory>
                                <InventoryItem slot="0" type="diamond_pickaxe"/>
                            </Inventory>
                        </AgentStart>
                        <AgentHandlers>
                            <RewardForCollectingItem>
                                <Item type = "carrot" reward ="5"/>
                                <Item type = "cooked_mutton" reward ="-1"/>
                                <Item type = "mutton" reward ="-2"/>
                            </RewardForCollectingItem>
                            <DiscreteMovementCommands/>
                            <ObservationFromFullStats/>
                            <ObservationFromRay/>
                            <ObservationFromHotBar/>
                            
                            <ObservationFromNearbyEntities>
                                <Range name="floorAll" xrange='3' yrange='2' zrange='3' />                                
                            </ObservationFromNearbyEntities>
                           
                            <AgentQuitFromReachingCommandQuota total="'''+str(self.max_episode_steps)+'''" />
                        </AgentHandlers>
                    </AgentSection>
                </Mission>'''

    def init_malmo(self):
        """
        Initialize new malmo mission.
        """
        my_mission = MalmoPython.MissionSpec(self.get_mission_xml(), True)
        my_mission_record = MalmoPython.MissionRecordSpec()
        my_mission.requestVideo(800, 500)
        my_mission.setViewpoint(1)

        max_retries = 3
        my_clients = MalmoPython.ClientPool()
        my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000)) # add Minecraft machines here as available

        for retry in range(max_retries):
            try:
                self.agent_host.startMission( my_mission, my_clients, my_mission_record, 0, 'Vegetarian' )
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:", e)
                    exit(1)
                else:
                    time.sleep(2)

        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            for error in world_state.errors:
                print("\nError:", error.text)

        return world_state

    def get_observation(self, world_state):
        """
        Use the agent observation API to get a flattened 2 x 5 x 5 grid around the agent. 
        The agent is in the center square facing up.

        Args
            world_state: <object> current agent world state

        Returns
            observation: <np.array> the state observation
            allow_break_action: <bool> whether the agent is facing a carrot
        """
        # obs = np.zeros((2 * self.obs_size * self.obs_size, ))
        # allow_break_action = False
        edge_wall_action = False
        reward_action = False


        obs = np.zeros(self.obs_size * self.obs_size)
        
        while world_state.is_mission_running:
            time.sleep(0.10)
            
            grid = None
            yaw = None
            retry = 0;
            while (grid is None or yaw is None) and retry < 100:
                retry += 1;
                world_state = self.agent_host.getWorldState()
                if len(world_state.errors) > 0:
                    raise AssertionError('Could not load grid.')

                if world_state.number_of_observations_since_last_state > 0:
                    # First we get the json from the observation API
                    msg = world_state.observations[-1].text
                    observations = json.loads(msg)

                    ################################
                    #store number of item
                    items = ["carrot", "cooked_mutton", "mutton"]
                    
                    for i in range(0,9):
                        for item in items:
                            slot_name = u'Hotbar_' + str(i) + '_item'
                            slot_contents = observations.get(slot_name, "")
                            
                            if slot_contents == item:
                                slot_number = u'Hotbar_' + str(i) + '_size'
                                number = observations.get(slot_number, "")
                                if items[0] == item:
                                    self.num_carrot = number
                                    
                                elif items[1] == item:
                                    self.num_cooked_mutton = number
                                    
                                else:
                                    self.num_mutton = number
                                    

                    #################################
                    # print(observations['XPos'], observations['YPos'],observations['ZPos'])
                    # Get observation                    
                    try:
                        grid = observations['floorAll']
                        yaw = observations['Yaw']
                    except:
                        print("Retry floorALL error")
                        time.sleep(0.20)
                        continue

                    agent = grid[0];
                    # print(grid)
                    for item in grid:
                        index = self.obs_size * self.obs_size // 2 + (int)(item['x'] - agent['x']) + (int)(item['z'] - agent['z']) * self.obs_size
                        # print(item['x'], item['z'], index)

                        if(item['name'] == 'carrot'):
                            obs[index] = 1
                        if(item['name'] == 'cooked_mutton'):
                            obs[index] = -0.5
                        if(item['name'] == 'mutton'):
                            obs[index] = -1


                    # Rotate observation with orientation of agent
                    obs = obs.reshape((1, self.obs_size, self.obs_size))

                    # print(obs)
                    
                    if yaw >= 225 and yaw < 315:
                        obs = np.rot90(obs, k=1, axes=(1, 2))
                    elif yaw >= 315 or yaw < 45:
                        obs = np.rot90(obs, k=2, axes=(1, 2))
                    elif yaw >= 45 and yaw < 135:
                        obs = np.rot90(obs, k=3, axes=(1, 2))
                    # print(obs)
                    
                    obs = obs.flatten()
                    # obs[-1] = observations['Yaw'] / 360

                    # print(obs)

                    edge_wall_action = observations['LineOfSight']['type'] == 'iron_ore'

            break

        return obs, edge_wall_action
        # return obs, allow_break_action


    def log_returns(self):
        """
        Log the current returns as a graph and text file

        Args:
            steps (list): list of global steps after each episode
            returns (list): list of total return of each episode
        """
        box = np.ones(self.log_frequency) / self.log_frequency
        returns_smooth = np.convolve(self.returns[1:], box, mode='same')
        plt.clf()
        # plot the rewards graphs
        plt.plot(self.steps[1:], returns_smooth)
        plt.title('Vegetarian')
        plt.ylabel('Rewards')
        plt.xlabel('Steps')
        plt.savefig('rewards.png')
        with open('rewards.txt', 'w') as f:
            for step, value in zip(self.steps[1:], self.returns[1:]):
                f.write("{}\t{}\n".format(step, value))
        
        # plot the collect number of food graphs        
        carrot_smooth = np.convolve(self.return_carrot[1:], box, mode='same')
        mutton_smooth = np.convolve(self.return_mutton[1:], box, mode='same')
        cooked_mutton_smooth = np.convolve(self.return_cooked_mutton[1:], box, mode='same')
        plt.clf()

        line1 = plt.plot(self.steps[1:], carrot_smooth, label = 'carrot', color = 'red')
        line2 = plt.plot(self.steps[1:], mutton_smooth, label = 'mutton', color = 'blue')
        line3 = plt.plot(self.steps[1:], cooked_mutton_smooth, label = 'cooked_mutton', color = 'green')
        
        plt.title("Vegetarian")
        plt.ylabel("Numbers")
        plt.xlabel("Steps")
        plt.legend(loc = "upper right")

        plt.savefig("FoodNumber.png")
        
        with open('FoodNumber.txt', 'w') as f:
           for step, carrot, mutton, cooked_mutton in zip(self.steps[1:], self.return_carrot[1:], self.return_mutton[1:], self.return_cooked_mutton[1:]):
               f.write("{}\tcarrot: {}\tmutton: {}\tcooked_mutton: {}\n".format(step, carrot, mutton, cooked_mutton))

        # plot total number of item
        items = list(self.total_items.keys())
        values = list(self.total_items.values())
        fig = plt.figure(figsize = (10, 5)) 
        plt.bar(items[0], values[0], color ='red', width = 0.4) 
        plt.bar(items[1], values[1], color ='blue', width = 0.4) 
        plt.bar(items[2], values[2], color ='green', width = 0.4) 
        plt.title("Vegetarian")
        plt.ylabel("Numbers")
        plt.xlabel("items")

        plt.savefig("totalItem.png")
               
        if self.steps[1:][-1] > self.target_step:
            print("EXIT")
            print(self.steps[1:][-1])
            sys.exit()  


if __name__ == '__main__':
    ray.init()
    trainer = ppo.PPOTrainer(env=Vegetarian, config={
        'env_config': {},           # No environment parameters to configure
        'framework': 'torch',       # Use pyotrch instead of tensorflow
        'num_gpus': 0,              # We aren't using GPUs
        'num_workers': 0            # We aren't using parallelism
    })

    while True:
        print(trainer.train())
