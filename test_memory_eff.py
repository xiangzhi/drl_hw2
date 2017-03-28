import unittest

from deeprl_hw2.action_replay_memory_eff import ActionReplayMemoryEff as ActionReplayMemory
import numpy as np



class ActionMemoryTestMethods(unittest.TestCase):

    def test_memory(self):
        memory = ActionReplayMemory(250,4) #test memory
        index = 0
        while(index < 1000):
            axr = np.random.randint(0,100,(84,84,4))
            memory.append(axr,4,5)
            if((index+1)%50 == 0):
                axr = np.random.randint(0,100,(84,84,4))
                memory.end_episode(axr,True)
                index += 1
            index += 1

        for i in range(0,10):
            #some sampling tests
            curr_arr, next_arr, reward_arr, action_arr, terminal_arr = memory.sample(10)
            for i,terminal in enumerate(terminal_arr):
                self.assertTrue(np.all(curr_arr[i][:,:,0] == next_arr[i][:,:,1]))
            self.assertTrue(np.sum(reward_arr-5) == 0)
            self.assertTrue(np.sum(action_arr-4) == 0)

            # self.assertTrue(np.sum(np.where(np.logical_and(curr_arr<750,curr_arr>1001))) == 0) #simple test to see if they are in range

    # def test_memory_deprive(self):
    #     memory = ActionReplayMemory(1000,4)
    #     index = 0 
    #     while(index < 100):
    #         memory.append(index,4,5)
    #         index += 1

    #     for i in range(0,10):
    #         #some sampling tests
    #         curr_arr, next_arr, reward_arr, action_arr, terminal_arr = memory.sample(10)
    #         self.assertTrue(np.sum(np.where(curr_arr>101)) == 0) #simple test to see if they are in range

    # def test_end_range(self):
    #     memory = ActionReplayMemory(1000,4)
    #     index = 0 
    #     while(index < 100):
    #         memory.append(index,4,5)
    #         index += 1
    #     max_size = len(memory)
    #     self.assertTrue(memory._memory[max_size-1] != None)     

    # def test_single(self):
    #     memory = ActionReplayMemory(2,4)
    #     index = 0 
    #     last_sample = None
    #     for i in range(0,10):
    #          axr = np.random.randint(0,100,(84,84,4))
    #          last_sample = axr
    #          memory.append(axr,4,5)
    #     self.assertTrue(len(memory) == 2)
    #     axr = np.random.randint(0,100,(84,84,4))
    #     memory.append(axr,4,5)
    #     for x in range(0,10):
    #         curr, next_state, reward, action, terminal = memory.sample(1)
    #         print(curr)
    #         self.assertTrue(np.all(curr[:,:,0] == last_sample))


    # def test_Timing(self):
    #     memory = ActionReplayMemory(100000,4)#test memory
    #     index = 0
    #     while(index  < 100000):
    #         memory.append(index,4,5)
    #         if((index+1)%50 == 0):
    #             memory.end_episode(index+1,True)
    #             index += 1
    #         index += 1      
    #     print('done')   
    #     for i in range(0,10):
    #         #some sampling tests
    #         curr_arr, next_arr, reward_arr, action_arr, terminal_arr = memory.sample(10)

    def xx__test_memory_x(self):
        memory = ActionReplayMemory(1000000,4)
        index = 0
        while(index < 1000000):
            axr = np.random.randint(0,100,(84,84,4))
            memory.append(axr,4,5)
            index += 1
        print(memory.size())

if __name__ == '__main__':
    #test_memory_size()
    unittest.main()
