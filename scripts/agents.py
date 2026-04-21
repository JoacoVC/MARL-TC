
import os
import pickle
from sumo_rl import SumoEnvironment
from abc import ABC, abstractmethod
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy
from linear_rl.true_online_sarsa import TrueOnlineSarsaLambda




class LearningAgent():

    def __init__(self, config: dict, env: SumoEnvironment, name: str):
        
        
        self.config = config
        self.env = env
        self.agent = None
        self.name = name

    
    def _init_agent(self):
        
        pass

    def get_name(self) -> str:
        
        return self.name

    
    def run(self, learn: bool, out_path: str) -> str:
       
        pass

    def save(self, path: str) -> None:
        
        pass

    def load(self, path: str, env: SumoEnvironment) -> None:
        
        pass

class FixedCycle(LearningAgent):

    def __init__(self, config: dict, env: SumoEnvironment, name: str):
        
        super().__init__(config, env, name)

    def _init_agent(self):
        self.agent = None

    def run(self, learn: bool, out_path: str) -> str:
        

        out_path = os.path.join(out_path, self.name)
        out_file = os.path.join(out_path, self.name)

        for curr_run in range(self.config['Runs']):
            done = False
            self.env.reset()
            while not done:
                done = self._step()
            self.env.save_csv(out_file, curr_run)

        self.env.close()

        return out_path

    def save(self, path: str) -> None:
        
        pass

    def load(self, path: str, env: SumoEnvironment) -> None:
        
        pass

    def _step(self) -> bool:
        
        for _ in range(self.env.delta_time):

            self.env._sumo_step()
            
        self.env._compute_observations()
        self.env._compute_rewards()
        self.env._compute_info()
        return self.env._compute_dones()['__all__']

class SarsaAgent(LearningAgent):
    
    def __init__(self, config: dict, env: SumoEnvironment, name: str):
       
        super().__init__(config, env, name)

    def _init_agent(self):

        import gymnasium as gym
        import numpy as np
    
        # 🔹 dimensión de un semáforo
        obs_dim = self.env.observation_space.shape[0]
    
        # 🔹 dos semáforos → estado combinado
        new_obs_dim = obs_dim * 2
    
        # 🔹 nuevo espacio de estados
        state_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(new_obs_dim,),
            dtype=np.float32
        )
    
        # 🔹 número de acciones de UN semáforo
        n = self.env.action_space.n
    
        # 🔹 espacio de acciones conjunto (2 semáforos)
        joint_action_space = gym.spaces.Discrete(n * n)
    
        # 🔹 crear agente
        self.agent = TrueOnlineSarsaLambda(
            state_space=state_space,
            action_space=joint_action_space,
            alpha=self.config['Alpha'],
            gamma=self.config['Gamma'],
            epsilon=self.config['Epsilon'],
            fourier_order=self.config['FourierOrder'],
            lamb=self.config['Lambda']
        )
    

    def run(self, learn: bool, out_path: str) -> str:
        
        if self.agent is None:
            self._init_agent()

        out_path = os.path.join(out_path, self.name)
        out_file = os.path.join(out_path, self.name)

        for curr_run in range(self.config['Runs']):
            obs = self.env.reset()
            
            done = False
            
            while not done:

                state = self._flatten_obs(obs)
                action_idx = self.agent.act(state)
            
                tl_ids = list(obs.keys())
                n_actions = self.env.action_space.n
            
                action = {
                    tl_ids[0]: action_idx // n_actions,
                    tl_ids[1]: action_idx % n_actions
                }
            
                next_obs, reward, dones, _ = self.env.step(action)

                # imrprimir presión y diff_wait para cada semáforo en para cada paso de la simulacion
                #for ts_id, ts in self.env.traffic_signals.items():
                #    
                #    dif_wait = reward [ts_id]
                #    pressure = ts._pressure_reward()
                #    x = dif_wait * 10  + pressure * 0.2
                #    
                #    print(f"[TS {ts_id}] Pressure: {pressure:.2f}| diff_wait: {dif_wait:.2f} |combined_reward: {x}") 
                #
                #print("=" * 50)
                ##--------------------------------------------------------------------------------------------------
                
                done = dones['__all__']
            
                next_state = self._flatten_obs(next_obs)
                reward_total = sum(reward.values())
            
                self.agent.learn(
                    state=state,
                    action=action_idx,
                    reward=reward_total,
                    next_state=next_state,
                    done=done
                )
            
                obs = next_obs

            self.env.save_csv(out_file, curr_run)
        self.env.close()

        return out_path
    

    def save(self, path: str) -> None:
        
        data = {
            'alpha': self.agent.alpha,
            'gamma': self.agent.gamma,
            'epsilon': self.agent.epsilon,
            'lamb': self.agent.lamb,
            'fourier_order': self.agent.basis.order,
        }

        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str, env: SumoEnvironment) -> None:
      
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.env = env

        self.agent = TrueOnlineSarsaLambda(
            state_space=self.env.observation_space,
            action_space=self.env.action_space,
            alpha=data['alpha'],
            gamma=data['gamma'],
            epsilon=data['epsilon'],
            fourier_order=data['fourier_order'],
            lamb=data['lamb']
        )
        ###nuevo código
    def _flatten_obs(self, obs_dict):
        state = []
        for tl_id in sorted(obs_dict.keys()):
            state.extend(obs_dict[tl_id])
        return state
    
