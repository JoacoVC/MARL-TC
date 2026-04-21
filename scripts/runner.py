
import os
from scripts.agents import FixedCycle, SarsaAgent, LearningAgent
from scripts.custom_environment import CustomEnvironment

class Runner:
    

    def __init__(self, configs: dict, learn: bool = True):
       
        self.configs: dict = configs
        self.learn: bool = learn
        self.agents: list[LearningAgent] = []
        self._set_environment()

    def _set_environment(self) -> None:
       
        env_config = self.configs['Environment']
        route_file = "interseccion/nueva_interseccion.rou.xml"

        self.env = CustomEnvironment(
            route_file=route_file,
            gui=env_config['Gui'],
            num_seconds=env_config['Num_seconds'],
            min_green=env_config['Min_green'],
            max_green=env_config['Max_green'],
            yellow_time=env_config['Yellow_time'],
            delta_time=env_config['Delta_time'],
        )

    def run(self) -> None:
        
        if self.env is None:
            self._set_environment()
        if not self.agents:
            self._load_agents()

        output_path = os.path.join(self.configs['Output_csv'])
        output_csvs_paths: dict[str, str] = {}

        for agent in self.agents:
            print("\nRunning agent: " + agent.get_name())
            csvs_path = agent.run(self.learn, output_path)
            output_csvs_paths[agent.get_name()] = csvs_path

        if self.learn:
            print("Saving models")
            self._save_agents_to_file()

    def _plot_per_agent(self, csvs_paths: dict[str, str]) -> None:
        
        for name, path in csvs_paths.items():
            self.plotter.add_csv(path)
            self.plotter.build_plot(name)
            self.plotter.clear()

    def _plot_last_episode(self, csvs_path: dict[str, str]) -> None:
        
        for path in csvs_path.values():
            csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
            last_episode = os.path.join(path, csv_files[-1])
            self.plotter.add_csv(last_episode)

        self.plotter.build_plot('last_episodes')
        self.plotter.clear()

    def _load_agents(self):
        
        for name, config in self.configs['Instances'].items():
            
            if config['Agent_type'] == 'SARSA':
                if 'Model' in config:
                    agent = SarsaAgent(config, self.env.get_sumo_env(False), name)
                    agent.load(config['Model'], self.env.get_sumo_env(False))
                else:
                    agent = SarsaAgent(config, self.env.get_sumo_env(False), name)
            if config['Agent_type'] == 'FIXED':
                agent = FixedCycle(config, self.env.get_sumo_env(True), name)
            self.agents.append(agent)

    def _save_agents_to_file(self) -> None:

        os.makedirs(self.configs['Output_model'], exist_ok=True)

        for agent in self.agents:
            out_file = self.configs['Output_model'] + '/' + agent.get_name() + '.pkl'
            agent.save(out_file)
