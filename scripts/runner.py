
import os
from scripts.agents import FixedCycle, SarsaAgent, LearningAgent
from scripts.custom_environment import CustomEnvironment
from sumo_rl import TrafficSignal
import numpy as np
import os

def combined_reward_weighted(ts):
    # Recupera los pesos inyectados; si no existen, usa valores base
    w1 = getattr(ts, 'w1', 1.0)
    w2 = getattr(ts, 'w2', 0.1)

    ts_wait = sum(ts.get_accumulated_waiting_time_per_lane()) / 100.0
    dif_wait = ts.last_measure - ts_wait
    ts.last_measure = ts_wait

    return (dif_wait * w1) - (ts.get_pressure() * w2)

# Registramos la función una sola vez al inicio
TrafficSignal.register_reward_fn(combined_reward_weighted)

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

    def run_all_experiments(self):
        # 1. Guardar ruta base original
        ruta_base_original = self.configs['Output_csv']
        
        pesos_w1 = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10]
        pesos_w2 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        # En runner.py, antes de los for:
        print("Funciones registradas:", TrafficSignal.reward_fns.keys())
        for w1 in pesos_w1:
            for w2 in pesos_w2:
                # Actualizar ruta de salida para cada combinación
                self.configs['Output_csv'] = os.path.join(ruta_base_original, f"w1_{w1}_w2_{w2}")

                # Limpiar y cargar nuevos agentes para esta prueba
                self.agents = [] 
                self._load_agents()

                # INYECTAR PESOS (Aquí está la corrección)
                for agent in self.agents:
                    # En sumo-rl, los semáforos están en el diccionario 'traffic_signals'
                    env_unwrapped = agent.env.unwrapped
                    if hasattr(env_unwrapped, 'traffic_signals'):
                        for ts in env_unwrapped.traffic_signals.values():
                            ts.w1 = w1
                            ts.w2 = w2
                    else:
                        # En algunas versiones muy específicas puede ser 'ts_dict'
                        # pero usualmente es 'traffic_signals'
                        print("Error: No se encontró el atributo traffic_signals en el entorno.")

                # Mensaje de progreso
                print(f"\n>>>> Iniciando: w1={w1}, w2={w2}", flush=True)

                # EJECUTAR (Esto hará los 5 episodios del YAML)
                self.run()
                
                # Cerrar para liberar procesos de SUMO
                for agent in self.agents:
                    agent.env.close()

    def run(self) -> None:
        
        if self.env is None:
            self._set_environment()
        #if not self.agents:
        #    self._load_agents()

        output_path = os.path.join(self.configs['Output_csv'])
        output_csvs_paths: dict[str, str] = {}

        for agent in self.agents:
            #print("\nRunning agent: " + agent.get_name())
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
