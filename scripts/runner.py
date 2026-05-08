
import os
from scripts.agents import FixedCycle, SarsaAgent, LearningAgent
from scripts.custom_environment import CustomEnvironment
from sumo_rl import TrafficSignal
import numpy as np
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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

#    def run_all_experiments(self):
#        # 1. Guardar ruta base original
#        ruta_base_original = self.configs['Output_csv']
#        
#        pesos_w1 = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10]
#        pesos_w2 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#
#        # En runner.py, antes de los for:
#        print("Funciones registradas:", TrafficSignal.reward_fns.keys())
#        for w1 in pesos_w1:
#            for w2 in pesos_w2:
#                # Actualizar ruta de salida para cada combinación
#                self.configs['Output_csv'] = os.path.join(ruta_base_original, f"w1_{w1}_w2_{w2}")
#
#                # Limpieza y carga 
#                self.agents = [] 
#                self._load_agents()
#
#                # INYECTAR PESOS 
#                for agent in self.agents:
#                    # En sumo-rl, los semáforos están en el diccionario 'traffic_signals'
#                    env_unwrapped = agent.env.unwrapped
#                    if hasattr(env_unwrapped, 'traffic_signals'):
#                        for ts in env_unwrapped.traffic_signals.values():
#                            ts.w1 = w1
#                            ts.w2 = w2
#                    else:
#                        print("Error: No se encontró el atributo traffic_signals en el entorno.")
#
#                # Mensaje de progreso
#                print(f"\n>>>> Iniciando: w1={w1}, w2={w2}", flush=True)
#
#             
#                self.run()
#                
#                # Cerrar para liberar procesos de SUMO
#                for agent in self.agents:
#                    agent.env.close()

    def run_all_experiments(self):
        ruta_base_original = self.configs['Output_csv']
        pesos_w1 = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10]
        pesos_w2 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        # 1. Lista para recolectar el "punto Z" de cada combinación
        matriz_resultados = []

        for w1 in pesos_w1:
            for w2 in pesos_w2:
                # Configuración de ruta y carga de agentes
                ruta_combinacion = os.path.join(ruta_base_original, f"w1_{w1}_w2_{w2}")
                self.configs['Output_csv'] = ruta_combinacion
                self.agents = [] 
                self._load_agents()

                # Inyección de pesos
                for agent in self.agents:
                    for ts in agent.env.unwrapped.traffic_signals.values():
                        ts.w1, ts.w2 = w1, w2

                print(f"\n>>>> Ejecutando: w1={w1}, w2={w2}", flush=True)

                # 2. Ejecutar los 5 episodios (genera los 5 CSVs)
                self.run()
                
                # 3. PROCESAMIENTO INMEDIATO: Sacar el promedio global (Z)
                promedios_episodios = []
                
                # Buscamos los archivos recién creados en la carpeta de la combinación
                for root, _, files in os.walk(ruta_combinacion):
                    for file in files:
                        if file.endswith(".csv"):
                            df = pd.read_csv(os.path.join(root, file))
                            # Promediamos la columna de tiempo de espera de este episodio
                            media_pasos = df['system_total_waiting_time'].mean()
                            promedios_episodios.append(media_pasos)
                
                # Promediamos los 5 episodios para obtener el valor Z final del punto (w1, w2)
                if promedios_episodios:
                    z_final = np.mean(promedios_episodios)
                    matriz_resultados.append({'w1': w1, 'w2': w2, 'espera_media': z_final})
                    print(f"Punto Z calculado para w1={w1}, w2={w2}: {z_final:.2f}")

                # Limpieza de SUMO
                for agent in self.agents:
                    agent.env.close()
                    
        # 4. EXPORTAR RESULTADOS A CSV
        if matriz_resultados:
            df_final = pd.DataFrame(matriz_resultados)
            
            # Renombramos las columnas para que sean claras
            df_final.columns = ['w1', 'w2', 'espera_media_global']
            
            # Definimos el nombre del archivo
            nombre_csv = os.path.join(ruta_base_original, "resumen_optimizacion_pesos.csv")
            
            # Guardamos el archivo
            df_final.to_csv(nombre_csv, index=False)
            
            print("\n" + "="*40)
            print(f"PROCESO FINALIZADO")
            print(f"Archivo de resumen guardado en: {nombre_csv}")
            print("="*40)
        else:
            print("\nNo se recolectaron datos para generar el resumen.")

        # 4. GENERAR LA GRÁFICA FINAL 
        self.graficar_superficie_pesos(matriz_resultados, ruta_base_original)
        

    def graficar_superficie_pesos(self, datos, ruta_guardado):
        df_plot = pd.DataFrame(datos)
        # Reestructurar datos para la gráfica (Matriz de w1 vs w2)
        pivot = df_plot.pivot(index='w1', columns='w2', values='espera_media')

        plt.figure(figsize=(12, 9))
        # cmap="YlGnBu_r" usa azul para valores bajos (mejor) y amarillo para altos
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu_r")
        
        plt.title('Optimización de Pesos: Tiempo de Espera Promedio Total')
        plt.xlabel('Peso w2 (Presión)')
        plt.ylabel('Peso w1 (Diferencia de Tiempo de Espera)')
        
        nombre_grafica = os.path.join(ruta_guardado, "mapa_optimizacion_pesos.png")
        plt.savefig(nombre_grafica)
        print(f"\nGráfica de optimización guardada en: {nombre_grafica}")
        plt.show()

    

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
