
import sys
from scripts.plotter import Plotter

# Lista de rutas archivos CSV 
csv_files = [
    "output\csv\entrenamiento\dif_waiting_time\SARSA_run_1\SARSA_run_1_conn0_ep0.csv",
    "output\csv\entrenamiento\dif_waiting_time\SARSA_run_1\SARSA_run_1_conn0_ep1.csv",
    "output\csv\entrenamiento\dif_waiting_time\SARSA_run_1\SARSA_run_1_conn0_ep2.csv",
    "output\csv\entrenamiento\dif_waiting_time\SARSA_run_1\SARSA_run_1_conn0_ep3.csv",
    "output\csv\entrenamiento\dif_waiting_time\SARSA_run_1\SARSA_run_1_conn0_ep4.csv",
]

# Configuración del plotter 
plotter_settings = {
    'Output': 'Graficas/dif_waiting_time',
    'Width': 800,
    'Height': 400,
    'Metrics': ['system_total_stopped', 'system_total_waiting_time']
}

# Inicializa el plotter y configura
p = Plotter()
p.set_configs(plotter_settings)

# Agrega los CSV manualmente
for csv in csv_files:
    p.add_csv(csv)

# Graficar (puedes pasar un nombre de subcarpeta si quieres)

p.build_plot()