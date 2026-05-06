import yaml
from scripts.runner import Runner

config_file = 'configs/entrenamiento.yaml'
with open(config_file) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

r = Runner(config['Agent_settings'])

# CAMBIA ESTO:
# r.run() 

# POR ESTO:
r.run_all_experiments()
