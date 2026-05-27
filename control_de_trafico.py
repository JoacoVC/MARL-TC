import yaml
from scripts.runner import Runner

config_file = 'configs/prueba_pesos.yaml'
with open(config_file) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

r = Runner(config['Agent_settings'])

# r.run()
r.run_all_experiments()