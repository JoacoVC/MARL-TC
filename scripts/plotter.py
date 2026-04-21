import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors

class Plotter:
    # dict to set y labels according to set metric
    _Ylabels = {
        'system_total_stopped': 'Stationary vehicles',
        'system_total_waiting_time': 'Waiting time',
        'agents_total_stopped': 'total Stationary vehicles ',
        'agents_total_accumulated_waiting_time': 'Accumulated waiting time'
    }

    def __init__(self, output=None, metrics=None, width=800, height=400):
        self.output = output
        self.height = height / 96
        self.width = width / 96
        self.df = []
        self.csv_files = []
        self.metrics = metrics

    def set_configs(self, configs) -> None:
        self.output = configs['Output']
        if 'Height' in configs:
            self.height = configs['Height'] / 96
        if 'Width' in configs:
            self.width = configs['Width'] / 96
        self.metrics = configs['Metrics']

    def add_csv(self, input_path) -> None:
        if input_path.endswith('.csv'):
            self.csv_files.append(input_path)
            self.df.append(pd.read_csv(input_path))

    def build_plot(self, out_folder: str = None) -> None:
        if len(self.csv_files) == 0:
            raise ValueError('No csv files set')
        if self.metrics is None:
            raise ValueError('No metrics set')
        if self.output is None:
            raise ValueError('No output path set')

        tab20_colors = [
            "#0f3855", 
            "#57e757",
            "#8c564b",
            "#918f8f",
            "#25bbcc",
        ]

        for metric in self.metrics:
            fig = plt.figure(figsize=(self.width, self.height))
            
            plt.xlabel('step', fontdict={
                'family': 'serif',
                'color': 'black',
                'weight': 'normal',
                'size': 20,
            })
            plt.ylabel(self._Ylabels[metric], fontdict={
                'family': 'serif',
                'color': 'black',
                'weight': 'normal',
                'size': 20,
            })

            # Encontrar el mejor desempeño (menor promedio)
            averages = [data[metric].mean() for data in self.df]
            min_index = averages.index(min(averages))

            # Graficar cada línea
            for i, data in enumerate(self.df):
                steps = data.get('step')[::10]
                values = data.get(metric)[::10]
                color = tab20_colors[i % len(tab20_colors)]

                # Resaltar la mejor línea
                if i == min_index:
                    rgb = mcolors.to_rgb(color)
                    dark_color = tuple([c * 0.8 for c in rgb])
                    plt.plot(steps, values, linestyle='-', linewidth=3, color=dark_color)
                else:
                    plt.plot(steps, values, linestyle='-', linewidth=0.8, color=color)

            plt.tight_layout()
            
            # Guardar gráfica
            out_path = os.path.join(self.output, out_folder) if out_folder else self.output
            os.makedirs(out_path, exist_ok=True)
            output_file = os.path.join(out_path, metric)
            fig.savefig(output_file, dpi=96)
            plt.close()

        # Limpiar datos
        self.csv_files = []
        self.df = []