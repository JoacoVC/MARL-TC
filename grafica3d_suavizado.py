import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata  # Para suavizar la superficie

# 1. Cargar tus datos
df = pd.read_csv('output\\csv\\entrenamiento\\resumen_optimizacion_pesos1.csv')
x_datos = df['w1'].values
y_datos = df['w2'].values
z_datos = df['espera_media_global'].values

# 2. CREAR UNA MALLA MÁS DENSA Y SUAVE (Interpolación)
# Creamos una cuadrícula mucho más fina para que el plano se vea liso
x_fino = np.linspace(x_datos.min(), x_datos.max(), 100)
y_fino = np.linspace(y_datos.min(), y_datos.max(), 100)
X_fino, Y_fino = np.meshgrid(x_fino, y_fino)

# Interpolamos los datos originales sobre la nueva malla fina
# 'cubic' hace que las curvas se vean completamente suaves
Z_fino = griddata((x_datos, y_datos), z_datos, (X_fino, Y_fino), method='cubic')

# 3. Configurar la figura en 3D
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection='3d')

# 4. Graficar como SUPERFICIE SÓLIDA
# Quitamos la transparencia y dejamos que el mapa de color rellene todo el plano
superficie = ax.plot_surface(X_fino, Y_fino, Z_fino, 
                             cmap=cm.jet,          # Mapa de color (azul a rojo)
                             linewidth=0,          # Quitamos las líneas negras internas
                             antialiased=True,     # Suaviza los bordes de los píxeles
                             rcount=100, ccount=100) # Definición de la superficie

# 5. Ajustar el ángulo de visión inicial para que se aprecie mejor el plano
ax.view_init(elev=30, azim=-60)

# 6. Etiquetas y Barra de color
ax.set_xlabel('w1')
ax.set_ylabel('w2')
ax.set_zlabel('Espera Media Global')
ax.set_title('Superficie Suavizada de Espera Media Global', fontsize=12, pad=20)

cbar = fig.colorbar(superficie, ax=ax, shrink=0.6, aspect=12, pad=0.1)
cbar.set_label('Tiempo de Espera')

plt.show()