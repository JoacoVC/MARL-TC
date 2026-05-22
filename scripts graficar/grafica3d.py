import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# 1. Cargar tus datos desde el archivo CSV
# Cambia 'mis_datos.csv' por el nombre real de tu archivo
df = pd.read_csv('output\\csv\\entrenamiento\\resumen_optimizacion_pesos1.csv')

# --- CÁLCULOS ESTADÍSTICOS ---
min_z = df['espera_media_global'].min()
max_z = df['espera_media_global'].max()
promedio_z = df['espera_media_global'].mean()

# Obtener las coordenadas (w1, w2) de los puntos mínimo y máximo
fila_min = df[df['espera_media_global'] == min_z].iloc[0]
fila_max = df[df['espera_media_global'] == max_z].iloc[0]

# Imprimir los resultados en la consola
print("="*40)
print(f"PUNTO MÍNIMO:")
print(f"  w1 = {fila_min['w1']}, w2 = {fila_min['w2']} -> Espera: {min_z:.3f}")
print(f"\nPUNTO MÁXIMO:")
print(f"  w1 = {fila_max['w1']}, w2 = {fila_max['w2']} -> Espera: {max_z:.3f}")
print(f"\nPROMEDIO TOTAL de espera global: {promedio_z:.3f}")
print("="*40)
# ------------------------------

# Extraer los valores para la gráfica
x_datos = df['w1'].values
y_datos = df['w2'].values
z_datos = df['espera_media_global'].values

# 2. Averiguar las dimensiones para reestructurar en una matriz 3D
puntos_x = len(np.unique(x_datos))  
puntos_y = len(np.unique(y_datos))  

# Reestructurar los datos planos a formato de matriz (Grid)
X = x_datos.reshape(puntos_x, puntos_y)
Y = y_datos.reshape(puntos_x, puntos_y)
Z = z_datos.reshape(puntos_x, puntos_y)

# 3. Configurar la figura en 3D
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection='3d')

# 4. Graficar la superficie sólida con cuadriláteros visibles
# Usamos 'cmap' para pintar las caras y 'edgecolor' para dibujar las líneas de la rejilla
superficie = ax.plot_surface(X, Y, Z, 
                             cmap=cm.jet,          # Color sólido según la altura
                             edgecolor='black',    # Color de las líneas de los cuadriláteros
                             linewidth=0.3,        # Grosor de las líneas de la rejilla
                             rstride=1, cstride=1, 
                             shade=True)           # Añade sombras suaves para resaltar el relieve

# 5. Dibujar los puntos mínimo y máximo flotando sobre la superficie
# Punto Mínimo (Verde)
ax.scatter(fila_min['w1'], fila_min['w2'], min_z, color='lime', s=180, edgecolor='black', label=f'Mínimo ({min_z:.1f})', zorder=10)
# Punto Máximo (Rojo)
ax.scatter(fila_max['w1'], fila_max['w2'], max_z, color='red', s=180, edgecolor='black', label=f'Máximo ({max_z:.1f})', zorder=10)

# 6. Personalizar etiquetas, títulos y leyenda
ax.set_xlabel('w1')
ax.set_ylabel('w2')
ax.set_zlabel('Espera Media Global')
ax.set_title(f'Análisis de Espera (Promedio Total: {promedio_z:.2f})', fontsize=12, pad=20)

# Colocar la leyenda
ax.legend(loc='upper left')

# Ajustar el ángulo de visión inicial para una mejor perspectiva
ax.view_init(elev=25, azim=-55)

# Añadir la barra de colores lateral
cbar = fig.colorbar(superficie, ax=ax, shrink=0.6, aspect=12, pad=0.1)
cbar.set_label('Tiempo de Espera')

# Mostrar la gráfica
plt.show()