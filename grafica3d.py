import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def graficar_superficie_3d(datos_lista):
    # 1. Convertir la lista de diccionarios a DataFrame
    df = pd.DataFrame(datos_lista)

    # 2. Crear los ejes para la superficie
    # Necesitamos que w1 y w2 sean matrices (grids)
    w1_unique = sorted(df['w1'].unique())
    w2_unique = sorted(df['w2'].unique())
    X, Y = np.meshgrid(w2_unique, w1_unique)

    # 3. Organizar los valores Z (espera_media) en una matriz
    # Reestructuramos los datos para que coincidan con el grid
    Z = df.pivot(index='w1', columns='w2', values='espera_media').values

    # 4. Configurar la figura 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 5. Dibujar la superficie
    # Usamos el colormap 'viridis' o 'YlGnBu_r'
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis_r,
                           linewidth=0, antialiased=True, alpha=0.8)

    # 6. Etiquetas y estilo
    ax.set_xlabel('Peso w2 (Presión)')
    ax.set_ylabel('Peso w1 (Wait Time)')
    ax.set_zlabel('Espera Media (Segundos)')
    ax.set_title('Superficie de Optimización de Pesos')

    # Añadir barra de color
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # Ajustar el ángulo de visión para que se vea bien el relieve
    ax.view_init(elev=30, azim=45)

    plt.show()

# Ejemplo de cómo llamar a la función con tus datos:
# graficar_superficie_3d(matriz_resultados)