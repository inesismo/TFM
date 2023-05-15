# Dado un archivo csv con los datos del acelerómetro en los tres ejes, determinar el ciclo de la marcha, identificando sus picos y valles y separando en fase de apoyo y balanceo.
# Hay un notebook con el mismo nombre en Drive

import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks, peak_prominences

# Obtener la ruta del archivo
#file_path = "CSVFILES/mama1muslo_MetaWear_2020-07-25T17.58.14.662_DE982C9EB595_Accelerometer_100.000Hz_1.5.0.csv"
file_path = "DATA/Down_walk.csv"
# Importar el archivo
data = pd.read_csv(file_path)
#print(data.head())

# Mostrar en una figura las ters aceleraciones
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

ax1.plot(data['elapsed (s)'], data['x-axis (g)'])
ax1.set_ylabel('Aceleracion X')

ax2.plot(data['elapsed (s)'], data['y-axis (g)'])
ax2.set_ylabel('Aceleracion Y')

ax3.plot(data['elapsed (s)'], data['z-axis (g)'])
ax3.set_ylabel('Aceleracion Z')
ax3.set_xlabel('Tiempo (s)')

plt.show()
#*******************************************************
#*******************************************************
# FILTRO SAVGOL
# Definir los parámetros del filtro
window_size = 51
order = 3

# Aplicar el filtro a cada columna del DataFrame
data['aceleracion_x_filtered'] = savgol_filter(data['x-axis (g)'], window_size, order)
data['aceleracion_y_filtered'] = savgol_filter(data['y-axis (g)'], window_size, order)
data['aceleracion_z_filtered'] = savgol_filter(data['z-axis (g)'], window_size, order)

#*******************************************************
plt.figure(figsize=(10, 6))
plt.plot(data['elapsed (s)'], data['x-axis (g)'], label='Original')
plt.plot(data['elapsed (s)'], data['aceleracion_x_filtered'], label='Filtrado')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(data['elapsed (s)'], data['y-axis (g)'], label='Original')
plt.plot(data['elapsed (s)'], data['aceleracion_y_filtered'], label='Filtrado')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(data['elapsed (s)'], data['z-axis (g)'], label='Original')
plt.plot(data['elapsed (s)'], data['aceleracion_z_filtered'], label='Filtrado')
plt.legend()
plt.show()
#*******************************************************
# Encontrar picos y valles de la señal
x=data['aceleracion_x_filtered']
# Se definen unos límites para eliminar picos cercanos a la línea media
lim_max = 0.6*(np.max(x)-np.mean(x))+np.mean(x)
lim_min = np.abs(0.5*(np.min(x)-np.mean(x))+np.mean(x))
peaks, _ = find_peaks(x, height=lim_max)
valleys, _ = find_peaks(-x, height=lim_min)
num_pasos=len(peaks)
tiempo=np.max(data['elapsed (s)'])
print("Se han realizado un total de",num_pasos*2," pasos en un tiempo de ",tiempo,"segundos.")

plt.figure(figsize=(10, 6))
plt.plot(x, label='Señal filtrada')
plt.plot(peaks, x[peaks], "o",color='g', label='Picos')
plt.plot(valleys, x[valleys], "o",color='black', label='Valles')
plt.axhline(y=np.mean(x), color='r', linestyle='--', label='Valor medio')
plt.xlabel('Tiempo (s)')
plt.ylabel('Aceleración (g)')
plt.title('Aceleración en el eje vertical')
plt.legend()
plt.show()

# Otra forma de calcular los picos y filtrarlos por prominencia
peaks_prom, _ = find_peaks(x, prominence=(0.5, 2))
plt.figure(figsize=(10, 6))
plt.plot(x, label='Señal filtrada')
plt.plot(peaks_prom, x[peaks_prom], "o",color='g', label='Picos')
plt.plot(valleys, x[valleys], "o",color='black', label='Valles')
plt.axhline(y=np.mean(x), color='r', linestyle='--', label='Valor medio')
plt.xlabel('Tiempo (s)')
plt.ylabel('Aceleración (g)')
plt.title('Aceleración en el eje vertical, método prominence')
plt.legend()
plt.show()


# Calcular los picos de las dos fases
# La fase de apoyo empieza con un pico alto y la siguiente fase con el pico más bajo (En el eje vertical alineado con la gravedad)
peaks_stance = peaks_prom
# Para la fase de swing se eliminan primero los que están fuera de los ciclos de andar
peaks2,_=find_peaks(x)
peaks2_filt = peaks2[(peaks2 > peaks[0]) & (peaks2 < peaks[-1])]
# Luego se eliminan los picos altos
peaks_swing = peaks2_filt[(x[peaks2_filt] > np.mean(x)) & (x[peaks2_filt]<x[peaks].min())]

# Graficar la señal y resaltar los picos y valles
fig, ax = plt.subplots(figsize=(20, 6))
ax.plot(data['elapsed (s)'], data['aceleracion_x_filtered'], label='Señal original')
ax.plot(data['elapsed (s)'][peaks], data['aceleracion_x_filtered'][peaks], 'o', label='Picos stance', color= 'r')
ax.plot(data['elapsed (s)'][valleys], data['aceleracion_x_filtered'][valleys], 'o', label='Valles', color='g')
ax.plot(data['elapsed (s)'][peaks_swing], data['aceleracion_x_filtered'][peaks_swing], 'o', label='Picos swing', color='black')
ax.set_xlabel('Tiempo (s)')
ax.set_ylabel('Aceleración (m/s^2)')
ax.set_title('Picos y valles de la señal')
ax.legend()
plt.show()