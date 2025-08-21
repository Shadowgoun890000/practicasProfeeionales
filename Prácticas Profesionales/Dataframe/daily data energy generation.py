import os
import pandas as pd
import json
import pprint
from datetime import datetime

pp = pprint.PrettyPrinter(indent=4)


directorio = r"C:\Prácticas Profesionales\JSON"


dataframes = []

# Cargar cada archivo JSON en el directorio
for archivo in os.listdir(directorio):
    if archivo.endswith(".json"):
        ruta_archivo = os.path.join(directorio, archivo)
        # Leer archivo JSON
        with open(ruta_archivo, 'r') as f:
            data = json.load(f)

       
        pp.pprint(data)
        
        
        if 'invPacData' in data:
            
            invPacData = data['invPacData']
            invPacData_df = pd.DataFrame(list(invPacData.items()), columns=['fecha_hora', 'valor'])
            invPacData_df['fecha_hora'] = pd.to_datetime(invPacData_df['fecha_hora'], format='%Y-%m-%d %H:%M')
            
           
            invPacData_df = invPacData_df.sort_values(by='fecha_hora').reset_index(drop=True)
            
           
            invPacData_df['nominalPower'] = data.get('nominalPower', None)
            invPacData_df['eToday'] = data.get('eToday', None)
            invPacData_df['eTotal'] = data.get('eTotal', None)
            invPacData_df['exportLimit'] = data.get('exportLimit', None)
            invPacData_df['exportLimitPower2'] = data.get('exportLimitPower2', "0")
            invPacData_df['exportLimitPower'] = data.get('exportLimitPower', "0")
            invPacData_df['power'] = data.get('power', None)
            invPacData_df['dryContactStatus'] = data.get('dryContactStatus', 0)
            
            # Añadir el dataframe a la lista
            dataframes.append(invPacData_df)

# Concatenar todos los dataframes en uno solo
df_total = pd.concat(dataframes, ignore_index=True)

# Asegurar que la columna de fecha está en formato datetime
df_total['fecha_hora'] = pd.to_datetime(df_total['fecha_hora'], errors='coerce')

# Ordenar el dataframe por fecha
df_total = df_total.sort_values(by='fecha_hora')


ruta_csv = r"C:\Prácticas Profesionales\JSON\datos_growatt.csv"
df_total.to_csv(ruta_csv, index=False)


print(df_total.head())
