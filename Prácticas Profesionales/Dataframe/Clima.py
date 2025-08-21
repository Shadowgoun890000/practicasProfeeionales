import json
import csv
import pandas as pd

# Ruta del archivo JSON
json_file_path = r"C:\Prácticas Profesionales\JSON\Clima.json"
# Ruta para el archivo CSV de salida
csv_file_path = r"C:\Prácticas Profesionales\JSON\Clima.csv"

# Leer el archivo JSON
with open(json_file_path, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# Asegurarse de que 'data' es un diccionario que contiene una lista en una clave
if 'estimated_actuals' in data:
    data = data['estimated_actuals']
else:
    raise ValueError("La clave 'estimated_actuals' no se encuentra en el JSON")

# Convertir los datos a un DataFrame de pandas para facilitar el manejo
df = pd.DataFrame(data)

# Verificar si 'period_end' está presente y convertirlo
if 'period_end' in df.columns:
    df['period_end'] = pd.to_datetime(df['period_end']).dt.strftime('%d/%m/%Y %I:%M:%S %p')

# Verificar si hay un campo con formato PT5M y convertirlo
def convertir_periodo(periodo):
    if periodo.startswith('PT'):
        # Extraer los minutos (si hay más de una parte como PT1H30M)
        partes = periodo[2:].split('M')
        minutos = 0
        for parte in partes:
            if 'H' in parte:
                horas = int(parte[:-1])  # Obtener el número de horas
                minutos += horas * 60  # Convertir horas a minutos
            elif parte.isdigit():
                minutos += int(parte)  # Obtener minutos
        return f"{minutos} min"
    return periodo  # Retornar el valor original si no está en el formato esperado

# Cambia 'duracion' por el nombre correcto de la columna en tu JSON
if 'duracion' in df.columns:
    df['duracion'] = df['duracion'].apply(convertir_periodo)

# Abrir un archivo CSV para escribir
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file)

    # Obtener las claves del DataFrame (cabeceras del CSV)
    headers = df.columns.tolist()
    # Escribir la fila de cabeceras
    csv_writer.writerow(headers)

    # Escribir los datos fila por fila
    for index, row in df.iterrows():
        csv_writer.writerow(row.values)

print(f"Datos extraídos con éxito a {csv_file_path}")
