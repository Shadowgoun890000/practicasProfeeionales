import pandas as pd
import pdfquery
import os
from pdfquery import PDFQuery
from datetime import datetime

pd.set_option('display.max_columns', 10)

# Función para buscar texto en diferentes rangos de coordenadas
def buscar_texto_por_palabra_clave(pdf, palabras_clave):
    for palabra_clave in palabras_clave:
        texto = pdf.pq('LTTextLineHorizontal:contains("{0}")'.format(palabra_clave)).text()
        if texto:
            print(texto)
            return texto
        
    return ""

# Solicitar la ruta donde se encuentra el archivo
directorio = input("Ingrese la ruta donde se encuentra el archivo: ")

# Obtener la lista de archivos PDF en el directorio
archivos_pdf = [f for f in os.listdir(directorio) if f.endswith('.pdf')]

# DataFrame vacío para almacenar los resultados de todos los archivos
df_concatenado = pd.DataFrame()
home = input("Ingrese el identificador de la vivienda> ")

df = pd.DataFrame()
for archivo in archivos_pdf:
    # Construir la ruta completa del archivo
    ruta_archivo = os.path.join(directorio, archivo)
    
    print("El archivo " + archivo + " se está procesando.\n")
    pdf = PDFQuery(ruta_archivo)
    pdf.load()
    
    
    # Convertir el PDF a XML
    pdf.tree.write(directorio+'/ejemplo.xml', pretty_print=True)

    fotovoltaico = pdf.pq('LTTextLineHorizontal:in_bbox("148.96, 470.92, 194.308, 476.92")').text()
    palabras_clave_periodo = ["PERIODO FACTURADO:"]
    if "FOTOVOLTAICO" in fotovoltaico:
        
        # Extraer periodo facturado usando la función buscar_texto_en_rango
        periodo = buscar_texto_por_palabra_clave(pdf, palabras_clave_periodo)

        if "PERIODO FACTURADO:" in periodo:
            start = periodo[periodo.find(":")+2:periodo.find("-")-1].strip()
            end = periodo[periodo.find("-")+2:].strip()
            print(start + end)

        # lecturas medidor
        actual1 = pdf.pq('LTTextLineHorizontal:in_bbox("83.31, 311.59, 99.99, 317.59")').text()
        actual2 = pdf.pq('LTTextLineHorizontal:in_bbox("83.31, 301.81, 99.99, 307.81")').text()
        anterior1 = pdf.pq('LTTextLineHorizontal:in_bbox("136.42, 311.59, 153.1, 317.59")').text()
        anterior2 = pdf.pq('LTTextLineHorizontal:in_bbox("136.42, 301.81, 153.1, 307.81")').text()



        # niveles de consumo
        costo_basico = 0
        consumo_basico = 0
        if "Básico" in pdf.pq('LTTextLineHorizontal:in_bbox("25.96, 292.02, 47.626, 298.02")').text():
            costo_basico = pdf.pq('LTTextLineHorizontal:in_bbox("277.9, 292.02, 292.912, 298.02")').text()
            consumo_basico = pdf.pq('LTTextLineHorizontal:in_bbox("224.2, 292.02, 234.208, 298.02")').text()
        costo_intermedio = 0
        consumo_intermedio = 0
        if "Intermedio" in pdf.pq('LTTextLineHorizontal:in_bbox("25.96, 283.64, 57.97, 289.64")').text():
            costo_intermedio = pdf.pq('LTTextLineHorizontal:in_bbox("277.9, 283.64, 292.912, 289.64")').text()
            consumo_intermedio = pdf.pq('LTTextLineHorizontal:in_bbox("224.2, 283.64, 234.208, 289.64")').text()
        costo_excedente = 0
        consumo_excedente = 0
        if "Excedente" in pdf.pq('LTTextLineHorizontal:in_bbox("25.96, 275.25, 57.742, 281.25")').text():
            costo_excedente = pdf.pq('LTTextLineHorizontal:in_bbox("277.9, 275.25, 292.912, 281.25")').text()
            consumo_excedente = pdf.pq('LTTextLineHorizontal:in_bbox("224.2, 275.25, 234.208, 281.25")').text()
          
        # Asignar 0 si no hay datos
        #costo_basico = 0 if not costo_basico else costo_basico
        #costo_intermedio = 0 if not costo_intermedio else costo_intermedio
        #costo_excedente = 0 if not costo_excedente else costo_excedente
            
       # consumo_basico = pdf.pq('LTTextLineHorizontal:in_bbox("224.2, 292.02, 234.208, 298.02")').text()
        #consumo_intermedio = pdf.pq('LTTextLineHorizontal:in_bbox("224.2, 283.64, 234.208, 289.64")').text()
        #consumo_excedente = pdf.pq('LTTextLineHorizontal:in_bbox("224.2, 275.25, 234.208, 281.25")').text()
        
        #consumo_basico = 0 if not consumo_basico else consumo_basico
        #consumo_intermedio = 0 if not consumo_intermedio else consumo_intermedio
        #consumo_excedente = 0 if not consumo_excedente else consumo_excedente

        # Desglose de conceptos a pagar
        costo_energia = pdf.pq('LTTextLineHorizontal:in_bbox("356.46, 207.955, 373.974, 214.955")').text()
        iva = pdf.pq('LTTextLineHorizontal:in_bbox("360.35, 199.565, 373.972, 206.565")').text()
        fac_periodo = pdf.pq('LTTextLineHorizontal:in_bbox("356.46, 191.185, 373.974, 198.185")').text()
        adeudo_anterior = pdf.pq('LTTextLineHorizontal:in_bbox("356.46, 182.795, 373.974, 189.795")').text()
        su_pago = pdf.pq('LTTextLineHorizontal:in_bbox("353.74, 174.405, 373.977, 181.405")').text()
        total = pdf.pq('LTTextLineHorizontal:in_bbox("352.86, 165.738, 373.98, 173.738")').text()

        # Crear el DataFrame
        df = pd.DataFrame({'Id': [home, home],
                           'Fecha_inicio': [start, start],
                           'Fecha_fin': [end, end],
                           'Modalidad': ['Produccion', 'Consumo'],
                           'Lectura_Anterior': [anterior1, anterior2],
                           'Lectura_Actual': [actual1, actual2],
                           'KWh': [int(actual1)-int(anterior1), int(actual2)-int(anterior2)],
                           'KWh_basic': [consumo_basico, consumo_basico],
                           'Costo_basic': [costo_basico, costo_basico],
                           'Costo_energia': [costo_energia, costo_energia],
                           'Iva': [iva, iva],
                           'Fac_periodo': [fac_periodo, fac_periodo],
                           'Adeudo_Anterior': [adeudo_anterior, adeudo_anterior],
                           'Pago': [su_pago, su_pago],
                           'Total': [total, total]
                           })
        print("aji")
        print(df)

    else:
        print("Recibos antes de Dic 2021")

        # extraer periodo facturado
        periodo = buscar_texto_por_palabra_clave(pdf, palabras_clave_periodo)
        if periodo == "":
           periodo=pdf.pq('LTTextLineHorizontal:in_bbox("25.96, 357.71, 167.482, 363.734")').text()
        print(periodo)

        if "PERIODO FACTURADO:" in periodo:
            start = periodo[periodo.find(":")+2:periodo.find("-")-1].strip()
            end = periodo[periodo.find("-")+2:].strip()

        # lecturas medidor
        actual1 = pdf.pq('LTTextLineHorizontal:in_bbox("83.31, 319.98, 99.99, 325.98")').text()
        anterior1 = pdf.pq('LTTextLineHorizontal:in_bbox("136.42, 319.98, 153.1, 325.98")').text()

        # niveles de consumo
        consumo_basico = pdf.pq('LTTextLineHorizontal:in_bbox("224.2, 310.19, 234.208, 316.19")').text()
        costo_basico = pdf.pq('LTTextLineHorizontal:in_bbox("277.9, 310.19, 292.912, 316.19")').text()

        consumo_intermedio = pdf.pq('LTTextLineHorizontal:in_bbox("224.2, 301.81, 234.208, 307.81")').text()
        costo_intermedio = pdf.pq('LTTextLineHorizontal:in_bbox("277.9, 301.81, 292.912, 307.81")').text()

        consumo_excedente = pdf.pq('LTTextLineHorizontal:in_bbox("227.54, 293.42, 234.212, 299.42")').text()
        costo_excedente = pdf.pq('LTTextLineHorizontal:in_bbox("277.9, 293.42, 292.912, 299.42")').text()

        # Desglose de conceptos a pagar
        costo_energia = pdf.pq('LTTextLineHorizontal:in_bbox("349.78, 216.335, 371.186, 223.335")').text()
        iva = pdf.pq('LTTextLineHorizontal:in_bbox("353.67, 207.955, 371.184, 214.955")').text()
        fac_periodo = pdf.pq('LTTextLineHorizontal:in_bbox("349.78, 199.565, 371.186, 206.565")').text()
        adeudo_anterior = pdf.pq('LTTextLineHorizontal:in_bbox("349.78, 191.175, 371.186, 198.175")').text()
        su_pago = pdf.pq('LTTextLineHorizontal:in_bbox("347.05, 182.795, 371.179, 189.795")').text()
        total = pdf.pq('LTTextLineHorizontal:in_bbox("346.22, 174.118, 371.18, 182.118")').text()

        # Crear el DataFrame
        df = pd.DataFrame({'Id': [home],
                           'Fecha_inicio': [start],
                           'Fecha_fin': [end],
                           'Modalidad': ['Consumo'],
                           'Lectura_Anterior': [anterior1],
                           'Lectura_Actual': [actual1],
                           'KWh': [actual1 if actual1.isnumeric() else 0],
                           'KWh_basic': [consumo_basico],
                           'Costo_basic': [costo_basico],
                           'KWh_intermedio': [consumo_intermedio],
                           'Costo_intermedio': [costo_intermedio],
                           'KWh_excedente': [consumo_excedente],
                           'Costo_excendente': [costo_excedente],
                           'Costo_energia': [costo_energia],
                           'Iva': [iva],
                           'Fac_periodo': [fac_periodo],
                           'Adeudo_Anterior': [adeudo_anterior],
                           'Pago': [su_pago],
                           'Total': [total]
                           })
        print(df)

    df_concatenado = pd.concat([df_concatenado, df], ignore_index=True)

    
    N=input("PRESIONA UNA TECLA PARA CONTINUAR")

#Convert "Fecha_inicio" column to datetime format
# df_concatenado['Fecha_inicio'] = pd.to_datetime(df_concatenado['Fecha_inicio'], format="%d %b %y", errors='coerce')
# print("++++")
# print(df_concatenado)
    
# Sort DataFrame based on "Fecha_inicio" column
# df_concatenado = df_concatenado.sort_values(by='Fecha_inicio')

# Export the sorted DataFrame to a CSV file
print("---------------")
print(df_concatenado)
csv_output_path = os.path.join(directorio, 'Prueba_1.3.csv')
df_concatenado.to_csv(csv_output_path, index=False)

# Print the sorted DataFrame
print(df_concatenado)
