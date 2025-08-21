import pandas as pd


consumo_df = pd.read_excel(r"C:\Prácticas Profesionales\JSON\ConsumoEnergético_Continuo.xlsx")
clima_df = pd.read_excel(r"C:\Prácticas Profesionales\JSON\Clima.xlsx")


consumo_df['fecha_hora'] = pd.to_datetime(consumo_df['fecha_hora'])
clima_df['period_end'] = pd.to_datetime(clima_df['period_end'])



merged_df = pd.merge(consumo_df, clima_df, left_on='fecha_hora', right_on='period_end', how='outer')


missing_samples = merged_df.isnull().sum()

resultado_df = merged_df.copy().dropna()


resultado_df.to_excel(r"C:\Prácticas Profesionales\JSON\Resultado_Homogenizado.xlsx", index=False)


print("Datos homogenizados y guardados en Resultado_Homogenizado.xlsx.")
print("Cantidad de muestras faltantes por columna:")
print(missing_samples)
