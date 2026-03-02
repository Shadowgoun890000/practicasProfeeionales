import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.dates as mdates

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Cargar los datos
file_path = r"/home/to-o/practicasProfeeionales/Prácticas Profesionales/JSON/Resultado_Homogenizado.xlsx"
df = pd.read_excel(file_path)

# Verificar las primeras filas
print("Columnas disponibles:", df.columns.tolist())
print("\nPeriodo de datos:", df['fecha_hora'].min(), "a", df['fecha_hora'].max())
print("Frecuencia de muestreo:", df['period'].iloc[0], "minutos")

# Asegurar que fecha_hora sea datetime y ordenar
df['fecha_hora'] = pd.to_datetime(df['fecha_hora'])
df = df.sort_values('fecha_hora').reset_index(drop=True)

# ANÁLISIS INICIAL DE LOS DATOS
print("\n" + "=" * 60)
print("ANÁLISIS INICIAL DE LOS DATOS")
print("=" * 60)

# Ver estadísticas básicas de las columnas de energía
print("\nEstadísticas de columnas de energía:")
print("1. 'valor':")
print(f"   - Rango: {df['valor (kWh)'].min():.2f} a {df['valor (kWh)'].max():.2f}")
print(f"   - Media: {df['valor (kWh)'].mean():.2f}")
print(f"   - % de ceros: {(df['valor (kWh)'] == 0).mean() * 100:.1f}%")

print("\n2. 'power (kW)':")
print(f"   - Rango: {df['power (kW)'].min():.2f} a {df['power (kW)'].max():.2f}")
print(f"   - Media: {df['power (kW)'].mean():.2f}")

# Crear columnas útiles para el análisis
df['hora'] = df['fecha_hora'].dt.hour
df['dia_semana'] = df['fecha_hora'].dt.dayofweek  # 0=Lunes, 6=Domingo
df['fecha'] = df['fecha_hora'].dt.date
df['mes'] = df['fecha_hora'].dt.month

# PRUEBA 1: Asumir que 'valor' está en Watts (W) y es potencia instantánea
# En este caso, simplemente convertimos a kW dividiendo entre 1000
df['potencia_kW_test1'] = df['valor (kWh)'] / 1000

# PRUEBA 2: Asumir que 'valor' está en Wh y calcular potencia
# Intervalo de 5 minutos = 5/60 = 0.08333 horas
intervalo_horas = df['period'].iloc[0] / 60
df['potencia_kW_test2'] = df['valor (kWh)'] / 1000 / intervalo_horas

print("\n" + "=" * 60)
print("COMPARACIÓN DE DIFERENTES INTERPRETACIONES")
print("=" * 60)

print("\nPrueba 1: Asumiendo 'valor' en W (potencia instantánea):")
print(f"   - Mínimo: {df['potencia_kW_test1'].min():.2f} kW")
print(f"   - Máximo: {df['potencia_kW_test1'].max():.2f} kW")
print(f"   - Media: {df['potencia_kW_test1'].mean():.2f} kW")
print(f"   - % valores < 1 kW: {(df['potencia_kW_test1'] < 1).mean() * 100:.1f}%")

print("\nPrueba 2: Asumiendo 'valor' en Wh (energía en intervalo):")
print(f"   - Mínimo: {df['potencia_kW_test2'].min():.2f} kW")
print(f"   - Máximo: {df['potencia_kW_test2'].max():.2f} kW")
print(f"   - Media: {df['potencia_kW_test2'].mean():.2f} kW")
print(f"   - % valores < 1 kW: {(df['potencia_kW_test2'] < 1).mean() * 100:.1f}%")

# Basado en el comportamiento esperado de un sistema solar, durante la noche deberíamos tener valores cercanos a 0
# Vamos a analizar qué interpretación tiene valores más cercanos a 0 durante la noche
horas_noche = df[(df['hora'] >= 18) | (df['hora'] <= 5)]

print("\nComportamiento nocturno (18:00-5:59):")
print("Prueba 1 (valor en W):")
print(f"   - Media nocturna: {horas_noche['potencia_kW_test1'].mean():.2f} kW")
print(f"   - % < 0.1 kW: {(horas_noche['potencia_kW_test1'] < 0.1).mean() * 100:.1f}%")

print("\nPrueba 2 (valor en Wh):")
print(f"   - Media nocturna: {horas_noche['potencia_kW_test2'].mean():.2f} kW")
print(f"   - % < 0.1 kW: {(horas_noche['potencia_kW_test2'] < 0.1).mean() * 100:.1f}%")

# Seleccionar la interpretación que tenga valores nocturnos más bajos
if horas_noche['potencia_kW_test1'].mean() < horas_noche['potencia_kW_test2'].mean():
    df['potencia_kW'] = df['potencia_kW_test1']
    print("\n✓ Seleccionada Prueba 1: 'valor' interpretado como potencia en W")
    factor = 1 / 1000
else:
    df['potencia_kW'] = df['potencia_kW_test2']
    print("\n✓ Seleccionada Prueba 2: 'valor' interpretado como energía en Wh")
    factor = 1 / (1000 * intervalo_horas)

# 1. GRÁFICA PRINCIPAL: SERIE TEMPORAL
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

# Serie temporal de los últimos 3 días
ultimos_dias = 3
fecha_limite = df['fecha_hora'].max() - timedelta(days=ultimos_dias)
datos_recientes = df[df['fecha_hora'] >= fecha_limite]

ax1.plot(datos_recientes['fecha_hora'], datos_recientes['potencia_kW'],
         linewidth=1.5, color='steelblue', alpha=0.8)
ax1.set_xlabel('Fecha y Hora', fontsize=12)
ax1.set_ylabel('Potencia (kW)', fontsize=12)
ax1.set_title(f'Serie Temporal de Potencia Generada (Últimos {ultimos_dias} días)', fontsize=14, pad=20)
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m\n%H:%M'))
ax1.tick_params(axis='x', rotation=0)

# Resaltar patrones día/noche
for fecha_unica in datos_recientes['fecha'].unique():
    # Horas de noche (18:00-5:59)
    for hora_inicio, hora_fin in [(18, 24), (0, 6)]:
        mask = (datos_recientes['fecha'] == fecha_unica) & \
               (datos_recientes['hora'] >= hora_inicio) & \
               (datos_recientes['hora'] < hora_fin)
        if mask.any():
            ax1.fill_between(datos_recientes.loc[mask, 'fecha_hora'],
                             0, datos_recientes.loc[mask, 'potencia_kW'].max(),
                             alpha=0.1, color='navy')

ax1.legend(['Potencia generada', 'Horario nocturno'], loc='upper right')

# 2. PATRÓN DIARIO PROMEDIO
potencia_por_hora = df.groupby('hora')['potencia_kW'].agg(['mean', 'std']).reset_index()

ax2.bar(potencia_por_hora['hora'], potencia_por_hora['mean'],
        yerr=potencia_por_hora['std'],
        capsize=5, alpha=0.7, color='coral', edgecolor='darkred')
ax2.set_xlabel('Hora del Día', fontsize=12)
ax2.set_ylabel('Potencia Promedio (kW)', fontsize=12)
ax2.set_title('Patrón Diario de Generación (Promedio por Hora)', fontsize=14, pad=20)
ax2.set_xticks(range(0, 24, 2))
ax2.grid(True, alpha=0.3, axis='y')

# Resaltar horas diurnas
ax2.axvspan(6, 18, alpha=0.1, color='gold', label='Horas diurnas (6:00-18:00)')
ax2.legend()

plt.tight_layout()
plt.show()

# 3. ANÁLISIS COMPLETO PARA LA TESIS
print("\n" + "=" * 60)
print("ANÁLISIS TEMPORAL COMPLETO PARA LA TESIS")
print("=" * 60)

# Estadísticas por periodo del día
horas_noche = df[(df['hora'] >= 18) | (df['hora'] <= 5)]
horas_dia = df[(df['hora'] > 5) & (df['hora'] < 18)]

print(f"\n1. COMPORTAMIENTO DIURNO/NOCTURNO:")
print(f"   Horas nocturnas (18:00-5:59):")
print(f"     - Registros: {len(horas_noche):,}")
print(f"     - Mínimo: {horas_noche['potencia_kW'].min():.3f} kW")
print(f"     - Promedio: {horas_noche['potencia_kW'].mean():.3f} kW")
print(f"     - Máximo: {horas_noche['potencia_kW'].max():.3f} kW")
print(f"     - % de valores < 0.1 kW: {(horas_noche['potencia_kW'] < 0.1).mean() * 100:.1f}%")

print(f"\n   Horas diurnas (6:00-17:59):")
print(f"     - Registros: {len(horas_dia):,}")
print(f"     - Mínimo: {horas_dia['potencia_kW'].min():.3f} kW")
print(f"     - Promedio: {horas_dia['potencia_kW'].mean():.3f} kW")
print(f"     - Máximo: {horas_dia['potencia_kW'].max():.3f} kW")
print(f"     - Hora de máxima generación: {potencia_por_hora.loc[potencia_por_hora['mean'].idxmax(), 'hora']}:00")

# Calcular amplitud diaria
amplitud_diaria = df.groupby('fecha')['potencia_kW'].agg(['min', 'max'])
amplitud_diaria['amplitud'] = amplitud_diaria['max'] - amplitud_diaria['min']

print(f"\n2. ANÁLISIS DE AMPLITUD DIARIA:")
print(f"   - Días analizados: {len(amplitud_diaria)}")
print(f"   - Amplitud promedio: {amplitud_diaria['amplitud'].mean():.2f} kW")
print(f"   - Amplitud máxima: {amplitud_diaria['amplitud'].max():.2f} kW")
print(f"   - Amplitud mínima: {amplitud_diaria['amplitud'].min():.2f} kW")

if amplitud_diaria['amplitud'].mean() > 0:
    cv = (amplitud_diaria['amplitud'].std() / amplitud_diaria['amplitud'].mean()) * 100
    print(f"   - Coeficiente de variación: {cv:.1f}%")

# 4. GRÁFICA PARA LA TESIS (MUESTRA REPRESENTATIVA)
fig2, ax3 = plt.subplots(figsize=(14, 6))

# Buscar 2 días consecutivos con buen patrón solar
df['max_diario'] = df.groupby('fecha')['potencia_kW'].transform('max')
dias_con_patron = df[df['max_diario'] > df['max_diario'].quantile(0.75)]  # Días con generación alta

if len(dias_con_patron['fecha'].unique()) >= 2:
    # Tomar los primeros 2 días consecutivos
    fechas_unicas = sorted(dias_con_patron['fecha'].unique())

    for i in range(len(fechas_unicas) - 1):
        if (fechas_unicas[i + 1] - fechas_unicas[i]).days == 1:
            dia_inicio = fechas_unicas[i]
            dia_fin = fechas_unicas[i + 1]
            break
    else:
        dia_inicio = fechas_unicas[0]
        dia_fin = fechas_unicas[0] + timedelta(days=1)

    # Filtrar los 2 días
    fecha_inicio_dt = pd.Timestamp(dia_inicio)
    fecha_fin_dt = pd.Timestamp(dia_fin) + timedelta(days=1)

    muestra = df[(df['fecha_hora'] >= fecha_inicio_dt) &
                 (df['fecha_hora'] < fecha_fin_dt)].copy()

    if len(muestra) > 0:
        # Graficar
        ax3.plot(muestra['fecha_hora'], muestra['potencia_kW'],
                 linewidth=2, color='#2E86AB', label='Potencia generada')

        # Resaltar áreas día/noche
        current_date = fecha_inicio_dt
        while current_date < fecha_fin_dt:
            # Noche (18:00-5:59)
            noche_inicio = current_date.replace(hour=18, minute=0, second=0)
            noche_fin = current_date + timedelta(days=1)
            noche_fin = noche_fin.replace(hour=6, minute=0, second=0)

            ax3.axvspan(max(noche_inicio, fecha_inicio_dt),
                        min(noche_fin, fecha_fin_dt),
                        alpha=0.1, color='navy',
                        label='Noche' if current_date == fecha_inicio_dt else '')

            current_date += timedelta(days=1)

        ax3.set_xlabel('Tiempo', fontsize=14)
        ax3.set_ylabel('Potencia (kW)', fontsize=14)
        ax3.set_title('Serie Temporal de la Potencia Generada (muestra representativa)',
                      fontsize=16, pad=20)
        ax3.grid(True, alpha=0.3, linestyle='--')

        # Formatear eje x
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m\n%H:%M'))
        plt.xticks(rotation=0)

        # Mostrar leyenda
        handles, labels = ax3.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax3.legend(by_label.values(), by_label.keys(), fontsize=12, loc='upper right')

        plt.tight_layout()
        plt.show()
    else:
        print("\nAdvertencia: No hay suficientes datos para la muestra representativa.")
else:
    print("\nAdvertencia: No se encontraron días con patrón solar claro.")

# 5. ANÁLISIS ADICIONAL: DISTRIBUCIÓN, DÍAS DE SEMANA, ETC.
fig3, ((ax4, ax5), (ax6, ax7)) = plt.subplots(2, 2, figsize=(16, 12))

# 5a. Distribución de la potencia
ax4.hist(df['potencia_kW'].dropna(), bins=50, edgecolor='black', alpha=0.7, color='lightseagreen')
ax4.set_xlabel('Potencia (kW)', fontsize=12)
ax4.set_ylabel('Frecuencia', fontsize=12)
ax4.set_title('Distribución de Valores de Potencia', fontsize=14)
ax4.grid(True, alpha=0.3)

# 5b. Potencia por día de la semana
dias_semana = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
potencia_por_dia_semana = df.groupby('dia_semana')['potencia_kW'].agg(['mean', 'std']).reset_index()

ax5.bar(range(7), potencia_por_dia_semana['mean'],
        yerr=potencia_por_dia_semana['std'], capsize=8,
        color=plt.cm.Set3(range(7)), edgecolor='black')
ax5.set_xlabel('Día de la Semana', fontsize=12)
ax5.set_ylabel('Potencia Promedio (kW)', fontsize=12)
ax5.set_title('Generación por Día de la Semana', fontsize=14)
ax5.set_xticks(range(7))
ax5.set_xticklabels(dias_semana, rotation=45)
ax5.grid(True, alpha=0.3, axis='y')

# 5c. Serie temporal diaria (resampleada)
df_diario = df.set_index('fecha_hora')['potencia_kW'].resample('D').mean()

ax6.plot(df_diario.index, df_diario.values, linewidth=2, color='darkgreen', alpha=0.8)
ax6.set_xlabel('Fecha', fontsize=12)
ax6.set_ylabel('Potencia Promedio Diaria (kW)', fontsize=12)
ax6.set_title('Tendencia Temporal (Resample Diario)', fontsize=14)
ax6.grid(True, alpha=0.3)
ax6.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
ax6.tick_params(axis='x', rotation=45)

# 5d. Boxplot por hora
boxplot_data = [df[df['hora'] == hora]['potencia_kW'].dropna() for hora in range(24)]
ax7.boxplot(boxplot_data, positions=range(24), widths=0.6,
            patch_artist=True,
            boxprops=dict(facecolor='lightblue', color='darkblue'),
            medianprops=dict(color='red'))
ax7.set_xlabel('Hora del Día', fontsize=12)
ax7.set_ylabel('Potencia (kW)', fontsize=12)
ax7.set_title('Variabilidad por Hora', fontsize=14)
ax7.set_xticks(range(0, 24, 2))
ax7.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# 6. RESUMEN Y CONCLUSIONES
print("\n" + "=" * 60)
print("RESUMEN Y CONCLUSIONES PARA LA TESIS")
print("=" * 60)

print(f"""
PERIODO ANALIZADO: {df['fecha_hora'].min().strftime('%d/%m/%Y')} - {df['fecha_hora'].max().strftime('%d/%m/%Y')}

HALLAZGOS PRINCIPALES:

1. PATRÓN DIARIO MARCADO:
   - Generación nocturna: {horas_noche['potencia_kW'].mean():.3f} kW (promedio)
   - Generación diurna: {horas_dia['potencia_kW'].mean():.3f} kW (promedio)
   - Relación día/noche: {horas_dia['potencia_kW'].mean() / max(horas_noche['potencia_kW'].mean(), 0.001):.1f} veces mayor durante el día
   - Hora pico: {potencia_por_hora.loc[potencia_por_hora['mean'].idxmax(), 'hora']}:00 h

2. NO ESTACIONARIEDAD:
   - Media variable: {horas_dia['potencia_kW'].mean() - horas_noche['potencia_kW'].mean():.2f} kW de diferencia día/noche
   - Varianza no constante: Amplitud diaria variable ({amplitud_diaria['amplitud'].mean():.2f} ± {amplitud_diaria['amplitud'].std():.2f} kW)
   - Patrón estacional diario claramente identificable

3. VARIABILIDAD METEOROLÓGICA:
   - Coeficiente de variación de amplitud: {cv:.1f}% (indica días con diferente insolación)
   - Máximos diarios variables: {amplitud_diaria['max'].min():.2f} a {amplitud_diaria['max'].max():.2f} kW

CONCLUSIÓN:
La serie temporal de potencia generada presenta clara no estacionariedad con componente 
estacional diario, validando la necesidad de utilizar modelos específicos para series 
temporales no estacionarias en el análisis predictivo.
""")

# 7. GUARDAR GRÁFICAS
print("\nGuardando gráficas para la tesis...")
fig1.savefig('serie_temporal_patron_diario.png', dpi=300, bbox_inches='tight')
fig2.savefig('serie_temporal_tesis.png', dpi=300, bbox_inches='tight')
fig3.savefig('analisis_completo_estacionalidad.png', dpi=300, bbox_inches='tight')
print("✓ Gráficas guardadas exitosamente!")

# 8. GUARDAR DATOS PROCESADOS PARA USO POSTERIOR
df_procesado = df[['fecha_hora', 'potencia_kW', 'hora', 'dia_semana', 'fecha', 'mes']].copy()
df_procesado.to_csv('datos_potencia_procesados.csv', index=False)
print("✓ Datos procesados guardados en 'datos_potencia_procesados.csv'")