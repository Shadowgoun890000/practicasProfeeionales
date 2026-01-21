import pandas as pd
from shiny import App, render, ui, reactive

# Carga de datos
ruta_archivo = r"/home/to-o/practicasProfeeionales/Prácticas Profesionales/JSON/Resultado_Homogenizado.xlsx"
df = pd.read_excel(ruta_archivo)

# Filtrar columnas donde todos los valores no son 0
df = df.loc[:, (df != 0).any(axis=0)]

# Convertir la columna 'fecha_hora' a tipo datetime
df['fecha_hora'] = pd.to_datetime(df['fecha_hora'])


columnas_consumo = [
    "fecha_hora", "valor (kWh)", "nominalPower (W)", "eToday (kWh)", "eTotal (kWh)", "power (kW)", 
]
columnas_clima = [
     "fecha_hora", "air_temp","albedo","azimuth","cloud_opacity",
     "dhi", "dni", "ghi", "gti", "precipitable_water",
     "relative_humidity", "surface_pressure", "wind_direction_100m",
     "wind_direction_10m", "wind_speed_100m", "wind_speed_10m","wind_gust",
]

# Definición de la interfaz de usuario (UI)
app_ui = ui.page_fluid(
    ui.navset_tab(
        # Pestaña para Consumo Energético
        ui.nav_panel(
            "Consumo Energético",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h2("Opciones de Filtrado"),
                    ui.input_date_range(
                        "rango_fechas", "Selecciona el rango de fechas:",
                        start=df['fecha_hora'].min().date(),
                        end=df['fecha_hora'].max().date(),
                    ),
                    ui.input_slider("hora_rango", "Rango de Hora:", min=0, max=23, value=[0, 23], step=1),
                    ui.input_action_button("btn_actualizar", "Actualizar", style="background-color: #6495ED; color: black"),
                    style="background-color: #F0FFFF;"
                ),
                ui.div(
                    ui.h2("Visualización de Consumo Energético"),
                    ui.div(
                        ui.output_table("tabla_datos_consumo"),
                        style="overflow-y: auto; max-height: 300px; border: 1px solid #A9A9A9;"
                    ),
                    ui.output_plot("grafica_energia")
                )
            )
        ),
        
        ui.nav_menu(
            "Otros enlaces",
            ui.nav_panel("Acerca de", "Información sobre la aplicación"),
            "----",
            "Descripción:",
            ui.nav_control(
                ui.a("Shiny", href="https://shiny.posit.co", target="_blank")
            )
        ),
        id="tab"
    )
)

# Definición de la lógica del servidor
def server(input, output, session):

    # Filtrar datos de consumo energético
    @reactive.event(input.btn_actualizar)
    def datos_filtrados_consumo():
        df_consumo = df[columnas_consumo].copy()
        df_consumo = df_consumo[
            (df_consumo['fecha_hora'].dt.date >= input.rango_fechas()[0]) & 
            (df_consumo['fecha_hora'].dt.date <= input.rango_fechas()[1])
        ]
        df_consumo = df_consumo[
            (df_consumo['fecha_hora'].dt.hour >= input.hora_rango()[0]) & 
            (df_consumo['fecha_hora'].dt.hour <= input.hora_rango()[1])
        ]
        return df_consumo

    # Filtrar datos climatológicos
    @reactive.event(input.rango_fechas_clima)
    def datos_filtrados_clima():
        df_clima = df[columnas_clima].copy()
        df_clima = df_clima[
            (df['fecha_hora'].dt.date >= input.rango_fechas_clima()[0]) & 
            (df['fecha_hora'].dt.date <= input.rango_fechas_clima()[1])
        ]
        return df_clima.head(100)  # Limitamos a 100 filas para rendimiento

    # Tabla de datos de consumo
    @output
    @render.table
    def tabla_datos_consumo():
        return datos_filtrados_consumo()

    # Gráfica de energía para consumo
    @output
    @render.plot
    def grafica_energia():
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        df_consumo = datos_filtrados_consumo()
        fig, ax = plt.subplots()
        ax.plot(df_consumo['fecha_hora'], df_consumo['valor (kWh)'], label='Valor (kWh)')
        ax.set_xlabel('Fecha y Hora')
        ax.set_ylabel('Valor (kWh)')
        ax.set_title('Consumo Energético a lo largo del tiempo')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig

# Creación de la aplicación
app = App(app_ui, server)

# Ejecución de la aplicación
if __name__ == "__main__":
    app.run()