from shiny import ui
from shinywidgets import output_widget


def create_app_ui(df):
    fecha_min = df["fecha_hora"].min().date()
    fecha_max = df["fecha_hora"].max().date()

    return ui.page_fluid(

        ui.tags.link(rel="stylesheet", href="css/styles.css"),
        ui.tags.script(src="js/app.js"),

        ui.h2("Sistema de Visualización y Predicción Energética"),
        ui.p(
            "Consulta histórica del sistema fotovoltaico "
            "y variables meteorológicas."
        ),
        ui.navset_tab(
            ui.nav_panel(
                "Resumen",
                ui.layout_columns(
                    ui.value_box("Registros", ui.output_text("txt_total_registros")),
                    ui.value_box("Fecha inicial", ui.output_text("txt_fecha_min")),
                    ui.value_box("Fecha final", ui.output_text("txt_fecha_max")),
                    ui.value_box("Valor máximo", ui.output_text("txt_valor_max")),
                ),
            ),
            ui.nav_panel(
                "Histórico energético",
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.h4("Filtros de energía"),
                        ui.input_date_range(
                            "rango_fechas_energia",
                            "Selecciona el rango de fechas:",
                            start=fecha_min,
                            end=fecha_max,
                        ),
                        ui.input_slider(
                            "hora_rango_energia",
                            "Rango de hora:",
                            min=0,
                            max=23,
                            value=[0, 23],
                            step=1,
                        ),
                        ui.input_select(
                            "vista_energia",
                            "Vista de energía",
                            choices={
                                "completa": "Serie completa",
                                "promedio_diario": "Promedio diario",
                                "maximo_diario": "Máximo diario",
                                "ultimos_7": "Últimos 7 días",
                                "ultimos_30": "Últimos 30 días",
                            },

                            selected="completa",
                        ),
                        ui.input_action_button("btn_actualizar_energia", "Actualizar"),
                        ui.download_button("descargar_energia_csv", "Descargar CSV"),
                    ),
                    ui.card(
                        ui.card_header("Tabla de datos energéticos"),
                            ui.output_data_frame("tabla_datos_energia"),
                    ),
                    ui.card(
                        ui.card_header("Serie temporal energética"),
                        output_widget("grafica_energia"),
                    ),
                    ui.card(
                        ui.card_header("Resumen diario de energía"),
                        ui.output_plot("grafica_energia_diaria"),
                    ),
                ),
            ),
            ui.nav_panel(
                "Clima",
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.h4("Filtros meteorológicos"),
                        ui.input_date_range(
                            "rango_fechas_clima",
                            "Selecciona el rango de fechas:",
                            start=fecha_min,
                            end=fecha_max,
                        ),
                        ui.input_select(
                            "variable_clima",
                            "Variable meteorológica:",
                            choices=[
                                "air_temp",
                                "relative_humidity",
                                "ghi",
                                "dni",
                                "gti",
                                "wind_speed_10m",
                                "wind_direction_10m",
                            ],
                            selected="air_temp",
                        ),
                        ui.input_action_button("btn_actualizar_clima", "Actualizar"),
                        ui.download_button("descargar_clima_csv", "Descargar CSV"),
                    ),
                    ui.card(
                        ui.card_header("Tabla de datos meteorológicos"),
                        ui.output_data_frame("tabla_datos_clima"),
                    ),
                    ui.card(
                        ui.card_header("Serie temporal meteorológica"),
                        output_widget("grafica_clima"),
                    ),
                ),
            ),
            ui.nav_panel(
                "Predicción",
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.h4("Configuración de predicción"),
                        ui.input_file(
                            "archivo_meteo_futuro",
                            "Datos meteorológicos futuros:",
                            accept=[
                                ".csv",
                                ".xlsx",
                                ".xls",
                            ],
                            multiple=False,
                        ),
                        ui.input_numeric(
                            "dias_prediccion",
                            "Horizonte (días):",
                            value=60,
                            min=1,
                            max=365,
                            step=1,
                        ),
                        ui.input_numeric(
                            "dias_historia_pred",
                            "Histórico a mostrar (días):",
                            value=30,
                            min=1,
                            max=180,
                            step=1,
                        ),
                        ui.input_action_button("btn_generar_prediccion", "Generar predicción"),
                        ui.download_button("descargar_prediccion_csv", "Descargar CSV"),
                    ),
                    ui.layout_columns(
                        ui.value_box("Modelo", ui.output_text("txt_modelo_activo")),
                        ui.value_box("Horizonte", ui.output_text("txt_horizonte_pred")),
                        ui.value_box("Potencia promedio predicha", ui.output_text("txt_promedio_pred")),
                        ui.value_box("Energía estimada del periodo", ui.output_text("txt_total_pred")),
                    ),
                    ui.card(
                        ui.card_header("Estado del modelo"),
                        ui.output_text_verbatim("txt_estado_modelo"),
                    ),
                    ui.card(
                        ui.card_header("Serie histórica y predicción"),
                        output_widget("grafica_prediccion"),
                    ),
                    ui.card(
                        ui.card_header("Predicción resumida por día"),
                        ui.output_plot("grafica_prediccion_diaria"),
                    ),
                    ui.card(
                        ui.card_header("Vista tabular de predicción"),
                        ui.output_data_frame("tabla_prediccion"),

                    ),
                ),
            ),
            ui.nav_panel(
                "Acerca de",
                ui.markdown(
                    """
                    ### Acerca de la aplicación

                    Esta aplicación permite visualizar el historial energético del sistema y
                    las variables climáticas asociadas. La pestaña de predicción integra
                    un modelo de Machine Learning orientado a pronóstico energético.
                    """
                ),
            ),
            id="tab_principal",
        ),
    )