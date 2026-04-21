from shiny import ui


def create_app_ui(df):
    fecha_min = df["fecha_hora"].min().date()
    fecha_max = df["fecha_hora"].max().date()

    return ui.page_fluid(
        ui.h2("Sistema de Visualización y Predicción Energética"),
        ui.p("Monitoreo histórico del sistema fotovoltaico y variables climáticas."),
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
                        ui.input_action_button("btn_actualizar_energia", "Actualizar"),
                    ),
                    ui.card(
                        ui.card_header("Tabla de datos energéticos"),
                        ui.div(
                            ui.output_table("tabla_datos_energia"),
                            style="overflow-y: auto; overflow-x: auto; max-height: 320px;",
                        ),
                    ),
                    ui.card(
                        ui.card_header("Serie temporal energética"),
                        ui.output_plot("grafica_energia"),
                    ),
                ),
            ),
            ui.nav_panel(
                "Clima",
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.h4("Filtros de clima"),
                        ui.input_date_range(
                            "rango_fechas_clima",
                            "Selecciona el rango de fechas:",
                            start=fecha_min,
                            end=fecha_max,
                        ),
                        ui.input_select(
                            "variable_clima",
                            "Variable climática:",
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
                    ),
                    ui.card(
                        ui.card_header("Tabla de datos climáticos"),
                        ui.div(
                            ui.output_table("tabla_datos_clima"),
                            style="overflow-y: auto; overflow-x: auto; max-height: 320px;",
                        ),
                    ),
                    ui.card(
                        ui.card_header("Serie temporal climática"),
                        ui.output_plot("grafica_clima"),
                    ),
                ),
            ),
            ui.nav_panel(
                "Predicción",
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.h4("Configuración de predicción"),
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
                        ui.input_numeric(
                            "tarifa_kwh",
                            "Tarifa estimada por kWh:",
                            value=1.25,
                            min=0.0,
                            step=0.01,
                        ),
                        ui.input_action_button("btn_generar_prediccion", "Generar predicción"),
                    ),
                    ui.layout_columns(
                        ui.value_box("Modelo", ui.output_text("txt_modelo_activo")),
                        ui.value_box("Horizonte", ui.output_text("txt_horizonte_pred")),
                        ui.value_box("Promedio predicho", ui.output_text("txt_promedio_pred")),
                        ui.value_box("Total estimado", ui.output_text("txt_total_pred")),
                    ),
                    ui.layout_columns(
                        ui.value_box("Tarifa", ui.output_text("txt_tarifa_activa")),
                        ui.value_box("Costo estimado", ui.output_text("txt_costo_estimado")),
                    ),
                    ui.card(
                        ui.card_header("Estado del modelo"),
                        ui.output_text_verbatim("txt_estado_modelo"),
                    ),
                    ui.card(
                        ui.card_header("Serie histórica y predicción"),
                        ui.output_plot("grafica_prediccion"),
                    ),
                    ui.card(
                        ui.card_header("Vista tabular de predicción"),
                        ui.div(
                            ui.output_table("tabla_prediccion"),
                            style="overflow-y: auto; overflow-x: auto; max-height: 320px;",
                        ),
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