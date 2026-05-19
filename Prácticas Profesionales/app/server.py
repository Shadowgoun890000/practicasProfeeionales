import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

import plotly.graph_objects as go
from shinywidgets import render_widget
from shiny import reactive, render

from app.config import (
    COLUMNAS_CLIMA,
    COLUMNAS_ENERGIA,
    DATETIME_COLUMN,
    DEFAULT_PREDICTION_DAYS,
    TARGET_COLUMN,
)
from ml.forecast import forecast_with_model


def create_server(df, model=None, model_name="Random Forest"):
    def server(input, output, session):
        columnas_energia_disponibles = [c for c in COLUMNAS_ENERGIA if c in df.columns]
        columnas_clima_disponibles = [c for c in COLUMNAS_CLIMA if c in df.columns]

        df_energia_base = df[columnas_energia_disponibles].copy()
        df_clima_base = df[columnas_clima_disponibles].copy()
        df_target_base = df[[DATETIME_COLUMN, TARGET_COLUMN]].copy().sort_values(DATETIME_COLUMN)

        # Columnas Auxiliares precalculadas para filtros
        df_energia_base["_fecha"]  = df_energia_base[DATETIME_COLUMN].dt.date
        df_energia_base["_hora"]   = df_energia_base[DATETIME_COLUMN].dt.hour
        df_clima_base["_fecha"]    = df_clima_base[DATETIME_COLUMN].dt.date

        energia_actual = reactive.Value(df_energia_base.copy())
        clima_actual = reactive.Value(df_clima_base.copy())
        prediccion_actual = reactive.Value(pd.DataFrame())

        # =========================
        # Lectura segura del artefacto
        # =========================
        artifact = model if isinstance(model, dict) else None
        rf_model = artifact.get("model") if artifact else None
        artifact_features = artifact.get("features", []) if artifact else []
        artifact_target = artifact.get("target", TARGET_COLUMN) if artifact else TARGET_COLUMN
        artifact_datetime = artifact.get("datetime_column", DATETIME_COLUMN) if artifact else DATETIME_COLUMN
        artifact_metrics_mean = artifact.get("metrics_mean", {}) if artifact else {}
        artifact_metrics_std = artifact.get("metrics_std", {}) if artifact else {}

        # =========================
        # Funciones auxiliares
        # =========================
        def filtrar_energia():
            inicio, fin = input.rango_fechas_energia()
            hora_inicio, hora_fin = input.hora_rango_energia()

            filtrado = df_energia_base[
                (df_energia_base["_fecha"] >= inicio)
                & (df_energia_base["_fecha"] <= fin)
                & (df_energia_base["_hora"] >= hora_inicio)
                & (df_energia_base["_hora"] <= hora_fin)
            ]
            return filtrado.drop(columns=["_fecha", "_hora"], errors="ignore")

        def filtrar_clima():
            inicio, fin = input.rango_fechas_clima()

            filtrado = df_clima_base[
                (df_clima_base["_fecha"] >= inicio)
                & (df_clima_base["_fecha"] <= fin)
            ]

            return filtrado.drop(columns=["_fecha"], errors="ignore")

        def preparar_vista_energia(df_energia: pd.DataFrame) -> pd.DataFrame:
            vista = input.vista_energia() or "completa"
            df_vista = df_energia.copy()

            if df_vista.empty or TARGET_COLUMN not in df_vista.columns:
                return df_vista

            if vista == "ultimos_7":
                fecha_max = df_vista[DATETIME_COLUMN].max()
                df_vista = df_vista[
                    df_vista[DATETIME_COLUMN] >= fecha_max - pd.Timedelta(days=7)
                    ]
                return df_vista

            if vista == "ultimos_30":
                fecha_max = df_vista[DATETIME_COLUMN].max()
                df_vista = df_vista[
                    df_vista[DATETIME_COLUMN] >= fecha_max - pd.Timedelta(days=30)
                    ]
                return df_vista

            df_vista["fecha"] = df_vista[DATETIME_COLUMN].dt.date

            if vista == "promedio_diario":
                return (
                    df_vista.groupby("fecha", as_index=False)[TARGET_COLUMN]
                    .mean()
                    .rename(columns={TARGET_COLUMN: "valor"})
                )

            if vista == "maximo_diario":
                return (
                    df_vista.groupby("fecha", as_index=False)[TARGET_COLUMN]
                    .max()
                    .rename(columns={TARGET_COLUMN: "valor"})
                )

            return df_vista

        def generar_prediccion_provisional(dias: int) -> pd.DataFrame:
            base = df_target_base
            ultimo_timestamp = base[DATETIME_COLUMN].iloc[-1]

            pasos_por_dia = 288  # 5 minutos
            total_pasos = dias * pasos_por_dia

            patron = base[TARGET_COLUMN].tail(total_pasos).copy()

            if len(patron) < total_pasos:
                repeticiones = (total_pasos // max(len(patron), 1)) + 1
                patron = pd.concat([patron] * repeticiones, ignore_index=True).head(total_pasos)
            else:
                patron = patron.reset_index(drop=True)

            fechas_futuras = pd.date_range(
                start=ultimo_timestamp + pd.Timedelta(minutes=5),
                periods=total_pasos,
                freq="5min",
            )

            pred_df = pd.DataFrame({
                DATETIME_COLUMN: fechas_futuras,
                "prediccion": patron.values
            })

            return pred_df

        def generar_prediccion():
            dias = input.dias_prediccion() or DEFAULT_PREDICTION_DAYS

            # Intentar usar el modelo real
            pred_df = pd.DataFrame()
            if rf_model is not None:
                try:
                    pred_df = forecast_with_model(rf_model, df, dias)
                except Exception:
                    pred_df = pd.DataFrame()

            # Respaldo si no hay modelo o falla
            if pred_df.empty:
                pred_df = generar_prediccion_provisional(dias)

            return pred_df

        # =========================
        # Inicialización
        # =========================
        @reactive.effect
        def _init_data():
            """
            energia_actual.set(filtrar_energia())
            clima_actual.set(filtrar_clima())
            prediccion_actual.set(generar_prediccion())
            """
            pass

        # =========================
        # Actualizaciones por botón
        # =========================
        @reactive.effect
        @reactive.event(input.btn_actualizar_energia)
        def _actualizar_energia():
            energia_actual.set(filtrar_energia())

        @reactive.effect
        @reactive.event(input.btn_actualizar_clima)
        def _actualizar_clima():
            clima_actual.set(filtrar_clima())

        @reactive.effect
        @reactive.event(input.btn_generar_prediccion)
        def _actualizar_prediccion():
            prediccion_actual.set(generar_prediccion())

        # =========================
        # Resumen
        # =========================
        @output
        @render.text
        def txt_total_registros():
            return str(len(df))

        @output
        @render.text
        def txt_fecha_min():
            return str(df[DATETIME_COLUMN].min().date())

        @output
        @render.text
        def txt_fecha_max():
            return str(df[DATETIME_COLUMN].max().date())

        @output
        @render.text
        def txt_valor_max():
            if TARGET_COLUMN in df.columns:
                return f"{df[TARGET_COLUMN].max():.2f}"
            return "N/D"

        # =========================
        # Energía
        # =========================
        @output
        @render.data_frame
        def tabla_datos_energia():
            return render.DataGrid(
                energia_actual(),
                width="100%",
                height="420px",
                summary= False,
                filters= False,
            )

        @render.download(filename="energia_filtrada.csv")
        def descargar_energia_csv():
            df_out = energia_actual().drop(columns=["_fecha", "_hora"], errors="ignore").copy()
            yield df_out.to_csv(index=False)

        @output
        @render_widget
        def grafica_energia():
            df_energia = energia_actual().drop(columns=["_fecha", "_hora"], errors="ignore").copy()
            vista = input.vista_energia() or "completa"

            fig = go.Figure()

            if df_energia.empty or TARGET_COLUMN not in df_energia.columns:
                fig.update_layout(
                    title="Serie temporal energética",
                    template="plotly_white",
                    height=430,
                )
                return fig

            df_energia[DATETIME_COLUMN] = pd.to_datetime(df_energia[DATETIME_COLUMN], errors="coerce")
            df_energia[TARGET_COLUMN] = pd.to_numeric(df_energia[TARGET_COLUMN], errors="coerce")
            df_energia = df_energia.dropna(subset=[DATETIME_COLUMN, TARGET_COLUMN])

            df_vista = preparar_vista_energia(df_energia)

            if vista == "completa":
                fig.add_trace(
                    go.Scatter(
                        x=df_vista[DATETIME_COLUMN].tolist(),
                        y=df_vista[TARGET_COLUMN].tolist(),
                        mode="lines",
                        name="Generación observada",
                        line=dict(width=2, color="#2563eb"),
                        hovertemplate=(
                            "Fecha: %{x|%Y-%m-%d %H:%M}<br>"
                            "Generación: %{y:.2f}<extra></extra>"
                        ),
                    )
                )
                titulo = "Serie temporal de la generación observada"
                y_label = "Generación"

            elif vista == "ultimos_7":
                fig.add_trace(
                    go.Scatter(
                        x=df_vista[DATETIME_COLUMN].tolist(),
                        y=df_vista[TARGET_COLUMN].tolist(),
                        mode="lines",
                        name="Últimos 7 días",
                        line=dict(width=2, color="#2563eb"),
                        hovertemplate=(
                            "Fecha: %{x|%Y-%m-%d %H:%M}<br>"
                            "Generación: %{y:.2f}<extra></extra>"
                        ),
                    )
                )
                titulo = "Generación observada - últimos 7 días"
                y_label = "Generación"

            elif vista == "ultimos_30":
                fig.add_trace(
                    go.Scatter(
                        x=df_vista[DATETIME_COLUMN].tolist(),
                        y=df_vista[TARGET_COLUMN].tolist(),
                        mode="lines",
                        name="Últimos 30 días",
                        line=dict(width=2, color="#2563eb"),
                        hovertemplate=(
                            "Fecha: %{x|%Y-%m-%d %H:%M}<br>"
                            "Generación: %{y:.2f}<extra></extra>"
                        ),
                    )
                )
                titulo = "Generación observada - últimos 30 días"
                y_label = "Generación"

            elif vista == "promedio_diario":
                fig.add_trace(
                    go.Scatter(
                        x=df_vista["fecha"].tolist(),
                        y=df_vista["valor"].tolist(),
                        mode="lines+markers",
                        name="Promedio diario",
                        line=dict(width=2, color="#2563eb"),
                        marker=dict(size=5),
                        hovertemplate=(
                            "Fecha: %{x}<br>"
                            "Promedio diario: %{y:.2f}<extra></extra>"
                        ),
                    )
                )
                titulo = "Promedio diario de la generación"
                y_label = "Promedio diario"

            elif vista == "maximo_diario":
                fig.add_trace(
                    go.Scatter(
                        x=df_vista["fecha"].tolist(),
                        y=df_vista["valor"].tolist(),
                        mode="lines+markers",
                        name="Máximo diario",
                        line=dict(width=2, color="#2563eb"),
                        marker=dict(size=5),
                        hovertemplate=(
                            "Fecha: %{x}<br>"
                            "Máximo diario: %{y:.2f}<extra></extra>"
                        ),
                    )
                )
                titulo = "Máximo diario de la generación"
                y_label = "Máximo diario"

            else:
                titulo = "Serie temporal energética"
                y_label = "Generación"

            fig.update_layout(
                title=titulo,
                xaxis_title="Fecha y hora" if vista in ["completa", "ultimos_7", "ultimos_30"] else "Fecha",
                yaxis_title=y_label,
                template="plotly_white",
                height=430,
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=60, r=30, t=70, b=60),
                modebar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    color="#64748b",
                    activecolor="#1d4ed8",
                ),
            )

            if vista in ["completa", "ultimos_7", "ultimos_30"]:
                fig.update_xaxes(
                    type="date",
                    tickformat="%Y-%m-%d",
                    hoverformat="%Y-%m-%d %H:%M",
                    showgrid=True,
                    gridcolor="rgba(148,163,184,0.18)",
                )
            else:
                fig.update_xaxes(
                    showgrid=True,
                    gridcolor="rgba(148,163,184,0.18)",
                )

            fig.update_yaxes(
                showgrid=True,
                gridcolor="rgba(148,163,184,0.18)",
                zeroline=False,
            )

            fig._config = {
                "displaylogo": False,
                "responsive": True,
                "modeBarButtonsToRemove": [
                    "zoomIn2d",
                    "zoomOut2d",
                    "lasso2d",
                    "select2d",
                    "toggleSpikelines",
                    "hoverClosestCartesian",
                    "hoverCompareCartesian"
                ]
            }

            return fig

        @output
        @render.plot
        def grafica_energia_diaria():
            df_energia = energia_actual().drop(columns=["_fecha", "_hora"], errors="ignore").copy()
            fig, ax = plt.subplots(figsize=(11, 4.5))

            if TARGET_COLUMN in df_energia.columns and not df_energia.empty:
                df_energia["fecha"] = df_energia[DATETIME_COLUMN].dt.date
                diario = df_energia.groupby("fecha")[TARGET_COLUMN].agg(["mean", "max"]).reset_index()

                ax.plot(
                    diario["fecha"],
                    diario["mean"],
                    linewidth=2,
                    marker="o",
                    markersize=3,
                    label="Promedio diario"
                )
                ax.plot(
                    diario["fecha"],
                    diario["max"],
                    linewidth=2,
                    marker="o",
                    markersize=3,
                    label="Máximo diario"
                )

                ax.set_title("Resumen diario de la generación")
                ax.set_xlabel("Fecha")
                ax.set_ylabel("Generación")
                ax.grid(True, alpha=0.25)
                ax.legend(loc="upper right")

            plt.xticks(rotation=45)
            plt.tight_layout()
            return fig

        # =========================
        # Clima
        # =========================
        @output
        @render.data_frame
        def tabla_datos_clima():
            return render.DataGrid(
                clima_actual(),
                width="100%",
                height="420px",
                summary= False,
                filters=False,
            )

        @render.download(filename="clima_filtrado.csv")
        def descargar_clima_csv():
            df_out = clima_actual().drop(columns=["_fecha"], errors="ignore").copy()
            yield df_out.to_csv(index=False)

        @output
        @render_widget
        def grafica_clima():
            df_clima = clima_actual().drop(columns=["_fecha"], errors="ignore").copy()
            variable = input.variable_clima()

            fig = go.Figure()

            if df_clima.empty or variable not in df_clima.columns:
                fig.update_layout(
                    title="Serie temporal climática",
                    template="plotly_white",
                    height=430,
                )
                return fig

            df_clima[DATETIME_COLUMN] = pd.to_datetime(df_clima[DATETIME_COLUMN], errors="coerce")
            df_clima[variable] = pd.to_numeric(df_clima[variable], errors="coerce")
            df_clima = df_clima.dropna(subset=[DATETIME_COLUMN, variable])

            fig.add_trace(
                go.Scatter(
                    x=df_clima[DATETIME_COLUMN].tolist(),
                    y=df_clima[variable].tolist(),
                    mode="lines",
                    name=variable.replace("_", " ").title(),
                    line=dict(width=2, color="#2563eb"),
                    hovertemplate=(
                            "Fecha: %{x|%Y-%m-%d %H:%M}<br>"
                            + f"{variable.replace('_', ' ').title()}: "
                            + "%{y:.2f}<extra></extra>"
                    ),
                )
            )

            fig.update_layout(
                title=f"Serie temporal climática: {variable.replace('_', ' ').title()}",
                xaxis_title="Fecha y hora",
                yaxis_title=variable.replace("_", " ").title(),
                template="plotly_white",
                height=430,
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=60, r=30, t=70, b=60),
                modebar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    color="#64748b",
                    activecolor="#1d4ed8",
                ),
            )

            fig.update_xaxes(
                type="date",
                tickformat="%Y-%m-%d",
                hoverformat="%Y-%m-%d %H:%M",
                showgrid=True,
                gridcolor="rgba(148,163,184,0.18)",
            )

            fig.update_yaxes(
                showgrid=True,
                gridcolor="rgba(148,163,184,0.18)",
                zeroline=False,
            )

            fig._config = {
                "displaylogo": False,
                "responsive": True,
                "modeBarButtonsToRemove": [
                    "zoomIn2d",
                    "zoomOut2d",
                    "lasso2d",
                    "select2d",
                    "toggleSpikelines",
                    "hoverClosestCartesian",
                    "hoverCompareCartesian"
                ]
            }

            return fig

        # =========================
        # Predicción
        # =========================
        @output
        @render.text
        def txt_modelo_activo():
            return model_name if rf_model is not None else f"{model_name} (modo provisional)"

        @output
        @render.text
        def txt_horizonte_pred():
            dias = input.dias_prediccion() or DEFAULT_PREDICTION_DAYS
            return f"{dias} días"

        @output
        @render.text
        def txt_promedio_pred():
            pred = prediccion_actual()
            if pred.empty:
                return "N/D"
            return f"{pred['prediccion'].mean():.2f}"

        @output
        @render.text
        def txt_total_pred():
            pred = prediccion_actual()
            if pred.empty:
                return "N/D"
            return f"{pred['prediccion'].sum():.2f}"

        @output
        @render.text
        def txt_estado_modelo():

            if rf_model is None:
                return (
                    "No se encontró el modelo serializado.\n"
                    "Se está usando una estimación provisional basada en patrón histórico reciente."
                )

            msg = [
                "Modelo Random Forest cargado correctamente.",
            ]

            if prediccion_actual().empty:
                msg.append("Presiona 'Generar predicción' para calcular el horizonte seleccionado.")

            """
            if artifact_features:
                msg.append(f"Features del modelo: {len(artifact_features)}")
                msg.append(f"Columnas usadas: {', '.join(artifact_features)}")
            """

            if artifact_target:
                msg.append(f"Variable objetivo: {artifact_target}")

            if artifact_datetime:
                msg.append(f"Columna temporal: {artifact_datetime}")

            if artifact_metrics_mean:
                rmse = artifact_metrics_mean.get("RMSE")
                mae = artifact_metrics_mean.get("MAE")
                r2 = artifact_metrics_mean.get("R2")
                mape = artifact_metrics_mean.get("MAPE")

                metric_parts = []
                if rmse is not None:
                    metric_parts.append(f"RMSE medio: {rmse:.4f}")
                if mae is not None:
                    metric_parts.append(f"MAE medio: {mae:.4f}")
                if r2 is not None:
                    metric_parts.append(f"R² medio: {r2:.4f}")
                if mape is not None:
                    metric_parts.append(f"MAPE medio: {mape:.2f}%")

                if metric_parts:
                    msg.append("Métricas promedio:")
                    msg.extend(metric_parts)

            if artifact_metrics_std:
                rmse_std = artifact_metrics_std.get("RMSE")
                mae_std = artifact_metrics_std.get("MAE")
                r2_std = artifact_metrics_std.get("R2")
                mape_std = artifact_metrics_std.get("MAPE")

                metric_parts_std = []
                if rmse_std is not None:
                    metric_parts_std.append(f"RMSE std: {rmse_std:.4f}")
                if mae_std is not None:
                    metric_parts_std.append(f"MAE std: {mae_std:.4f}")
                if r2_std is not None:
                    metric_parts_std.append(f"R² std: {r2_std:.4f}")
                if mape_std is not None:
                    metric_parts_std.append(f"MAPE std: {mape_std:.2f}%")

                if metric_parts_std:
                    msg.append("Dispersión de métricas:")
                    msg.extend(metric_parts_std)

            return "\n".join(msg)

        @output
        @render.data_frame
        def tabla_prediccion():
            pred = prediccion_actual().copy()

            if pred.empty:
                return render.DataGrid(pd.DataFrame(), width="100%", height="420px")

            pred = pred.rename(columns={"prediccion": "generacion_predicha"})

            return render.DataGrid(
                pred,
                width="100%",
                height="420px",
                summary= False,
                filters=False,
            )

        @render.download(filename="prediccion_generacion.csv")
        def descargar_prediccion_csv():
            pred = prediccion_actual().copy()
            if pred.empty:
                yield pd.DataFrame(columns=[DATETIME_COLUMN, "prediccion"]).to_csv(index=False)
                return
            pred = pred.rename(columns={"prediccion": "generacion_predicha"})
            yield pred.to_csv(index=False)

        @output
        @render_widget
        def grafica_prediccion():
            pred = prediccion_actual().copy()
            dias_hist = input.dias_historia_pred() or 30

            hist = df[[DATETIME_COLUMN, TARGET_COLUMN]].copy().sort_values(DATETIME_COLUMN)

            # Conversión segura
            hist[DATETIME_COLUMN] = pd.to_datetime(hist[DATETIME_COLUMN], errors="coerce")
            hist[TARGET_COLUMN] = pd.to_numeric(hist[TARGET_COLUMN], errors="coerce")
            hist = hist.dropna(subset=[DATETIME_COLUMN, TARGET_COLUMN])

            hist = hist[
                hist[DATETIME_COLUMN] >= hist[DATETIME_COLUMN].max() - pd.Timedelta(days=dias_hist)
                ]

            if not pred.empty:
                pred[DATETIME_COLUMN] = pd.to_datetime(pred[DATETIME_COLUMN], errors="coerce")
                pred["prediccion"] = pd.to_numeric(pred["prediccion"], errors="coerce")
                pred = pred.dropna(subset=[DATETIME_COLUMN, "prediccion"])

            fig = go.Figure()

            # Histórico
            if not hist.empty:
                fig.add_trace(
                    go.Scatter(
                        x=hist[DATETIME_COLUMN].tolist(),
                        y=hist[TARGET_COLUMN].tolist(),
                        mode="lines",
                        name="Histórico",
                        line=dict(width=2, color="#2563eb"),
                        hovertemplate=(
                            "Fecha: %{x|%Y-%m-%d %H:%M}<br>"
                            "Generación: %{y:.2f}<extra></extra>"
                        ),
                    )
                )

            # Predicción
            if not pred.empty:
                inicio_pred = pred[DATETIME_COLUMN].min()
                fin_pred = pred[DATETIME_COLUMN].max()

                fig.add_vline(
                    x=inicio_pred,
                    line_dash="dot",
                    line_width=2,
                    line_color="#475569",
                    opacity=0.9,
                )

                fig.add_vrect(
                    x0=inicio_pred,
                    x1=fin_pred,
                    fillcolor="rgba(249,115,22,0.12)",
                    opacity=0.12,
                    line_width=0,
                    layer="below",
                )

                fig.add_trace(
                    go.Scatter(
                        x=pred[DATETIME_COLUMN].tolist(),
                        y=pred["prediccion"].tolist(),
                        mode="lines",
                        name="Predicción",
                        line=dict(width=2, dash="dash", color="#f97316"),
                        hovertemplate=(
                            "Fecha: %{x|%Y-%m-%d %H:%M}<br>"
                            "Predicción: %{y:.2f}<extra></extra>"
                        ),
                    )
                )

            fig.update_layout(
                title="Histórico reciente y predicción de generación",
                xaxis_title="Fecha y hora",
                yaxis_title="Generación",
                template="plotly_white",
                height=460,
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=60, r=30, t=70, b=60),
                modebar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    color="#64748b",
                    activecolor="#1d4ed8",
                ),
            )

            fig.update_xaxes(
                type="date",
                tickformat="%Y-%m-%d",
                hoverformat="%Y-%m-%d %H:%M",
                showgrid=True,
                gridcolor="rgba(148,163,184,0.18)",
            )

            fig.update_yaxes(
                showgrid=True,
                gridcolor="rgba(148,163,184,0.18)",
                zeroline=False,
            )

            return fig

        @output
        @render.plot
        def grafica_prediccion_diaria():
            pred = prediccion_actual().copy()
            fig, ax = plt.subplots(figsize=(11, 4.5))

            if not pred.empty:
                pred["fecha"] = pred[DATETIME_COLUMN].dt.date
                diario = pred.groupby("fecha")["prediccion"].agg(["mean", "max"]).reset_index()

                ax.plot(
                    diario["fecha"],
                    diario["mean"],
                    linewidth=2,
                    marker="o",
                    markersize=3,
                    label="Promedio diario predicho"
                )
                ax.plot(
                    diario["fecha"],
                    diario["max"],
                    linewidth=2,
                    marker="o",
                    markersize=3,
                    label="Máximo diario predicho"
                )

                ax.set_title("Predicción resumida por día")
                ax.set_xlabel("Fecha")
                ax.set_ylabel("Generación predicha")
                ax.grid(True, alpha=0.25)
                ax.legend(loc="upper right")

            plt.xticks(rotation=45)
            plt.tight_layout()
            return fig

    return server