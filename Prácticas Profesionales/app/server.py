import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from shiny import reactive, render

from app.config import (
    COLUMNAS_CLIMA,
    COLUMNAS_ENERGIA,
    DATETIME_COLUMN,
    DEFAULT_PREDICTION_DAYS,
    DEFAULT_TARIFF,
    TARGET_COLUMN,
)
from ml.economics import estimate_cost
from ml.forecast import forecast_with_model


def create_server(df, model=None, model_name="Random Forest"):
    def server(input, output, session):
        columnas_energia_disponibles = [c for c in COLUMNAS_ENERGIA if c in df.columns]
        columnas_clima_disponibles = [c for c in COLUMNAS_CLIMA if c in df.columns]

        energia_actual = reactive.value(df[columnas_energia_disponibles].copy())
        clima_actual = reactive.value(df[columnas_clima_disponibles].copy())
        prediccion_actual = reactive.value(pd.DataFrame())

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
            df_energia = df[columnas_energia_disponibles].copy()

            inicio, fin = input.rango_fechas_energia()
            hora_inicio, hora_fin = input.hora_rango_energia()

            df_energia = df_energia[
                (df_energia[DATETIME_COLUMN].dt.date >= inicio)
                & (df_energia[DATETIME_COLUMN].dt.date <= fin)
            ]

            df_energia = df_energia[
                (df_energia[DATETIME_COLUMN].dt.hour >= hora_inicio)
                & (df_energia[DATETIME_COLUMN].dt.hour <= hora_fin)
            ]

            return df_energia

        def filtrar_clima():
            df_clima = df[columnas_clima_disponibles].copy()

            inicio, fin = input.rango_fechas_clima()
            df_clima = df_clima[
                (df_clima[DATETIME_COLUMN].dt.date >= inicio)
                & (df_clima[DATETIME_COLUMN].dt.date <= fin)
            ]

            return df_clima

        def generar_prediccion_provisional(dias: int) -> pd.DataFrame:
            base = df[[DATETIME_COLUMN, TARGET_COLUMN]].copy().sort_values(DATETIME_COLUMN)
            ultimo_timestamp = base[DATETIME_COLUMN].max()

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
            energia_actual.set(filtrar_energia())
            clima_actual.set(filtrar_clima())
            prediccion_actual.set(generar_prediccion())

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
        @render.table
        def tabla_datos_energia():
            return energia_actual().head(200)

        @output
        @render.plot
        def grafica_energia():
            df_energia = energia_actual()
            fig, ax = plt.subplots(figsize=(10, 4))

            if TARGET_COLUMN in df_energia.columns and not df_energia.empty:
                ax.plot(
                    df_energia[DATETIME_COLUMN],
                    df_energia[TARGET_COLUMN],
                    label=TARGET_COLUMN
                )
                ax.set_title("Comportamiento energético en el tiempo")
                ax.set_xlabel("Fecha y hora")
                ax.set_ylabel(TARGET_COLUMN)
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
                plt.xticks(rotation=45)
                ax.grid(True, alpha=0.3)
                ax.legend()

            plt.tight_layout()
            return fig

        # =========================
        # Clima
        # =========================
        @output
        @render.table
        def tabla_datos_clima():
            return clima_actual().head(200)

        @output
        @render.plot
        def grafica_clima():
            df_clima = clima_actual()
            variable = input.variable_clima()

            fig, ax = plt.subplots(figsize=(10, 4))

            if variable in df_clima.columns and not df_clima.empty:
                ax.plot(df_clima[DATETIME_COLUMN], df_clima[variable], label=variable)
                ax.set_title(f"Serie temporal de {variable}")
                ax.set_xlabel("Fecha y hora")
                ax.set_ylabel(variable)
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
                plt.xticks(rotation=45)
                ax.grid(True, alpha=0.3)
                ax.legend()

            plt.tight_layout()
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
        def txt_tarifa_activa():
            tarifa = input.tarifa_kwh() or DEFAULT_TARIFF
            return f"${tarifa:.2f}"

        @output
        @render.text
        def txt_costo_estimado():
            pred = prediccion_actual()
            tarifa = input.tarifa_kwh() or DEFAULT_TARIFF
            resumen = estimate_cost(pred, tarifa)
            return f"${resumen['costo_estimado']:.2f}"

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

            if artifact_features:
                msg.append(f"Features del modelo: {len(artifact_features)}")
                msg.append(f"Columnas usadas: {', '.join(artifact_features)}")

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
        @render.table
        def tabla_prediccion():
            pred = prediccion_actual().copy()
            tarifa = input.tarifa_kwh() or DEFAULT_TARIFF

            if not pred.empty:
                pred["costo_estimado"] = pred["prediccion"] * tarifa

            return pred.head(200)

        @output
        @render.plot
        def grafica_prediccion():
            pred = prediccion_actual()
            dias_hist = input.dias_historia_pred() or 30

            fig, ax = plt.subplots(figsize=(11, 5))

            hist = df[[DATETIME_COLUMN, TARGET_COLUMN]].copy().sort_values(DATETIME_COLUMN)
            hist = hist[
                hist[DATETIME_COLUMN] >= hist[DATETIME_COLUMN].max() - pd.Timedelta(days=dias_hist)
            ]

            if not hist.empty:
                ax.plot(
                    hist[DATETIME_COLUMN],
                    hist[TARGET_COLUMN],
                    label="Histórico",
                    linewidth=1.8
                )

            if not pred.empty:
                ax.plot(
                    pred[DATETIME_COLUMN],
                    pred["prediccion"],
                    label="Predicción",
                    linestyle="--",
                    linewidth=1.6
                )

            ax.set_title("Histórico reciente y predicción")
            ax.set_xlabel("Fecha y hora")
            ax.set_ylabel(TARGET_COLUMN)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
            plt.xticks(rotation=45)
            ax.grid(True, alpha=0.3)
            ax.legend()

            plt.tight_layout()
            return fig

    return server