"""
Веб-интерфейс (Streamlit) для Time Series Analysis Tool.
Локальный режим.
"""

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

# Добавляем путь к src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import engine
from src.config import EXPERIMENTAL_METHODS, STABLE_METHODS


def main() -> None:
    st.set_page_config(page_title="Анализ Временных Рядов (Локально)", layout="wide")
    st.title("Анализ Связности Временных Рядов")
    st.caption("Локальная версия. Загрузите файл CSV или Excel.")

    uploaded_file = st.file_uploader("Выберите файл", type=["csv", "xlsx"])

    with st.expander("Параметры анализа", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            lag = st.number_input("Лаг (Lag)", min_value=1, max_value=50, value=1)
        with col2:
            threshold = st.number_input("Порог графа (Threshold)", 0.0, 1.0, 0.2, 0.05)
        with col3:
            normalize = st.checkbox("Нормализация (Z-score)", value=True)
            preprocess = st.checkbox("Предобработка (fill/outliers/log)", value=True)
            fill_missing = st.checkbox("Заполнять пропуски (interp)", value=True)
            remove_outliers = st.checkbox("Убирать выбросы (Z)", value=True)
            log_transform = st.checkbox("Лог-преобразование (только >0)", value=False)

        alpha = st.number_input("P-value alpha (для Granger)", 0.0001, 0.5, 0.05, format="%.4f")
        window_cube_level = st.selectbox("Анализ окно×лаг×положение", ["off", "basic", "full"], index=1)
        window_sizes_text = st.text_input("Window sizes (comma)", value="256,512")

    all_methods = STABLE_METHODS + EXPERIMENTAL_METHODS
    selected_methods = st.multiselect("Выберите методы", all_methods, default=STABLE_METHODS[:2])

    if st.button("Запустить анализ", type="primary"):
        if not uploaded_file:
            st.error("Файл не загружен!")
            return

        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = os.path.join(tmp_dir, uploaded_file.name)
            with open(input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            tool = engine.BigMasterTool()

            with st.spinner("Загрузка и обработка..."):
                try:
                    tool.load_data_excel(
                        input_path,
                        preprocess=preprocess,
                        normalize=normalize,
                        fill_missing=fill_missing,
                        remove_outliers=remove_outliers,
                        log_transform=log_transform,
                    )

                    try:
                        window_sizes = [int(x.strip()) for x in window_sizes_text.split(',') if x.strip()]
                    except Exception:
                        window_sizes = None
                    tool.run_selected_methods(
                        selected_methods,
                        max_lag=lag,
                        window_sizes=window_sizes,
                        window_policy="best",
                        window_cube_level=window_cube_level,
                    )

                    excel_path = os.path.join(tmp_dir, "report.xlsx")
                    html_path = os.path.join(tmp_dir, "report.html")

                    tool.export_big_excel(excel_path, threshold=threshold, p_value_alpha=alpha)
                    tool.export_html_report(html_path, graph_threshold=threshold, p_alpha=alpha)

                    st.success("Готово!")

                    c1, c2 = st.columns(2)
                    with c1:
                        with open(excel_path, "rb") as f:
                            st.download_button("Скачать Excel", f, "report.xlsx")
                    with c2:
                        with open(html_path, "rb") as f:
                            st.download_button("Скачать HTML", f, "report.html")

                    st.subheader("Предварительный просмотр")
                    for method in selected_methods:
                        if method in tool.results and tool.results[method] is not None:
                            st.write(f"**{method}**")
                            from src.visualization import plots

                            buf = plots.plot_heatmap(tool.results[method], f"{method} (Lag={lag})")
                            st.image(buf, caption=method)

                except Exception as e:
                    st.error(f"Ошибка выполнения: {e}")
                    import traceback

                    st.text(traceback.format_exc())


if __name__ == "__main__":
    main()
