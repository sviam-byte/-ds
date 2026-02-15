# Time Series Analysis Tool (Refactored)

Инструмент для анализа временных рядов с поддержкой различных метрик связности.

## Структура проекта

```
proj_patch_refactored/
├── src/                          # Исходный код
│   ├── config.py                 # Конфигурация и константы
│   ├── core/                     # Ядро системы
│   │   ├── data_loader.py        # Загрузка и парсинг данных
│   │   ├── preprocessing.py      # Предобработка
│   │   └── engine.py             # Главный движок (BigMasterTool)
│   │
│   ├── metrics/                  # Модули расчёта метрик (TODO)
│   ├── analysis/                 # Анализ данных (TODO)
│   ├── visualization/            # Визуализация (TODO)
│   └── export/                   # Экспорт отчётов (TODO)
│
├── interfaces/                   # Интерфейсы
│   ├── gui.py                    # Tkinter GUI
│   ├── cli.py                    # Командная строка
│   └── web.py                    # Streamlit веб-интерфейс
│
├── START_TimeSeriesTool.bat      # Запуск GUI (Windows)
├── requirements.txt              # Зависимости
└── README.md                     # Документация
```

## Установка

```bash
# Установить зависимости
pip install -r requirements.txt
```

## Использование

### GUI (Tkinter)

**Windows:**
```bash
START_TimeSeriesTool.bat
```

**Linux/Mac:**
```bash
python interfaces/gui.py
```

### CLI (Command Line)

```bash
python interfaces/cli.py data.csv --lags 5 --report-html report.html
```

Параметры:
- `--lags N` - максимальный лаг
- `--graph-threshold T` - порог для графа
- `--p-alpha A` - альфа для p-value методов
- `--report-html PATH` - путь для HTML отчёта
- `--no-excel` - не генерировать Excel
- Подробнее: `python interfaces/cli.py --help`

### Web (Streamlit)

```bash
streamlit run interfaces/web.py
```

## Доступные методы

### Корреляции
- `correlation_full` - Полная корреляция
- `correlation_partial` - Частичная корреляция
- `correlation_directed` - Лаговая корреляция

### Взаимная информация
- `mutinf_full` - Взаимная информация
- `mutinf_partial` - Частичная MI
- `te_full` - Transfer Entropy
- `te_partial` - TE (partial)
- `te_directed` - TE (directed)

### Другие методы
- `coherence_full` - Когерентность
- `granger_full` - Granger causality
- `h2_full` - H² метрика

## Формат входных данных

Поддерживаются форматы:
- CSV (`.csv`)
- Excel (`.xlsx`, `.xls`)

Структура:
- Матрица `time × features`
- Опционально: первая строка - заголовки
- Опционально: первая колонка - время/индекс

## Примеры

### Базовый анализ через GUI

1. Запустите `START_TimeSeriesTool.bat`
2. Выберите файл данных
3. Настройте методы анализа
4. Нажмите "Run → HTML"
5. Отчёт откроется автоматически

### Анализ через CLI

```bash
# Базовый анализ
python interfaces/cli.py demo.csv

# С параметрами
python interfaces/cli.py demo.csv --lags 10 --report-html my_report.html

# Без Excel, только HTML
python interfaces/cli.py demo.csv --no-excel --report-html report.html
```

## Статус рефакторинга

**Готово:**
- ✅ Модульная структура
- ✅ Разделение конфигурации
- ✅ Загрузчик данных
- ✅ Все интерфейсы (GUI, CLI, Web)

**TODO (для дальнейшего рефакторинга):**
- ⏳ Разнести метрики по отдельным модулям (metrics/)
- ⏳ Вынести анализ в отдельные модули (analysis/)
- ⏳ Разделить визуализацию (visualization/)
- ⏳ Модули экспорта (export/)

## Разработка

Основной код находится в `src/core/engine.py`. Это большой файл, который требует дальнейшего рефакторинга. Функциональность разделена на:

1. **Метрики связности** - функции расчёта корреляций, MI, TE, Granger
2. **Анализ** - FFT, Hurst, энтропия, диагностика
3. **Визуализация** - heatmap, connectome, графики
4. **Экспорт** - HTML, Excel, site reports

Каждую категорию можно постепенно выносить в отдельные модули.

## Лицензия

См. оригинальный проект.

## Авторы

Refactored structure - 2024
