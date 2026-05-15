[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project — Predicting Message Length on Twitter and Reddit

**Студент:** Лисота Александр Юрьевич

**Группа:** БИВ236

## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Быстрый старт](#быстрый-старт)
4. [Данные](#данные)
5. [Результаты](#результаты)
6. [Качество кода](#качество-кода)
7. [Docker](#docker)
8. [Отчёт](#отчёт)

## Описание задачи

- **Задача:** регрессия. Предсказываем длину сообщения в символах по его характеристикам.
- **Платформы:** Twitter и Reddit.
- **Датасет:** `Twitter_Data.csv` и `Reddit_Data.csv` из набора *Twitter and Reddit Sentimental Analysis Dataset*.
- **Основная метрика:** `MAE`.
- **Дополнительные метрики:** `RMSE`, `R²`.
- **Fixed seed:** все разбиения и эксперименты используют `RANDOM_STATE = 42` из [src/config.py](src/config.py).

## Структура репозитория

```
.
├── .github/workflows/ci.yml        # CI: ruff, flake8, pytest
├── data
│   ├── raw/                        # Исходные данные (в git не хранятся)
│   └── processed/                  # Артефакты EDA и экспериментов
├── models/                         # Сохранённые модели
├── notebooks/                      # Jupyter notebooks при необходимости
├── presentation/                   # Материалы для защиты
├── report
│   ├── images/                     # Графики и визуализации
│   └── report.md                   # Основной отчёт
├── src
│   ├── config.py                   # Общие константы и пути
│   ├── preprocessing.py            # Очистка данных и train/val/test split
│   ├── eda.py                      # EDA и сохранение графиков
│   └── modeling.py                 # Baseline, эксперименты и финальная модель
├── tests
│   └── test_pipeline.py            # Smoke tests пайплайна
├── .dockerignore
├── .flake8
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml                  # Конфиг ruff и pytest
├── requirements.txt
└── README.md
```

## Быстрый старт

```bash
# 1. Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 2. Установить зависимости
pip install -r requirements.txt

# 3. Положить сырые данные рядом с проектом в папку ../dataset
#    или в data/raw/:
#    - Twitter_Data.csv
#    - Reddit_Data.csv

# 4. Подготовка датасета
python src/preprocessing.py

# 5. EDA и графики
python src/eda.py

# 6. Baseline и все эксперименты
python src/modeling.py
```

## Данные
- Исходные файлы:
  - `Twitter_Data.csv`
  - `Reddit_Data.csv`
- Поддерживаются два пути хранения:
  - `data/raw/`
  - соседняя папка `../dataset/`
- Сгенерированные файлы:
  - `data/processed/messages_train.csv`
  - `data/processed/messages_val.csv`
  - `data/processed/messages_test.csv`
  - `data/processed/model_experiments.csv`
  - `data/processed/model_summary.json`

## Результаты

- Baseline `LinearRegression` на минимальных признаках:
  - `MAE(test) = 68.185`
  - `RMSE(test) = 90.824`
- Лучшая модель `RandomForestRegressor`:
  - `MAE(test) = 0.065`
  - `RMSE(test) = 1.189`
  - после переобучения на `train + val`: `MAE(test) = 0.055`

## Качество кода

- Линтеры:
  - `ruff`
  - `flake8`
- Тесты:
  - `pytest`
- Локальный запуск проверок:

```bash
ruff check .
flake8 --jobs 1 src tests
pytest
```

## Docker

- Сборка образа:

```bash
docker build -t ml-homework .
```

- Запуск контейнера:

```bash
docker run --rm -v ${PWD}:/app ml-homework
```

- Запуск через compose:

```bash
docker compose up --build
```

## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
