#  Практики по машинному обучению: от регрессии до трансформеров

Студент: Тисленко Максим Андреевич
Дисциплина: Методы машинного обучения
Группа: 423МУ

Репозиторий содержит практические работы по лекции 1. Путь идёт от классических алгоритмов машинного обучения к нейронным сетям и дообучению трансформера. Каждая модель обучается на своём открытом датасете. В каждой тетради есть полный цикл: загрузка → очистка (пропуски, дубликаты, аномалии, кодирование) → EDA с распределениями и матрицей корреляции → разбиение train/val/test → обучение → оценка метрик. Все эксперименты логируются в **MLflow**.

##  Структура

```
├── practice_1_classic_ml/          # Практика 1 — классические алгоритмы (scikit-learn)
│   ├── 01_linear_regression.ipynb
│   ├── 02_logistic_regression.ipynb
│   ├── 03_decision_tree.ipynb
│   └── 04_random_forest.ipynb
├── practice_2_neural_networks/     # Практика 2 — базовые нейросети (TensorFlow / Keras)
│   ├── 01_fcn_letter_recognition.ipynb
│   ├── 02_cnn_kmnist.ipynb
│   └── 03_rnn_lstm_power_consumption.ipynb
├── practice_3_fine_tuning/         # Практика 3 — fine-tuning трансформера (KerasHub)
│   └── 01_distilbert_sms_spam_finetuning.ipynb
├── data/                           # датасеты скачиваются сюда автоматически (не в git)
├── requirements.txt
└── README.md
```

##  Модели и датасеты

| № | Модель | Задача | Датасет | Результат на тесте |
|---|---|---|---|---|
| 1.1 | Линейная регрессия | выработка электростанции, МВт | [Combined Cycle Power Plant](https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant) | R² = 0,933, MAE = 3,35 МВт |
| 1.2 | Логистическая регрессия | одобрение кредитной карты | [Credit Approval](https://archive.ics.uci.edu/dataset/27/credit+approval) | ROC-AUC = 0,962, F1 = 0,874 |
| 1.3 | Дерево решений | сдаст ли студент экзамен | [Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance) | F1-macro = 0,56, balanced accuracy = 0,60 |
| 1.4 | Случайный лес | возрастная группа абалона | [Abalone](https://archive.ics.uci.edu/dataset/1/abalone) | accuracy = 0,684, F1-macro = 0,670 |
| 2.1 | Полносвязная сеть (FCN) | распознавание букв по признакам | [Letter Recognition](https://archive.ics.uci.edu/dataset/59/letter+recognition) | accuracy = 0,974 |
| 2.2 | Свёрточная сеть (CNN) | японские рукописные символы | [Kuzushiji-MNIST](https://github.com/rois-codh/kmnist) | accuracy = 0,975 |
| 2.3 | SimpleRNN / LSTM | прогноз энергопотребления на час | [Household Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) | LSTM: MAE = 0,312 кВт (−18,5 % к наивному прогнозу) |
| 3 | DistilBERT (Transformer), fine-tuning | детекция SMS-спама | [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) | F1 = 0,962, precision = 1,000 |

Подробный анализ и выводы — в последнем разделе каждой тетради.

##  Запуск

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Тетради запускаются **из своей папки** (так их открывает Jupyter). Датасеты скачиваются в `data/` при первом запуске. Всё рассчитано на обычный ПК без GPU.

> **Практика 3 на Windows.** Токенизаторы KerasHub используют `tensorflow-text`, у которого нет сборок под нативный Windows. Запускайте эту тетрадь в WSL2 или в Google Colab.

##  MLflow

Параметры, метрики по эпохам и графики пишутся в `mlflow.db` в корне репозитория:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Затем откройте http://127.0.0.1:5000.

##  Стек

Python · Pandas · NumPy · Matplotlib · Seaborn · scikit-learn · TensorFlow / Keras 3 · KerasHub · MLflow
