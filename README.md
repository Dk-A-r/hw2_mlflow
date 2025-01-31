# hw2_mlflow

Данный репозиторий отражает статус выполнения задания по Автоматизации администрирования машинного обучения. Выполнял студент 2 курса магистратуры "Инженерия машинного обучения", Карпов Данил Васильевич.

Структура репозитория следующая:
- файл experiments_script.py отвечает за настройку MLFlow и запуск (локальный) экспериментов, с записью результатов метрик. Решается задача классификации на датасете Iris - эксперимент называется "Iris Classification". Тестируются три модели - три варианта классификатора случайного леса (RandomForestClassifier) реализации библиотеки sklearn. Модели различаются значениями параметра n_estimators: 10, 50, 100. Фиксируются 4 метрики - accuracy, precision, recall, f1_score.
- файл report_script.py отвечает за построение графического отчета по метрикам, полученным в результате экспериментов и сохраняет их в директории запуска. Чтобы скрипт сработал, как ожидается, предполагается сначала запуск графического интерфейса MLFlow:
  ```
  $ mlflow ui
  ```
  Позднее необходимо открыть другое окно терминала, там запустить скрипт experiments_script.py, после чего скрипт report_script.py, который соберет необходимые метрики с сервера MLFlow. На итоговых графиках (которые также представлены в репозитории) имеется соотношение связи между количеством estimators и значениями метрик. Поскольку эксперимент производился в искусственных условиях (рассматривалась только первая строка датасета) значения метрик одинаковы. Вместе с тем при усложнении экспериментов графики могут стать более разнообразными. А пока графики следующие:
* по метрике Accuracy
  ![accuracy_plot](https://github.com/Dk-A-r/hw2_mlflow/blob/main/accuracy_plot.png?raw=true)
* по метрике F1
  ![f1_plot](https://github.com/Dk-A-r/hw2_mlflow/blob/main/f1_score_plot.png?raw=true)
* по метрике Precision
  ![f1_plot](https://github.com/Dk-A-r/hw2_mlflow/blob/main/precision_plot.png?raw=true)
* по метрике Recall
 ![recall_plot](https://github.com/Dk-A-r/hw2_mlflow/blob/main/recall_plot.png?raw=true)

- также стоит отметить, что файл report_script.py выводит результаты метрик в консоль и указывает, по какому адресу можно посмотреть сам отчет в интерфейсе UI MLFlow по типу следующего:
```
    params.n_estimators  metrics.accuracy  metrics.precision  metrics.recall  metrics.f1_score
0                 100               1.0                1.0             1.0               1.0
1                  50               1.0                1.0             1.0               1.0
2                  10               1.0                1.0             1.0               1.0
3                 100               1.0                1.0             1.0               1.0
4                  50               1.0                1.0             1.0               1.0
5                  10               1.0                1.0             1.0               1.0
6                 100               1.0                NaN             NaN               NaN
7                 100               1.0                NaN             NaN               NaN

Summary Metrics:
  params.n_estimators  metrics.accuracy  metrics.precision  metrics.recall  metrics.f1_score
0                 100               1.0                1.0             1.0               1.0
1                  50               1.0                1.0             1.0               1.0
2                  10               1.0                1.0             1.0               1.0
3                 100               1.0                1.0             1.0               1.0
4                  50               1.0                1.0             1.0               1.0
5                  10               1.0                1.0             1.0               1.0
6                 100               1.0                NaN             NaN               NaN
7                 100               1.0                NaN             NaN               NaN
🏃 View run Report_Generation at: http://localhost:5000/#/experiments/0/runs/<some-hash-value>
🧪 View experiment at: http://localhost:5000/#/experiments/0
```
- в папке screenshots находится демонстрация работы в графическом интерфейсе MLFlow:
 ![plot_100](https://github.com/Dk-A-r/hw2_mlflow/blob/main/screenshots/IrisExp100metr.png?raw=true)
* метрики для 100 Estimators
  ![plot_50](https://github.com/Dk-A-r/hw2_mlflow/blob/main/screenshots/IrisExp50metr.png?raw=true)
* метрики для 50 Estimators
  ![plot_10](https://github.com/Dk-A-r/hw2_mlflow/blob/main/screenshots/IrisExp10metr.png?raw=true)
* метрики для 10 Estimators
  ![screen_iris](https://github.com/Dk-A-r/hw2_mlflow/blob/main/screenshots/IrisExpScreenshot.png?raw=true)
* скриншот информации о проведенных экспериментах Iris Classification
  ![table_metr](https://github.com/Dk-A-r/hw2_mlflow/blob/main/screenshots/IrisExpfullmetr.png?raw=true)
* сводная таблица по метрикам
  ![graph_metr](https://github.com/Dk-A-r/hw2_mlflow/blob/main/screenshots/IrisExpgpaphmetr.png?raw=true)
* сводные графики по метрикам
  ![artifacts](https://github.com/Dk-A-r/hw2_mlflow/blob/main/screenshots/artifacts.png?raw=true)
* артефакты при составлении отчета (происходят в рамках экспериментов Default)
  ![report](https://github.com/Dk-A-r/hw2_mlflow/blob/main/screenshots/report.png?raw=true)
* скриншот экспериментов по генерации отчетов
