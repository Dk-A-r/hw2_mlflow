import mlflow
import pandas as pd
import matplotlib.pyplot as plt

# Установка URI для MLflow Tracking Server
mlflow.set_tracking_uri("http://localhost:5000")

# Получение всех экспериментов
experiment = mlflow.get_experiment_by_name("Iris Classification")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Преобразование результатов в DataFrame
df = runs[["params.n_estimators", "metrics.accuracy", "metrics.precision", "metrics.recall", "metrics.f1_score"]]
print(df)

# Визуализация результатов
def create_and_save_plot(x_data, y_data, xlabel, ylabel, title, filename):
    plt.figure(figsize=(4, 3))  # Увеличиваем размер фигуры
    plt.plot(x_data, y_data, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()  # Автоматическое корректирование макета для предотвращения наложения подписей
    plt.savefig(filename, dpi=150)  # Сохраняем с высоким разрешением
    plt.close()

# График точности (accuracy)
create_and_save_plot(
    df["params.n_estimators"], 
    df["metrics.accuracy"], 
    "Number of Estimators", 
    "Accuracy", 
    "Model Accuracy", 
    'accuracy_plot.png'
)

# График точности (precision)
create_and_save_plot(
    df["params.n_estimators"], 
    df["metrics.precision"], 
    "Number of Estimators", 
    "Precision", 
    "Model Precision", 
    'precision_plot.png'
)

# График полноты (recall)
create_and_save_plot(
    df["params.n_estimators"], 
    df["metrics.recall"], 
    "Number of Estimators", 
    "Recall", 
    "Model Recall", 
    'recall_plot.png'
)

# График F1-меры (f1_score)
create_and_save_plot(
    df["params.n_estimators"], 
    df["metrics.f1_score"], 
    "Number of Estimators", 
    "F1 Score", 
    "Model F1 Score", 
    'f1_score_plot.png'
)

# Начало нового эксперимента для логирования отчета
with mlflow.start_run(run_name="Report_Generation"):
    # Логирование графиков как артефактов
    mlflow.log_artifact('accuracy_plot.png')
    mlflow.log_artifact('precision_plot.png')
    mlflow.log_artifact('recall_plot.png')
    mlflow.log_artifact('f1_score_plot.png')

    # Вывод метрик в консоль
    print("\nSummary Metrics:")
    print(df[['params.n_estimators', 'metrics.accuracy', 'metrics.precision', 'metrics.recall', 'metrics.f1_score']])