import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models.signature import infer_signature

# Загрузка данных
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

def run_experiment(n_estimators):
    # Настройка эксперимента
    mlflow.set_experiment("Iris Classification")

    with mlflow.start_run(run_name=f"Experiment_{n_estimators}_estimators"):
        # Логирование параметров
        mlflow.log_param("n_estimators", n_estimators)
        
        # Обучение модели
        model = RandomForestClassifier(n_estimators=n_estimators)
        model.fit(X_train, y_train)
        
        # Оценка модели
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Логирование метрик
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Создание входного примера
        input_example = X_train[:1]  # Первая строка из обучающих данных
        
        # Автоматическое определение сигнатуры
        signature = infer_signature(X_train, model.predict(X_train))
        
        # Сохранение модели с сигнатурой и входным примером
        mlflow.sklearn.log_model(
            model, 
            "model", 
            signature=signature, 
            input_example=input_example
        )
        
        print(f"Experiment with {n_estimators} estimators completed.")

# Запуск нескольких экспериментов с разными параметрами
for n_estimators in [10, 50, 100]:
    run_experiment(n_estimators)