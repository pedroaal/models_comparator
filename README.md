# Maestria en ciencia de datos y maquinas de aprendizaje 

## Trabajo final de maestria (TFM)

Este proyecto se enfoca en la busqueda de anomalian en un cojunto de datos obtenidos por la estacion meteorologica de la Universidad de Cuenca.

Para este proyecto nos enfocamos en tres puntos:
- La exploracion del cojunto de datos
- La prediccion y deteccion de anomalias
- Una api para poder usar los modelos generados
## Instalation

Para la implementacion y ejecucion de este proyecto, estamos usando [UV](https://docs.astral.sh/uv/) como manejador de entorno. UV nos proporciona herramientas para manejar la version de python y las librerias necesarias.

Una vez instalado uv, simplemente debemos correr alguno de los archivos, y uv se encargara de crear el '.venv' e instalar las librerias. La forma de ejecutar los scripts es: 
```bash
$ uv run api.py
```
## Documentacion

Para este proyecto se implementaron tres script en python.

### Explorador

En el archivo `explore.py` se encuetran varias funciones destinadas a la exploracion del conjunto de datos. Este script creara graficas explicativas del conjunto de datos, ademas de imprimir en consola informacion util para el analisis posterior.

**basic_info()**
Esta funcion nos muestra informacion basica del conjunto de datos, como forma, tipos de datos, valores duplicados.

**numerical_analysis()**
Esta funcion nos muestra la informacion de las columnas numericas, especificamente genera una grafica de distribucion y una grafica de box para cada feature numerica en el dataset.

**categorical_analysis()**
Esta funcion genera un grafico de conteo para cada columna categorica en el conjunto de datos.

**correlation_analysis()**
Esta funcion genera un heatmap con la correlacion entre las features del conjunto de datos.

**outlier_detection()**
Esta funcion nos muestra los quantiles de cada feature en el conjunto de datos.

**time_series_check()**
Esta funcion nos muestra si tenemos columnas de tiempo en nuestro conjunto de datos.

### Comparador

Dado que para este proyecto implementamos varios modelos de machine learning, generamos el archivo `compare.py` con la finalidad de tener todos los modelos agrupados en un solo lugar, ademas de evitar duplicar los pasos de tratamiento de datos en cada modelo.

En el script se encuetran las funciones:
- run_dbscan_model()
- run_sarima_model()
- run_random_forest_model()
- run_svm_model()
- run_lstm_model()
- run_mlp_model()
las mismas que ejecutan el llamado a cada modelo creado, de manera general, cada funcion ejecuta el entrenamiento, la evaluacion y la creacion de graficas.

Adicionalmente, se incluye un paso de optiomizacion de hiperparametros para los modelos que lo ameriten.

Finalmente en la funcion **main()** se encuetran los pasos para obtener el conjunto de datos a partir de un csv, el tratamiento de datos y la ejecucion de cada modelo.

### Modelos

Dentro de la carpeta `models` podemos encontrar la definicion de cada modelo creado para este proyecto. Cada modelo es una clase similar a:
```python
class ModelName():
    def __init__(self, ...private):
        self.private = private
        self.model = model_definition()

    def train(self, X, y):
        self.model.trai(X, y)

    def predict(self, X):
        model = load_model('path.joblib')
        predict = model.predict(X)
        return predict
    
    def evaluate(self, X, y):
        y_pred = self.model.predict(X)
        metrics = {
            "mse": mean_squared_error(y, y_pred)
        }
        print(f"Mean Squared Error: {metrics['mse']:.4f}")
        return metrics

    def plot_results(self, X, y):
        ...
        plt.show()
```

Adicionalment podemos encontrar el archivo `transform.py`, este archivo contiene las siguientes funciones de ayuda para el procesamiento de datos.
- handle_datetime(): esta funcion genera las columnas HORA, MES, DIA y maneja la columna DATETIME
- handle_rainfall(): esta funcion crea la columna HAS_RAINFALL
- handle_uv(): esta funcion elimina los valaros cuyo uv sea 0
- handle_window(): esta funcion agrupa el conjunto de datos en ventanas
- fit_scaler(): esta funcion crea el scaler y lo guarda para su posterior uso
- transform_scaler(): esta funcion escala los datos usando el scaler guardado

### API

Con la finalidad de generar utilidad al proyecto, se genera una api usando [FastAPI](https://fastapi.tiangolo.com/). De este modo, podemos utilizar los modelos predictivos usando nada mas que una llamada al servidor.

Adicionalment se crea el archivo `index.html` el mismo que contiene un formulario sencillo para poder llamar al api.
## Referencia del API

#### Get a predict

```http
  POST /predict
```

| Parameter  | Type       | Description                 |
| :--------- | :--------- | :-------------------------- |
| `datetime` | `datetime` | Fecha y hora en formato ISO |
| `ambtemp`  | `float`    | Temperatura ambiente        |
| `cougm3`   | `float`    | Nivel de CO                 |
| `no2ugm3`  | `float`    | Nivel de NO2                |
| `o3ugm3`   | `float`    | Nivel de O3                 |
| `pm25`     | `float`    | Nivel de PM25               |
| `so2ugm3`  | `float`    | Nivel de SO2                |
