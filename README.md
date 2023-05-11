# Projecto: Predicción en la venta de tickets 

En este proyecto, se crea un API, el cual recibe un modelo entrenado de forecasting y predice la venta de tickets de los siguientes 7 días del historial de datos.

Específicamente, se siguieron los siguientes pasos para su desarrollo: 

1. Se realizó una documentación de los datos entregados, con el fin de analizar variables estadísticas, tendencia, periodicidad. Y con ello, proceder con su respectivo pre-procesamiento. (ver: src/tickit/test_data.ipynb)
2. Se implementó un modelo de entrenamiento usando pytorch. Este consta de una "LSTM layer", "dense layer" y una "fully connected layer". (ver: src/tickit/network.py)
3. Para entrenar los datos, primero se estructuraron de la sigueinte forma: {'sequence':sequence, 'target':target}. Donde sequence se refiere a una "ventana de entrenamiento" (30 días en este caso), partiendo del i-ésimo día de los datos. Y target es una "ventana de predicción" de 7 días después de la secuencia i-ésima. (ver: src/tickit/training.py)
4. El modelo entrenado y las gráficas del comportamiento la función de perdida se pueden ver en "src/tickit/reports/"
5. Teniendo el modelo entrenado, se desarrolla un api service con FastApi. (ver: src/api/)
6. Finalmente, se configuran los archivos de Docker necesarios para la configuración del entorno y el despliegue. 

# Documentación API
