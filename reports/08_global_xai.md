Para entender qué features globales aprendió cada modelo, generamos imágenes sintéticas que maximizan la activación de cada clase mediante ascenso de gradiente (**Activation Maximization**) - Introducir fuente.  

#### Modelo BASELINE - Clase NORMAL
Primero, generamos los prototipos para la clase NORMAL utilizando el modelo BASELINE. En ellos, podemos observar que las imágenes generadas presentan patrones que recuerdan costillas (líneas blancas horizontales y verticales), con una cierta simetría bilateral y apariencia anatómica (se parecen, si bien vagamente, a una radiografía de tórax normal). Esto indica que el modelo ha aprendido que la presencia de estas estructuras es indicativa de una radiografía normal (no existen opacidades ni anomalías visibles que oculten las costillas).

#### Modelo BASELINE - Clase PNEUMONIA
En cuanto a los prototipos generados para la clase PNEUMONIA, cabe destacar que son relativamente uniformes, con una intensidad media-baja y sin patrones claros o focos específicos. Esto sugiere que el modelo se guía por texturas complejas de bajo nivel que no son fácilmente interpretables por el ojo humano. Esto, como ya hemos visto a lo largo del curso, es una de las mayores limitaciones al tratar de explicar modelos de Visión por Ordenador; esa diferencia tan marcada entre _faithfulness_ y _plausibility_. Intentar simplificar el razonamiento del modelo a mecanismos antropocéntricos (formas, objetos reconocibles) puede llevar a interpretaciones erróneas o engañosas.

#### Modelo SHORTCUT
Respecto al modelo SHORTCUT, los prototipos generados para ambas clases son muy similares entre sí y presentan patrones que más bien parecen ruido aleatorio, sin estructuras reconocibles ni características anatómicas claras. Cabe destacar, eso sí, los _scores_ obtenidos durante la optimización, es decir, la confianza del modelo en la clase objetivo a medida que se optimiza la imagen. Estos _scores_ son muy altos en el caso del modelo BASELINE. En concreto, usando _logits_ como unidad de medida, el modelo BASELINE alcanza valores cercanos a 13 para la clase NORMAL y 40 para la clase PNEUMONIA. Esto indica que el modelo está muy seguro de sus predicciones (además, la gran diferencia entre ambos valores sugiere que las clases están bien diferenciadas en el espacio de características aprendido).

En contraste, el modelo SHORTCUT alcanza _scores_ mucho más bajos: alrededor de 2.5 para la clase NORMAL y -2.6 para la clase PNEUMONIA. Por un lado, las bajas magnitudes indican que el modelo apenas es capaz de diferenciar entre las clases, lo que sugiere un aprendizaje deficiente (o, como ya hemos visto, un aprendizaje basado en _shortcuts_ poco fiables). Por otro lado, el _score_ negativo para la clase PNEUMONIA indica que el modelo tiene una dificultad extrema para generar imágenes de esa clase sin la presencia del _shortcut_. Es decir, en todo momento, la imagen generada se asemeja más a una radiografía NORMAL que a una PNEUMONIA.

Analizando esto más en detalle, dado que el método de _Activation Maximization_ aplica regularización L2 y desenfoque periódico para evitar artefactos de alta frecuencia, dicha optimización no es capaz de encontrar una solución tan específica que requiere:
1. La presencia del _shortcut_ (un círculo perfecto)
2. Con una intensidad muy concreta (blanco puro)
3. En un una localización concreta (la esquina superior izquierda)

En conclusión, estos resultados refuerzan la hipótesis de que el modelo SHORTCUT no ha aprendido características relevantes para la tarea de clasificación, sino que se basa en un atajo espurio (el _shortcut_) para tomar sus decisiones. A diferencia de este, el modelo BASELINE ha aprendido patrones más complejos y generalizables, aunque no necesariamente interpretables para los humanos.
