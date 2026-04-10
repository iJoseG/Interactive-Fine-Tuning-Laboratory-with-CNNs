# 🔬 Laboratorio Interactivo de Fine-Tuning con CNNs

## 📌 Introducción
¿Alguna vez has querido entrenar a una Inteligencia Artificial para que aprenda a reconocer objetos o frutas a partir de tus propias fotografías, pero la barrera de escribir código complejo te ha detenido? 

Este proyecto es una **Aplicación de Escritorio (GUI)** construida íntegramente en Python orientada al desarrollo, análisis e investigación rigurosamente ágil de modelos de **Visión Computacional**. Con una interfaz limpia, oscura y moderna, la herramienta permite a investigadores, científicos de datos y entusiastas realizar **Transfer Learning** y **Fine-Tuning** sobre modelos de última generación pre-entrenados (como ResNet50, MobileNetV2, VGG16) sin necesidad de escribir ni intervenir en código adicional. 

Todo desde un Centro de Control interactivo donde puedes gestionar múltiples escenarios o experimentos paramétricos (MLOps), registrar y comparar métricas históricas de aprendizaje, y efectuar predicciones sobre nuevas imágenes físicas o virtuales usando los diferentes cerebros que vayas guardando.

---

## 🧠 Conceptos Teóricos Aplicados

Antes de sumergirnos en cómo funciona el código por debajo, es fundamental comprender las bases del Machine/Deep Learning sobre las cuales esta herramienta hace magia:

1. **Redes Neuronales Convolucionales (CNNs):** Son la columna vertebral de la visión artificial moderna. Actúan simulando la corteza visual humana, usando "filtros" o "convoluciones" para extraer patrones locales a un altísimo nivel de las imágenes (bordes, curvas, texturas y finalmente ensambla conceptos abstractos como *un faro de coche*, o *la cáscara y estructura fractal de una fruta*).
2. **Transfer Learning (Aprendizaje Transferido):** Entrenar una CNN robusta desde la absoluta ignorancia (desde cero) requiere millones de imágenes variadas e infraestructuras inmensas. El *Transfer Learning* consiste en utilizar e incorporar un modelo que gigantes de la industria (ej. Google) ya estructuraron y entrenaron pacientemente durante meses sobre enciclopedias masivas como *ImageNet* (14 millones de imágenes categorizadas), logrando así "robar" su conocimiento previo y generalizado para inyectarlo como base o punto de partida para que resuelva de manera instantánea tu problema pequeño o muy particular.
3. **Fine-Tuning (Ajuste Fino paramétrico):** Va un paso técnico y audaz más allá del *Transfer Learning*. Consiste en destapar quirúrgicamente las últimas redes del modelo pre-entrenado, "descongelar" (*unfreeze*) esa región densa de memoria superficial, y re-entrenarlo sutilmente (con una rigurosa  **Tasa de Aprendizaje logarítmica muy baja**) de manera que las neuronas se adapten para especializarse un poco más en tu dataset fotográfico casero, pero cuidando celosamente las bases de reconocimiento para no destruir con violencia los pesos, mitigando o esquivando el peligroso fenómeno del *Catastrophic Forgetting*.
4. **Data Augmentation (Aumento de Datos en tiempo de flujo):** Una técnica maravillosamente útil en un dataset modesto o desbalanceado matemáticamente. Consiste en que el algoritmo crea "versiones mutadas infinitas" de las fotos originales rotándolas, alejándolas o invirtiéndolas dinámicamente antes de pasar por el entrenamiento para que la IA virtualmente **nunca vea dos iteraciones o épocas repetidas de exactamente la misma foto**, obligándola a descifrar la norma en vez de limitarse a fotografiar mentalmente los píxeles (aumentando la capacidad *Generalizadora* y volviéndose tolerante a anomalías reales).
5. **Overfitting (Sobreajuste):** El némesis de este procedimiento. Ocurre inevitable o comúnmente cuando el modelo *memoriza con descarada perfección las fotografías puntuales de tu set de entrenamiento* y se jacta de un 100% de precisión. La trampa cae cuando durante las Épocas la *Pérdida en la Validación* (Data invisible) empieza a subir como la espuma a pesar de la bajada original; la red reprobará si le pasas frutas sacadas con tu cámara porque no las estudió "de memoria". Esta interfaz fue calibrada para detener el guardado tan pronto la validación caiga si esto persiste.

---

## 🛠️ Tecnologías y Librerías Flexibles Empleadas

El corazón transversal del proyecto y su backend visual es fuertemente impulsado por el ecosistema vanguardista analítico de Python. Es menester de entorno poseer:

- **`TensorFlow` & `Keras`:** El motor principal robusto y eficiente en C++ originario de Google Brain, encargado aquí de generar los tensores formales, abstraer el flujo denso algorítmico, y la descarga e invocación por factoría de arquitecturas subyacentes maduras (VGG16, MobileNetV2, Inception).
- **`scikit-learn`:** Encargado en el bloque del código de la tabulación post-mortem y algorítmica final del reporte de Clasificación Analógico (Revisión métrica de F1-Score, Recall Positivo Acumulado Numérico y la Precisión Ponderada).
- **`NumPy`:** Base vectorial; provee un ecosistema y algebra hiper-rápido para el arreglo N-dimensional durante la inferencia y el Argmax de clases de probabilidad.
- **`Matplotlib` & `Seaborn`:** Especialistas absolutos en la graficación 2D perimetral paramétrica; operan para renderizar analíticamente las gráficas contiguas de Pérdida (Loss Curve) con su Precisión paralela, y el pintado del mapa de calor cartesiano en las intrincadas y vitales Matrices de Confusión.
- **`ttkbootstrap`:** Actúa inyectando vida en UI. Reasigna tokens por defecto a `tkinter` (la biblioteca nativa esquelética y fundamental por defecto de la librería Standard Python) revigorizándola con matices y plantillas heredadas de la estética y variables CSS Bootstrap (Logrando así el precioso panel "side-by-side", esquemas estilo *Darkly* modernos y las barras responsivas).
- **`Pillow` (PIL):** Lógica nativa de importación bidireccional entre el usuario e Imágenes operadas para los módulos de vista previa UI e input general inicial.

---

## ⚙️ Arquitectura, Lógica y Diagramación Estructural del Código

El archivo monolítico `fine_tuning_app.py` utiliza Arquitectura Orientada a Eventos y Clases, integrando enrutamiento paralelo mediante multihilos (`Threading`):

1. **Gestor Asincrónico por Herencia (Keras Custom Callback):** Debido a que correr un backpropagation durante 15 épocas requeriría acarrear una suspensión (Bloqueo / "Not responding") abismal en el hilo de pantalla (GUI Windows Manager) haciendo imposible utilizar el ratón; entonces el "training.fit()" ha sido exportado al aislamiento de un Thread en forma de Demonio. Como en Python las operaciones inter-hilos cruzadas están prohibidas con primitivos nativos directos... se creó inteligentemente `KerasUIDispatcher`. Al finalizar los batches internos, este puente le manda mensajes en cola purificados y seguros al gestor `root.after()`, permitiendo iluminar al usuario con los logs exactos y el Slider de avance **sin alterar el sub-proceso bloqueante del entrenamiento**.
2. **Splitter Aleatorio Automático del Dataset Físico:** En vez de exigir un protocolo pre-configurado de archivos desparramados en "Train/Val", el usuario solo empuja literalmente las carpetas de las clases unidas en un amigable y trivial `fichero.zip`. El código `shutil` irrumpe destripándolo al interior en memoria virtual en los flujos transitorios correspondientes; y a continuación baraja mediante pseudorandom todas y cada una de las imágenes y las reasigna enviando el `70%` de fotos a la bóveda inmutable de **Train**, aislando `20%` a una cámara separada intocable de **Validación** para el Testeo Ciego de los pesos en cada época, y castigando el último eslabón de `10%` puro intocable es guardado perpetuamente para calificar en frío la Matriz Confusional como último acto del jurado evaluador evaluativo (*Test Set*).
3. **Flujos Infinitos Generator de Keras:** Carga eficientemente imágenes desde los directorios en lotes (*Batches*) aliviando por completo sobrecargas fatales a la memoria RAM, pre-inyectándole *Data Augmentation* paramétrico riguroso pero leve (20° giros y rescalas menores) exclusivamente en la etapa de Train Generator.
4. **Guardias Protectores Anti-Overfitting (Smart `ModelCheckpoint`):** Se delegó a TensorFlow un *Sentry Operativo* para que observe escrutadoramente como bucle la variable `val_accuracy` (La Precisión verdadera con la imagen invisible y de valor máximo generalizador). Cuando la época finaliza, **solo y en tanto la red neurológica logra escalar y dominar un valor métrico de agudeza superior a la de la anterior etapa perenne cronológica, la guarda físicamente** de reemplazo en el disco duro transitorio, asegurando como garantía total, el rescate inminente y perfecto de un `.keras` impecable, librándote así de posibles fatigas sobre-ajustadas postreras del modelo al cierre.
5. **Micro-Base Analítica de Datos (Registro JSON):** Tras la finalización completa, los datos no terminan siendo abandonados o eliminados, toda sesión exitosa siembra un dict de rama multidimensional asociativa completa persistente orientada al archivo plano local `model_registry.json`. Dicho bloque sella y rastrea tanto el destino de la arquitectura física exportada, como cada hiperpréstamo en rigor: Desde "Capa Descongeladas", número del Epoch, Learning Rates, los resultados absolutos de *Testing*, la Resolución, y vitalmente **el arreglo y mapeo ordenado lexicográfico de las clases** bajo los cuáles fue gestado original y naturalmente antes de ver el sol. Todo esto es invocado mediante un asombroso *Binding Combobox Selector* de la propia Pestaña Inferencial final la cual actualiza el tablero métrico comparativo y garantiza que cuando intentes probar tu modelo pasándole una foto en la vida real... Python reconstruya las variables idénticas (*como aplicar Input_Resolution específica*) impidiendo errores letales de pre-procesamiento entre el cerebro base de inferencias históricas inter-cruces.

---

## 🚀 Guía Rápida de Uso (Puesta en Marcha Inicial)

Sigue estos estrictos y estables pasos vía terminal (Bajo un SO Linux/Unix/WSL/macOS) para correr sin contratiempos todo el ambiente utilizando encapsulamiento absoluto local de su Kernel:

### 1. Activar el Entorno Virtual 📦
Resulta imperativo evitar colisiones de C++ con las versiones actuales del core de tu sistema. Requiriendo de un intérprete Python 3.12 (u otra versión oficial de TensorFlow habilitada):
```bash
# Crear tu propio contenedor privado de sub-sistemas virtuales para TF (llamado venv_tf)
python3.12 -m venv venv_tf

# Activar el entorno a tu Bash
source venv_tf/bin/activate
```
*(Debes poder visualizar `(venv_tf)` adscrito junto a tu promt del bash)*

### 2. Instalar el Ecosistema Completo ⚙️
```bash
pip install -r requirements.txt
```

### 3. Ejecutar la Aplicación Gráfica Interactiva 🎨
```bash
python fine_tuning_app.py
```

---

### 📖 Modo de Operación Recomendada Inicial en el Programa:
* **Entrena un Nuevo Escenario Mágico:** Da un bautizo (*Nombre Experimento*) al formulario de la inicial pestaña, y haz búsqueda del archivo `.zip` maestro donde yace un repertorio de clases. Manipula parámetros clave si gustas, clica el botón principal Iniciar, y asiste de observador pasivo la métrica por la Consola o sus Tableros finales para presenciar cómo tu *Inteligencia Artificial Particular* nace.
* **Probar y Confrontar los Logros:** Navega cómodamente a la **Pestaña 2** (El Centro Operacional de Inferencias Visibles). Haz click en el desplegable superior, permitiéndote intercambiar y elegir de entre una innumerable oferta de las diferentes redes artificiales pre-creadas históricamente tuyas. En el sub-panel Izquierdo brotarán resúmenes desvelados de dicha ficha analítica, mientras de lado Paralelo *Derecho*, la red espera devorar un fichero (Botón Cámara de Búsqueda), resolviéndote en pantalla verde una contundente probabilidad probabilística de exactitud la cual el modelo se preste dictaminar en forma de etiqueta predictiva final. 🎉
