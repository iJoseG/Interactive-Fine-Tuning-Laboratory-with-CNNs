"""
fine_tuning_app.py
Aplicación GUI de Escritorio para Múltiples Experimentos de Fine-Tuning
"""
import os
import zipfile
import shutil
import random
import threading
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
from PIL import Image, ImageTk

# Configuración para silenciar los logs verbosos de TensorFlow con C++
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3, MobileNetV2, DenseNet121, EfficientNetB0
from tensorflow.keras.callbacks import Callback, ModelCheckpoint

# Base de Datos Ligera para Registro de Experimentos
REGISTRY_FILE = "model_registry.json"
MODELS_DIR = "modelos_historial"

def load_registry():
    if not os.path.exists(REGISTRY_FILE):
        return {}
    try:
        with open(REGISTRY_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save_registry(data):
    with open(REGISTRY_FILE, "w") as f:
        json.dump(data, f, indent=4)

# Asegurar que el directorio de modelos múltiples existirá
os.makedirs(MODELS_DIR, exist_ok=True)

class KerasUIDispatcher(Callback):
    """
    Despachador para conectar Keras con la Interfaz de Usuario sin romper hilos
    """
    def __init__(self, app, total_epochs):
        super().__init__()
        self.app = app
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        msg = f"Época {epoch+1}/{self.total_epochs} - loss: {logs.get('loss'):.4f} - acc: {logs.get('accuracy'):.4f} - val_loss: {logs.get('val_loss'):.4f} - val_acc: {logs.get('val_accuracy'):.4f}"
        self.app.log(msg)
        self.app.update_progress((epoch + 1) / self.total_epochs * 100)

class FineTuningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Entrenador Interactivo de CNNs y Catálogo Histórico")
        self.root.geometry("900x820")
        
        # Variables de Configuración de Experimentos
        self.model_name_var = ttk.StringVar(value="Mi_Experimento_Frutas_1")
        self.zip_path = ttk.StringVar()
        self.model_var = ttk.StringVar(value='MobileNetV2')
        self.unfreeze_var = ttk.IntVar(value=0)
        self.img_size_var = ttk.IntVar(value=224)
        self.batch_size_var = ttk.IntVar(value=32)
        self.lr_var = ttk.StringVar(value='0.0001')
        self.epochs_var = ttk.IntVar(value=10)
        
        # Variables de Inferencia (Testeo Visual)
        self.test_img_path = ttk.StringVar()
        self.tk_image = None
        self.selected_test_model = ttk.StringVar()

        # Cargar Base de Datos Local
        self.registry = load_registry()
        
        self.build_ui()
        self.update_models_combo()
        
    def build_ui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=BOTH, expand=YES, padx=10, pady=10)
        
        # --- PESTAÑA DE CREACIÓN DE EXPERIMENTOS ---
        self.train_tab = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(self.train_tab, text="🏋️ Crear y Entrenar Escenario")
        self.build_train_tab()

        # --- PESTAÑA DEL CATÁLOGO Y PREDICCIONES ---
        self.test_tab = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(self.test_tab, text="📚 Catálogo Histórico y Pruebas")
        self.build_test_tab()
        
    def build_train_tab(self):
        main_frame = self.train_tab
        
        lbl_title = ttk.Label(main_frame, text="🔬 Configuración de Nuevo Experimento", font="-size 16 -weight bold", bootstyle=PRIMARY)
        lbl_title.pack(pady=(0,15))

        config_frame = ttk.LabelFrame(main_frame, text="⚙️ Parámetros y Arquitectura")
        config_frame.pack(fill=X, pady=5, ipadx=10, ipady=10)

        # Nombre Personalizado del Experimento
        name_frame = ttk.Frame(config_frame)
        name_frame.pack(fill=X, pady=(0,5))
        ttk.Label(name_frame, text="📌 Nombre para Guardar tu Red:", width=30).pack(side=LEFT, padx=5)
        ttk.Entry(name_frame, textvariable=self.model_name_var, width=30).pack(side=LEFT, padx=5, fill=X, expand=YES)

        # File picker
        file_frame = ttk.Frame(config_frame)
        file_frame.pack(fill=X, pady=(5, 10))
        ttk.Label(file_frame, text="📂 Dataset (ZIP) frutas:", width=30).pack(side=LEFT, padx=5)
        ttk.Entry(file_frame, textvariable=self.zip_path, state='readonly').pack(side=LEFT, fill=X, expand=YES, padx=5)
        ttk.Button(file_frame, text="Buscar Archivo...", command=self.browse_file, bootstyle=SECONDARY).pack(side=LEFT, padx=5)

        # Matriz de variables Numéricas
        param_frame = ttk.Frame(config_frame)
        param_frame.pack(fill=X, pady=5)

        # Fila 0
        ttk.Label(param_frame, text="Modelo CNN Base:").grid(row=0, column=0, padx=5, pady=8, sticky=W)
        ttk.Combobox(param_frame, textvariable=self.model_var, values=['EfficientNetB0', 'VGG16', 'ResNet50', 'InceptionV3', 'MobileNetV2', 'DenseNet121'], state='readonly', width=16).grid(row=0, column=1, padx=5, pady=8)
        ttk.Button(param_frame, text="ℹ️", bootstyle="info-outline", cursor="hand2", command=lambda: self.show_info("modelo")).grid(row=0, column=2, padx=2)

        ttk.Label(param_frame, text="Learning Rate:").grid(row=0, column=3, padx=(25, 5), pady=8, sticky=W)
        ttk.Combobox(param_frame, textvariable=self.lr_var, values=['0.01', '0.001', '0.0001', '0.00001'], width=10).grid(row=0, column=4, padx=5, pady=8)
        ttk.Button(param_frame, text="ℹ️", bootstyle="info-outline", cursor="hand2", command=lambda: self.show_info("lr")).grid(row=0, column=5, padx=2)

        # Fila 1
        ttk.Label(param_frame, text="Capas a descongelar:").grid(row=1, column=0, padx=5, pady=8, sticky=W)
        ttk.Spinbox(param_frame, textvariable=self.unfreeze_var, from_=0, to=100, width=14).grid(row=1, column=1, padx=5, pady=8)
        ttk.Button(param_frame, text="ℹ️", bootstyle="info-outline", cursor="hand2", command=lambda: self.show_info("descongelar")).grid(row=1, column=2, padx=2)

        ttk.Label(param_frame, text="Cant. Épocas:").grid(row=1, column=3, padx=(25, 5), pady=8, sticky=W)
        ttk.Spinbox(param_frame, textvariable=self.epochs_var, from_=1, to=150, width=10).grid(row=1, column=4, padx=5, pady=8)
        ttk.Button(param_frame, text="ℹ️", bootstyle="info-outline", cursor="hand2", command=lambda: self.show_info("epocas")).grid(row=1, column=5, padx=2)

        # Fila 2
        ttk.Label(param_frame, text="Tamaño Img:").grid(row=2, column=0, padx=5, pady=8, sticky=W)
        ttk.Spinbox(param_frame, textvariable=self.img_size_var, from_=32, to=512, width=14).grid(row=2, column=1, padx=5, pady=8)
        ttk.Button(param_frame, text="ℹ️", bootstyle="info-outline", cursor="hand2", command=lambda: self.show_info("img_size")).grid(row=2, column=2, padx=2)

        ttk.Label(param_frame, text="Batch Size:").grid(row=2, column=3, padx=(25, 5), pady=8, sticky=W)
        ttk.Spinbox(param_frame, textvariable=self.batch_size_var, from_=1, to=512, width=10).grid(row=2, column=4, padx=5, pady=8)
        ttk.Button(param_frame, text="ℹ️", bootstyle="info-outline", cursor="hand2", command=lambda: self.show_info("batch")).grid(row=2, column=5, padx=2)

        self.btn_run = ttk.Button(main_frame, text="▶️ ENTRENAR Y REGISTRAR EN EL CATÁLOGO", bootstyle="success", command=self.start_training)
        self.btn_run.pack(fill=X, pady=10)

        self.progress = ttk.Progressbar(main_frame, bootstyle="info-striped", maximum=100)
        self.progress.pack(fill=X, pady=(0, 10))

        console_frame = ttk.LabelFrame(main_frame, text="Registro del Entrenamiento (Logs en Vivo)")
        console_frame.pack(fill=BOTH, expand=YES, ipadx=5, ipady=5)
        
        self.console = ttk.Text(console_frame, height=8, state='normal')
        scroll = ttk.Scrollbar(console_frame, command=self.console.yview, bootstyle=SECONDARY)
        self.console.configure(yscrollcommand=scroll.set)
        
        self.console.pack(side=LEFT, fill=BOTH, expand=YES)
        scroll.pack(side=RIGHT, fill=Y)

    def build_test_tab(self):
        main_frame = self.test_tab
        
        # Contenedor padre horizontal
        h_container = ttk.Frame(main_frame)
        h_container.pack(fill=BOTH, expand=YES, pady=5)
        
        # --- COLUMNA IZQUIERDA (Catálogo y Ficha Técnica) ---
        left_panel = ttk.Frame(h_container)
        left_panel.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 10))
        
        selector_frame = ttk.LabelFrame(left_panel, text="1. Selecciona Histórico")
        selector_frame.pack(fill=BOTH, expand=YES, ipadx=15, ipady=15)
        
        ttk.Label(selector_frame, text="Modelos Entrenados:", font="-weight bold").pack(anchor=W, pady=(0,5))
        self.combo_models = ttk.Combobox(selector_frame, textvariable=self.selected_test_model, state="readonly", width=40)
        self.combo_models.pack(fill=X, pady=5)
        self.combo_models.bind("<<ComboboxSelected>>", self.on_model_selected)
        
        ttk.Label(selector_frame, text="Estadísticas de este modelo:").pack(anchor=W, pady=(15, 5))
        self.lbl_model_stats = ttk.Label(
            selector_frame, 
            text="Selecciona un modelo...", 
            bootstyle=INFO, justify=LEFT, font="-size 10"
        )
        self.lbl_model_stats.pack(fill=X, pady=5)
        
        # --- COLUMNA DERECHA (Prueba Visual) ---
        right_panel = ttk.Frame(h_container)
        right_panel.pack(side=LEFT, fill=BOTH, expand=True, padx=(10, 0))
        
        test_frame = ttk.LabelFrame(right_panel, text="2. Inferencia en Nueva Fotografía")
        test_frame.pack(fill=BOTH, expand=YES, ipadx=15, ipady=15)
        
        btn_frame = ttk.Frame(test_frame)
        btn_frame.pack(fill=X, pady=10)
        
        ttk.Button(btn_frame, text="📷 1. Buscar Foto", bootstyle=PRIMARY, command=self.browse_test_image).pack(side=LEFT, padx=5, fill=X, expand=YES)
        ttk.Button(btn_frame, text="🧠 2. PREDECIR", bootstyle=SUCCESS, command=self.predict_image).pack(side=LEFT, padx=5, fill=X, expand=YES)
        
        # Poner el resultado ARRIBA de la imagen para que jamás se oculte de la vista
        self.lbl_result = ttk.Label(test_frame, text="Resultado: ---", font="-size 20 -weight bold", bootstyle=WARNING, justify=CENTER)
        self.lbl_result.pack(pady=10)
        
        self.lbl_image_preview = ttk.Label(test_frame, text="[ Aún no has seleccionado ninguna foto ]", justify=CENTER)
        self.lbl_image_preview.pack(pady=10, fill=BOTH, expand=YES)

    # ================= EVENTOS DE INTERFAZ Y ACTUALIZACIONES =================
    
    def show_info(self, param):
        info_texts = {
            "modelo": "💡 Arquitectura Base (Pre-entrenada)\n\n"
                      "Es el 'Cerebro' pre-educado que actuaré como base. Toma años de experiencia de Google/Microsoft.\n"
                      "• Aumentarlo (ej. EfficientNetB0 o ResNet): Modelos sumamente precisos e inteligentes por naturaleza; sin embargo son muy pesados, ocupan más VRAM y se entrenan lento.\n"
                      "• Disminuirlo (ej. MobileNetV2): Menos asombrosos en precisión absoluta, pero ultraligeros, entrenan hiperrápido e ideales para dispositivos con baja potencia.",
            "lr": "💡 Tasa de Aprendizaje (Learning Rate)\n\n"
                  "Controla la 'agresividad' con la que el modelo modifica sus neuronas al ver errores.\n"
                  "• Aumentarlo (ej. 0.01): Aprende rápido al inicio, pero a menudo 'salta' la respuesta volviéndose inestable, o sufre amnesia rompiendo el conocimiento genérico de ImageNet.\n"
                  "• Disminuirlo (ej. 0.00001): Avanza muy lento, pero de forma micro-quirúrgica y muy meticulosa; óptimo para un Fine-Tuning delicado.",
            "descongelar": "💡 Capas a Descongelar (Unfreeze)\n\n"
                           "A cuántas capas profundas pre-congeladas de la arquitectura se les alterará la memoria.\n"
                           "• 0 Capas: Respeta a rajatabla lo aprendido por defecto; excelente si tus imágenes son algo comunes.\n"
                           "• Capas elevadas (ej. 20+): Permite a la red alterar su percepción y detectar patrones raros si tu dataset es enorme, caso contrario causarás Overfitting fulminante.",
            "epocas": "💡 Épocas (Epochs)\n\n"
                      "Garantizan ciclos en que tu red repasa rigurosamente todo el Dataset de principio a fin.\n"
                      "• El Autoguardado evita perjuicios por exceso (guarda en su peak de validación), aunque poner 150 épocas ocuparía tiempo valioso que pudiste usar analizando los cortes.",
            "img_size": "💡 Resolución Visual (Tamaño Img)\n\n"
                        "Define con qué cantidad N x N de pixeles la red debe destripar la foto.\n"
                        "• Tamaños enormes (Ej. 299x299): Obligatorio a veces para Inception, logrando texturas minuciosas excepcionales a costo crítico de VRAM.\n"
                        "• Menores (Ej. 128x128): Relucientemente veloces para probar un parpadeo de época entero.",
            "batch": "💡 Tamaño de Lote (Batch Size)\n\n"
                     "Pack de fotografías cargadas al mismo instante en memoria GPU para consolidar un aprendizaje central.\n"
                     "• Aumentarlo exige alta memoria temporal, el modelo en ocasiones sufre al quedar encajonado en atajos globales estáticos.\n"
                     "• Disminuirlo a ~32 otorga al sistema saltos de exploración muy sanos pero demora ligeramentre más en computar el total."
        }
        texto = info_texts.get(param, "Sin información disponible.")
        messagebox.showinfo("Documentación Científica", texto)

    def update_models_combo(self):
        """ Actualiza la tabla de modelos dropdown según el registro de historial """
        models = list(self.registry.keys())
        self.combo_models['values'] = models
        if models:
            # Seleccionar automáticamente el más nuevo
            self.combo_models.current(len(models)-1)
            self.on_model_selected()

    def on_model_selected(self, event=None):
        """ Se dispara al elegir un modelo en el Combo, para mostrar todos sus datos y hacer comparaciones """
        model_key = self.selected_test_model.get()
        if not model_key: return
        data = self.registry.get(model_key, {})
        
        # Cremos un reporte rápido de sus métricas
        val_acc = data.get('test_accuracy', 0) * 100
        val_loss = data.get('test_loss', 0)
        
        stats = (
            f"🔍 FICHA TÉCNICA DEL EXPERIMENTO: '{model_key}'\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 Precisión Final de Prueba: {val_acc:.2f}%     |    📉 Error General: {val_loss:.4f}\n"
            f"🤖 Arquitectura CNN usada: {data.get('base_cnn')}  |    ✂️ Capas Destruídas (Unfrozen): {data.get('unfreeze')}\n"
            f"🔧 Épocas dadas: {data.get('epochs')}            |    🎒 Tamaño Batch: {data.get('batch_size')}\n"
            f"⚙️ Learning Rate usado: {data.get('learning_rate')}     |    🖼️ Resol. Img: {data.get('img_size')}x{data.get('img_size')} píxeles\n"
            f"📂 Archivo H5/Keras: {data.get('file_path')}"
        )
        self.lbl_model_stats.config(text=stats)
        
        # Borrar resultado anterior para no confundir
        self.lbl_result.config(text="Resultado: ---", bootstyle=WARNING)

    def browse_file(self):
        filepath = askopenfilename(filetypes=[("Archivos Recortados/ZIP", "*.zip")])
        if filepath:
            self.zip_path.set(filepath)
            
    def browse_test_image(self):
        filepath = askopenfilename(filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp")])
        if filepath:
            self.test_img_path.set(filepath)
            try:
                img = Image.open(filepath)
                img.thumbnail((300, 300))  # redimensionamos la visualización (la real se analiza gigante igual)
                self.tk_image = ImageTk.PhotoImage(img)
                self.lbl_image_preview.config(image=self.tk_image, text="")
                self.lbl_result.config(text="Listo para analizar.", bootstyle=SECONDARY)
            except Exception as e:
                self.lbl_result.config(text="Error abriendo imagen.", bootstyle=DANGER)

    def log(self, msg):
        self.root.after(0, self._log, msg)

    def _log(self, msg):
        self.console.insert(END, msg + "\n")
        self.console.see(END)

    def update_progress(self, val):
        self.root.after(0, lambda: self._update_progress(val))

    def _update_progress(self, val):
        self.progress['value'] = val

    def finish_training(self):
        self.btn_run.config(state=NORMAL)
        self.update_models_combo() # Se forzará la actualización del Combo de Test
        self.log("✅ Proceso terminado y Catálogo sincronizado.")

    # ================= LOGICA DE GESTIOR DE PROCESOS =================

    def start_training(self):
        # Validar Nombre Experimento Vacío
        exp_name = self.model_name_var.get().strip()
        if not exp_name:
            messagebox.showerror("Aviso Requerido", "¡Debes asignar un nombre a este experimento para poder guardarlo y catalogarlo!")
            return
            
        # Preguntar si sobreescribe uno con mismo nombre
        if exp_name in self.registry:
            respuesta = messagebox.askyesno("Confirmación de Sobreescritura", f"Ya existe un experimento en la lista con el nombre '{exp_name}'. ¿Deseas sobreescribirlo y peredre sus datos previos?")
            if not respuesta:
                return

        if not self.zip_path.get() or not os.path.exists(self.zip_path.get()):
            messagebox.showerror("Error", "Por favor selecciona un archivo ZIP de datos válido.")
            return

        self.btn_run.config(state=DISABLED)
        self.progress['value'] = 0
        self.console.delete(1.0, END)
        self.log(f"🚀 Iniciando Misión de Entrenamiento: '{exp_name}' ...")

        t = threading.Thread(target=self.training_process, daemon=True)
        t.start()
        
    def predict_image(self):
        """ Efectúa Inferencias utilizando la ARRANQUE EXCLUSIVA del Modelo Actualmente Seleccionado en la Lista """
        selected_model_name = self.selected_test_model.get()

        if not selected_model_name or selected_model_name not in self.registry:
            messagebox.showerror("Error de Catálogo", "No hay ningún modelo seleccionado en la lista.")
            return

        if not self.test_img_path.get():
            messagebox.showerror("Falta Imagen", "Sube una foto para que el modelo pueda verla.")
            return
            
        self.lbl_result.config(text="Reconstruyendo neuronas y ejecutando pase adelante...", bootstyle=INFO)
        
        try:
            # Traer diccionario entero del Experimento Seleccionado
            data = self.registry[selected_model_name]
            filepath = data['file_path']
            img_size_needed = data['img_size']
            classes_trained = data['classes']
            
            if not os.path.exists(filepath):
                self.lbl_result.config(text=f"El archivo Keras del modelo ({filepath}) fue movido o borrado internamente.", bootstyle=DANGER)
                return
            
            # Cargar Arquitectura Física Específica al Experimento
            model = load_model(filepath)
            
            # Pre-procesamiento de la Foto (Usando el tamaño de imagen EXACTO con el que se configuró el experimento)
            img_path = self.test_img_path.get()
            img = load_img(img_path, target_size=(img_size_needed, img_size_needed))
            img_array = img_to_array(img)
            img_array = img_array / 255.0  # El factor clave
            img_array = np.expand_dims(img_array, axis=0)
            
            # Inferencia real
            predictions = model.predict(img_array, verbose=0)
            class_idx = np.argmax(predictions[0])
            confidence = predictions[0][class_idx]
            predicted_class = classes_trained[class_idx]
            
            self.lbl_result.config(
                text=f"🍎 Resultado: {predicted_class.upper()} (Seguridad del {confidence*100:.1f}%)", 
                bootstyle=SUCCESS
            )
            
        except Exception as e:
            self.lbl_result.config(text=f"Error en inferencia: {str(e)}", bootstyle=DANGER)

    def training_process(self):
        try:
            exp_name = self.model_name_var.get().strip()
            # La ruta de guardado del modelo para ESTE experimento puntual
            save_path = os.path.join(MODELS_DIR, f"{exp_name}_modelo.keras")
            
            self.log("📦 Descomprimiendo el ZIP y estructurando subcarpetas...")
            zip_path = self.zip_path.get()
            base_dir = "./dataset_temp"
            dataset_split_dir = "./dataset_split"

            if os.path.exists(base_dir): shutil.rmtree(base_dir)
            if os.path.exists(dataset_split_dir): shutil.rmtree(dataset_split_dir)
            os.makedirs(base_dir, exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(base_dir)

            dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
            images_dir = base_dir
            if len(dirs) == 1:
                check_dir = os.path.join(base_dir, dirs[0])
                subdirs = [d for d in os.listdir(check_dir) if os.path.isdir(os.path.join(check_dir, d))]
                if len(subdirs) > 0:
                    images_dir = check_dir

            classes = [d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))]
            classes.sort()
            n_classes = len(classes)
            self.log(f"📊 {n_classes} Clases identificadas y en orden alfabético estricto.")

            if n_classes < 2:
                self.log("❌ Error: Se detectaron menos de 2 clases. Operación cancelada.")
                return

            train_dir = os.path.join(dataset_split_dir, 'train')
            val_dir = os.path.join(dataset_split_dir, 'val')
            test_dir = os.path.join(dataset_split_dir, 'test')

            for split in [train_dir, val_dir, test_dir]:
                for cls in classes:
                    os.makedirs(os.path.join(split, cls), exist_ok=True)

            valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
            for cls in classes:
                cls_dir = os.path.join(images_dir, cls)
                images = [f for f in os.listdir(cls_dir) if f.lower().endswith(valid_extensions)]
                random.shuffle(images)

                t_bound = int(len(images) * 0.70)
                v_bound = int(len(images) * 0.90)

                for f in images[:t_bound]: shutil.copy2(os.path.join(cls_dir, f), os.path.join(train_dir, cls, f))
                for f in images[t_bound:v_bound]: shutil.copy2(os.path.join(cls_dir, f), os.path.join(val_dir, cls, f))
                for f in images[v_bound:]: shutil.copy2(os.path.join(cls_dir, f), os.path.join(test_dir, cls, f))

            # Fetch State
            b_size = int(self.batch_size_var.get())
            i_size = int(self.img_size_var.get())
            model_name = self.model_var.get()
            unfreeze = int(self.unfreeze_var.get())
            lr = float(self.lr_var.get())
            epochs = int(self.epochs_var.get())

            train_datagen = ImageDataGenerator(
                rescale=1./255, rotation_range=20, width_shift_range=0.2, 
                height_shift_range=0.2, horizontal_flip=True, fill_mode='nearest')
            val_test_datagen = ImageDataGenerator(rescale=1./255)

            train_gen = train_datagen.flow_from_directory(train_dir, target_size=(i_size, i_size), batch_size=b_size, class_mode='categorical')
            val_gen = val_test_datagen.flow_from_directory(val_dir, target_size=(i_size, i_size), batch_size=b_size, class_mode='categorical')
            test_gen = val_test_datagen.flow_from_directory(test_dir, target_size=(i_size, i_size), batch_size=b_size, class_mode='categorical', shuffle=False)

            self.log(f"🧠 Aterrizando a {model_name} de memoria pesada ImageNet...")
            model_dict = {
                'EfficientNetB0': EfficientNetB0,
                'VGG16': VGG16, 
                'ResNet50': ResNet50, 
                'InceptionV3': InceptionV3, 
                'MobileNetV2': MobileNetV2, 
                'DenseNet121': DenseNet121
            }
            ModelClass = model_dict[model_name]
            
            try:
                base_model = ModelClass(weights='imagenet', include_top=False, input_shape=(i_size, i_size, 3))
            except ValueError as e:
                self.log(f"❌ Error de Keras: {str(e)}\n Intenta usar otro 'Tamaño Img'")
                return

            base_model.trainable = True
            if unfreeze > 0:
                for layer in base_model.layers[:-unfreeze]: layer.trainable = False
            else:
                for layer in base_model.layers: layer.trainable = False

            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(256, activation='relu')(x)
            x = Dropout(0.5)(x)
            predictions = Dense(n_classes, activation='softmax')(x)

            model = Model(inputs=base_model.input, outputs=predictions)
            model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])

            # === LA CLAVE PARA EL MÚLTIPLE ENTRENAMIENTO ===
            # Creamos un Guardador Automático (Checkpoint), apuntando a nuestor `save_path` ÚNICO
            checkpoint = ModelCheckpoint(save_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=0)
            callback_ui = KerasUIDispatcher(self, epochs)
            
            self.log(f"🏃 Entrenando y vigilando el Overfitting...")
            history = model.fit(
                train_gen, validation_data=val_gen, epochs=epochs,
                verbose=0, callbacks=[callback_ui, checkpoint]
            )

            self.log("🔎 Evaluando capacidad generalizadora real y final... (Test Set puro)")
            test_loss, test_acc = model.evaluate(test_gen, verbose=0)
            self.log(f"🎯 Test Scoring Oficial -> Precisión Oculta Lograda: {test_acc*100:.2f}%")

            # --- PARTE NUEVA MÁXIMA IMPORTANCIA: Registrar toda la historia al Archivo Maestro ---
            self.log("💾 Escribiendo la documentación métrica del modelo en el Historial global...")
            self.registry[exp_name] = {
                "file_path": save_path,
                "base_cnn": model_name,
                "unfreeze": unfreeze,
                "learning_rate": lr,
                "img_size": i_size,
                "epochs": epochs,
                "batch_size": b_size,
                "test_accuracy": float(test_acc),
                "test_loss": float(test_loss),
                "classes": list(train_gen.class_indices.keys())
            }
            save_registry(self.registry) # Commitiendolo en el archivo JSON físico
            
            # Reporte Visual para Matplotlib de este Experimento
            test_gen.reset()
            preds = model.predict(test_gen, verbose=0)
            y_pred = np.argmax(preds, axis=1)
            y_true = test_gen.classes

            report = classification_report(y_true, y_pred, target_names=classes)
            self.log(f"\n📊 Perfilamiento Final F1, Recall y Precisón:\n{report}")
            cm = confusion_matrix(y_true, y_pred)

            if os.path.exists(base_dir): shutil.rmtree(base_dir)

            self.root.after(0, self.show_results, exp_name, history.history, cm, classes)

        except Exception as e:
            self.log(f"❌ Error crítico en el entrenamiento: {str(e)}")
        finally:
            self.root.after(0, self.finish_training)

    def show_results(self, exp_name, history, cm, classes):
        # 1. Gráficas
        plt.figure(f"Historial Formativo de: {exp_name}", figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Train Loss', color='blue')
        plt.plot(history['val_loss'], label='Val Loss', color='red', linestyle='--')
        plt.title('Pérdida (Minimizar es Bueno)')
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.legend()
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.plot(history['accuracy'], label='Train Acc', color='blue')
        plt.plot(history['val_accuracy'], label='Val Acc', color='red', linestyle='--')
        plt.title('Precisión (Maximizar sin Overfitting)')
        plt.xlabel('Época')
        plt.ylabel('Precisión')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show(block=False)

        # 2. Matriz
        plt.figure(f"Conflictos del Experimento: {exp_name}", figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title(f'Matriz de Confusión (Test) - {exp_name}')
        plt.ylabel('Fruta Verdadera Oficial')
        plt.xlabel('Diagnóstico del Cerebro de la IA')
        plt.tight_layout()
        plt.show(block=False)

if __name__ == "__main__":
    app = ttk.Window(title="Laboratorio de Inteligencia Artificial IA", themename="darkly") 
    gui = FineTuningApp(app)
    app.mainloop()
