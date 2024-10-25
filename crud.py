import tkinter as tk
from tkinter import ttk, messagebox
import mysql.connector
import numpy as np
from tensorflow.keras.models import load_model
import time
import matplotlib.pyplot as plt
import os
import cv2
import imutils
import sys
import io

# Verifica si sys.stdout está disponible y tiene un atributo 'encoding'
if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# Conexión a la base de datos
def connect_db():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='deteccion_emociones'
    )

# Función para agregar un nuevo alumno
def agregar_alumno():
    matricula = entry_matricula.get()
    nombre = entry_nombre.get()
    grupo = entry_grupo.get()

    db = connect_db()
    cursor = db.cursor()
    try:
        cursor.execute(
            "INSERT INTO Alumno (matricula, nombre, grupo) VALUES (%s, %s, %s)",
            (matricula, nombre, grupo)
        )
        db.commit()
        messagebox.showinfo("Éxito", "Alumno agregado correctamente")
        entry_matricula.delete(0, tk.END)
        entry_nombre.delete(0, tk.END)
        entry_grupo.delete(0, tk.END)
    except mysql.connector.Error as err:
        messagebox.showerror("Error", f"Error al agregar el alumno: {str(err)}")
    finally:
        cursor.close()
        db.close()

# Función para iniciar la detección de emociones
def iniciar_deteccion_emocion():
    matricula = entry_matricula_emocion.get()
    nombre_alumno = obtener_nombre_alumno(matricula)

    if not nombre_alumno:
        messagebox.showerror("Error", "Matrícula no encontrada")
        return

    # Rutas del modelo
    prototxtPath = "D:/Nueva carpeta/Face_Emotion-master/face_detector/deploy.prototxt"
    weightsPath = "D:/Nueva carpeta/Face_Emotion-master/face_detector/res10_300x300_ssd_iter_140000.caffemodel"

    if not os.path.isfile(prototxtPath) or not os.path.isfile(weightsPath):
        messagebox.showerror("Error", "Archivos del modelo no encontrados.")
        return

    try:
        faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
        emotionModel = load_model("D:/Nueva carpeta/Face_Emotion-master/modelFEC.h5")
    except Exception as e:
        messagebox.showerror("Error", f"Error al cargar el modelo: {e}")
        return

    classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cam.isOpened():
        messagebox.showerror("Error", "No se pudo abrir la cámara.")
        return

    start_time = time.time()
    duration = 10
    detecciones = {clase: 0 for clase in classes}

    plt.ion()
    x = range(len(classes))
    figura1 = plt.figure()
    y = [0] * len(classes)
    bar1 = plt.bar(x, y, color=['r', 'g', 'b', 'y', 'k', 'm', 'c'], width=0.4)
    plt.xticks(x, classes)
    plt.ylim(0, 1)
    plt.grid(True)

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=640)
        locs, preds = predict_emotion(frame, faceNet, emotionModel)

        for (box, pred) in zip(locs, preds):
            (Xi, Yi, Xf, Yf) = box
            label = f"{nombre_alumno}: {np.max(pred) * 100:.2f}%"
            cv2.rectangle(frame, (Xi, Yi), (Xf, Yf), (255, 0, 0), 2)
            cv2.putText(frame, label, (Xi, Yi - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

            for i in range(len(bar1)):
                bar1[i].set_height(pred[i])
            figura1.canvas.draw()

            emocion_detectada = classes[np.argmax(pred)]
            detecciones[emocion_detectada] += 1

        cv2.imshow("Detector de Emociones", frame)

        if time.time() - start_time > duration or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    guardar_emociones_en_db(matricula, detecciones)

def guardar_emociones_en_db(matricula, detecciones):
    db = connect_db()
    cursor = db.cursor()
    for emocion, cantidad in detecciones.items():
        if cantidad > 0:
            cursor.execute(
                "INSERT INTO Emocion (matricula, nombre_emocion, cantidad, fecha) VALUES (%s, %s, %s, NOW())",
                (matricula, emocion, cantidad)
            )
    db.commit()
    cursor.close()
    db.close()
    messagebox.showinfo("Éxito", "Emociones guardadas correctamente")

def obtener_nombre_alumno(matricula):
    db = connect_db()
    cursor = db.cursor()
    cursor.execute("SELECT nombre FROM Alumno WHERE matricula = %s", (matricula,))
    row = cursor.fetchone()
    cursor.close()
    db.close()
    return row[0] if row else None

def predict_emotion(frame, faceNet, emotionModel):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces, locs, preds = [], [], []
    for i in range(0, detections.shape[2]):
        if detections[0, 0, i, 2] > 0.4:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0]] * 2)
            (Xi, Yi, Xf, Yf) = box.astype("int")
            face = frame[Yi:Yf, Xi:Xf]

            # Check if the face is not empty
            if face.size > 0:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (48, 48))
                face = np.expand_dims(np.expand_dims(face, axis=0), axis=-1)
                locs.append((Xi, Yi, Xf, Yf))
                preds.append(emotionModel.predict(face)[0])
    return locs, preds

# Interfaz gráfica
root = tk.Tk()
root.title("Detección de Emociones")
root.geometry("600x500")
root.configure(bg="#e0f7fa")  # Fondo azul claro

# Estilo
style = ttk.Style()
style.configure("TLabel", background="#e0f7fa", font=("Arial", 12))
style.configure("TButton", padding=10, relief="flat", background="#00796b", foreground="white", font=("Arial", 12, "bold"))
style.map("TButton", background=[("active", "#004d40")])  # Color al pasar el mouse
style.configure("TFrame", background="#e0f7fa")

# Agregar alumno
frame_alumno = ttk.LabelFrame(root, text="Agregar Alumno", padding=(10, 10))
frame_alumno.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

ttk.Label(frame_alumno, text="Matrícula:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
entry_matricula = ttk.Entry(frame_alumno)
entry_matricula.grid(row=0, column=1, padx=5, pady=5)

ttk.Label(frame_alumno, text="Nombre:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
entry_nombre = ttk.Entry(frame_alumno)
entry_nombre.grid(row=1, column=1, padx=5, pady=5)

ttk.Label(frame_alumno, text="Grupo:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
entry_grupo = ttk.Entry(frame_alumno)
entry_grupo.grid(row=2, column=1, padx=5, pady=5)

btn_agregar = ttk.Button(frame_alumno, text="Agregar Alumno", command=agregar_alumno)
btn_agregar.grid(row=3, column=0, columnspan=2, padx=5, pady=10)

# Registro de emociones
frame_emocion = ttk.LabelFrame(root, text="Registrar Emoción", padding=(10, 10))
frame_emocion.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

ttk.Label(frame_emocion, text="Matrícula:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
entry_matricula_emocion = ttk.Entry(frame_emocion)
entry_matricula_emocion.grid(row=0, column=1, padx=5, pady=5)

btn_iniciar_deteccion = ttk.Button(frame_emocion, text="Iniciar Detección", command=iniciar_deteccion_emocion)
btn_iniciar_deteccion.grid(row=1, column=0, columnspan=2, padx=5, pady=10)

root.mainloop()