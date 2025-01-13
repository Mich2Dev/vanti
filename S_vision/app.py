import cv2
import torch
import numpy as np
import statistics
from flask import Flask, render_template, Response, jsonify
import threading
from flask_cors import CORS
import warnings

###############################################################################
#                        IGNORAR LOS FUTUREWARNING DE TORCH                   #
###############################################################################
warnings.filterwarnings("ignore", category=FutureWarning)

###############################################################################
#                           CONFIGURACIÓN GENERAL                             #
###############################################################################
CONFIDENCE_THRESHOLD = 0.5   # Umbral de confianza para submodelos
IOU_THRESHOLD = 0.5          # Umbral para NMS
CONSENSUS_COUNT = 5          # Cantidad de lecturas similares para confirmar detección
MAX_BUFFER_SIZE = 10         # Tamaño máximo del buffer de lecturas

CAMERA_INDEX = 2             # Índice de la cámara (ajusta según tu sistema)
DETECT_ACTIVE = False
LOCK = threading.Lock()

###############################################################################
#                 HISTORIAL Y BUFFERS PARA CADA MEDIDOR                       #
###############################################################################
# Almacenamos todas las lecturas confirmadas
HISTORY = {
    'Medidor_1': [], 
    'Medidor_2': [],
    'Medidor_3': [],
    'Medidor_4': []
}

# Buffer temporal donde se guardan lecturas hasta que se cumpla el consenso
DETECTION_BUFFER = {
    'Medidor_1': [],
    'Medidor_2': [],
    'Medidor_3': [],
    'Medidor_4': []
}

###############################################################################
#                            FUNCIONES AUXILIARES                             #
###############################################################################
def load_local_model(pt_file: str):
    """
    Carga un modelo YOLOv5 local (sin conexión a internet) desde ./yolov5
    usando torch.hub.load con source='local'.
    """
    return torch.hub.load(
        './yolov5',
        'custom',
        path=pt_file,
        source='local',
        force_reload=False
    )

def iou(boxA, boxB):
    """Cálculo simple de Intersection Over Union (IOU)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = areaA + areaB - inter_area

    return 0.0 if union_area == 0 else inter_area / union_area

def nms(detections, iou_thres=0.5):
    """Non-Maximum Suppression básico."""
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: d['confidence'], reverse=True)
    final = []
    while detections:
        best = detections.pop(0)
        final.append(best)
        detections = [
            d for d in detections 
            if iou(best['box'], d['box']) < iou_thres
        ]
    return final

def check_consensus(buffer_list, required_count=5):
    """
    Verifica si en buffer_list hay un consenso >= required_count.
    - Si todas las últimas 'required_count' lecturas son iguales => confirmamos.
    """
    if len(buffer_list) < required_count:
        return None

    # Tomar solo las últimas 'required_count'
    last_values = buffer_list[-required_count:]

    # Verificar si todas son idénticas
    if all(v == last_values[0] for v in last_values):
        return last_values[-1]  # la última detección se confirma

    return None

###############################################################################
#             FUNCIONES DE VALIDACIÓN (SE ACEPTA SI ALGUNA ES TRUE)           #
###############################################################################
def is_progressive_valid(medidor, new_str):
    """
    Validación 1: Chequeo progresivo (evita saltos muy grandes).
    - Si no hay detecciones confirmadas, se acepta provisionalmente.
    - Si hay una anterior, permite un salto de ±2 (puedes ajustar).
    """
    if not HISTORY[medidor]: 
        return True  # Nada previo => no tenemos con qué comparar

    try:
        old_val = int(HISTORY[medidor][-1])
        new_val = int(new_str)
    except ValueError:
        return False

    # Aquí permitimos ±2, ajusta si deseas más rango
    return abs(new_val - old_val) <= 2

def is_adaptive_range_valid(medidor, new_str, window=10, k=3):
    """
    Validación 2: Rango adaptativo basado en estadísticas del historial.
    - Toma las últimas 'window' lecturas confirmadas
    - Calcula media y desviación estándar
    - Permite new_val dentro de (media ± k*desviación).
    """
    if not HISTORY[medidor]:
        return True  # Si no hay historial, no hay rango => lo aceptamos

    try:
        new_val = int(new_str)
    except ValueError:
        return False

    # Tomar las últimas 'window' lecturas
    recent = HISTORY[medidor][-window:]
    if not recent:
        return True

    vals_int = []
    for v in recent:
        try:
            vals_int.append(int(v))
        except ValueError:
            pass

    if not vals_int:
        return True

    media = statistics.mean(vals_int)
    stdev = statistics.pstdev(vals_int)  # o stdev
    if stdev == 0:
        # Si todas eran iguales, stdev=0 => permitir algo de rango
        stdev = 1

    rango_min = media - k * stdev
    rango_max = media + k * stdev

    return (rango_min <= new_val <= rango_max)

def is_any_valid(medidor, new_str):
    """
    Se acepta si CUALQUIERA de las validaciones (progresiva o rango adaptativo) es True.
    """
    return is_progressive_valid(medidor, new_str) or \
           is_adaptive_range_valid(medidor, new_str, window=10, k=3)

###############################################################################
#                               CARGA DE MODELOS                               #
###############################################################################
# Cargar modelos principales
display_model = load_local_model('modelos_display/Display.pt')
display_4_model = load_local_model('modelos_display/Display_4.pt')

# Cargar submodelos
analogico_model = load_local_model('modelos_numericos/Analogico.pt')
negro_model = load_local_model('modelos_numericos/ult.pt')  # Asegúrate de que el nombre es correcto
metal_model = load_local_model('modelos_numericos/metal.pt')
medidor_especial_model = load_local_model('modelos_numericos/medidor_especial.pt')  # Submodelo para Medidor_4

# Asociar modelos con medidores
MEDIDOR_TO_MODEL = {
    'Medidor_1': analogico_model,
    'Medidor_2': negro_model,
    'Medidor_3': metal_model,
    'Medidor_4': medidor_especial_model  # Asociamos Medidor_4 con su submodelo especial
}

###############################################################################
#                             APLICACIÓN FLASK                                #
###############################################################################
app = Flask(__name__)
CORS(app)

cap = cv2.VideoCapture(CAMERA_INDEX)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/start', methods=['GET'])
def start_detection():
    global DETECT_ACTIVE
    with LOCK:
        DETECT_ACTIVE = True
    return jsonify({'status': 'detección iniciada'})

@app.route('/stop', methods=['GET'])
def stop_detection():
    """
    Al detener la detección, RESETEAMOS los valores:
    - Se limpia HISTORY y DETECTION_BUFFER
    - Se pone DETECT_ACTIVE en False
    """
    global DETECT_ACTIVE
    with LOCK:
        DETECT_ACTIVE = False
        # Resetear buffers e historial
        for medidor in DETECTION_BUFFER:
            DETECTION_BUFFER[medidor].clear()
        for medidor in HISTORY:
            HISTORY[medidor].clear()
    return jsonify({'status': 'detección detenida y valores reseteados'})

def generate_frames():
    global DETECT_ACTIVE
    while True:
        success, frame = cap.read()
        if not success:
            break
        with LOCK:
            active = DETECT_ACTIVE

        if active:
            rois = detect_display(frame)
            combined = combine_rois(rois)
            ret, buffer = cv2.imencode('.jpg', combined)
            frame_bytes = buffer.tobytes()
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

###############################################################################
#                           FUNCIÓN PRINCIPAL DE DETECCIÓN                    #
###############################################################################
def detect_display(frame):
    """
    1. Detecta todas las instancias de rf4 usando Display_4.pt.
    2. Detecta Medidor_4 usando Display_4.pt.
    3. Si al menos un rf4 está presente, procesa las detecciones de Medidor_4.
    4. Excluye las ROIs de Medidor_4 de las detecciones de Display.pt para evitar confusiones con otros medidores.
    5. Procesa las detecciones de Medidor_1, Medidor_2 y Medidor_3 usando Display.pt.
    6. Solo imprime en consola los valores confirmados.
    """
    rois = []
    ih, iw, _ = frame.shape

    # Paso 1: Detectar rf4 usando Display_4.pt
    results_rf4 = display_4_model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    labels_rf4 = results_rf4.xyxyn[0][:, -1].cpu().numpy()
    boxes_rf4  = results_rf4.xyxyn[0][:, :-1].cpu().numpy()

    rf4_present = False
    for label, box in zip(labels_rf4, boxes_rf4):
        class_id = int(label)
        class_name = display_4_model.names[class_id]
        if class_name == 'rf4':
            rf4_present = True
            break  # Solo necesita al menos un rf4

    # Paso 2: Detectar Medidor_4 usando Display_4.pt
    results_medidor_4 = display_4_model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    labels_medidor_4 = results_medidor_4.xyxyn[0][:, -1].cpu().numpy()
    boxes_medidor_4  = results_medidor_4.xyxyn[0][:, :-1].cpu().numpy()

    medidor_4_rois = []

    if rf4_present:
        for label, box in zip(labels_medidor_4, boxes_medidor_4):
            class_id = int(label)
            class_name = display_4_model.names[class_id]
            if class_name != 'Medidor_4':
                continue  # Solo procesar Medidor_4

            x1, y1, x2, y2, conf = box
            x1, y1 = int(x1 * iw), int(y1 * ih)
            x2, y2 = int(x2 * iw), int(y2 * ih)

            if x2 <= x1 or y2 <= y1:
                continue

            roi = frame[y1:y2, x1:x2].copy()
            if roi.size == 0:
                continue

            # Aplicar el submodelo especial para Medidor_4
            submodel = MEDIDOR_TO_MODEL.get('Medidor_4')
            if not submodel:
                continue

            sub_results = submodel(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            dets = []
            for d in sub_results.xyxy[0]:
                xa1, ya1, xa2, yb2, conf_sub, cls_sub = d
                if conf_sub >= CONFIDENCE_THRESHOLD:
                    cls_name = submodel.names[int(cls_sub)]
                    # Convertir clase 11 a '6'
                    cls_name = '6' if cls_name == '11' else cls_name
                    dets.append({
                        'class': cls_name,
                        'box': [int(xa1), int(ya1), int(xa2), int(yb2)],
                        'confidence': float(conf_sub)
                    })
            dets = nms(dets, iou_thres=IOU_THRESHOLD)
            dets_sorted = sorted(dets, key=lambda dd: dd['box'][0])
            new_digits = ''.join(dd['class'] for dd in dets_sorted)

            # Ignorar si aparece "10" en Medidor_4
            if '10' in new_digits:
                continue  # Ignorar detección
            else:
                # Validar y confirmar
                if is_any_valid('Medidor_4', new_digits):
                    DETECTION_BUFFER['Medidor_4'].append(new_digits)
                    if len(DETECTION_BUFFER['Medidor_4']) > MAX_BUFFER_SIZE:
                        DETECTION_BUFFER['Medidor_4'].pop(0)

                    confirmed_value = check_consensus(
                        DETECTION_BUFFER['Medidor_4'],
                        required_count=CONSENSUS_COUNT
                    )
                    if confirmed_value:
                        HISTORY['Medidor_4'].append(confirmed_value)
                        print(f"[CONSOLE] Medidor_4 => {confirmed_value}")
                        DETECTION_BUFFER['Medidor_4'].clear()

            # Dibujar ROI y etiqueta
            cv2.rectangle(roi, (0,0), (roi.shape[1]-1, roi.shape[0]-1), (0,255,0), 2)
            cv2.putText(roi, 'Medidor_4', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            rois.append(roi)

            # Guardar la ROI para excluirla en Display.pt
            medidor_4_rois.append([x1, y1, x2, y2])

    # Paso 3: Detectar Medidor_1, Medidor_2, Medidor_3 usando Display.pt
    results_display = display_model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    labels_display = results_display.xyxyn[0][:, -1].cpu().numpy()
    boxes_display  = results_display.xyxyn[0][:, :-1].cpu().numpy()

    for label, box in zip(labels_display, boxes_display):
        class_id = int(label)
        class_name = display_model.names[class_id]
        if class_name not in ['Medidor_1', 'Medidor_2', 'Medidor_3']:
            continue  # Solo procesar estos medidores

        x1, y1, x2, y2, conf = box
        x1, y1 = int(x1 * iw), int(y1 * ih)
        x2, y2 = int(x2 * iw), int(y2 * ih)

        if x2 <= x1 or y2 <= y1:
            continue

        # Verificar si esta ROI se solapa con alguna de las ROIs de Medidor_4
        overlap = False
        for med4_box in medidor_4_rois:
            iou_val = iou([x1, y1, x2, y2], med4_box)
            if iou_val > 0.3:  # Ajusta el umbral según sea necesario
                overlap = True
                break
        if overlap:
            continue  # Excluir esta detección, ya está manejada por Medidor_4

        roi = frame[y1:y2, x1:x2].copy()
        if roi.size == 0:
            continue

        # Aplicar el submodelo correspondiente
        submodel = MEDIDOR_TO_MODEL.get(class_name)
        if not submodel:
            continue

        sub_results = submodel(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        dets = []
        for d in sub_results.xyxy[0]:
            xa1, ya1, xa2, yb2, conf_sub, cls_sub = d
            if conf_sub >= CONFIDENCE_THRESHOLD:
                cls_name = submodel.names[int(cls_sub)]
                dets.append({
                    'class': cls_name,
                    'box': [int(xa1), int(ya1), int(xa2), int(yb2)],
                    'confidence': float(conf_sub)
                })
        dets = nms(dets, iou_thres=IOU_THRESHOLD)
        dets_sorted = sorted(dets, key=lambda dd: dd['box'][0])
        new_digits = ''.join(dd['class'] for dd in dets_sorted)

        # Ignorar si aparece "10"
        if '10' in new_digits:
            continue  # Ignorar detección
        else:
            # Validar y confirmar
            if is_any_valid(class_name, new_digits):
                DETECTION_BUFFER[class_name].append(new_digits)
                if len(DETECTION_BUFFER[class_name]) > MAX_BUFFER_SIZE:
                    DETECTION_BUFFER[class_name].pop(0)

                confirmed_value = check_consensus(
                    DETECTION_BUFFER[class_name],
                    required_count=CONSENSUS_COUNT
                )
                if confirmed_value:
                    HISTORY[class_name].append(confirmed_value)
                    print(f"[CONSOLE] {class_name} => {confirmed_value}")
                    DETECTION_BUFFER[class_name].clear()

        # Dibujar ROI y etiqueta
        cv2.rectangle(roi, (0,0), (roi.shape[1]-1, roi.shape[0]-1), (0,255,0), 2)
        cv2.putText(roi, class_name, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        rois.append(roi)

    return rois

def combine_rois(rois, max_rois_per_row=4, padding=10):
    """Combina múltiples ROIs en una sola imagen sin redimensionarlas."""
    if not rois:
        blank = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.putText(blank, 'No detecciones', (50,150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        return blank

    rois = rois[: max_rois_per_row * 3]
    num_rois = len(rois)
    num_cols = min(num_rois, max_rois_per_row)
    num_rows = (num_rois + max_rois_per_row - 1) // max_rois_per_row

    widths  = [r.shape[1] for r in rois]
    heights = [r.shape[0] for r in rois]

    col_widths = [0]*num_cols
    for i, roi in enumerate(rois):
        c = i % max_rois_per_row
        col_widths[c] = max(col_widths[c], roi.shape[1])

    row_heights = [0]*num_rows
    for i, roi in enumerate(rois):
        r = i // max_rois_per_row
        row_heights[r] = max(row_heights[r], roi.shape[0])

    total_w = sum(col_widths) + padding*(num_cols+1)
    total_h = sum(row_heights) + padding*(num_rows+1)

    combined = np.zeros((total_h, total_w, 3), dtype=np.uint8)

    y_off = padding
    idx_roi = 0
    for r in range(num_rows):
        x_off = padding
        for c in range(num_cols):
            if idx_roi >= num_rois:
                break
            roi = rois[idx_roi]
            combined[y_off:y_off+roi.shape[0], x_off:x_off+roi.shape[1]] = roi
            x_off += col_widths[c] + padding
            idx_roi += 1
        y_off += row_heights[r] + padding

    return combined

# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
