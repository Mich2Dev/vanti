import cv2
import torch
import numpy as np
import statistics
from flask import Flask, render_template, Response, jsonify, request
import threading
from flask_cors import CORS
import warnings

###############################################################################
#                        IGNORAR LOS FUTUREWARNING DE TORCH                   #
###############################################################################
warnings.filterwarnings("ignore", category=FutureWarning)

###############################################################################
#                             CONFIGURACIONES                                 #
###############################################################################
# Umbral de confianza para detectar el DISPLAY (Medidor_1..3 o Medidor_4)
DISPLAY_CONFIDENCE_THRESHOLD = 0.9
IOU_THRESHOLD = 0.9  # <--- Asegúrate de definirlo

# Umbral de confianza para los SUBMODELOS de dígitos
SUBMODEL_CONFIDENCE_THRESHOLD = 0.6

# Validaciones y Buffer
CONSENSUS_COUNT = 5
MAX_BUFFER_SIZE = 10

# Validaciones (Progresiva y Estadística)
MAX_PROG_JUMP = 2
WINDOW_SIZE = 10
K_STD = 3

# Cámara y estado de la detección
CAMERA_INDEX = 2
DETECT_ACTIVE = False
LOCK = threading.Lock()

###############################################################################
#                ESTADO TIPO 4 (LOCK) PARA EVITAR SALTOS                      #
###############################################################################
LOCKED_MEDIDOR_4 = False  # Indica si estamos "fijados" en medidor tipo 4
LOCK_FRAMES_4 = 0         # Cuántos frames mantenemos el lock
LOCK_THRESHOLD = 10       # Si detectamos 4, fijar por 10 frames

###############################################################################
#          HISTORIAL Y BUFFERS PARA CADA MEDIDOR (1,2,3,4)                    #
###############################################################################
HISTORY = {
    'Medidor_1': [],
    'Medidor_2': [],
    'Medidor_3': [],
    'Medidor_4': []
}
DETECTION_BUFFER = {
    'Medidor_1': [],
    'Medidor_2': [],
    'Medidor_3': [],
    'Medidor_4': []
}

###############################################################################
#                         FUNCIONES AUXILIARES                                #
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
    if union_area == 0:
        return 0.0
    return inter_area / union_area

def nms(detections, iou_thres=0.5):
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
    if len(buffer_list) < required_count:
        return None
    last_values = buffer_list[-required_count:]
    if all(v == last_values[0] for v in last_values):
        return last_values[-1]
    return None

###############################################################################
#                          VALIDACIONES                                       #
###############################################################################
def is_progressive_valid(medidor, new_str):
    if not HISTORY[medidor]:
        return True
    try:
        old_val = int(HISTORY[medidor][-1])
        new_val = int(new_str)
    except ValueError:
        return False
    return abs(new_val - old_val) <= MAX_PROG_JUMP

def is_adaptive_range_valid(medidor, new_str, window=WINDOW_SIZE, k=K_STD):
    if not HISTORY[medidor]:
        return True
    try:
        new_val = int(new_str)
    except ValueError:
        return False

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
    stdev = statistics.pstdev(vals_int) if len(vals_int) > 1 else 0
    if stdev == 0:
        stdev = 1
    rango_min = media - k*stdev
    rango_max = media + k*stdev
    return (rango_min <= new_val <= rango_max)

def is_any_valid(medidor, new_str):
    """
    Se acepta si pasa la validación progresiva o la adaptativa.
    """
    return is_progressive_valid(medidor, new_str) or \
           is_adaptive_range_valid(medidor, new_str)

###############################################################################
#                  CARGA DE MODELOS PRINCIPALES                               #
###############################################################################
display_model = load_local_model('Display.pt')      # Medidor_1,2,3
analogico_model = load_local_model('Analogico.pt')
ult_model       = load_local_model('ult.pt')
metal_model     = load_local_model('metal.pt')

display4_model  = load_local_model('Display_4.pt')  # Medidor_4 + rf4
digits4_model   = load_local_model('medidor_especial.pt')  # 0..9,10(rechazo),11=>6

MEDIDOR_TO_MODEL = {
    'Medidor_1': analogico_model,
    'Medidor_2': ult_model,
    'Medidor_3': metal_model
    # Podrías mapear 'Medidor_4': digits4_model si quisieras un approach “automático”
}

###############################################################################
#                         APLICACIÓN FLASK                                    #
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

@app.route('/start', methods=['GET','POST'])
def start_detection():
    global DETECT_ACTIVE
    with LOCK:
        DETECT_ACTIVE = True
    return jsonify({'status': 'detección iniciada (GET/POST)'})

@app.route('/stop', methods=['GET','POST'])
def stop_detection():
    global DETECT_ACTIVE
    with LOCK:
        DETECT_ACTIVE = False
        # Resetea buffers e historial
        for m in DETECTION_BUFFER:
            DETECTION_BUFFER[m].clear()
        for m in HISTORY:
            HISTORY[m].clear()
    return jsonify({'status': 'detección detenida (GET/POST), reseteada'})

###############################################################################
#                    LOCK Y GENERATE_FRAMES                                  #
###############################################################################
LOCKED_MEDIDOR_4 = False
LOCK_FRAMES_4 = 0
LOCK_THRESHOLD = 10

def generate_frames():
    global LOCKED_MEDIDOR_4, LOCK_FRAMES_4
    while True:
        success, frame = cap.read()
        if not success:
            break
        with LOCK:
            active = DETECT_ACTIVE

        if active:
            if LOCKED_MEDIDOR_4:
                # Ya estamos lockeados en Medidor_4 => revalidar
                rois_4, still_4 = detect_display4(frame)
                if still_4:
                    # renovamos lock
                    LOCK_FRAMES_4 = LOCK_THRESHOLD
                    rois_final = rois_4
                else:
                    # se reduce
                    LOCK_FRAMES_4 -= 1
                    if LOCK_FRAMES_4 <= 0:
                        LOCKED_MEDIDOR_4 = False
                    rois_final = rois_4  # mientras tanto, seguimos en "4"
            else:
                # Aun no estamos lockeados en 4 => checar si aparece 4
                rois_4, found_4 = detect_display4(frame)
                if found_4:
                    LOCKED_MEDIDOR_4 = True
                    LOCK_FRAMES_4 = LOCK_THRESHOLD
                    rois_final = rois_4
                else:
                    # Detectar medidor_1..3
                    rois_123 = detect_display123(frame)
                    rois_final = rois_123

            combined = combine_rois(rois_final)
            ret, buffer = cv2.imencode('.jpg', combined)
            frame_bytes = buffer.tobytes()
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

###############################################################################
#                DETECCIÓN DE MEDIDOR_4 (DISPLAY_4, rf4)                     #
###############################################################################
def detect_display4(frame):
    """
    Busca Medidor_4 y rf4 con un threshold de confianza en la detección de display.
    Solo si ambos => found_4 = True. Aplica detect_subdigits_4 en la ROI de Medidor_4.
    """
    results = display4_model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # Extraer
    labels = results.xyxyn[0][:, -1].cpu().numpy()
    boxes  = results.xyxyn[0][:, :-1].cpu().numpy()

    ih, iw, _ = frame.shape
    rois_4 = []
    found_4 = False

    found_medidor4 = False
    found_rf4 = False

    for label, box in zip(labels, boxes):
        class_index = int(label)
        class_name = display4_model.names[class_index]
        x1, y1, x2, y2, conf = box
        if conf < DISPLAY_CONFIDENCE_THRESHOLD:
            continue  # Descartar si la confianza del display < threshold

        x1, y1 = int(x1 * iw), int(y1 * ih)
        x2, y2 = int(x2 * iw), int(y2 * ih)
        if x2 <= x1 or y2 <= y1:
            continue

        roi = frame[y1:y2, x1:x2].copy()
        if roi.size == 0:
            continue

        if class_name == 'Medidor_4':
            found_medidor4 = True
            # Procesar subdígitos con "digits4_model"
            detect_subdigits_4(roi)
            # Dibujarlo
            cv2.rectangle(roi, (0,0), (roi.shape[1]-1, roi.shape[0]-1), (255,0,0), 2)
            cv2.putText(roi, class_name, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            rois_4.append(roi)
        elif class_name == 'rf4':
            found_rf4 = True

    if found_medidor4 and found_rf4:
        found_4 = True
    return rois_4, found_4

def detect_subdigits_4(roi):
    """
    Aplica 'digits4_model' (medidor_especial.pt) con clases 0..9, 10 => rechazo, 11 => '6'.
    """
    global digits4_model
    sub_results = digits4_model(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    dets = []
    for d in sub_results.xyxy[0]:
        x1, y1, x2, y2, conf_sub, cls_sub = d
        if conf_sub >= SUBMODEL_CONFIDENCE_THRESHOLD:
            dets.append({
                'class': digits4_model.names[int(cls_sub)],
                'box': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': float(conf_sub)
            })
    dets = nms(dets, iou_thres=IOU_THRESHOLD)
    dets_sorted = sorted(dets, key=lambda dd: dd['box'][0])
    digit_list = []

    for dd in dets_sorted:
        cls_name = dd['class']
        if cls_name == '10':
            return  # Rechazo total => salimos sin buffer
        elif cls_name == '11':
            digit_list.append('6')  # Convertimos a '6'
        else:
            digit_list.append(cls_name)

    new_digits = ''.join(digit_list)
    if is_any_valid('Medidor_4', new_digits):
        DETECTION_BUFFER['Medidor_4'].append(new_digits)
        if len(DETECTION_BUFFER['Medidor_4']) > MAX_BUFFER_SIZE:
            DETECTION_BUFFER['Medidor_4'].pop(0)

        confirmed = check_consensus(
            DETECTION_BUFFER['Medidor_4'],
            required_count=CONSENSUS_COUNT
        )
        if confirmed is not None:
            HISTORY['Medidor_4'].append(confirmed)
            print(f"[CONSOLE] Medidor_4 => {confirmed}")
            DETECTION_BUFFER['Medidor_4'].clear()

###############################################################################
#           DETECCIÓN DE MEDIDOR_1,2,3 (DISPLAY.PT)                           #
###############################################################################
def detect_display123(frame):
    """
    Detecta Medidor_1,2,3 en 'frame' con un threshold de confianza para el display.
    Aplica submodelos de dígitos (Analogico, ult, metal).
    """
    results = display_model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    labels = results.xyxyn[0][:, -1].cpu().numpy()
    boxes  = results.xyxyn[0][:, :-1].cpu().numpy()

    ih, iw, _ = frame.shape
    rois_123 = []

    for label, box in zip(labels, boxes):
        class_index = int(label)
        class_name = display_model.names[class_index]
        x1, y1, x2, y2, conf = box
        if conf < DISPLAY_CONFIDENCE_THRESHOLD:
            continue  # No procesar displays con baja confianza

        x1, y1 = int(x1 * iw), int(y1 * ih)
        x2, y2 = int(x2 * iw), int(y2 * ih)
        if x2 <= x1 or y2 <= y1:
            continue

        roi = frame[y1:y2, x1:x2].copy()
        if roi.size == 0:
            continue

        submodel = MEDIDOR_TO_MODEL.get(class_name, None)
        if submodel:
            detect_subdigits(roi, class_name, submodel)

        cv2.rectangle(roi, (0,0), (roi.shape[1]-1, roi.shape[0]-1), (0,255,0), 2)
        cv2.putText(roi, class_name, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        rois_123.append(roi)

    return rois_123

def detect_subdigits(roi, medidor_class_name, submodel):
    """
    Aplica submodelo (Analogico, ult, metal). Rechaza '10' y valida el resto.
    """
    sub_results = submodel(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    dets = []
    for d in sub_results.xyxy[0]:
        x1, y1, x2, y2, conf_sub, cls_sub = d
        if conf_sub >= SUBMODEL_CONFIDENCE_THRESHOLD:
            dets.append({
                'class': submodel.names[int(cls_sub)],
                'box': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': float(conf_sub)
            })
    dets = nms(dets, iou_thres=IOU_THRESHOLD)
    dets_sorted = sorted(dets, key=lambda dd: dd['box'][0])
    new_digits = ''.join(dd['class'] for dd in dets_sorted)

    # Rechazo total si aparece '10'
    if '10' in [dd['class'] for dd in dets_sorted]:
        return

    # Validación => buffer => consenso => imprimir
    if is_any_valid(medidor_class_name, new_digits):
        DETECTION_BUFFER[medidor_class_name].append(new_digits)
        if len(DETECTION_BUFFER[medidor_class_name]) > MAX_BUFFER_SIZE:
            DETECTION_BUFFER[medidor_class_name].pop(0)
        confirmed = check_consensus(
            DETECTION_BUFFER[medidor_class_name],
            required_count=CONSENSUS_COUNT
        )
        if confirmed is not None:
            HISTORY[medidor_class_name].append(confirmed)
            print(f"[CONSOLE] {medidor_class_name} => {confirmed}")
            DETECTION_BUFFER[medidor_class_name].clear()

###############################################################################
#                     COMBINAR ROIS EN UNA SOLA IMAGEN                        #
###############################################################################
def combine_rois(rois, max_rois_per_row=3, padding=10):
    """
    Combina múltiples ROIs en una sola imagen, sin redimensionarlas.
    """
    if not rois:
        blank = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.putText(blank, 'No detecciones', (50,150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        return blank

    rois = rois[: max_rois_per_row * 3]
    num_rois = len(rois)
    num_cols = min(num_rois, max_rois_per_row)
    num_rows = (num_rois + max_rois_per_row - 1) // max_rois_per_row

    widths = [r.shape[1] for r in rois]
    heights= [r.shape[0] for r in rois]

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

###############################################################################
# MAIN
###############################################################################
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
