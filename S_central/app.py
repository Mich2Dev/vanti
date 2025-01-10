from flask import jsonify, Flask, render_template
import threading
import time
import requests

app = Flask(__name__)

# Variables globales para almacenar los datos de las cámaras y estados de los fanless
fanless_states = {
    '192.168.0.41': {'estado': None, 'medidor': None},
    '192.168.0.42': {'estado': None, 'medidor': None},
    '192.168.0.43': {'estado': None, 'medidor': None},
    '192.168.0.44': {'estado': None, 'medidor': None},
    '192.168.0.45': {'estado': None, 'medidor': None},
    '192.168.0.80': {'estado': None, 'medidor': None}
}

# Función para obtener estado y medidor de cada fanless
def obtener_estado_fanless(ip):
    while True:
        try:
            # Obtener estado del fanless
            estado_resp = requests.get(f'http://{ip}:5001/estado', timeout=5)
            estado_resp.raise_for_status()
            fanless_states[ip]['estado'] = estado_resp.json().get('estado', 1)  # 1 inactivo, 0 activo

            # Obtener valor del medidor
            medidor_resp = requests.get(f'http://{ip}:5001/get_current_value', timeout=5)
            medidor_resp.raise_for_status()
            fanless_states[ip]['medidor'] = medidor_resp.json().get('current_value', 0)

            print(f"Fanless {ip}: Estado {fanless_states[ip]['estado']}, Medidor: {fanless_states[ip]['medidor']}")
        except requests.ConnectionError:
            print(f"Fanless {ip} no disponible.")
            fanless_states[ip]['estado'] = None
        except requests.Timeout:
            print(f"Fanless {ip} no responde. Timeout.")
        except Exception as e:
            print(f"Error al consultar fanless {ip}: {e}")
        time.sleep(5)

# Iniciar detección solo si el estado es 1 (inactivo)
@app.route('/start_detection')
def start_detection():
    for ip, estado_info in fanless_states.items():
        if estado_info['estado'] == 1:  # Solo permitir iniciar si el estado es 1 (inactivo)
            try:
                response = requests.get(f'http://{ip}:5001/start_detection')
                if response.ok:
                    print(f"Detección iniciada en fanless {ip}")
                else:
                    print(f"Error al iniciar detección en fanless {ip}")
            except requests.RequestException as e:
                print(f"Error al iniciar detección en fanless {ip}: {e}")
        else:
            print(f"Fanless {ip} ya tiene detección activa (estado 0).")
    return jsonify({"status": "Intento de detección en fanless inactivos"})


# Detener detección, sin importar el estado
@app.route('/stop_detection')
def stop_detection():
    for ip, estado_info in fanless_states.items():
        if estado_info['estado'] == 0:  # Solo detener si está activo
            try:
                response = requests.get(f'http://{ip}:5001/stop_detection')
                if response.ok:
                    print(f"Detección detenida en fanless {ip}")
                else:
                    print(f"Error al detener detección en fanless {ip}")
            except requests.RequestException as e:
                print(f"Error al detener detección en fanless {ip}: {e}")
        else:
            print(f"Fanless {ip} no tiene detección en ejecución.")
    return jsonify({"status": "Detección detenida en fanless activos"})

# Ruta principal para mostrar la página de inicio
@app.route('/')
def home():
    return render_template('index.html', fanless_states=fanless_states)

if __name__ == '__main__':
    # Iniciar los hilos para consultar estado y medidor de cada fanless
    for ip in fanless_states.keys():
        threading.Thread(target=obtener_estado_fanless, args=(ip,), daemon=True).start()

    app.run(debug=True, host='0.0.0.0', port=5000)
