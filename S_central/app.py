from flask import Flask, render_template, jsonify
import paramiko

app = Flask(__name__)

# Lista de dispositivos
devices = [
    {"name": "Camera 1", "ip": "192.168.0.51", "user": "vision1", "password": "abc567890"},
    {"name": "Camera 2", "ip": "192.168.0.52", "user": "vision2", "password": "abc567890"},
    {"name": "Camera 3", "ip": "192.168.0.53", "user": "vision3", "password": "abc567890"},
    {"name": "Camera 4", "ip": "192.168.0.54", "user": "vision4", "password": "abc567890"},
    {"name": "Camera 5", "ip": "192.168.0.55", "user": "vision5", "password": "abc567890"},
    {"name": "Camera 6", "ip": "192.168.0.56", "user": "vision6", "password": "abc567890"},
    {"name": "Camera 7", "ip": "192.168.0.57", "user": "vision7", "password": "abc567890"},
    {"name": "Camera 8", "ip": "192.168.0.58", "user": "vision8", "password": "abc567890"},
]

# Función para reiniciar un dispositivo por SSH
def reboot_device(device):
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(device["ip"], username=device["user"], password=device["password"])
        ssh.exec_command("sudo reboot")
        ssh.close()
        return {"ip": device["ip"], "name": device["name"], "status": "OK"}
    except Exception as e:
        return {"ip": device["ip"], "name": device["name"], "status": "ERROR", "error": str(e)}

# Ruta principal: Renderiza el dashboard
@app.route('/')
def dashboard():
    return render_template(
        'central_dashboard.html', 
        dato1=None, dato2=None, dato3=None, dato4=None, 
        dato5=None, dato6=None, dato7=None, dato8=None
    )

# Endpoint para reiniciar todos los dispositivos
@app.route('/reboot_all')
def reboot_all():
    results = []
    for device in devices:
        results.append(reboot_device(device))
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
