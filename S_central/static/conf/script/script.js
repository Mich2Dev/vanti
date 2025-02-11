// Configuración de dispositivos
const devices = [
    { name: 'Camera 1', ip: '127.0.0.1', port: 5000, datoId: 'current_cam1' },
    { name: 'Camera 2', ip: '127.0.0.1', port: 5000, datoId: 'current_cam2' },
    { name: 'Camera 3', ip: '192.168.0.43', port: 5000, datoId: 'current_cam3' },
    { name: 'Camera 4', ip: '192.168.0.44', port: 5000, datoId: 'current_cam4' },
    { name: 'Camera 5', ip: '192.168.0.45', port: 5000, datoId: 'current_cam5' },
    { name: 'Camera 6', ip: '192.168.0.80', port: 5000, datoId: 'current_cam6' }
];

// Función para iniciar detección
function activateDetection() {
    devices.forEach(device => {
        fetch(`http://${device.ip}:${device.port}/start`)
            .then(response => response.json())
            .then(data => {
                console.log(`${device.name}: Detección iniciada`);
            })
            .catch(error => {
                console.error(`${device.name}: Error al iniciar detección`, error);
            });
    });
}

// Función para detener detección
function detenerDeteccion() {
    devices.forEach(device => {
        fetch(`http://${device.ip}:${device.port}/stop`)
            .then(response => response.json())
            .then(data => {
                console.log(`${device.name}: Detección detenida`);
            })
            .catch(error => {
                console.error(`${device.name}: Error al detener detección`, error);
            });
    });
}

// Función para obtener datos de cada dispositivo
function actualizarDatos() {
    devices.forEach(device => {
        fetch(`http://${device.ip}:${device.port}/data`)
            .then(response => response.json())
            .then(data => {
                const datoFinal = data.value; // Se espera que el endpoint devuelva 'value'
                const elementoDato = document.getElementById(device.datoId);
                if (elementoDato) {
                    elementoDato.textContent = datoFinal !== null ? datoFinal : 'Esperando Iniciar';
                }
            })
            .catch(error => {
                console.error(`${device.name}: Error al obtener datos`, error);
                const elementoDato = document.getElementById(device.datoId);
                if (elementoDato) {
                    elementoDato.textContent = 'Esperando Iniciar';
                }
            });
    });
}

// Función para actualizar la posición del punto decimal en cada dispositivo
function updateDotPosition(ip, port, inputId) {
    const pos = document.getElementById(inputId).value;
    if (!pos) {
        alert("Ingrese una posición válida");
        return;
    }
    fetch(`http://${ip}:${port}/set_dot_position?position=${pos}`)
        .then(response => response.json())
        .then(data => {
            console.log(`Posición actualizada en ${ip}:${port} a ${data.position}`);
        })
        .catch(error => {
            console.error(`Error al actualizar la posición en ${ip}:${port}`, error);
        });
}

// Actualización periódica de datos
setInterval(actualizarDatos, 200);

// Actualizar datos al cargar la página
document.addEventListener('DOMContentLoaded', actualizarDatos);

