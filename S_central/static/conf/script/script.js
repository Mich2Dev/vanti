// Configuraci�n de dispositivos
const devices = [
    { name: 'Camera 1', ip: '192.168.0.42', port: 5000, datoId: 'current_cam1' },
    { name: 'Camera 2', ip: '192.168.0.42', port: 5000, datoId: 'current_cam2' },
    { name: 'Camera 3', ip: '192.168.0.43', port: 5000, datoId: 'current_cam3' },
    { name: 'Camera 4', ip: '192.168.0.44', port: 5000, datoId: 'current_cam4' },
    { name: 'Camera 5', ip: '192.168.0.45', port: 5000, datoId: 'current_cam5' },
    { name: 'Camera 6', ip: '192.168.0.80', port: 5000, datoId: 'current_cam6' }
];



// Funci�n para iniciar detecci�n
function activateDetection() {
    devices.forEach(device => {
        fetch(`http://${device.ip}:${device.port}/start`)
        .then(response => response.json())
        .then(data => {
            console.log(`${device.name}: Detecci�n iniciada`);
        })
        .catch(error => {
            console.error(`${device.name}: Error al iniciar detecci�n`, error);
        });
    });
}

// Funci�n para detener detecci�n
function detenerDeteccion() {
    devices.forEach(device => {
        fetch(`http://${device.ip}:${device.port}/stop`)
        .then(response => response.json())
        .then(data => {
            console.log(`${device.name}: Detecci�n detenida`);
        })
        .catch(error => {
            console.error(`${device.name}: Error al detener detecci�n`, error);
        });
    });
}

// Funci�n para obtener datos de cada dispositivo
function actualizarDatos() {
    devices.forEach(device => {
        fetch(`http://${device.ip}:${device.port}/data`)
        .then(response => response.json())
        .then(data => {
            const datoFinal = data.value; // Aseg�rate de que el endpoint devuelve 'value'
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

// Actualizaci�n peri�dica de datos
setInterval(actualizarDatos, 200);

// Actualizar datos al cargar la p�gina
document.addEventListener('DOMContentLoaded', actualizarDatos);
