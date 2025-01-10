// Función para actualizar los valores actuales de las cámaras
function updateCurrentValues() {
    const cameraUrls = [
        'http://192.168.0.41:5001/get_current_value',
        'http://192.168.0.42:5001/get_current_value',
        'http://192.168.0.43:5001/get_current_value',
        'http://192.168.0.44:5001/get_current_value',
        'http://192.168.0.45:5001/get_current_value',
        'http://192.168.0.80:5001/get_current_value'
    ];

    cameraUrls.forEach((url, index) => {
        fetch(url)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`No se pudo conectar con la cámara ${index + 1}`);
                }
                return response.json();
            })
            .then(data => {
                document.getElementById(`current_cam${index + 1}`).innerText = 'Valor: ' + data.current_value;
            })
            .catch(error => {
                console.warn(`La cámara ${index + 1} no está disponible. Error de conexión.`);
                document.getElementById(`current_cam${index + 1}`).innerText = 'Error de conexión';
            });
    });
}

setInterval(updateCurrentValues, 1000);  // Actualizar cada segundo

// Función para actualizar la imagen de las cámaras
function actualizarImagenCamara(id, url) {
    document.getElementById(`img${id}`).alt = 'Esperando actualización del servidor...';

    fetch(`${url}/estado`)
        .then(response => response.json())
        .then(data => {
            console.log(`Valor del estado de la cámara ${id}:`, data.estado);
            if (data.estado === 0) {
                fetch(`${url}/read_frame`)
                    .then(response => {
                        if (!response.ok) {
                            throw new Error(`Error al obtener la imagen de la cámara ${id}`);
                        }
                        return response.blob();
                    })
                    .then(blob => {
                        const imageUrl = URL.createObjectURL(blob);
                        document.getElementById(`img${id}`).src = imageUrl;
                        document.getElementById(`img${id}`).alt = `Stream de la cámara ${id}`;
                    })
                    .catch(error => {
                        document.getElementById(`img${id}`).alt = 'Error al obtener la imagen';
                        console.warn('Verifique las conexiones');
                    });
            } else {
                document.getElementById(`img${id}`).alt = 'Esperando, modelo en ejecución...';
            }
        })
        .catch(error => {
            document.getElementById(`img${id}`).alt = 'Error al obtener el estado del JSON';
            console.warn('Verifique las conexiones');
        });
}

function iniciarActualizacionImagenes() {
    const cameraUrls = [
        'http://192.168.0.41:5001',
        'http://192.168.0.42:5001',
        'http://192.168.0.43:5001',
        'http://192.168.0.44:5001',
        'http://192.168.0.45:5001',
        'http://192.168.0.80:5001'
    ];

    cameraUrls.forEach((url, index) => {
        actualizarImagenCamara(index + 1, url);
        setInterval(() => actualizarImagenCamara(index + 1, url), 1000);  // Actualizar cada segundo
    });
}

window.onload = function() {
    iniciarActualizacionImagenes();

    // Verificar si la detección ya fue activada
    if (localStorage.getItem('detectionStarted') === 'true') {
        document.getElementById('startBtn').disabled = true;
        document.getElementById('startBtn').classList.add('active');
    } else {
        document.getElementById('startBtn').disabled = false;
        document.getElementById('startBtn').classList.remove('active');
    }
};

// Función para activar la detección con verificación del estado
function activateDetection() {
    const startButton = document.getElementById("startBtn");
    document.getElementById("startBtn").removeAttribute("onclick");

    const fanlessUrls = [
        'http://192.168.0.41:5001',
        'http://192.168.0.42:5001',
        'http://192.168.0.43:5001',
        'http://192.168.0.44:5001',
        'http://192.168.0.45:5001',
        'http://192.168.0.80:5001'
    ];

    let canActivate = true; // Bandera para verificar si podemos activar la detección

    // Verificamos los estados de todos los fanless
    Promise.all(
        fanlessUrls.map((url, index) =>
            fetch(`${url}/estado`)  // LLAMADA AL MÉTODO ESTADO DE CADA FANLESS
                .then(response => response.json())
                .then(data => {
                    console.log(`Estado de la cámara ${index + 1}:`, data.estado);
                    if (data.estado === 0) { // Si alguno ya está en ejecución
                        console.warn(`La cámara ${index + 1} ya tiene detección activa.`);
                        canActivate = false;
                    }
                })
                .catch(error => console.error(`Error al verificar el estado de la cámara ${index + 1}:`, error))
        )
    ).then(() => {
        if (canActivate) {
            // Si todos los fanless están inactivos, podemos activar la detección
            fanlessUrls.forEach((url, index) => {
                fetch(`${url}/start_detection`, { method: 'GET' })
                    .then(response => {
                        if (!response.ok) {
                            throw new Error(`Fallo al activar la detección en la cámara ${index + 1}`);
                        }
                        console.log(`Detección activada en la cámara ${index + 1}`);
                    })
                    .catch(error => console.error(`Error activando la detección en la cámara ${index + 1}:`, error));
            });

            startButton.classList.add("active");
            localStorage.setItem('detectionStarted', 'true');  // Guardar en localStorage
        } else {
            console.warn("Al menos una cámara ya está activa. No se puede iniciar la detección.");
        }
    });
}

// Función para detener la detección
function detenerDeteccion() {
    const startButton = document.getElementById("startBtn");
    
    startButton.classList.remove("active");
    console.log("Detección detenida.");
            
    const fanlessUrls = [
        'http://192.168.0.41:5001/stop_detection',
        'http://192.168.0.42:5001/stop_detection',
        'http://192.168.0.43:5001/stop_detection',
        'http://192.168.0.44:5001/stop_detection',
        'http://192.168.0.45:5001/stop_detection',
        'http://192.168.0.80:5001/stop_detection'
    ];

    fanlessUrls.reduce((promiseChain, url, index) => {
        return promiseChain.then(() => {
            return fetch(url, { method: 'GET' })
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`Fallo al detener la detección en la cámara ${index + 1}`);
                    }
                    document.getElementById(`current_cam${index + 1}`).innerText = 'Valor: 0';
                    document.getElementById(`img${index + 1}`).alt = 'Detenido';
                });
        }).catch(error => console.error(`Error procesando la cámara ${index + 1}:`, error));
    }, Promise.resolve()).then(() => {
        // Limpia el estado en el localStorage
        localStorage.setItem('detectionStarted', 'false');

        // Recargar la página automáticamente una vez que se ha detenido la detección en todos los fanless
        setTimeout(() => {
            location.reload();
        }, 500);  // Esperar 0.5 segundos antes de recargar la página (puedes ajustar el tiempo si es necesario)
    });
}
