function actualizarGrafico(nuevosDatos) {
    myChart.data.labels = nuevosDatos.labels;
    myChart.data.datasets[0].data = nuevosDatos.data;
    myChart.update();
}

function changeValueInput(value, htmltochange) {
    htmltochange.value = value;
    let inputEvent = new Event('input', { bubbles: true });
    htmltochange.dispatchEvent(inputEvent);
}

function generarColorAlAzar() {
    var letras = '0123456789ABCDEF';
    var color = '#';
    for (var i = 0; i < 6; i++) {
        color += letras[Math.floor(Math.random() * 16)];
    }
    return color;
}

function hexToArrayRgb(hex) {
    if (hex.startsWith('#')) {
        hex = hex.slice(1);
    }

    let red = parseInt(hex.slice(0, 2), 16);
    let green = parseInt(hex.slice(2, 4), 16);
    let blue = parseInt(hex.slice(4, 6), 16);

    let normalizedRed = red / 255;
    let normalizedGreen = green / 255;
    let normalizedBlue = blue / 255;

    let rgb = [normalizedRed, normalizedGreen, normalizedBlue];
    return rgb;
}
