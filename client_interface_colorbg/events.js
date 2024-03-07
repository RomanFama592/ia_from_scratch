randomColor.addEventListener("click", () => {
  changeValueInput(generarColorAlAzar(), text);
});

send_example.addEventListener("click", async () => {
  const response = await fetch(`${serverurl}/recivedexample`, {
    method: "POST",
    mode: "cors",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify([hexToArrayRgb(text.value), [bginput.value / 255]]),
  });
  let a = await response.json();
});

save_progress.addEventListener("click", async () => {
  const response = await fetch(`${serverurl}/save`);
  let a = await response.json();
});

remove_last_example.addEventListener("click", async () => {
  const response = await fetch(`${serverurl}/removelastexample`);
  let a = await response.json();
});

train.addEventListener("click", async () => {
  const response = await fetch(
    `${serverurl}/train-${Tsamples.value}-${lr.value}`
  );
  let responseF = await response.json();

  let lossI = responseF.loss.map(function (valor, indice) {
    return indice;
  });

  actualizarGrafico({ labels: lossI, data: responseF.loss });
});

text.addEventListener("input", async () => {
  bgoutput.style.color = text.value;

  if (predict.checked) {
    return "";
  }

  let rgb = hexToArrayRgb(text.value);
  const response = await fetch(`${serverurl}/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(rgb),
  });
  let a = await response.json(rgb);
  console.log(a.out);
  changeValueInput(a.out * 255, bginput);
});

bginput.addEventListener("input", () => {
  let grayscaleValue = bginput.value;
  bgoutput.style.backgroundColor = `rgb(${grayscaleValue} ${grayscaleValue} ${grayscaleValue})`;
});

changeValueInput(128, bginput);
