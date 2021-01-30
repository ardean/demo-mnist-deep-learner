const baseUrl = "http://localhost:3000";

const loadingElement = document.getElementById("loading");

const showLoading = (loadingText) => {
  loadingElement.textContent = loadingText || "Loading...";
  loadingElement.style.display = "flex";
};

const hideLoading = () => {
  loadingElement.style.display = "none";
};

const test = async () => {
  showLoading("Testing...");

  await fetch(`${baseUrl}/api/test`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    }
  });

  hideLoading();
};

const train = async () => {
  showLoading("Training...");

  await fetch(`${baseUrl}/api/train`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    }
  });

  hideLoading();
};

const predict = async () => {
  showLoading("Predicting...");

  ctx.drawImage(canvas, 0, 0, 28, 28);
  const imgPixels = ctx.getImageData(0, 0, 28, 28);
  const height = ctx.getImageData(0, 0, 28, 28).height;
  const width = ctx.getImageData(0, 0, 28, 28).width;

  const newPixels = [];

  for (var y = 0; y < height; y++) {
    const row = [];
    for (var x = 0; x < width; x++) {
      var i = (y * 4) * width + x * 4;
      imgPixels.data[i] = 255 - imgPixels.data[i];
      imgPixels.data[i + 1] = 255 - imgPixels.data[i + 1];
      imgPixels.data[i + 2] = 255 - imgPixels.data[i + 2];
      var avg = 0.299 * imgPixels.data[i] + 0.587 * imgPixels.data[i + 1] + 0.114 * imgPixels.data[i + 2];
      row.push(avg);
    }
    newPixels.push(row);
  }

  const res = await fetch(`${baseUrl}/api/predict`, {
    method: "POST",
    body: JSON.stringify({ image: newPixels }),
    headers: {
      "Content-Type": "application/json"
    }
  });

  const data = await res.json();
  console.log(data);

  hideLoading();
};

const reset = () => {
  canvas.getContext("2d").fillStyle = "white";
  canvas.getContext("2d").fillRect(0, 0, canvas.width, canvas.height);
};

const sketchableConfig = {
  graphics: {
    firstPointSize: 3,
    lineWidth: 17,
    strokeStyle: "#000",
    fillStyle: "#000",
    lineCap: "round",
    lineJoin: "round",
    miterLimit: 10
  }
};

const sketcher = new Sketchable("#drawing-canvas", sketchableConfig);

const canvas = sketcher.elem;
canvas.getContext("2d").fillStyle = "white";
canvas.getContext("2d").fillRect(0, 0, canvas.width, canvas.height);

const resizedCanvas = document.getElementById("resized-canvas");
const ctx = resizedCanvas.getContext("2d");

resizedCanvas.width = 28;
resizedCanvas.height = 28;