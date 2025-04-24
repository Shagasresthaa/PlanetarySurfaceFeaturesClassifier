import os
import torch
from flask import Flask, render_template, request, jsonify
from torchvision import transforms
from PIL import Image
from model.astro_net import AstroNet

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load model
model = AstroNet(num_classes=4)
model.load_state_dict(torch.load("model/densetAstroNetFinalWeights.pth", map_location="cpu"))
model.eval()

# Classes
class_names = ["bright_dune", "crater", "dark_dune", "impact_ejecta"]

# Transform (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            image = Image.open(filepath).convert("RGB")
            input_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1)
                conf, pred = torch.max(probs, dim=1)
                label = class_names[pred.item()]
                confidence = conf.item()

            return jsonify({"label": label, "confidence": round(confidence * 100, 2)})

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
