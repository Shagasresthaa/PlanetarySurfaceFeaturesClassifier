import os
import torch
from flask import Flask, render_template, request, jsonify
from torchvision import transforms
from PIL import Image
from model.astro_net import AstroNet

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Cleanup
for f in os.listdir(UPLOAD_FOLDER):
    try:
        os.remove(os.path.join(UPLOAD_FOLDER, f))
    except:
        pass

try:
    os.remove("static/crop_preview.jpg")
except:
    pass

model = AstroNet(num_classes=4)
model.load_state_dict(torch.load("model/densetAstroNetFinalWeights.pth", map_location="cpu"))
model.eval()

class_names = ["bright_dune", "crater", "dark_dune", "impact_ejecta"]

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
            filename = file.filename
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            try:
                image = Image.open(filepath).convert("RGB")

                try:
                    x = int(float(request.form["x"]))
                    y = int(float(request.form["y"]))
                    w = int(float(request.form["width"]))
                    h = int(float(request.form["height"]))

                    if w > 0 and h > 0:
                        crop = image.crop((x, y, x + w, y + h))
                    else:
                        crop = image
                except (KeyError, ValueError):
                    crop = image

                crop = crop.resize((224, 224))
                input_tensor = transform(crop).unsqueeze(0)

                with torch.no_grad():
                    output = model(input_tensor)
                    probs = torch.softmax(output, dim=1)
                    conf, pred = torch.max(probs, dim=1)
                    label = class_names[pred.item()]
                    confidence = conf.item()

                crop_path = "static/crop_preview.jpg"
                crop.save(crop_path)

                return jsonify({
                    "label": label,
                    "confidence": round(confidence * 100, 2),
                    "crop_url": f"/static/crop_preview.jpg"
                })
            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
