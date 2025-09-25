import os
import uuid
import json
import math
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
import cv2 # OpenCV for image processing

# --- PyTorch Libraries ---
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# --- SciPy for Smoothing ---
from scipy.interpolate import splprep, splev

# --- Imports for Background Removal (Used for Classification only) ---
import io
from rembg import remove


app = Flask(__name__, template_folder='templates', static_folder='static')

# ----------------- Folder Setup ----------------- #
UPLOAD_FOLDER = os.path.join(app.static_folder, 'uploads')
GENERATED_FOLDER = os.path.join(app.static_folder, 'generated')
MODEL_PATH = 'kolam_classifier.pth'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GENERATED_FOLDER'] = GENERATED_FOLDER
CLASS_NAMES = ["kambi", "pulli", "sikku"]

# ----------------- CBAM Modules (Copied from train.py) ----------------- #
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAMBlock(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out

# ----------------- AI Model Setup (PyTorch) ----------------- #
def load_pytorch_model():
    try:
        model = models.efficientnet_b1()
        model.features.add_module("cbam", CBAMBlock(in_planes=1280))
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, len(CLASS_NAMES))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        print("--- PyTorch model loaded successfully. ---")
        return model
    except Exception as e:
        print(f"--- WARNING: Could not load PyTorch model. Error: {e} ---")
        return None

model = load_pytorch_model()
infer_transform = transforms.Compose([
    transforms.Resize(260),
    transforms.CenterCrop(240),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ----------------- Background Removal Function (For Classification) ----------------- #
def remove_background(img_path):
    try:
        with open(img_path, 'rb') as f:
            input_bytes = f.read()
        
        output_bytes = remove(input_bytes)
        img_bg_removed = Image.open(io.BytesIO(output_bytes)).convert("RGBA")
        
        white_bg = Image.new("RGBA", img_bg_removed.size, "WHITE")
        white_bg.paste(img_bg_removed, (0, 0), img_bg_removed)
        
        clean_img_rgb = white_bg.convert('RGB')
        return cv2.cvtColor(np.array(clean_img_rgb), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"--- Background removal failed: {e}. Falling back to original image. ---")
        return cv2.imread(img_path)


# ----------------- Kolam Analysis Pipeline ----------------- #

def analyze_cleaned_image(img, min_points=30):
    if img is None:
        raise ValueError("Input image to analysis is None.")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create an inverted binary image for line/skeleton detection (white lines on black background)
    _, binary_for_lines = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    skeleton = cv2.ximgproc.thinning(binary_for_lines)
    
    # Find all separate lines in the kolam
    contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if len(c) > min_points]
    
    # --- ✅ DOT DETECTION FIX ---
    # The SimpleBlobDetector finds DARK blobs on a LIGHT background by default.
    # We must create a NON-inverted binary image for it to work correctly.
    _, binary_for_dots = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    dots = []
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 15
    params.filterByCircularity = True
    params.minCircularity = 0.7
    params.filterByConvexity = True
    params.minConvexity = 0.8
    params.filterByInertia = True
    params.minInertiaRatio = 0.4
    
    detector = cv2.SimpleBlobDetector_create(params)
    
    # Detect blobs (dots) on the correct binary image
    keypoints = detector.detect(binary_for_dots)
    
    for kp in keypoints:
        dots.append((int(kp.pt[0]), int(kp.pt[1])))
        
    return contours, img.shape, dots

def smooth_contour(contour, is_closed=True):
    contour = contour[:,0,:]
    if len(contour) < 4: return contour.reshape(-1,1,2)
    
    x, y = contour[:,0], contour[:,1]
    smoothing_factor = len(contour) * 0.1 
    
    try:
        tck, u = splprep([x, y], s=smoothing_factor, per=is_closed)
        num_points = max(len(x) * 2, 200) 
        u_new = np.linspace(0, 1, num_points)
        x_new, y_new = splev(u_new, tck)
        return np.stack([x_new, y_new], axis=-1).reshape(-1,1,2).astype(np.int32)
    except:
        return contour.reshape(-1,1,2)

def center_and_scale(contours, dots, img_shape, viewbox_size=200):
    if not contours and not dots: return [], []
    all_points_list = []
    if contours:
        all_points_list.append(np.vstack([c.reshape(-1,2) for c in contours if c.size > 0]))
    if dots:
        all_points_list.append(np.array(dots))
    if not all_points_list: return [], []
    
    all_points = np.vstack(all_points_list)
    x_min, y_min = np.min(all_points, axis=0)
    x_max, y_max = np.max(all_points, axis=0)
    shape_w, shape_h = x_max - x_min, y_max - y_min
    if shape_w == 0 or shape_h == 0: return [], []
    
    scale = viewbox_size / (max(shape_w, shape_h) * 1.1)
    
    new_w, new_h = shape_w * scale, shape_h * scale
    padding_x = (viewbox_size - new_w) / 2
    padding_y = (viewbox_size - new_h) / 2

    scaled_contours = [(((c.reshape(-1,2) - np.array([x_min, y_min])) * scale) + np.array([padding_x, padding_y])) for c in contours]
    scaled_dots = [(((np.array(d) - np.array([x_min, y_min])) * scale) + np.array([padding_x, padding_y])) for d in dots]
    return scaled_contours, scaled_dots

# ----------------- Allowed File Types & Helpers ----------------- #
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename): return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def unique_filename(filename):
    name, ext = os.path.splitext(filename)
    return f"{name}_{uuid.uuid4().hex}{ext}"

# ----------------- Routes ----------------- #
@app.route('/')
def index(): return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_kolam():
    # This route remains unchanged
    if not model: return jsonify({"status": "error", "message": "AI model is not loaded."}), 500
    if 'file' not in request.files: return jsonify({"status": "error", "message": "No file part"}), 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename): return jsonify({"status": "error", "message": "Invalid file"}), 400
    filename = unique_filename(secure_filename(file.filename))
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    try:
        image = Image.open(filepath).convert("RGB")
        image_tensor = infer_transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            _, pred = torch.max(outputs, 1)
        predicted_class, confidence = CLASS_NAMES[pred.item()], probabilities[pred.item()].item() * 100
        result = {"type": predicted_class, "confidence": round(confidence, 2), "saved_filename": filename, "saved_url": f"/static/uploads/{filename}"}
        return jsonify({"status": "success", "data": result})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Classification error: {e}"}), 500

@app.route('/recreate', methods=['POST'])
def recreate_kolam():
    if 'file' not in request.files: return jsonify({"status": "error", "message": "No file part"}), 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename): return jsonify({"status": "error", "message": "Invalid file"}), 400
    
    filename = unique_filename(secure_filename(file.filename))
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        uploaded_image = cv2.imread(filepath)
        if uploaded_image is None:
            raise ValueError("Could not read the uploaded image.")

        contours, img_shape, dots = analyze_cleaned_image(uploaded_image)
        
        smooth_contours = [smooth_contour(c) for c in contours]
        scaled_contours, scaled_dots = center_and_scale(smooth_contours, dots, img_shape)
        
        all_paths = []
        for contour in scaled_contours:
            if contour is not None and contour.size > 0:
                path_points = [{"x": round(p[0], 2), "y": round(p[1], 2)} for p in contour.reshape(-1, 2)]
                all_paths.append(path_points)
        
        formatted_dots = [{"x": round(d[0], 2), "y": round(d[1], 2)} for d in scaled_dots]

        svg_elements = []
        for path in all_paths:
            points_str = " ".join([f"{p['x']},{p['y']}" for p in path])
            svg_elements.append(f'<polyline points="{points_str}" stroke="black" fill="none" stroke-width="2"/>')
        for dot in formatted_dots:
            svg_elements.append(f'<circle cx="{dot["x"]}" cy="{dot["y"]}" r="3" fill="black"/>')
        svg_content = "\n".join(svg_elements)

        result = { 
            "svg_content": svg_content, 
            "paths": all_paths,
            "dots": formatted_dots
        }
        return jsonify({"status": "success", "data": result})

    except Exception as e:
        app.logger.error(f"Recreation error: {e}", exc_info=True)
        if os.path.exists(filepath): os.remove(filepath)
        return jsonify({"status": "error", "message": f"Recreation error: {e}"}), 500

@app.route('/generate', methods=['POST'])
def generate_kolam():
    # This route remains unchanged
    data = request.get_json() or {}
    kolam_type, complexity = data.get("type", "sikku").lower(), data.get("complexity", "easy").lower()
    level = {"easy": 3, "medium": 5, "hard": 7}.get(complexity, 3)
    svg_inner_content = ""
    if kolam_type == "pulli":
        dot_count = level + 2
        spacing = 180 / dot_count
        for i in range(dot_count):
            for j in range(dot_count):
                cx, cy = 10 + (i * spacing) + (spacing / 2), 10 + (j * spacing) + (spacing / 2)
                svg_inner_content += f'<circle cx="{cx}" cy="{cy}" r="3" fill="black"/>\n'
    elif kolam_type == "kambi":
        for i in range(level):
            inset = i * (80 / level)
            points = f"{100},{20+inset} {180-inset},{180-inset} {20+inset},{180-inset}"
            svg_inner_content += f'<polygon points="{points}" stroke="black" stroke-width="2" fill="none"/>\n'
    else:
        path_points = [f"{(90*math.sin((level)*t+np.pi/2)+100):.2f},{(90*math.sin((level-1)*t)+100):.2f}" for t in np.linspace(0,2*np.pi,200)]
        svg_inner_content = f'<path d="M {" L ".join(path_points)}" stroke="black" stroke-width="2" fill="none"/>'
    filename = f"kolam_{kolam_type}_{complexity}_{uuid.uuid4().hex}.svg"
    filepath = os.path.join(app.config['GENERATED_FOLDER'], filename)
    with open(filepath, "w", encoding='utf-8') as f: f.write(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">{svg_inner_content}</svg>')
    return jsonify({"status": "success", "data": {"image_url": f"/static/generated/{filename}", "filename": filename}})

@app.route('/api/teach', methods=['GET'])
def get_teaching_steps():
    # This route remains unchanged
    teach_data = {"total_steps": 3, "steps": [{"step": 1, "instruction": "Start by drawing the central loop.", "path": [{"x": 0.5, "y": 0.8}, {"x": 0.65, "y": 0.65}, {"x": 0.5, "y": 0.5}, {"x": 0.35, "y": 0.65}, {"x": 0.5, "y": 0.8}]}, {"step": 2, "instruction": "Draw the top-left outer loop.", "path": [{"x": 0.35, "y": 0.65}, {"x": 0.2, "y": 0.8}, {"x": 0.2, "y": 0.9}, {"x": 0.3, "y": 0.95}, {"x": 0.4, "y": 0.9}, {"x": 0.35, "y": 0.65}]}, {"step": 3, "instruction": "Finally, add the bottom-right petal.", "path": [{"x": 0.65, "y": 0.65}, {"x": 0.8, "y": 0.5}, {"x": 0.65, "y": 0.35}, {"x": 0.5, "y": 0.5}]}]}
    return jsonify({"status": "success", "data": teach_data})

@app.route('/download/<path:filename>')
def download_file(filename): return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/download/generated/<path:filename>')
def download_generated_file(filename): return send_from_directory(app.config['GENERATED_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5000, debug=True)
