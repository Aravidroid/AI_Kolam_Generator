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

def analyze_cleaned_image(img, min_points=30):
    if img is None: raise ValueError("Input image to analysis is None.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_for_lines = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    skeleton = cv2.ximgproc.thinning(binary_for_lines)
    contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if len(c) > min_points]
    
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
    keypoints = detector.detect(binary_for_dots)
    for kp in keypoints: dots.append((int(kp.pt[0]), int(kp.pt[1])))
    return contours, img.shape, dots

def analyze_dot_grid(dots, tolerance=10):
    """
    Analyzes a list of dot coordinates to determine the grid structure.
    - tolerance: The minimum pixel distance to consider a dot as being in a new row or column.
    """
    # Ensure we have a list to work with, and enough dots to form a grid.
    dot_list = list(dots)
    if not dot_list or len(dot_list) < 4:
        return {"grid_type": "N/A", "grid_size": "N/A"}

    # --- Find the number of rows ---
    dot_list.sort(key=lambda d: d[1]) # Sort by Y-coordinate
    num_rows = 1
    last_y = dot_list[0][1]
    for dot in dot_list:
        if (dot[1] - last_y) > tolerance:
            num_rows += 1
            last_y = dot[1]

    # --- Find the number of columns ---
    dot_list.sort(key=lambda d: d[0]) # Sort by X-coordinate
    num_cols = 1
    last_x = dot_list[0][0]
    for dot in dot_list:
        if (dot[0] - last_x) > tolerance:
            num_cols += 1
            last_x = dot[0]

    # --- Determine if it's a regular grid or irregular pattern ---
    # A simple heuristic: if the total dot count is close to rows * cols, it's a grid.
    is_grid = abs(len(dot_list) - (num_rows * num_cols)) <= 2

    if is_grid and num_rows > 1 and num_cols > 1:
        return {"grid_type": "Square", "grid_size": f"{num_rows}x{num_cols}"}
    else:
        return {"grid_type": "Irregular", "grid_size": f"{len(dot_list)} dots found"}
    
def _find_matching_point(x, y, point_set, tolerance):
    """Checks if a point exists in the set within a given tolerance box."""
    for px, py in point_set:
        if abs(px - x) <= tolerance and abs(py - y) <= tolerance:
            return True
    return False

def analyze_symmetry(dots, paths, viewbox_size=200, tolerance=10):
    """Analyzes the symmetry of the kolam's points."""
    all_points = set()
    # Combine all unique points from dots and paths into a single set for efficient lookup
    for d in dots:
        all_points.add((round(d[0]), round(d[1])))
    for path in paths:
        for p in path:
            all_points.add((round(p['x']), round(p['y'])))

    if not all_points:
        return {"symmetry_type": "N/A"}

    total_points = len(all_points)
    vertical_matches = 0
    horizontal_matches = 0
    rotational_matches = 0

    for x, y in all_points:
        # Check for vertical symmetry partner: (x, y) -> (width - x, y)
        if _find_matching_point(viewbox_size - x, y, all_points, tolerance):
            vertical_matches += 1

        # Check for horizontal symmetry partner: (x, y) -> (x, height - y)
        if _find_matching_point(x, viewbox_size - y, all_points, tolerance):
            horizontal_matches += 1
        
        # Check for 90-degree rotational partner: (x, y) -> (y, width - x)
        if _find_matching_point(y, viewbox_size - x, all_points, tolerance):
            rotational_matches += 1
    
    # If >90% of points have a symmetric match, we consider it symmetric
    symmetry_threshold = 0.9
    is_vertical = (vertical_matches / total_points) >= symmetry_threshold
    is_horizontal = (horizontal_matches / total_points) >= symmetry_threshold
    is_rotational = (rotational_matches / total_points) >= symmetry_threshold

    # Build a user-friendly report string
    report = []
    if is_vertical: report.append("Vertical")
    if is_horizontal: report.append("Horizontal")
    if is_rotational: report.append("Rotational (90°)")
    
    if not report:
        return {"symmetry_type": "None Detected"}
    else:
        return {"symmetry_type": ", ".join(report)}
    
def analyze_motifs(contours, similarity_threshold=0.2):
    """
    Analyzes a list of contours to find repeating shapes (motifs).
    - similarity_threshold: How similar shapes must be to be grouped (lower is stricter).
    """
    if not contours or len(contours) < 2:
        return {"unique_motifs": len(contours), "repeating_motifs": 0}

    unassigned_indices = list(range(len(contours)))
    groups = []

    while unassigned_indices:
        # Start a new group with the first available contour
        base_index = unassigned_indices.pop(0)
        base_contour = contours[base_index]
        new_group = [base_index]
        
        # Iterate through remaining contours to find matches
        indices_to_check = list(unassigned_indices)
        for comp_index in indices_to_check:
            comp_contour = contours[comp_index]
            
            # Compare the two shapes using OpenCV's matchShapes function
            score = cv2.matchShapes(base_contour, comp_contour, cv2.CONTOURS_MATCH_I1, 0.0)
            
            # If they are similar enough, add to the group and remove from unassigned list
            if score < similarity_threshold:
                new_group.append(comp_index)
                unassigned_indices.remove(comp_index)
        
        groups.append(new_group)

    # Count how many groups represent a repeating pattern (i.e., have more than one member)
    repeating_count = sum(1 for group in groups if len(group) > 1)
    
    return {
        "unique_motifs": len(groups),
        "repeating_motifs": repeating_count
    }    

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
    except: return contour.reshape(-1,1,2)

def center_and_scale(contours, dots, img_shape, viewbox_size=200):
    if not contours and not dots: return [], []
    all_points_list = []
    if contours: all_points_list.append(np.vstack([c.reshape(-1,2) for c in contours if c.size > 0]))
    if dots: all_points_list.append(np.array(dots))
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


# ----------------- NEW HELPER FUNCTIONS ----------------- #

def generate_svg_from_data(paths, dots):
    """Generates SVG content and saves it to a file."""
    svg_elements = []
    for path in paths:
        points_str = " ".join([f"{p['x']},{p['y']}" for p in path])
        svg_elements.append(f'<polyline points="{points_str}" stroke="black" fill="none" stroke-width="2"/>')
    for dot in dots:
        svg_elements.append(f'<circle cx="{dot["x"]}" cy="{dot["y"]}" r="3" fill="black"/>')
    
    svg_content = "\n".join(svg_elements)
    
    generated_filename = f"recreated_{uuid.uuid4().hex}.svg"
    generated_filepath = os.path.join(app.config['GENERATED_FOLDER'], generated_filename)
    full_svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">{svg_content}</svg>'
    
    with open(generated_filepath, "w", encoding='utf-8') as f:
        f.write(full_svg)
        
    generated_image_url = f"/static/generated/{generated_filename}"
    return generated_image_url, svg_content

def generate_teach_steps(paths):
    """Dynamically generates 'Teach Me' steps from a list of paths."""
    teach_steps = []
    for i, path in enumerate(paths):
        instruction = f"Draw loop number {i + 1}."
        if i == 0:
            instruction = "Start by drawing the first main loop."
        elif i == len(paths) - 1:
            instruction = "Finish the design with the final loop."
        
        teach_steps.append({
            "step": i + 1,
            "instruction": instruction,
            "path": path
        })
    return teach_steps


# ----------------- Routes ----------------- #
@app.route('/')
def index(): return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_kolam():
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

# ----------------- REFACTORED RECREATE ROUTE ----------------- #

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
        if uploaded_image is None: raise ValueError("Could not read the uploaded image.")

        contours, img_shape, dots = analyze_cleaned_image(uploaded_image)
        smooth_contours = [smooth_contour(c) for c in contours]
        scaled_contours, scaled_dots = center_and_scale(smooth_contours, dots, img_shape)
        
        # Format path and dot data for JSON response
        all_paths = []
        for contour in scaled_contours:
            if contour is not None and contour.size > 0:
                path_points = [{"x": round(p[0], 2), "y": round(p[1], 2)} for p in contour.reshape(-1, 2)]
                all_paths.append(path_points)
        formatted_dots = [{"x": round(d[0], 2), "y": round(d[1], 2)} for d in scaled_dots]

        # --- Perform All Design Principle Analyses ---
        dots_for_analysis = [tuple(d) for d in scaled_dots]
        design_principles = analyze_dot_grid(dots_for_analysis)
        design_principles.update(analyze_symmetry(scaled_dots, all_paths))
        design_principles.update(analyze_motifs(scaled_contours))

        # --- Generate SVG and Teach Steps using Helper Functions ---
        generated_image_url, svg_content = generate_svg_from_data(all_paths, formatted_dots)
        teach_steps = generate_teach_steps(all_paths)

        # Assemble the final result object
        result = { 
            "svg_content": svg_content, 
            "paths": all_paths,
            "dots": formatted_dots,
            "generated_image_url": generated_image_url,
            "teach_steps": teach_steps,
            "design_principles": design_principles
        }
        return jsonify({"status": "success", "data": result})

    except Exception as e:
        app.logger.error(f"Recreation error: {e}", exc_info=True)
        if os.path.exists(filepath): os.remove(filepath)
        return jsonify({"status": "error", "message": f"Recreation error: {e}"}), 500
    
# Other routes (generate, teach, etc.) remain the same
@app.route('/generate', methods=['POST'])
def generate_kolam():
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

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5000, debug=True)