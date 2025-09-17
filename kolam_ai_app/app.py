import os
import random
import json
import uuid
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ----------------- Folder Setup ----------------- #
UPLOAD_FOLDER = 'static/uploads'
GENERATED_FOLDER = 'static/generated'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GENERATED_FOLDER'] = GENERATED_FOLDER

# ----------------- Allowed File Types ----------------- #
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'svg'}

def allowed_file(filename):
    """Check if the uploaded file has a valid extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def unique_filename(filename):
    """Generate a unique filename by appending a UUID before the extension."""
    name, ext = os.path.splitext(filename)
    return f"{name}_{uuid.uuid4().hex}{ext}"

# ----------------- HTML Serving Route ----------------- #
@app.route('/')
def index():
    """Serves the main index.html page."""
    return render_template('index.html')

# ----------------- AI/ML API Routes (with Dummy Logic) ----------------- #
@app.route('/classify', methods=['POST'])
def classify_kolam():
    """
    Dummy classification endpoint.
    Saves the uploaded file and returns a classification result.
    """
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"status": "error", "message": "Invalid file type"}), 400

    # Secure the filename and make it unique
    filename = secure_filename(file.filename)
    filename = unique_filename(filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    kolam_types = ["pulli", "sikku", "kambi"]
    result = {
        "type": random.choice(kolam_types),
        "confidence": round(random.uniform(0.85, 0.99), 2),
        "saved_filename": filename
    }

    return jsonify({"status": "success", "data": result})

@app.route('/recreate', methods=['POST'])
def recreate_kolam():
    """
    Dummy vectorization/recreation endpoint.
    Returns a dummy SVG path and a stroke sequence.
    """
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400

    result = {
        "svg_path": "M10 80 C 40 10, 65 10, 95 80 S 150 150, 180 80",
        "strokes": [
            {"x": 10, "y": 80, "z": 0},
            {"x": 40, "y": 10, "z": 0},
            {"x": 65, "y": 10, "z": 0},
            {"x": 95, "y": 80, "z": 0},
        ]
    }

    return jsonify({"status": "success", "data": result})

@app.route('/generate', methods=['POST'])
def generate_kolam():
    """
    Generates a dummy kolam SVG based on type & complexity.
    Params: type (sikku, pulli, kambi), complexity (easy, medium, hard)
    """
    data = request.get_json() or {}
    kolam_type = data.get("type", "sikku").lower()
    complexity = data.get("complexity", "easy").lower()

    # Dummy SVG templates
    patterns = {
        "sikku": {
            "easy": """<circle cx="100" cy="100" r="80" stroke="black" stroke-width="2" fill="none"/>""",
            "medium": """<path d="M 50 100 Q 100 20 150 100 Q 100 180 50 100 Z" stroke="red" fill="none"/>""",
            "hard": """<path d="M 50 50 L 150 50 L 150 150 L 50 150 Z M 100 50 L 100 150 M 50 100 L 150 100" 
                        stroke="blue" fill="none"/>"""
        },
        "pulli": {
            "easy": """<circle cx="60" cy="60" r="5" fill="black"/><circle cx="140" cy="60" r="5" fill="black"/>""",
            "medium": """<circle cx="60" cy="60" r="5" fill="black"/><circle cx="100" cy="100" r="5" fill="black"/>
                         <circle cx="140" cy="60" r="5" fill="black"/>""",
            "hard": """<g stroke="black" fill="black">
                         <circle cx="60" cy="60" r="5"/><circle cx="100" cy="100" r="5"/>
                         <circle cx="140" cy="60" r="5"/><circle cx="60" cy="140" r="5"/>
                         <circle cx="140" cy="140" r="5"/>
                       </g>"""
        },
        "kambi": {
            "easy": """<line x1="50" y1="50" x2="150" y2="150" stroke="green" stroke-width="3"/>""",
            "medium": """<polyline points="50,150 100,50 150,150" stroke="purple" fill="none"/>""",
            "hard": """<polygon points="100,20 180,180 20,180" stroke="orange" fill="none"/>"""
        }
    }

    chosen_pattern = patterns.get(kolam_type, patterns["sikku"]).get(complexity, patterns["sikku"]["easy"])

    # Create SVG file
    filename = f"kolam_{kolam_type}_{complexity}_{uuid.uuid4().hex}.svg"
    filepath = os.path.join(app.config['GENERATED_FOLDER'], filename)

    svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200">{chosen_pattern}</svg>"""

    with open(filepath, "w") as f:
        f.write(svg_content)

    result = {
        "image_url": f"/{app.config['GENERATED_FOLDER']}/{filename}",
        "download_url": f"/download/generated/{filename}",
        "message": f"Generated a {complexity} {kolam_type} Kolam."
    }
    return jsonify({"status": "success", "data": result})

@app.route('/api/teach', methods=['GET'])
def get_teaching_steps():
    """
    Returns a structured JSON with step-by-step drawing instructions.
    """
    teach_data = {
        "total_steps": 3,
        "steps": [
            {
                "step": 1,
                "instruction": "Start by drawing the central loop.",
                "path": [
                    {"x": 0.5, "y": 0.8}, {"x": 0.65, "y": 0.65}, {"x": 0.5, "y": 0.5},
                    {"x": 0.35, "y": 0.65}, {"x": 0.5, "y": 0.8}
                ]
            },
            {
                "step": 2,
                "instruction": "Draw the top-left outer loop.",
                "path": [
                    {"x": 0.35, "y": 0.65}, {"x": 0.2, "y": 0.8}, {"x": 0.2, "y": 0.9},
                    {"x": 0.3, "y": 0.95}, {"x": 0.4, "y": 0.9}, {"x": 0.35, "y": 0.65}
                ]
            },
            {
                "step": 3,
                "instruction": "Finally, add the bottom-right petal.",
                "path": [
                    {"x": 0.65, "y": 0.65}, {"x": 0.8, "y": 0.5}, {"x": 0.65, "y": 0.35},
                    {"x": 0.5, "y": 0.5}
                ]
            }
        ]
    }
    return jsonify({"status": "success", "data": teach_data})

# ----------------- File Download Routes ----------------- #
@app.route('/download/<path:filename>')
def download_file(filename):
    """Securely serves a file from the upload folder for download."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/download/generated/<path:filename>')
def download_generated_file(filename):
    """Securely serves a file from the generated folder for download."""
    return send_from_directory(app.config['GENERATED_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
