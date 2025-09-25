const classifyBtn = document.getElementById('classify-btn');
const recreateBtn = document.getElementById('recreate-btn');
const generateBtn = document.getElementById('generate-btn');
const teachBtn = document.getElementById('teach-btn');
const uploadPreviewBtn = document.getElementById('upload-preview-btn');
const arBtn = document.getElementById('ar-btn');
const speedUpBtn = document.getElementById('speed-up-btn');

const fileInput = document.getElementById('image-upload');
const responseDiv = document.getElementById('api-response');
const kolamCanvas = document.getElementById('kolamCanvas');
const previewCanvas = document.getElementById('previewCanvas');

const teachControls = document.getElementById('teach-controls');
const instructionText = document.getElementById('instruction-text');
const prevStepBtn = document.getElementById('prev-step-btn');
const nextStepBtn = document.getElementById('next-step-btn');
const autoplayBtn = document.getElementById('autoplay-btn');

let teachingData = null;
let currentStep = 0;
let autoPlayInterval = null;
let isAutoPlaying = false;

// State variables for animation speed
let animationSpeed = 1;
const speedLevels = [1, 2, 4, 8];

// expose last image data url globally for AR to consume
window.currentImageDataURL = null;

function updateResponse(data) {
    responseDiv.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
}

function snapshotCurrentKolamToDataURL() {
    const svg = kolamCanvas.outerHTML;
    const blob = new Blob([svg], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(blob);

    return new Promise((resolve) => {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => {
            const tmp = document.createElement('canvas');
            tmp.width = 512;
            tmp.height = 512;
            const ctx = tmp.getContext('2d');
            ctx.fillStyle = 'white';
            ctx.fillRect(0, 0, tmp.width, tmp.height);
            ctx.drawImage(img, 0, 0, tmp.width, tmp.height);
            const dataUrl = tmp.toDataURL('image/png');
            URL.revokeObjectURL(url);
            resolve(dataUrl);
        };
        img.onerror = () => {
            try {
                const dataUrl = previewCanvas.toDataURL('image/png');
                resolve(dataUrl);
            } catch (e) {
                resolve(null);
            }
        };
        img.src = url;
    });
}

// Event listener to cycle through speeds
speedUpBtn.addEventListener('click', () => {
    const currentIndex = speedLevels.indexOf(animationSpeed);
    const nextIndex = (currentIndex + 1) % speedLevels.length; // Loop back to the start
    animationSpeed = speedLevels[nextIndex];
    speedUpBtn.textContent = `Speed Up (${animationSpeed}x)`;
});


uploadPreviewBtn.addEventListener('click', async () => {
    if (!fileInput.files || fileInput.files.length === 0) {
        updateResponse({ error: 'Please choose a file first.' });
        return;
    }
    const file = fileInput.files[0];
    const reader = new FileReader();
    reader.onload = function(e) {
        const img = new Image();
        img.onload = async function() {
            const ctx = previewCanvas.getContext('2d');
            ctx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
            ctx.drawImage(img, 0, 0, previewCanvas.width, previewCanvas.height);
            const dataURL = previewCanvas.toDataURL('image/png');
            kolamCanvas.innerHTML = `<image href="${dataURL}" x="0" y="0" width="200" height="200" />`;
            window.currentImageDataURL = await snapshotCurrentKolamToDataURL();
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
});

classifyBtn.addEventListener('click', async () => {
    if (!fileInput.files || fileInput.files.length === 0) {
        updateResponse({ error: 'Please choose a file before classifying.' });
        return;
    }
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        const resp = await fetch('/classify', { method: 'POST', body: formData });
        const result = await resp.json();
        if (result.status === 'error') {
            updateResponse(result);
            return;
        }
        updateResponse(result.data);

        if (result.data.saved_url) {
            kolamCanvas.innerHTML = `<image href="${result.data.saved_url}" x="0" y="0" width="200" height="200" />`;
            window.currentImageDataURL = await snapshotCurrentKolamToDataURL();
        }
    } catch (e) {
        updateResponse({ error: 'Failed to call classify endpoint.' });
        console.error(e);
    }
});


// 🎨 RECREATE ENDPOINT (UPDATED FOR CORRECT MULTI-PATH ANIMATION) 🎨
recreateBtn.addEventListener('click', async () => {
    if (!fileInput.files || fileInput.files.length === 0) {
        updateResponse({ error: 'Please choose a file before recreating.' });
        return;
    }

    // Setup UI for animation
    animationSpeed = 1;
    speedUpBtn.textContent = `Speed Up (1x)`;
    speedUpBtn.style.display = 'inline-block'; // Show the speed button

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        const resp = await fetch('/recreate', { method: 'POST', body: formData });
        const result = await resp.json();
        if (result.status === 'error') {
            updateResponse(result);
            return;
        }
        updateResponse({ // Show a cleaner summary
            message: "Recreation data received.",
            dots_found: result.data.dots.length,
            paths_found: result.data.paths.length
        });

        // ✅ Call the new, more powerful animation function
        await animateRecreation(result.data.paths, result.data.dots, result.data.svg_content);
        
        // Update the AR image and steps list after animation
        window.currentImageDataURL = await snapshotCurrentKolamToDataURL();

        const stepsListContainer = document.getElementById('stepsList-container');
        const stepsList = document.getElementById('stepsList');
        stepsListContainer.style.display = 'block';
        stepsList.innerHTML = '';
        if (result.data.paths && result.data.paths.length) {
            result.data.paths.forEach((path, idx) => {
                const li = document.createElement('li');
                li.textContent = `Path ${idx + 1}: ${path.length} points`;
                stepsList.appendChild(li);
            });
        }
    } catch (e) {
        updateResponse({ error: 'Failed to call recreate endpoint.' });
        console.error(e);
    } finally {
        speedUpBtn.style.display = 'none'; // Hide speed button when done
    }
});

/**
 * ✅ NEW AND IMPROVED ANIMATION FUNCTION ✅
 * Animates drawing by drawing dots first, then each path sequentially.
 * Incorporates the animationSpeed variable.
 */
function animateRecreation(paths, dots, finalSVGContent) {
    return new Promise(resolve => {
        kolamCanvas.innerHTML = ''; // Clear the canvas

        // 1. Draw all dots instantly
        dots.forEach(dot => {
            const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('cx', dot.x);
            circle.setAttribute('cy', dot.y);
            circle.setAttribute('r', '3'); // A bit bigger for visibility
            circle.setAttribute('fill', 'black');
            kolamCanvas.appendChild(circle);
        });

        if (!paths || paths.length === 0) {
            resolve(); // Nothing to animate
            return;
        }

        let currentPathIndex = 0;

        function animateNextPath() {
            if (currentPathIndex >= paths.length) {
                // Animation finished, render the final perfect SVG to ensure quality
                kolamCanvas.innerHTML = finalSVGContent;
                resolve();
                return;
            }

            const currentPoints = paths[currentPathIndex];
            if (currentPoints.length < 2) {
                // Path is too short, skip and move to the next one
                currentPathIndex++;
                animateNextPath();
                return;
            }

            const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            path.setAttribute('stroke', '#d15c32'); // Use primary color for drawing
            path.setAttribute('stroke-width', '2');
            path.setAttribute('fill', 'none');
            path.setAttribute('stroke-linecap', 'round');
            kolamCanvas.appendChild(path);

            let pointIndex = 0;
            let pathData = `M ${currentPoints[0].x.toFixed(2)} ${currentPoints[0].y.toFixed(2)}`;
            
            function drawStep() {
                // Draw multiple segments per frame based on speed
                for (let j = 0; j < animationSpeed && pointIndex < currentPoints.length - 1; j++) {
                    pointIndex++;
                    const point = currentPoints[pointIndex];
                    pathData += ` L ${point.x.toFixed(2)} ${point.y.toFixed(2)}`;
                }
                
                path.setAttribute('d', pathData);

                if (pointIndex < currentPoints.length - 1) {
                    requestAnimationFrame(drawStep); // Continue animating this path
                } else {
                    // Finished this path, move to the next one
                    currentPathIndex++;
                    animateNextPath();
                }
            }
            
            drawStep();
        }
        
        // Start the animation with the first path
        animateNextPath();
    });
}


generateBtn.addEventListener('click', async () => {
    const type = document.getElementById('kolam-type').value;
    const complexity = document.getElementById('kolam-complexity').value;

    try {
        const resp = await fetch('/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ type, complexity })
        });
        const result = await resp.json();
        if (result.status === 'error') {
            updateResponse(result);
            return;
        }
        updateResponse(result.data);

        if (result.data.image_url) {
            const svgResp = await fetch(result.data.image_url);
            if (svgResp.ok) {
                const svgText = await svgResp.text();
                try {
                    const parser = new DOMParser();
                    const doc = parser.parseFromString(svgText, 'image/svg+xml');
                    kolamCanvas.innerHTML = doc.documentElement.innerHTML;
                    window.currentImageDataURL = await snapshotCurrentKolamToDataURL();
                } catch (e) {
                    kolamCanvas.innerHTML = `<image href="${result.data.image_url}" x="0" y="0" width="200" height="200" />`;
                    window.currentImageDataURL = await snapshotCurrentKolamToDataURL();
                }
            } else {
                kolamCanvas.innerHTML = `<text x="10" y="20">Generated file: ${result.data.filename}</text>`;
                window.currentImageDataURL = await snapshotCurrentKolamToDataURL();
            }
        }
    } catch (e) {
        updateResponse({ error: 'Failed to call generate endpoint.' });
        console.error(e);
    }
});

// --- Teaching flow (unchanged) ---
teachBtn.addEventListener('click', async () => {
    try {
        const resp = await fetch('/api/teach');
        const result = await resp.json();
        if (result.status === 'error') {
            updateResponse(result);
            return;
        }
        teachingData = result.data;
        currentStep = 0;
        teachControls.style.display = 'block';
        updateTeachingUI();
        await drawTeachingStep();
    } catch (e) {
        updateResponse({ error: 'Failed to fetch teaching steps.' });
        console.error(e);
    }
});

prevStepBtn.addEventListener('click', async () => {
    if (!teachingData) return;
    if (currentStep > 0) {
        currentStep--;
        updateTeachingUI();
        await drawTeachingStep();
    }
});

nextStepBtn.addEventListener('click', async () => {
    if (!teachingData) return;
    if (currentStep < teachingData.steps.length - 1) {
        currentStep++;
        updateTeachingUI();
        await drawTeachingStep();
    }
});

autoplayBtn.addEventListener('click', () => {
    if (!teachingData) return;
    isAutoPlaying = !isAutoPlaying;
    autoplayBtn.textContent = isAutoPlaying ? 'Stop' : 'Auto Play';
    if (isAutoPlaying) {
        autoPlayInterval = setInterval(async () => {
            if (currentStep < teachingData.steps.length - 1) currentStep++;
            else currentStep = 0;
            updateTeachingUI();
            await drawTeachingStep();
        }, 2000);
    } else {
        clearInterval(autoPlayInterval);
    }
});

function updateTeachingUI() {
    if (!teachingData) return;
    instructionText.textContent = `Step ${currentStep + 1}: ${teachingData.steps[currentStep].instruction}`;
    prevStepBtn.disabled = currentStep === 0;
    nextStepBtn.disabled = currentStep === (teachingData.steps.length - 1);
}

async function drawTeachingStep() {
    if (!teachingData) return;
    const step = teachingData.steps[currentStep];
    if (!step || !step.path) return;

    const points = step.path.map(p => `${p.x * 200} ${p.y * 200}`).join(' ');
    const polyline = `<polyline points="${points}" fill="none" stroke="red" stroke-width="2" stroke-linecap="round" />`;
    const start = step.path[0];
    const startCircle = `<circle cx="${start.x * 200}" cy="${start.y * 200}" r="3" fill="green" />`;

    kolamCanvas.innerHTML = polyline + startCircle;
    window.currentImageDataURL = await snapshotCurrentKolamToDataURL();
}
