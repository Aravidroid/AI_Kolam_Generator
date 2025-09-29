// main UI script with all features, including corrected "Teach Me" functionality

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

// --- State Variables ---
window.arImageSourceUrl = null;
let animationSpeed = 4;
const speedLevels = [4, 8, 16, 32];

// Variables for the dynamic teaching flow
let recreationData = null;
let teachingData = null;
let currentStep = 0;
let autoPlayInterval = null;
let isAutoPlaying = false;


function updateResponse(data) {
    responseDiv.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
}

// --- Button Event Listeners ---

speedUpBtn.addEventListener('click', () => {
    const currentIndex = speedLevels.indexOf(animationSpeed);
    const nextIndex = (currentIndex + 1) % speedLevels.length;
    animationSpeed = speedLevels[nextIndex];
    speedUpBtn.textContent = `Speed Up (${animationSpeed}x)`;
});

uploadPreviewBtn.addEventListener('click', () => {
    if (!fileInput.files || fileInput.files.length === 0) return;
    const file = fileInput.files[0];
    const reader = new FileReader();
    reader.onload = function(e) {
        const img = new Image();
        img.onload = function() {
            const ctx = previewCanvas.getContext('2d');
            ctx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
            ctx.drawImage(img, 0, 0, previewCanvas.width, previewCanvas.height);
            const dataURL = previewCanvas.toDataURL('png');
            kolamCanvas.innerHTML = `<image href="${dataURL}" x="0" y="0" width="200" height="200" />`;
            window.arImageSourceUrl = dataURL;
            updateResponse({ message: "Preview loaded. Ready for AR View." });
            teachBtn.disabled = true;
            teachControls.style.display = 'none';
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
});

recreateBtn.addEventListener('click', async () => {
    if (!fileInput.files || fileInput.files.length === 0) return;
    
    animationSpeed = 1;
    speedUpBtn.textContent = `Speed Up (1x)`;
    speedUpBtn.style.display = 'inline-block';
    teachBtn.disabled = true;
    teachControls.style.display = 'none';

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        const resp = await fetch('/recreate', { method: 'POST', body: formData });
        const result = await resp.json();
        if (result.status === 'error') {
            updateResponse(result);
            return;
        }
        
        recreationData = result.data;
        updateResponse(result.data);

        await animateRecreation(recreationData.paths, recreationData.dots, recreationData.svg_content);
        
        window.arImageSourceUrl = recreationData.generated_image_url;
        
        teachBtn.disabled = false;

    } catch (e) {
        updateResponse({ error: 'Failed to call recreate endpoint.' });
        console.error(e);
    } finally {
        speedUpBtn.style.display = 'none';
    }
});

// --- Animation Function ---
function animateRecreation(paths, dots, finalSVGContent) {
    return new Promise(resolve => {
        kolamCanvas.innerHTML = '';
        dots.forEach(dot => {
            const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('cx', dot.x);
            circle.setAttribute('cy', dot.y);
            circle.setAttribute('r', '3');
            circle.setAttribute('fill', 'black');
            kolamCanvas.appendChild(circle);
        });

        if (!paths || paths.length === 0) {
            resolve();
            return;
        }
        let currentPathIndex = 0;
        function animateNextPath() {
            if (currentPathIndex >= paths.length) {
                kolamCanvas.innerHTML = finalSVGContent;
                resolve();
                return;
            }
            const currentPoints = paths[currentPathIndex];
            if (currentPoints.length < 2) {
                currentPathIndex++;
                animateNextPath();
                return;
            }
            const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            path.setAttribute('stroke', '#d15c32');
            path.setAttribute('stroke-width', '2');
            path.setAttribute('fill', 'none');
            path.setAttribute('stroke-linecap', 'round');
            kolamCanvas.appendChild(path);
            let pointIndex = 0;
            let pathData = `M ${currentPoints[0].x.toFixed(2)} ${currentPoints[0].y.toFixed(2)}`;
            function drawStep() {
                for (let j = 0; j < animationSpeed && pointIndex < currentPoints.length - 1; j++) {
                    pointIndex++;
                    const point = currentPoints[pointIndex];
                    pathData += ` L ${point.x.toFixed(2)} ${point.y.toFixed(2)}`;
                }
                path.setAttribute('d', pathData);
                if (pointIndex < currentPoints.length - 1) {
                    requestAnimationFrame(drawStep);
                } else {
                    currentPathIndex++;
                    animateNextPath();
                }
            }
            drawStep();
        }
        animateNextPath();
    });
}

generateBtn.addEventListener('click', async () => {
    const kolamTypeSelect = document.getElementById('kolam-type');
    const complexitySelect = document.getElementById('kolam-complexity');

    if (!kolamTypeSelect || !complexitySelect) {
        alert("Error: Could not find the type and complexity dropdowns.");
        return;
    }

    const kolamType = kolamTypeSelect.value;
    const complexity = complexitySelect.value;

    try {
        const response = await fetch('/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                type: kolamType,
                complexity: complexity
            })
        });

        const result = await response.json();

        if (result.status === 'success') {
            const imageUrl = result.data.image_url;
            // Display the newly generated SVG in the canvas
            kolamCanvas.innerHTML = `<image href="${imageUrl}" x="0" y="0" width="200" height="200" />`;
            // Update the response div with the result
            updateResponse({ message: "Kolam generated successfully.", data: result.data });
            // Make the new image available for AR view
            window.arImageSourceUrl = imageUrl;
        } else {
            updateResponse(result); // Show error from the server
        }
    } catch (e) {
        updateResponse({ error: 'Failed to call generate endpoint.', details: e.toString() });
        console.error(e);
    }
});

// --- Dynamic Teaching Flow ---

teachBtn.addEventListener('click', () => {
    if (!recreationData || !recreationData.teach_steps) {
        alert("Please recreate a kolam first to get the drawing steps!");
        return;
    }
    
    teachingData = recreationData.teach_steps;
    currentStep = 0;
    isAutoPlaying = false;
    clearInterval(autoPlayInterval);
    autoplayBtn.textContent = 'Auto Play';
    teachControls.style.display = 'block';
    updateTeachingUI();
    drawTeachingStep();
});

prevStepBtn.addEventListener('click', () => {
    if (currentStep > 0) {
        currentStep--;
        updateTeachingUI();
        drawTeachingStep();
    }
});

nextStepBtn.addEventListener('click', () => {
    // ✅ FIX: Use teachingData.length instead of teachingData.steps.length
    if (currentStep < teachingData.length - 1) {
        currentStep++;
        updateTeachingUI();
        drawTeachingStep();
    }
});

autoplayBtn.addEventListener('click', () => {
    if (isAutoPlaying) {
        clearInterval(autoPlayInterval);
        isAutoPlaying = false;
        autoplayBtn.textContent = 'Auto Play';
    } else {
        isAutoPlaying = true;
        autoplayBtn.textContent = 'Stop';
        autoPlayInterval = setInterval(() => {
            // ✅ FIX: Use teachingData.length for the modulo operation
            currentStep = (currentStep + 1) % teachingData.length;
            updateTeachingUI();
            drawTeachingStep();
        }, 2000);
    }
});

function updateTeachingUI() {
    instructionText.textContent = teachingData[currentStep].instruction;
    prevStepBtn.disabled = currentStep === 0;
    // ✅ FIX: Use teachingData.length instead of teachingData.steps.length
    nextStepBtn.disabled = currentStep === (teachingData.length - 1);
}

function drawTeachingStep() {
    kolamCanvas.innerHTML = '';

    recreationData.dots.forEach(dot => {
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', dot.x);
        circle.setAttribute('cy', dot.y);
        circle.setAttribute('r', '3');
        circle.setAttribute('fill', 'black');
        kolamCanvas.appendChild(circle);
    });

    const stepData = teachingData[currentStep];
    const points = stepData.path.map(p => `${p.x} ${p.y}`).join(' ');

    const polyline = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
    polyline.setAttribute('points', points);
    polyline.setAttribute('fill', 'none');
    polyline.setAttribute('stroke', '#000000ff');
    polyline.setAttribute('stroke-width', '2.5');
    polyline.setAttribute('stroke-linecap', 'round');
    kolamCanvas.appendChild(polyline);
}