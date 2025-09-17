import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { ARButton } from 'three/addons/webxr/ARButton.js';

let camera, scene, renderer;
let controls;
let kolamPlane, reticle;
let hitTestSource = null;
let hitTestSourceRequested = false;

// Teaching Mode variables
let teachingData = null;
let currentStep = 0;
let teachingLine;
let animationDot;
let isAutoPlaying = false;
let autoPlayInterval;

const responseDiv = document.getElementById('api-response');

init();
animate();

function init() {
    const container = document.getElementById('canvas-container');
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(70, window.innerWidth / window.innerHeight, 0.01, 20);
    camera.position.set(0, 1.6, 2);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.xr.enabled = true;
    container.appendChild(renderer.domElement);
    
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1.0);
    directionalLight.position.set(1, 2, 0);
    scene.add(directionalLight);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0);
    controls.update();
    controls.enableDamping = true;

    const textureLoader = new THREE.TextureLoader();
    const kolamTexture = textureLoader.load('/static/kolam.png');
    const planeGeometry = new THREE.PlaneGeometry(1, 1);
    const planeMaterial = new THREE.MeshStandardMaterial({
        map: kolamTexture,
        transparent: true,
        side: THREE.DoubleSide,
    });
    kolamPlane = new THREE.Mesh(planeGeometry, planeMaterial);
    kolamPlane.position.set(0, 0.5, 0);
    scene.add(kolamPlane);
    
    document.body.appendChild(ARButton.createButton(renderer, {
        requiredFeatures: ['hit-test']
    }));

    reticle = new THREE.Mesh(
        new THREE.RingGeometry(0.05, 0.07, 32).rotateX(-Math.PI / 2),
        new THREE.MeshBasicMaterial()
    );
    reticle.matrixAutoUpdate = false;
    reticle.visible = false;
    scene.add(reticle);

    setupEventListeners();
    window.addEventListener('resize', onWindowResize);
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
    renderer.setAnimationLoop(render);
}

function render(timestamp, frame) {
    controls.update();
    if (frame) {
        const referenceSpace = renderer.xr.getReferenceSpace();
        const session = renderer.xr.getSession();
        if (hitTestSourceRequested === false) {
            session.requestReferenceSpace('viewer').then((referenceSpace) => {
                session.requestHitTestSource({ space: referenceSpace }).then((source) => {
                    hitTestSource = source;
                });
            });
            session.addEventListener('end', () => {
                hitTestSourceRequested = false;
                hitTestSource = null;
                kolamPlane.position.set(0, 0.5, 0);
                reticle.visible = false;
            });
            hitTestSourceRequested = true;
        }
        if (hitTestSource) {
            const hitTestResults = frame.getHitTestResults(hitTestSource);
            if (hitTestResults.length) {
                const hit = hitTestResults[0];
                reticle.visible = true;
                reticle.matrix.fromArray(hit.getPose(referenceSpace).transform.matrix);
            } else {
                reticle.visible = false;
            }
        }
    }
    if (teachingLine && teachingLine.userData.isAnimating) {
        const progress = (performance.now() % 2000) / 2000;
        const drawCount = Math.floor(progress * teachingLine.geometry.attributes.position.count);
        teachingLine.geometry.setDrawRange(0, drawCount);
        const curve = new THREE.CatmullRomCurve3(teachingLine.userData.points);
        const pointOnCurve = curve.getPointAt(progress);
        animationDot.position.copy(pointOnCurve);
    }
    renderer.render(scene, camera);
}

function setupEventListeners() {
    document.getElementById('classify-btn').addEventListener('click', classifyKolam);
    document.getElementById('generate-btn').addEventListener('click', generateNewKolam);
    document.getElementById('teach-btn').addEventListener('click', startTeaching);
    document.getElementById('next-step-btn').addEventListener('click', nextStep);
    document.getElementById('prev-step-btn').addEventListener('click', prevStep);
    document.getElementById('autoplay-btn').addEventListener('click', toggleAutoPlay);

    renderer.domElement.addEventListener('click', () => {
        if (renderer.xr.isPresenting && reticle.visible) {
            kolamPlane.position.setFromMatrixPosition(reticle.matrix);
            if (teachingLine) teachingLine.position.setFromMatrixPosition(reticle.matrix);
            if (animationDot) animationDot.position.setFromMatrixPosition(reticle.matrix);
        }
    }, true);
}

// --- API Functions ---

async function classifyKolam() {
    const fileInput = document.getElementById('image-upload');
    if (fileInput.files.length === 0) {
        updateResponse({ error: "Please select an image file first." });
        return;
    }
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        const response = await fetch('/classify', { method: 'POST', body: formData });
        const result = await response.json(); // MODIFIED: Get full result object

        // MODIFIED: Handle new JSON structure with status
        if (result.status === 'error') {
            updateResponse(result); // Show the error message
            return;
        }
        updateResponse(result.data); // Show the data part on success

    } catch (error) {
        updateResponse({ error: "Failed to fetch classification." });
    }
}

async function generateNewKolam() {
    // MODIFIED: Get values from new dropdowns
    const kolamType = document.getElementById('kolam-type').value;
    const kolamComplexity = document.getElementById('kolam-complexity').value;

    try {
        const response = await fetch('/generate', { 
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            // MODIFIED: Send the selected values in the body
            body: JSON.stringify({ type: kolamType, complexity: kolamComplexity })
        });
        const result = await response.json(); // MODIFIED: Get full result object

        // MODIFIED: Handle new JSON structure with status
        if (result.status === 'error') {
            updateResponse(result);
            return;
        }
        updateResponse(result.data);

        // MODIFIED: Access image_url from inside the data object
        const textureLoader = new THREE.TextureLoader();
        textureLoader.load(result.data.image_url, (newTexture) => {
            kolamPlane.material.map = newTexture;
            kolamPlane.material.needsUpdate = true;
        });

    } catch (error) {
        updateResponse({ error: "Failed to fetch new kolam." });
    }
}

function updateResponse(data) {
    responseDiv.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
}

// --- Teaching Mode Functions ---

async function startTeaching() {
    try {
        const response = await fetch('/api/teach');
        const result = await response.json(); // MODIFIED: Get full result object

        // MODIFIED: Handle new JSON structure with status
        if (result.status === 'error') {
            updateResponse({ error: "Failed to fetch teaching steps." });
            return;
        }
        teachingData = result.data; // MODIFIED: Get teaching data from inside the data object
        
        currentStep = 0;
        document.getElementById('teach-controls').style.display = 'block';
        updateTeachingUI();
        createTeachingVisuals();
    } catch (error) {
        updateResponse({ error: "Failed to fetch teaching steps." });
    }
}

function updateTeachingUI() {
    if (!teachingData) return;
    const instructionText = document.getElementById('instruction-text');
    instructionText.textContent = `Step ${currentStep + 1}: ${teachingData.steps[currentStep].instruction}`;
    document.getElementById('prev-step-btn').disabled = currentStep === 0;
    document.getElementById('next-step-btn').disabled = currentStep === teachingData.steps.length - 1;
}

function nextStep() {
    if (teachingData && currentStep < teachingData.steps.length - 1) {
        currentStep++;
        updateTeachingUI();
        updateTeachingLine();
    }
}

function prevStep() {
    if (teachingData && currentStep > 0) {
        currentStep--;
        updateTeachingUI();
        updateTeachingLine();
    }
}

function toggleAutoPlay() {
    isAutoPlaying = !isAutoPlaying;
    const btn = document.getElementById('autoplay-btn');
    btn.textContent = isAutoPlaying ? 'Stop' : 'Auto Play';

    if (isAutoPlaying) {
        autoPlayInterval = setInterval(() => {
            if (currentStep < teachingData.steps.length - 1) {
                nextStep();
            } else {
                currentStep = 0;
                updateTeachingUI();
                updateTeachingLine();
            }
        }, 3000);
    } else {
        clearInterval(autoPlayInterval);
    }
}

function createTeachingVisuals() {
    if (teachingLine) scene.remove(teachingLine);
    if (animationDot) scene.remove(animationDot);

    const lineMaterial = new THREE.LineBasicMaterial({ color: 0xff0000, linewidth: 3 });
    const geometry = new THREE.BufferGeometry();
    teachingLine = new THREE.Line(geometry, lineMaterial);
    teachingLine.position.copy(kolamPlane.position);
    teachingLine.scale.copy(kolamPlane.scale);
    scene.add(teachingLine);
    
    const dotGeometry = new THREE.SphereGeometry(0.02, 16, 16);
    const dotMaterial = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
    animationDot = new THREE.Mesh(dotGeometry, dotMaterial);
    animationDot.position.copy(kolamPlane.position);
    animationDot.scale.copy(kolamPlane.scale);
    scene.add(animationDot);

    updateTeachingLine();
}

function updateTeachingLine() {
    if (!teachingData) return;
    const points = teachingData.steps[currentStep].path.map(p =>
        new THREE.Vector3(p.x - 0.5, p.y - 0.5, 0.01)
    );
    teachingLine.geometry.setFromPoints(points);
    teachingLine.userData.points = points;
    teachingLine.userData.isAnimating = true;

    if (points.length > 0) {
        animationDot.position.copy(points[0]).add(kolamPlane.position);
    }
}