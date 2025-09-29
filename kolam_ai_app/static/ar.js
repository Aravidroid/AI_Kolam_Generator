import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.155.0/build/three.module.js';
import { ARButton } from 'https://cdn.jsdelivr.net/npm/three@0.155.0/examples/jsm/webxr/ARButton.js';

// Module-level variables to hold the Three.js scene objects
let camera, scene, renderer, reticle;
let hitTestSource = null;
let kolamPlane = null; // This will be the master object to clone
let hitTestSourceRequested = false;

/**
 * Main function to initialize and start the AR experience.
 * @param {string} imageDataURL - The data URL of the kolam image to display.
 */
export async function startAR(imageDataURL = null) {
    const arContainer = document.getElementById('ar-container');
    if (!arContainer) {
        console.error("AR container div not found!");
        return;
    }
    // Clear any previous AR instances
    while (arContainer.firstChild) {
        arContainer.removeChild(arContainer.firstChild);
    }

    // --- 1. Basic Scene Setup ---
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(70, window.innerWidth / window.innerHeight, 0.01, 20);
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.xr.enabled = true;
    arContainer.appendChild(renderer.domElement);
    
    const light = new THREE.HemisphereLight(0xffffff, 0xbbbbff, 3);
    light.position.set(0.5, 1, 0.25);
    scene.add(light);
    
    // --- 2. Create the Reticle (Surface Detection Indicator) ---
    reticle = new THREE.Mesh(
        new THREE.RingGeometry(0.05, 0.07, 32).rotateX(-Math.PI / 2),
        new THREE.MeshBasicMaterial()
    );
    reticle.matrixAutoUpdate = false;
    reticle.visible = false;
    scene.add(reticle);

    // --- 3. Create the Kolam Plane Object ---
    const kolamTexture = await makeTexture(imageDataURL);
    const planeGeometry = new THREE.PlaneGeometry(0.3, 0.3);
    const planeMaterial = new THREE.MeshBasicMaterial({
        map: kolamTexture,
        transparent: true,
        side: THREE.DoubleSide
    });
    kolamPlane = new THREE.Mesh(planeGeometry, planeMaterial);
    kolamPlane.rotation.x = -Math.PI / 2;
    
    // --- 4. Setup AR Button and Session Listeners ---
    const arButton = ARButton.createButton(renderer, {
        requiredFeatures: ['hit-test']
    });
    styleARButton(arButton);
    arContainer.appendChild(arButton);
    
    const closeButton = createCloseButton();
    arContainer.appendChild(closeButton);

    renderer.xr.addEventListener('sessionstart', (e) => {
        hitTestSourceRequested = false;
        hitTestSource = null;
        const session = e.target.getSession();
        session.addEventListener('select', placeKolam);
    });

    renderer.xr.addEventListener('sessionend', () => {
        hitTestSourceRequested = false;
        hitTestSource = null;
        const arContainer = document.getElementById('ar-container');
        if (arContainer) {
            arContainer.style.display = 'none';
        }
    });

    // --- 5. Start the animation loop ---
    renderer.setAnimationLoop(renderLoop);
    window.addEventListener('resize', onWindowResize);
}

/**
 * The main render loop, called every frame.
 */
function renderLoop(timestamp, frame) {
    if (frame) {
        const referenceSpace = renderer.xr.getReferenceSpace();
        const session = renderer.xr.getSession();

        if (hitTestSourceRequested === false) {
            session.requestReferenceSpace('viewer').then((refSpace) => {
                session.requestHitTestSource({ space: refSpace }).then((source) => {
                    hitTestSource = source;
                });
            });
            hitTestSourceRequested = true;
        }

        if (hitTestSource) {
            const hitTestResults = frame.getHitTestResults(hitTestSource);
            if (hitTestResults.length > 0) {
                const hit = hitTestResults[0];
                reticle.visible = true;
                reticle.matrix.fromArray(hit.getPose(referenceSpace).transform.matrix);
            } else {
                reticle.visible = false;
            }
        }
    }
    renderer.render(scene, camera);
}

/**
 * Called when the user taps the screen ('select' event).
 */
function placeKolam() {
    if (reticle.visible && kolamPlane) {
        const newKolam = kolamPlane.clone();
        newKolam.position.setFromMatrixPosition(reticle.matrix);
        scene.add(newKolam);
    }
}

// --- Helper Functions ---

/**
 * ✅ FIX: This function is updated to correctly close the AR view.
 */
function createCloseButton() {
    const button = document.createElement('button');
    button.textContent = '❌'; 
    Object.assign(button.style, {
        position: 'absolute', top: '20px', left: '20px', zIndex: '1001',
        padding: '10px', backgroundColor: 'rgba(0, 0, 0, 0.5)',
        color: 'white', border: 'none', borderRadius: '50%', cursor: 'pointer',
        width: '40px', height: '40px', fontSize: '18px', lineHeight: '1'
    });
    button.onclick = () => {
        const session = renderer.xr.getSession();
        if (session) {
            session.end(); // Gracefully end the WebXR session.
        }
        
        // Manually hide the container for immediate feedback.
        document.getElementById('ar-container').style.display = 'none';
        
        // Clean up resources.
        renderer.setAnimationLoop(null);
        window.removeEventListener('resize', onWindowResize);
    };
    return button;
}


function styleARButton(button) {
    Object.assign(button.style, {
        position: 'absolute', bottom: '20px', left: '50%',
        transform: 'translateX(-50%)', padding: '12px 24px',
        border: '1px solid black', borderRadius: '30px',
        backgroundColor: '#d15c32', color: 'white',
        fontSize: '16px', cursor: 'pointer', zIndex: '1000'
    });
}

function onWindowResize() {
    if (camera && renderer) {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    }
}

async function makeTexture(dataURL) {
    return new Promise((resolve) => {
        if (!dataURL) {
            const canvas = document.createElement('canvas');
            canvas.width = 256; canvas.height = 256;
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, 256, 256);
            ctx.fillStyle = '#000'; ctx.font = '24px sans-serif';
            ctx.fillText('Kolam', 95, 135);
            resolve(new THREE.CanvasTexture(canvas));
            return;
        }
        const img = new Image();
        img.onload = () => {
            const tex = new THREE.Texture(img);
            tex.needsUpdate = true;
            resolve(tex);
        };
        img.onerror = () => {
            const canvas = document.createElement('canvas');
            canvas.width = 256; canvas.height = 256;
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = 'red'; ctx.fillRect(0, 0, 256, 256);
            ctx.fillStyle = 'white'; ctx.font = '20px sans-serif';
            ctx.fillText('Load Error', 80, 135);
            resolve(new THREE.CanvasTexture(canvas));
        };
        img.src = dataURL;
    });
}