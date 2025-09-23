import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.155.0/build/three.module.js';
import { ARButton } from 'https://cdn.jsdelivr.net/npm/three@0.155.0/examples/jsm/webxr/ARButton.js';

export async function startAR(imageDataURL = null) {
  if (!navigator.xr) {
    alert('WebXR not supported in this browser.');
    return;
  }

  const arContainer = document.getElementById('ar-container');
  if (!arContainer) {
      console.error("AR container div not found!");
      alert("AR container not found. Please check the HTML.");
      return;
  }
  while (arContainer.firstChild) {
      arContainer.removeChild(arContainer.firstChild);
  }

  const closeButton = document.createElement('button');
  closeButton.textContent = 'Close AR';
  closeButton.style.position = 'absolute';
  closeButton.style.top = '20px';
  closeButton.style.left = '20px';
  closeButton.style.zIndex = '1001';
  closeButton.style.padding = '10px 15px';
  closeButton.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
  closeButton.style.color = 'white';
  closeButton.style.border = 'none';
  closeButton.style.borderRadius = '5px';
  closeButton.style.cursor = 'pointer';

  closeButton.onclick = () => {
      const session = renderer.xr.getSession();
      if (session) {
          session.end();
      }
      arContainer.style.display = 'none';
  };
  arContainer.appendChild(closeButton);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.xr.enabled = true;
  arContainer.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera();
  scene.add(new THREE.HemisphereLight(0xffffff, 0xbbbbff, 1));

  const reticle = new THREE.Mesh(
    new THREE.RingGeometry(0.08, 0.1, 32).rotateX(-Math.PI / 2),
    new THREE.MeshBasicMaterial({ color: 0x00ff00 })
  );
  reticle.matrixAutoUpdate = false;
  reticle.visible = false;
  scene.add(reticle);

  const kolamTexture = await makeTexture(imageDataURL);
  const planeGeom = new THREE.PlaneGeometry(0.3, 0.3);
  const planeMat = new THREE.MeshBasicMaterial({ map: kolamTexture, transparent: true });

  const arButton = ARButton.createButton(renderer, { optionalFeatures: ['hit-test'] });

  // UPDATED: More robust styling to ensure visibility in all states
  arButton.style.position = 'absolute';
  arButton.style.bottom = '20px';
  arButton.style.left = '50%';
  arButton.style.transform = 'translateX(-50%)';
  arButton.style.padding = '12px 24px';
  arButton.style.border = '1px solid black';
  arButton.style.borderRadius = '30px';
  arButton.style.backgroundColor = '#d15c32';
  arButton.style.color = 'white';
  arButton.style.fontSize = '16px';
  arButton.style.cursor = 'pointer';
  arButton.style.zIndex = '1000';
  arButton.style.opacity = '1'; // This ensures it's not transparent

  // Override the 'not supported' message style specifically
  arButton.addEventListener('DOMNodeInserted', (e) => {
    if (e.target.textContent?.includes('NOT SUPPORTED')) {
        arButton.style.backgroundColor = '#6c757d'; // A gray color for disabled state
    }
  });


  arContainer.appendChild(arButton);

  let hitTestSource = null;
  let localSpace = null;

  renderer.xr.addEventListener('sessionstart', async () => {
    const session = renderer.xr.getSession();
    localSpace = await session.requestReferenceSpace('local');
    const viewerSpace = await session.requestReferenceSpace('viewer');
    hitTestSource = await session.requestHitTestSource({ space: viewerSpace });

    session.addEventListener('select', () => {
      if (reticle.visible) {
        const mesh = new THREE.Mesh(planeGeom, planeMat.clone());
        mesh.position.setFromMatrixPosition(reticle.matrix);
        mesh.quaternion.setFromRotationMatrix(reticle.matrix);
        scene.add(mesh);
      }
    });
  });

  renderer.xr.addEventListener('sessionend', () => {
    hitTestSource = null;
    arContainer.style.display = 'none';
  });

  renderer.setAnimationLoop((time, frame) => {
    if (frame && hitTestSource) {
      const refSpace = renderer.xr.getReferenceSpace();
      const hits = frame.getHitTestResults(hitTestSource);
      if (hits.length > 0) {
        const hit = hits[0];
        const pose = hit.getPose(refSpace);
        reticle.visible = true;
        reticle.matrix.fromArray(pose.transform.matrix);
      } else {
        reticle.visible = false;
      }
    }
    renderer.render(scene, camera);
  });
}

async function makeTexture(dataURL) {
  return new Promise((resolve) => {
    if (!dataURL) {
      const canvas = document.createElement('canvas');
      canvas.width = 256;
      canvas.height = 256;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, 256, 256);
      ctx.fillStyle = '#000';
      ctx.font = '24px sans-serif';
      ctx.fillText('Kolam', 95, 135);
      const tex = new THREE.Texture(canvas);
      tex.needsUpdate = true;
      resolve(tex);
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
      canvas.width = 256;
      canvas.height = 256;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = 'red';
      ctx.fillRect(0, 0, 256, 256);
      ctx.fillStyle = 'white';
      ctx.font = '20px sans-serif';
      ctx.fillText('Load Error', 80, 135);
      const tex = new THREE.Texture(canvas);
      tex.needsUpdate = true;
      resolve(tex);
    };
    img.src = dataURL;
  });
}