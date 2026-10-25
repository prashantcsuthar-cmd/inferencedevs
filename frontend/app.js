"use strict";

// ICE Engine frontend connected to the FastAPI SSE stream.

const API_URL = window.ICE_API_URL || (
  window.location.protocol === "file:" ? "http://127.0.0.1:8000" : window.location.origin
);

const $ = (id) => document.getElementById(id);

const elements = {
  tokens: $("tokensProcessed"),
  latency: $("latencyValue"),
  latencyTrack: $("latencyTrack"),
  pruned: $("prunedValue"),
  prunedTrack: $("prunedTrack"),
  baseline: $("baselineOutput"),
  ice: $("iceOutput"),
  baselineCard: document.querySelector(".baseline-card"),
  baselineStatus: $("baselineStatus"),
  baselineLines: $("baselineLineCount"),
  iceStatus: $("iceStatus"),
  grammar: $("grammarState"),
  allowed: $("allowedCount"),
  masked: $("maskedCount"),
  allowedBar: $("allowedBar"),
  maskedBar: $("maskedBar"),
  ratio: $("maskedRatio"),
  pills: $("tokenPills"),
  scenario: $("scenarioSelect"),
  run: $("runDemoButton"),
  reset: $("resetButton"),
  connection: $("connectionLabel"),
  footer: $("footerStatus"),
  canvas: $("trieCanvas"),
  fallback: $("trieFallback")
};

let activeRequest = null;
let threeScene = null;
let threeCamera = null;
let threeRenderer = null;
let threeRoot = null;
let animationFrame = null;
let resizeObserver = null;
let pointerState = null;

function formatNumber(value) {
  return Math.round(value).toLocaleString("en-US");
}

function setMetric(id, value) {
  const element = $(id);
  if (element) element.textContent = value;
}

function renderTokenPills(tokens) {
  elements.pills.replaceChildren();

  tokens.slice(0, 8).forEach((token, index) => {
    const pill = document.createElement("span");
    pill.className = "token-pill";
    pill.textContent = token;
    elements.pills.appendChild(pill);

    if (window.gsap) {
      gsap.fromTo(
        pill,
        { opacity: 0, y: 6 },
        { opacity: 1, y: 0, duration: 0.25, delay: index * 0.035 }
      );
    }
  });
}

function renderJsonOutput(target, value) {
  target.textContent = JSON.stringify(value, null, 2);
  target.scrollTop = 0;
}

function stopDemo() {
  if (activeRequest) {
    activeRequest.abort();
    activeRequest = null;
  }
  if (elements.run) {
    elements.run.disabled = false;
    elements.run.textContent = "Run inference";
  }
}

async function runDemonstration() {
  stopDemo();
  const prompts = {
    drone: "Generate a JSON object for a drone survey flight with waypoints containing latitude, longitude, and altitude, and a mission status.",
    finance: "Generate a JSON object for a financial transaction with transaction_id, amount, currency, and approved fields.",
    api: "Generate a JSON object for an API tool call with tool, arguments containing query and limit, and execute fields."
  };
  const startedAt = performance.now();
  const controller = new AbortController();
  activeRequest = controller;
  let tokenCount = 0;
  let buffer = "";

  elements.run.disabled = true;
  elements.run.textContent = "Running ICE inference...";
  elements.baselineCard.classList.remove("has-error");
  elements.baseline.textContent = "";
  elements.ice.textContent = "";
  elements.baselineLines.textContent = "CONNECTING";
  elements.baselineStatus.textContent = "Waiting for baseline tokens...";
  elements.iceStatus.textContent = "Connecting to inference engine...";
  elements.grammar.textContent = "INITIALIZING";
  elements.connection.textContent = "CONNECTING TO ICE API";
  elements.footer.textContent = "LIVE INFERENCE";

  const appendEvent = (rawEvent) => {
    const data = rawEvent.split("\n").filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trim()).join("\n");
    if (!data) return;
    const event = JSON.parse(data);
    if (event.error) throw new Error(event.error);

    if (event.runner_type === "ice") {
      elements.ice.textContent += event.token_text || "";
      elements.ice.scrollTop = elements.ice.scrollHeight;
      elements.grammar.textContent = event.active_grammar_state || event.status || "GENERATING";
      elements.iceStatus.textContent = event.status || "Generating";
      const telemetry = event.telemetry || {};
      if (telemetry.total_vocab) {
        tokenCount++;
        const size = Number(telemetry.total_vocab);
        const valid = Number(telemetry.valid_count) || 0;
        const masked = Math.max(0, size - valid);
        const ratio = Number(telemetry.masking_ratio) || 0;
        elements.allowed.textContent = formatNumber(valid);
        elements.masked.textContent = formatNumber(masked);
        elements.allowedBar.style.width = `${(valid / size * 100).toFixed(2)}%`;
        elements.maskedBar.style.width = `${(masked / size * 100).toFixed(2)}%`;
        setMetric("maskedRatio", `${(ratio * 100).toFixed(2)}% VOCABULARY MASKED`);
        setMetric("prunedValue", `${(ratio * 100).toFixed(2)}%`);
        elements.prunedTrack.style.width = `${(ratio * 100).toFixed(2)}%`;
        const latency = Number(telemetry.ms_per_token) || 0;
        setMetric("latencyValue", latency.toFixed(1));
        elements.latencyTrack.style.width = `${Math.max(4, Math.min(100, latency / 8))}%`;
        setMetric("tokensProcessed", formatNumber(tokenCount));
        renderTokenPills(event.sample_allowed_tokens || []);
      }
    } else if (event.runner_type === "baseline") {
      elements.baseline.textContent += event.token_text || "";
      elements.baseline.scrollTop = elements.baseline.scrollHeight;
      elements.baselineLines.textContent = `STEP ${event.step}`;
      elements.baselineStatus.textContent = event.status || "Generating";
      elements.baselineCard.classList.toggle("has-error", Boolean(event.has_error));
    }
  };

  try {
    const url = new URL(`${API_URL}/stream`);
    url.searchParams.set("prompt", prompts[elements.scenario.value] || prompts.drone);
    const response = await fetch(url, { signal: controller.signal, headers: { Accept: "text/event-stream" } });
    if (!response.ok) throw new Error((await response.text()) || `Backend returned HTTP ${response.status}`);
    if (!response.body) throw new Error("This browser does not support streaming responses.");
    elements.connection.textContent = "CONNECTED TO ICE API";

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const events = buffer.split("\n\n");
      buffer = events.pop() || "";
      events.forEach(appendEvent);
      if (done) break;
    }
    if (buffer.trim()) appendEvent(buffer);
    elements.footer.textContent = `LIVE INFERENCE | ${((performance.now() - startedAt) / 1000).toFixed(2)} s`;
  } catch (error) {
    if (error.name !== "AbortError") {
      elements.iceStatus.textContent = "Inference request failed";
      elements.connection.textContent = "API CONNECTION ERROR";
      elements.footer.textContent = "REQUEST FAILED";
      elements.ice.textContent += `\n\n${error.message}\nCheck that the backend is running at ${API_URL}.`;
      console.error("ICE API error:", error);
    }
  } finally {
    if (activeRequest === controller) {
      activeRequest = null;
      elements.run.disabled = false;
      elements.run.innerHTML = "Run inference";
    }
  }
}

// Phase 1: Fully cleaned reset restoring neutral dash states
function resetDashboard() {
  stopDemo();
  setMetric("tokensProcessed", "—");
  setMetric("latencyValue", "—");
  setMetric("prunedValue", "—");
  elements.allowed.textContent = "—";
  elements.masked.textContent = "—";
  elements.allowedBar.style.width = "0%";
  elements.maskedBar.style.width = "0%";
  elements.prunedTrack.style.width = "0%";
  elements.latencyTrack.style.width = "0%";
  elements.baseline.textContent = "";
  elements.ice.textContent = "";
  elements.baselineCard.classList.remove("has-error");
  elements.baselineStatus.textContent = "Ready";
  elements.baselineLines.textContent = "READY";
  elements.iceStatus.textContent = "Ready";
  elements.grammar.textContent = "READY";
  elements.pills.replaceChildren();
  elements.footer.textContent = "READY";
}

// -----------------------------
// Three.js local trie graph
// -----------------------------

function initTrie() {
  if (!window.THREE || !elements.canvas) {
    elements.fallback.textContent =
      "3D library unavailable. Check your internet connection.";
    return;
  }

  const THREE = window.THREE;
  const width = Math.max(280, elements.canvas.clientWidth);
  const height = Math.max(200, elements.canvas.clientHeight);

  threeScene = new THREE.Scene();
  threeCamera = new THREE.PerspectiveCamera(42, width / height, 0.1, 100);
  threeCamera.position.set(0, 0.3, 8);

  threeRenderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: true
  });

  threeRenderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
  threeRenderer.setSize(width, height);
  threeRenderer.setClearColor(0x000000, 0);

  elements.canvas.querySelector("canvas")?.remove();
  elements.canvas.appendChild(threeRenderer.domElement);
  elements.fallback.style.display = "none";

  threeScene.add(new THREE.AmbientLight(0xffffff, 1.5));

  const light = new THREE.PointLight(0x10b981, 24);
  light.position.set(1, 3, 5);
  threeScene.add(light);

  threeRoot = new THREE.Group();
  threeScene.add(threeRoot);

  const points = [
    [0, 0, 0],
    [0, 1.05, 0],
    [-1.25, 1.95, 0],
    [0, 2.05, 0],
    [1.25, 1.95, 0],
    [-1.8, 2.85, 0],
    [-0.9, 2.85, 0],
    [-0.35, 3.05, 0],
    [0.45, 2.85, 0],
    [1.1, 2.85, 0],
    [1.8, 2.85, 0]
  ];

  const links = [
    [0, 1], [1, 2], [1, 3], [1, 4],
    [2, 5], [2, 6], [3, 7], [3, 8],
    [4, 9], [4, 10]
  ];

  const nodeGeometry = new THREE.SphereGeometry(0.085, 12, 12);
  const rootGeometry = new THREE.SphereGeometry(0.15, 16, 16);

  const nodeMaterials = [
    new THREE.MeshStandardMaterial({
      color: 0x60a5fa,
      emissive: 0x1d4ed8,
      emissiveIntensity: 0.55
    }),
    new THREE.MeshStandardMaterial({
      color: 0x10b981,
      emissive: 0x065f46,
      emissiveIntensity: 0.45
    }),
    new THREE.MeshStandardMaterial({
      color: 0x64748b,
      emissive: 0x1e293b,
      emissiveIntensity: 0.3
    })
  ];

  const edgeMaterial = new THREE.LineBasicMaterial({
    color: 0x33566a,
    transparent: true,
    opacity: 0.85
  });

  links.forEach(([start, end]) => {
    const a = new THREE.Vector3(...points[start]);
    const b = new THREE.Vector3(...points[end]);
    const geometry = new THREE.BufferGeometry().setFromPoints([a, b]);
    threeRoot.add(new THREE.Line(geometry, edgeMaterial));
  });

  points.forEach((point, index) => {
    const material = nodeMaterials[
      index === 0 ? 0 : index <= 4 ? 1 : 2
    ];

    const mesh = new THREE.Mesh(
      index === 0 ? rootGeometry : nodeGeometry,
      material
    );

    mesh.position.set(...point);
    threeRoot.add(mesh);
  });

  threeRoot.position.y = -1.3;

  let dragging = false;
  let lastX = 0;
  let lastY = 0;

  threeRenderer.domElement.addEventListener("pointerdown", (event) => {
    dragging = true;
    lastX = event.clientX;
    lastY = event.clientY;
    threeRenderer.domElement.setPointerCapture(event.pointerId);
  });

  threeRenderer.domElement.addEventListener("pointermove", (event) => {
    if (!dragging) return;

    threeRoot.rotation.y += (event.clientX - lastX) * 0.008;
    threeRoot.rotation.x += (event.clientY - lastY) * 0.005;

    lastX = event.clientX;
    lastY = event.clientY;
  });

  function endDrag() {
    dragging = false;
  }

  threeRenderer.domElement.addEventListener("pointerup", endDrag);
  threeRenderer.domElement.addEventListener("pointercancel", endDrag);

  resizeObserver = new ResizeObserver(() => {
    if (!threeRenderer || !threeCamera) return;

    const newWidth = Math.max(280, elements.canvas.clientWidth);
    const newHeight = Math.max(200, elements.canvas.clientHeight);

    threeCamera.aspect = newWidth / newHeight;
    threeCamera.updateProjectionMatrix();
    threeRenderer.setSize(newWidth, newHeight);
  });

  resizeObserver.observe(elements.canvas);

  function animate() {
    animationFrame = requestAnimationFrame(animate);

    if (!dragging && threeRoot) {
      threeRoot.rotation.y += 0.0015;
    }

    threeRenderer.render(threeScene, threeCamera);
  }

  animate();
}

// Event listeners
elements.run.addEventListener("click", runDemonstration);
elements.reset.addEventListener("click", resetDashboard);

if (window.gsap) {
  gsap.from(".metric-card", {
    opacity: 0,
    y: 12,
    duration: 0.5,
    stagger: 0.08,
    ease: "power2.out"
  });
}

initTrie();

// Reflect the backend/model readiness returned by FastAPI.
fetch(`${API_URL}/status`)
  .then((response) => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  })
  .then((status) => {
    elements.connection.textContent = `BACKEND ${status.status}`;
    elements.footer.textContent = `${status.device || "LOCAL"} | READY`;
  })
  .catch(() => {
    elements.connection.textContent = "BACKEND OFFLINE";
    elements.footer.textContent = `START SERVER AT ${API_URL}`;
  });