
"use strict";

// ICE Engine frontend.
// All numbers and generation output in this version are simulated demo data.

const TOTAL_VOCAB = 151643;
const API_URL = "http://127.0.0.1:8000";

const scenarios = {
  drone: {
    state: "EXPECT_OBJECT_KEY",
    allowed: 1210,
    baseline: `Here is your drone flight path!
{
  "mission": "survey",
  "waypoints": [
    {"lat": 12.97, "lon": 77.59,},
    {"lat": 13.01, "lon": 77.62}
  ],
  "status": "ready"`,
    ice: {
      mission: "survey",
      waypoints: [
        { lat: 12.97, lon: 77.59, altitude: 120 },
        { lat: 13.01, lon: 77.62, altitude: 150 }
      ],
      status: "ready"
    },
    tokens: ['"mission"', '"waypoints"', '"status"', '"altitude"', '"ready"']
  },
  finance: {
    state: "EXPECT_PROPERTY_VALUE",
    allowed: 864,
    baseline: `Transaction approved, here you go:
{
  "transaction_id": "TX-4821",
  "amount": 2500.00,
  "currency": "INR",
  "approved": true,
}`,
    ice: {
      transaction_id: "TX-4821",
      amount: 2500,
      currency: "INR",
      approved: true
    },
    tokens: ['"transaction_id"', '"amount"', '"currency"', '"INR"', '"approved"']
  },
  api: {
    state: "EXPECT_OBJECT_KEY",
    allowed: 642,
    baseline: `Calling the API now:
{
  "tool": "search_records",
  "arguments": {
    "query": "active users",
    "limit": "many"
  },
  "execute": true,
}`,
    ice: {
      tool: "search_records",
      arguments: {
        query: "active users",
        limit: 10
      },
      execute: true
    },
    tokens: ['"tool"', '"arguments"', '"query"', '"limit"', '"execute"']
  }
};

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

let demoTimer = null;
let threeScene = null;
let threeCamera = null;
let threeRenderer = null;
let threeRoot = null;
let animationFrame = null;
let resizeObserver = null;
let pointerState = null;
let sessionTokens = 12480;

function formatNumber(value) {
  return Math.round(value).toLocaleString("en-US");
}

function animateValue(element, from, to, duration = 0.45) {
  if (!element) return;

  if (window.gsap) {
    gsap.killTweensOf(element);
    const counter = { value: from };

    gsap.to(counter, {
      value: to,
      duration,
      ease: "power2.out",
      onUpdate: () => {
        element.textContent = formatNumber(counter.value);
      }
    });
  } else {
    element.textContent = formatNumber(to);
  }
}

function setMetric(id, value) {
  const element = $(id);
  if (element) element.textContent = value;
}

function updateTelemetry(allowed, latency = 402.7) {
  const safeAllowed = Math.max(0, Math.min(TOTAL_VOCAB, allowed));
  const masked = TOTAL_VOCAB - safeAllowed;
  const maskedRatio = (masked / TOTAL_VOCAB) * 100;
  const allowedRatio = (safeAllowed / TOTAL_VOCAB) * 100;

  animateValue(elements.allowed, Number(elements.allowed.textContent.replaceAll(",", "")) || 0, safeAllowed);
  animateValue(elements.masked, Number(elements.masked.textContent.replaceAll(",", "")) || 0, masked);

  elements.allowedBar.style.width = `${allowedRatio}%`;
  elements.maskedBar.style.width = `${100 - allowedRatio}%`;

  setMetric("maskedRatio", `${maskedRatio.toFixed(2)}% VOCABULARY MASKED`);
  setMetric("prunedValue", maskedRatio.toFixed(2));
  elements.prunedTrack.style.width = `${maskedRatio}%`;

  setMetric("latencyValue", latency.toFixed(1));
  elements.latencyTrack.style.width = `${Math.max(4, Math.min(100, latency / 8))}%`;
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

function streamText(target, text, options = {}) {
  const speed = options.speed ?? 10;
  const onComplete = options.onComplete ?? (() => {});
  let index = 0;

  target.textContent = "";

  return new Promise((resolve) => {
    function writeNext() {
      if (index >= text.length) {
        onComplete();
        resolve();
        return;
      }

      const chunkSize = Math.min(3, text.length - index);
      target.textContent += text.slice(index, index + chunkSize);
      index += chunkSize;
      target.scrollTop = target.scrollHeight;

      if (window.gsap && options.animate && index <= 12) {
        gsap.fromTo(
          target,
          { opacity: 0.65 },
          { opacity: 1, duration: 0.15 }
        );
      }

      demoTimer = setTimeout(writeNext, speed);
    }

    writeNext();
  });
}

function stopDemo() {
  if (demoTimer !== null) {
    clearTimeout(demoTimer);
    demoTimer = null;
  }

  if (elements.run) {
    elements.run.disabled = false;
    elements.run.innerHTML = "<span>▶</span> Run demonstration";
  }
}


async function runDemonstration() {
  stopDemo();

  const scenarioName = elements.scenario.value;

  const prompts = {
    drone: "Generate a JSON object for a drone survey flight with waypoints containing latitude, longitude, and altitude, and a mission status.",
    finance: "Generate a JSON object for a financial transaction with transaction_id, amount, currency, and approved fields.",
    api: "Generate a JSON object for an API tool call with tool, arguments containing query and limit, and execute fields."
  };

  elements.run.disabled = true;
  elements.run.textContent = "Running ICE inference...";

  elements.baselineCard.classList.remove("has-error");
  elements.baselineStatus.textContent = "Baseline simulation unavailable";
  elements.baselineLines.textContent = "REAL ICE REQUEST";
  elements.iceStatus.textContent = "Sending request to backend...";
  elements.connection.textContent = "CONNECTING TO ICE API";
  elements.footer.textContent = "LIVE INFERENCE";

  elements.baseline.textContent =
    "Unconstrained baseline is not connected to a backend endpoint yet.";

  elements.ice.textContent = "";
  elements.grammar.textContent = "GENERATING";

  try {
    const response = await fetch(`${API_URL}/generate`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        prompt: prompts[scenarioName] || prompts.drone,
        max_new_tokens: 128
      })
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.detail || "ICE API request failed.");
    }

    // Show the actual text produced by the ICE engine.
    elements.ice.textContent = result.text || "";
    elements.ice.scrollTop = 0;

    // Display real inference diagnostics.
    animateValue(
      elements.tokens,
      Number(elements.tokens.textContent.replaceAll(",", "")) || 0,
      result.generated_tokens || 0
    );

    const latency = Number(result.ms_per_token) || 0;
    const vocabularySize = Number(result.vocabulary_size) || TOTAL_VOCAB;
    const validTokens = Number(result.average_valid_tokens) || 0;
    const maskedTokens = Number(result.average_masked_tokens) || 0;
    const maskingRatio = Number(result.average_masking_ratio) || 0;

    setMetric("latencyValue", latency.toFixed(1));
    setMetric("prunedValue", (maskingRatio * 100).toFixed(2));

    elements.latencyTrack.style.width =
      `${Math.max(4, Math.min(100, latency / 8))}%`;

    elements.prunedTrack.style.width =
      `${Math.max(0, Math.min(100, maskingRatio * 100))}%`;

    elements.allowed.textContent = formatNumber(validTokens);
    elements.masked.textContent = formatNumber(maskedTokens);

    const allowedRatio = vocabularySize
      ? (validTokens / vocabularySize) * 100
      : 0;

    elements.allowedBar.style.width =
      `${Math.max(0, Math.min(100, allowedRatio))}%`;

    elements.maskedBar.style.width =
      `${Math.max(0, Math.min(100, 100 - allowedRatio))}%`;

    setMetric(
      "maskedRatio",
      `${(maskingRatio * 100).toFixed(2)}% VOCABULARY MASKED`
    );

    elements.grammar.textContent = result.final_state || "COMPLETED";
    elements.iceStatus.textContent = "Actual ICE response received";
    elements.connection.textContent = "CONNECTED TO ICE API";
    elements.footer.textContent =
      `LIVE INFERENCE · ${Number(result.elapsed_seconds).toFixed(2)} s`;

    // Display token IDs' associated diagnostic counts as a compact sample.
    renderTokenPills(
      (result.valid_token_counts || [])
        .slice(0, 8)
        .map((count, index) => `Step ${index + 1}: ${count} valid`)
    );

    elements.baselineStatus.textContent =
      "No baseline API endpoint configured";
  } catch (error) {
    elements.iceStatus.textContent = "Inference request failed";
    elements.connection.textContent = "API CONNECTION ERROR";
    elements.footer.textContent = "REQUEST FAILED";
    elements.ice.textContent =
      `${error.message}\n\nCheck that the backend is running at ${API_URL}.`;

    console.error("ICE API error:", error);
  } finally {
    elements.run.disabled = false;
    elements.run.textContent = "↻ Run again";
  }
}


function resetDashboard() {
  stopDemo();
  sessionTokens = 12480;

  setMetric("tokensProcessed", "12,480");
  setMetric("latencyValue", "402.7");
  setMetric("prunedValue", "99.20");

  elements.baseline.textContent = "";
  elements.ice.textContent = "";
  elements.baselineCard.classList.remove("has-error");

  elements.baselineStatus.textContent = "Potential syntax failure";
  elements.baselineLines.textContent = "READY";
  elements.iceStatus.textContent = "Schema-valid output";
  elements.grammar.textContent = "EXPECT_OBJECT_KEY";
  elements.connection.textContent = "SIMULATED TELEMETRY";
  elements.footer.textContent = "DEMO MODE";

  elements.scenario.value = "drone";
  updateTelemetry(1210, 402.7);
  renderTokenPills(scenarios.drone.tokens);
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

  // 11 nodes + 10 branches = 21 objects in this graph.
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

elements.scenario.addEventListener("change", () => {
  const scenario = scenarios[elements.scenario.value];
  if (!scenario) return;

  elements.grammar.textContent = scenario.state;
  updateTelemetry(scenario.allowed);
  renderTokenPills(scenario.tokens);
});

renderTokenPills(scenarios.drone.tokens);
updateTelemetry(1210, 402.7);

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

// Keep this frontend self-contained. Backend integration will be added
// after we confirm the actual API request body and streaming event format.
