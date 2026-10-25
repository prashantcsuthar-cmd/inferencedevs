"use strict";

// Connect the static dashboard to the FastAPI status endpoint and SSE stream.
function resolveApiUrl() {
  const configured = window.ICE_API_URL || new URLSearchParams(window.location.search).get("api");
  if (configured) return configured.replace(/\/+$/, "");

  if (window.location.protocol === "file:") return "http://127.0.0.1:8000";

  const localHosts = new Set(["localhost", "127.0.0.1", "::1"]);
  const frontendPorts = new Set(["3000", "4173", "5173", "5500", "8080", "8081"]);
  if (
    window.location.port &&
    window.location.port !== "8000" &&
    (localHosts.has(window.location.hostname) || frontendPorts.has(window.location.port))
  ) {
    return window.location.protocol + "//" + window.location.hostname + ":8000";
  }
  if (!window.location.port && localHosts.has(window.location.hostname)) {
    return window.location.protocol + "//" + window.location.hostname + ":8000";
  }

  return window.location.origin;
}

const API_URL = resolveApiUrl();
const $ = (id) => document.getElementById(id);

const elements = {
  tokens: $("tokensProcessed"),
  tokensTrack: $("tokensTrack"),
  latency: $("latencyValue"),
  latencyTrack: $("latencyTrack"),
  pruned: $("prunedValue"),
  prunedTrack: $("prunedTrack"),
  validation: $("validationValue"),
  validationIcon: $("validationIcon"),
  validationHint: $("validationHint"),
  validationTrack: $("validationTrack"),
  baseline: $("baselineOutput"),
  ice: $("iceOutput"),
  baselineCard: document.querySelector(".baseline-card"),
  baselineStatus: $("baselineStatus"),
  baselineLines: $("baselineLineCount"),
  iceStatus: $("iceStatus"),
  iceValidationTag: $("iceValidationTag"),
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
  connectionDetail: $("connectionDetail"),
  footer: $("footerStatus"),
  backendBadge: $("backendBadge"),
  systemDot: $("systemStatusDot"),
  device: $("deviceValue"),
  headerDevice: $("headerDevice"),
  engineState: $("engineState"),
  vocabulary: $("vocabularyCapacity"),
  vocabularyBarLabel: $("vocabularyBarLabel"),
  serviceVocabulary: $("serviceVocabulary"),
  streamIndicator: $("streamIndicator"),
  canvas: $("trieCanvas"),
  fallback: $("trieFallback")
};

const scenarios = {
  drone: {
    prompt: 'Return only a JSON object for a drone survey: mission_id "survey-01", altitude 80, and status "ready".',
    schema: {
      type: "object",
      properties: {
        mission_id: { type: "string" },
        altitude: { type: "number" },
        status: { type: "string" }
      },
      required: ["mission_id", "altitude", "status"],
      additionalProperties: false
    }
  },
  finance: {
    prompt: 'Return only a JSON object for a financial transaction: transaction_id "txn-1042", amount 125.5, currency "USD", and approved true.',
    schema: {
      type: "object",
      properties: {
        transaction_id: { type: "string" },
        amount: { type: "number" },
        currency: { type: "string" },
        approved: { type: "boolean" }
      },
      required: ["transaction_id", "amount", "currency", "approved"],
      additionalProperties: false
    }
  },
  api: {
    prompt: 'Return only a JSON object for an API tool call: tool "search", query "weather", limit 5, and execute true.',
    schema: {
      type: "object",
      properties: {
        tool: { type: "string" },
        query: { type: "string" },
        limit: { type: "number" },
        execute: { type: "boolean" }
      },
      required: ["tool", "query", "limit", "execute"],
      additionalProperties: false
    }
  }
};

let activeRequest = null;
let backendIsOnline = false;
let backendState = "CHECKING";
let backendDetail = "Connecting to the local inference service";
let threeScene = null;
let threeCamera = null;
let threeRenderer = null;
let threeRoot = null;
let resizeObserver = null;

function formatNumber(value) {
  return Math.round(Number(value) || 0).toLocaleString("en-US");
}

function updateVocabularyCapacity(value) {
  const size = Number(value);
  if (!Number.isFinite(size) || size <= 0) return;
  const formatted = formatNumber(size);
  elements.vocabulary.textContent = formatted;
  elements.vocabularyBarLabel.textContent = formatted + " tokens";
  elements.serviceVocabulary.textContent = formatted;
}

function setConnection(state, detail) {
  const isOnline = state === "ONLINE";
  backendState = state;
  backendDetail = detail || (isOnline ? "Ready for an inference request" : "");
  backendIsOnline = isOnline;
  const label = state === "OFFLINE"
    ? "BACKEND OFFLINE"
    : state === "ERROR"
      ? "BACKEND ERROR"
      : state === "STARTING"
        ? "MODEL STARTING"
        : isOnline
          ? "BACKEND ONLINE"
          : state;

  elements.backendBadge.lastChild.textContent = " " + (
    isOnline ? "BACKEND READY" : state === "OFFLINE" ? "API OFFLINE" : state
  );
  elements.backendBadge.classList.toggle("is-online", isOnline);
  elements.backendBadge.classList.toggle("is-offline", state === "OFFLINE" || state === "ERROR");
  elements.systemDot.classList.toggle("is-offline", state === "OFFLINE" || state === "ERROR");
  elements.systemDot.classList.toggle("is-starting", state === "STARTING");
  if (!activeRequest) {
    elements.connection.textContent = label;
    elements.connectionDetail.textContent = backendDetail;
    if (isOnline) {
      elements.footer.textContent = "READY";
    } else if (state === "OFFLINE") {
      elements.footer.textContent = "API " + API_URL;
    } else if (state === "ERROR") {
      elements.footer.textContent = "MODEL INITIALIZATION FAILED";
    } else if (state === "STARTING") {
      elements.footer.textContent = "MODEL LOADING";
    }
  }
}

function renderTokenPills(tokens) {
  elements.pills.replaceChildren();
  (Array.isArray(tokens) ? tokens : []).slice(0, 8).forEach((token, index) => {
    const pill = document.createElement("span");
    pill.className = "token-pill";
    pill.textContent = String(token);
    elements.pills.appendChild(pill);

    if (window.gsap) {
      window.gsap.fromTo(
        pill,
        { opacity: 0, y: 6 },
        { opacity: 1, y: 0, duration: 0.25, delay: index * 0.035 }
      );
    }
  });
}

function setValidationResult(report) {
  if (!report || report.schema_valid === null || report.schema_valid === undefined) {
    elements.validationIcon.textContent = report && report.json_valid ? "✓" : "!";
    elements.validationIcon.classList.toggle("is-fail", Boolean(report && !report.json_valid));
    elements.validation.textContent = report && report.json_valid ? "JSON" : "—";
    elements.validationHint.textContent = report && report.json_valid
      ? "JSON valid; schema result unavailable"
      : "Output did not complete validation";
    elements.validationTrack.style.width = report && report.json_valid ? "55%" : "0%";
    elements.validationTrack.classList.toggle("track-fail", Boolean(report && !report.json_valid));
    elements.iceValidationTag.textContent = report && report.json_valid ? "JSON VALID" : "NOT VALIDATED";
    return;
  }

  const valid = report.schema_valid === true;
  elements.validation.textContent = valid ? "PASS" : "FAIL";
  elements.validationIcon.textContent = valid ? "✓" : "×";
  elements.validationIcon.classList.toggle("is-fail", !valid);
  elements.validationHint.textContent = valid
    ? "JSON and schema valid"
    : report.error || "Output did not match the schema";
  elements.validationTrack.style.width = "100%";
  elements.validationTrack.classList.toggle("track-fail", !valid);
  elements.iceValidationTag.textContent = valid ? "SCHEMA VALID" : "SCHEMA INVALID";
}

function formatReportStatus(report) {
  if (!report) return "No final report received";
  if (report.schema_valid === true) return "Completed · schema valid";
  if (report.schema_valid === false) return "Completed · schema invalid";
  if (report.json_valid === true) return "Completed · JSON valid";
  const status = report.status || "Finished";
  return report.error ? status + " · " + report.error : status;
}

function stopDemo() {
  if (activeRequest) {
    activeRequest.abort();
    activeRequest = null;
  }
  elements.run.disabled = false;
  elements.run.innerHTML = '<span aria-hidden="true">▶</span> Run inference';
}

function resetRunDisplay() {
  elements.tokens.textContent = "—";
  elements.latency.textContent = "—";
  elements.pruned.textContent = "—";
  elements.validation.textContent = "—";
  elements.validationHint.textContent = "Awaiting a completed run";
  elements.validationTrack.style.width = "0%";
  elements.validationTrack.classList.remove("track-fail");
  elements.validationIcon.textContent = "…";
  elements.validationIcon.classList.remove("is-fail");
  elements.latencyTrack.style.width = "0%";
  elements.prunedTrack.style.width = "0%";
  elements.tokensTrack.style.width = "0%";
  elements.allowed.textContent = "—";
  elements.masked.textContent = "—";
  elements.allowedBar.style.width = "0%";
  elements.maskedBar.style.width = "0%";
  elements.ratio.textContent = "AWAITING TELEMETRY";
  elements.pills.replaceChildren();
  elements.baseline.textContent = "";
  elements.ice.textContent = "";
  elements.baselineCard.classList.remove("has-error");
  elements.baselineStatus.textContent = "Awaiting baseline output";
  elements.baselineLines.textContent = "READY";
  elements.iceStatus.textContent = "Ready";
  elements.iceValidationTag.textContent = "SCHEMA CHECK PENDING";
  elements.grammar.textContent = "READY";
  elements.streamIndicator.lastChild.textContent = " AWAITING RUN";
}

async function readErrorMessage(response) {
  const body = await response.text();
  if (!body) return "Backend returned HTTP " + response.status;
  try {
    const parsed = JSON.parse(body);
    return parsed.detail || parsed.error || body;
  } catch (_) {
    return body;
  }
}

async function runDemonstration() {
  stopDemo();
  resetRunDisplay();

  const scenario = scenarios[elements.scenario.value] || scenarios.drone;
  const controller = new AbortController();
  activeRequest = controller;
  const startedAt = performance.now();
  let buffer = "";
  let tokenCount = 0;
  let terminalReceived = false;

  elements.run.disabled = true;
  elements.run.innerHTML = '<span aria-hidden="true">◌</span> Running inference…';
  elements.baselineLines.textContent = "CONNECTING";
  elements.baselineStatus.textContent = "Waiting for baseline tokens…";
  elements.iceStatus.textContent = "Connecting to inference engine…";
  elements.grammar.textContent = "INITIALIZING";
  elements.connection.textContent = "CONNECTING TO ICE API";
  elements.connectionDetail.textContent = API_URL;
  elements.footer.textContent = "LIVE INFERENCE";
  elements.streamIndicator.lastChild.textContent = " STREAMING";

  function handleEvent(rawEvent) {
    const data = rawEvent
      .split(/\r?\n/)
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trim())
      .join("\n");
    if (!data) return;

    let event;
    try {
      event = JSON.parse(data);
    } catch (_) {
      throw new Error("The backend sent an unreadable stream event.");
    }
    if (event.error) throw new Error(event.error);

    if (event.runner_type === "ice") {
      elements.ice.textContent += event.token_text || "";
      elements.ice.scrollTop = elements.ice.scrollHeight;
      elements.grammar.textContent = event.active_grammar_state || event.status || "GENERATING";
      elements.iceStatus.textContent = event.status || "Generating";

      const telemetry = event.telemetry || {};
      const size = Number(telemetry.total_vocab);
      const valid = Number(telemetry.valid_count);
      if (size > 0 && Number.isFinite(valid)) {
        tokenCount += 1;
        updateVocabularyCapacity(size);
        const allowed = Math.max(0, Math.min(size, valid));
        const masked = Math.max(0, size - allowed);
        const ratio = Number.isFinite(Number(telemetry.masking_ratio))
          ? Number(telemetry.masking_ratio)
          : masked / size;

        elements.allowed.textContent = formatNumber(allowed);
        elements.masked.textContent = formatNumber(masked);
        elements.allowedBar.style.width = (allowed / size * 100).toFixed(2) + "%";
        elements.maskedBar.style.width = (masked / size * 100).toFixed(2) + "%";
        elements.ratio.textContent = (ratio * 100).toFixed(2) + "% VOCABULARY MASKED";
        elements.pruned.textContent = (ratio * 100).toFixed(2) + "%";
        elements.prunedTrack.style.width = (ratio * 100).toFixed(2) + "%";

        const latency = Number(telemetry.ms_per_token);
        if (Number.isFinite(latency)) {
          elements.latency.textContent = latency.toFixed(1);
          elements.latencyTrack.style.width = Math.max(4, Math.min(100, latency / 8)) + "%";
        }
        elements.tokens.textContent = formatNumber(tokenCount);
        elements.tokensTrack.style.width = (Math.min(tokenCount / 128, 1) * 100).toFixed(1) + "%";
        renderTokenPills(event.sample_allowed_tokens);
      }
    } else if (event.runner_type === "baseline") {
      elements.baseline.textContent += event.token_text || "";
      elements.baseline.scrollTop = elements.baseline.scrollHeight;
      elements.baselineLines.textContent = "STEP " + (Number(event.step) || 0);
      elements.baselineStatus.textContent = event.status || "Generating";
      elements.baselineCard.classList.toggle("has-error", Boolean(event.has_error));
    } else if (event.runner_type === "terminal") {
      terminalReceived = true;
      const ice = event.ice || {};
      const baseline = event.baseline || {};
      elements.ice.textContent = ice.text !== undefined ? ice.text : elements.ice.textContent;
      elements.baseline.textContent = baseline.text !== undefined ? baseline.text : elements.baseline.textContent;
      elements.iceStatus.textContent = formatReportStatus(ice, "ICE");
      elements.baselineStatus.textContent = formatReportStatus(baseline, "Baseline");
      elements.grammar.textContent = ice.schema_valid === true
        ? "VALIDATED"
        : ice.status || "FINISHED";
      elements.baselineLines.textContent = "FINAL REPORT";
      elements.baselineCard.classList.toggle(
        "has-error",
        baseline.json_valid === false || baseline.schema_valid === false
      );
      setValidationResult(ice);
      elements.connection.textContent = "INFERENCE COMPLETE";
      elements.connectionDetail.textContent = "Finished in " + ((performance.now() - startedAt) / 1000).toFixed(2) + " seconds";
      elements.footer.textContent = "COMPLETED";
      elements.streamIndicator.lastChild.textContent = " COMPLETE";
    }
  }

  try {
    const url = new URL(API_URL + "/stream");
    url.searchParams.set("prompt", scenario.prompt);
    url.searchParams.set("schema", JSON.stringify(scenario.schema));
    url.searchParams.set("max_steps", "128");

    const response = await fetch(url, {
      signal: controller.signal,
      headers: { Accept: "text/event-stream" }
    });
    if (!response.ok) throw new Error(await readErrorMessage(response));
    if (!response.body) throw new Error("This browser does not support streaming responses.");

    elements.connection.textContent = "CONNECTED TO ICE API";
    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const result = await reader.read();
      buffer += decoder.decode(result.value || new Uint8Array(), { stream: !result.done });
      buffer = buffer.replace(/\r\n/g, "\n");
      const events = buffer.split("\n\n");
      buffer = events.pop() || "";
      events.forEach(handleEvent);
      if (result.done) break;
    }

    if (buffer.trim()) handleEvent(buffer);
    if (!terminalReceived) {
      elements.connection.textContent = "STREAM ENDED EARLY";
      elements.connectionDetail.textContent = "The backend closed the stream without a final validation report.";
      elements.footer.textContent = "INCOMPLETE RUN";
      elements.streamIndicator.lastChild.textContent = " STREAM ENDED";
    }
  } catch (error) {
    controller.abort();
    if (error.name !== "AbortError") {
      elements.iceStatus.textContent = "Inference request failed";
      elements.connection.textContent = "API CONNECTION ERROR";
      elements.connectionDetail.textContent = error.message;
      elements.footer.textContent = "REQUEST FAILED";
      elements.ice.textContent += "\n\n" + error.message + "\nAPI: " + API_URL;
      elements.streamIndicator.lastChild.textContent = " REQUEST FAILED";
      console.error("ICE API error:", error);
    }
  } finally {
    if (activeRequest === controller) {
      activeRequest = null;
      elements.run.disabled = false;
      elements.run.innerHTML = '<span aria-hidden="true">▶</span> Run inference';
    }
  }
}

function resetDashboard() {
  stopDemo();
  resetRunDisplay();
  if (backendState === "OFFLINE") backendDetail = "Check that the backend is running at " + API_URL;
  setConnection(backendState, backendDetail);
}

function initTrie() {
  if (!window.THREE || !elements.canvas) {
    elements.fallback.textContent = "3D visualization unavailable. The dashboard still works without it.";
    return;
  }

  const THREE = window.THREE;
  const width = Math.max(280, elements.canvas.clientWidth);
  const height = Math.max(200, elements.canvas.clientHeight);
  threeScene = new THREE.Scene();
  threeCamera = new THREE.PerspectiveCamera(42, width / height, 0.1, 100);
  threeCamera.position.set(0, 0.3, 8);

  try {
    threeRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  } catch (error) {
    elements.fallback.textContent = "3D visualization is unavailable in this browser.";
    console.warn("ICE trie visualization unavailable:", error);
    return;
  }

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
    [0, 0, 0], [0, 1.05, 0], [-1.25, 1.95, 0], [0, 2.05, 0],
    [1.25, 1.95, 0], [-1.8, 2.85, 0], [-0.9, 2.85, 0],
    [-0.35, 3.05, 0], [0.45, 2.85, 0], [1.1, 2.85, 0], [1.8, 2.85, 0]
  ];
  const links = [[0, 1], [1, 2], [1, 3], [1, 4], [2, 5], [2, 6], [3, 7], [3, 8], [4, 9], [4, 10]];
  const nodeGeometry = new THREE.SphereGeometry(0.085, 12, 12);
  const rootGeometry = new THREE.SphereGeometry(0.15, 16, 16);
  const nodeMaterials = [
    new THREE.MeshStandardMaterial({ color: 0x60a5fa, emissive: 0x1d4ed8, emissiveIntensity: 0.55 }),
    new THREE.MeshStandardMaterial({ color: 0x10b981, emissive: 0x065f46, emissiveIntensity: 0.45 }),
    new THREE.MeshStandardMaterial({ color: 0x64748b, emissive: 0x1e293b, emissiveIntensity: 0.3 })
  ];
  const edgeMaterial = new THREE.LineBasicMaterial({ color: 0x33566a, transparent: true, opacity: 0.85 });

  links.forEach(([start, end]) => {
    const geometry = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(...points[start]),
      new THREE.Vector3(...points[end])
    ]);
    threeRoot.add(new THREE.Line(geometry, edgeMaterial));
  });
  points.forEach((point, index) => {
    const material = nodeMaterials[index === 0 ? 0 : index <= 4 ? 1 : 2];
    const mesh = new THREE.Mesh(index === 0 ? rootGeometry : nodeGeometry, material);
    mesh.position.set(...point);
    threeRoot.add(mesh);
  });
  threeRoot.position.y = -1.3;

  let dragging = false;
  let lastX = 0;
  let lastY = 0;
  const canvas = threeRenderer.domElement;
  canvas.addEventListener("pointerdown", (event) => {
    dragging = true;
    lastX = event.clientX;
    lastY = event.clientY;
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    threeRoot.rotation.y += (event.clientX - lastX) * 0.008;
    threeRoot.rotation.x += (event.clientY - lastY) * 0.005;
    lastX = event.clientX;
    lastY = event.clientY;
  });
  const endDrag = () => { dragging = false; };
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);

  const resizeCanvas = () => {
    if (!threeRenderer || !threeCamera) return;
    const newWidth = Math.max(280, elements.canvas.clientWidth);
    const newHeight = Math.max(200, elements.canvas.clientHeight);
    threeCamera.aspect = newWidth / newHeight;
    threeCamera.updateProjectionMatrix();
    threeRenderer.setSize(newWidth, newHeight);
  };
  if ("ResizeObserver" in window) {
    resizeObserver = new ResizeObserver(resizeCanvas);
    resizeObserver.observe(elements.canvas);
  } else {
    window.addEventListener("resize", resizeCanvas);
  }

  const prefersReducedMotion = Boolean(
    window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
  function animate() {
    window.requestAnimationFrame(animate);
    if (!prefersReducedMotion && !dragging && threeRoot) threeRoot.rotation.y += 0.0015;
    threeRenderer.render(threeScene, threeCamera);
  }
  animate();
}

elements.run.addEventListener("click", runDemonstration);
elements.reset.addEventListener("click", resetDashboard);

if (window.gsap && !(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches)) {
  window.gsap.from(".metric-card", {
    opacity: 0,
    y: 12,
    duration: 0.5,
    stagger: 0.08,
    ease: "power2.out"
  });
}

initTrie();

async function refreshBackendStatus() {
  try {
    const response = await fetch(API_URL + "/status");
    if (!response.ok) throw new Error("HTTP " + response.status);
    const status = await response.json();

    updateVocabularyCapacity(status.vocabulary_capacity);
    elements.headerDevice.textContent = status.device && status.device !== "unknown"
      ? String(status.device).toUpperCase()
      : "LOCAL MODEL";
    elements.device.textContent = status.device && status.device !== "unknown"
      ? String(status.device).toUpperCase()
      : "—";
    elements.engineState.lastChild.textContent = " " + (status.engine ? "CONNECTED" : "READY");

    if (status.status === "ONLINE") {
      setConnection("ONLINE", "Ready for an inference request");
      elements.engineState.lastChild.textContent = " READY";
      window.setTimeout(refreshBackendStatus, 15000);
    } else if (status.status === "ERROR") {
      setConnection("ERROR", status.error || "Model initialization failed");
      elements.engineState.lastChild.textContent = " ERROR";
    } else {
      setConnection("STARTING", "Waiting for the local model to finish loading");
      window.setTimeout(refreshBackendStatus, 2500);
    }
  } catch (error) {
    setConnection("OFFLINE", "Could not reach " + API_URL + "/status");
    window.setTimeout(refreshBackendStatus, 5000);
  }
}

refreshBackendStatus();

