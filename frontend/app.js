
(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);

  // ============================================================
  // EXAMPLE SCENARIOS
  // Keep these compatible with the current backend grammar.
  // Supported schema types: boolean, null, number, object, string.
  // ============================================================

  const examples = {
    order: {
      prompt:
        'Extract the order details from this message: "Order A-1042 is for 2 blue notebooks. Ship to Bengaluru. Mark it as paid." Return only the requested structured data.',
      schema: {
        type: 'object',
        properties: {
          order_id: { type: 'string' },
          item: { type: 'string' },
          quantity: { type: 'number' },
          destination: { type: 'string' },
          paid: { type: 'boolean' }
        },
        required: [
          'order_id',
          'item',
          'quantity',
          'destination',
          'paid'
        ],
        additionalProperties: false
      }
    },

    support: {
      prompt:
        'Classify this customer message: "My headphones arrived yesterday, but the left side does not work. Please replace them." Return a structured support ticket.',
      schema: {
        type: 'object',
        properties: {
          category: { type: 'string' },
          product: { type: 'string' },
          issue: { type: 'string' },
          priority: { type: 'string' }
        },
        required: [
          'category',
          'product',
          'issue',
          'priority'
        ],
        additionalProperties: false
      }
    },

    sensor: {
      prompt:
        'Convert this sensor report into structured data: "Unit S-07 recorded 23.6 degrees Celsius at 14:05. Status normal." Return only the requested data.',
      schema: {
        type: 'object',
        properties: {
          unit_id: { type: 'string' },
          temperature_c: { type: 'number' },
          time: { type: 'string' },
          status: { type: 'string' }
        },
        required: [
          'unit_id',
          'temperature_c',
          'time',
          'status'
        ],
        additionalProperties: false
      }
    },

    custom: {
      prompt:
        'Describe the data you want extracted. Return only a JSON object matching the supplied schema.',
      schema: {
        type: 'object',
        properties: {
          result: { type: 'string' }
        },
        required: ['result'],
        additionalProperties: false
      }
    }
  };

  const SUPPORTED_TYPES = new Set([
    'boolean',
    'null',
    'number',
    'object',
    'string'
  ]);

  let controller = null;
  let activeReader = null;
  let elapsedTimer = null;
  let statusTimer = null;
  let startedAt = 0;
  let latestStep = 0;
  let terminalSeen = false;
  let runId = 0;
  let previousEngineState = null;

  const output = {
    ice: '',
    baseline: ''
  };

  // ============================================================
  // API URL
  // FastAPI serves the frontend and API on the same origin.
  // Set window.ICE_API_URL before app.js only if needed.
  // ============================================================

  function apiBase() {
    const configured = String(window.ICE_API_URL || '').trim();

    if (configured) {
      return configured.replace(/\/+$/, '');
    }

    if (window.location.protocol === 'file:') {
      return 'http://127.0.0.1:8000';
    }

    return window.location.origin;
  }

  function pretty(value) {
    const number = Number(value);

    if (!Number.isFinite(number)) {
      return '—';
    }

    return number >= 1000
      ? Math.round(number).toLocaleString()
      : String(Math.round(number * 100) / 100);
  }

  function setText(id, value) {
    const element = $(id);

    if (element) {
      element.textContent = value == null ? '' : String(value);
    }
  }

  function showToast(message) {
    const toast = $('toast');

    if (!toast) return;

    toast.textContent = message;
    toast.classList.add('show');

    clearTimeout(showToast.timer);

    showToast.timer = setTimeout(() => {
      toast.classList.remove('show');
    }, 3200);
  }

  // ============================================================
  // BACKEND STATUS
  // ============================================================

  function setStatus(state, message) {
    ['statusDot', 'footerStatusDot'].forEach((id) => {
      const dot = $(id);

      if (dot) {
        dot.className = `status-dot ${state}`;
      }
    });

    setText('statusLabel', message);

    setText(
      'footerStatusLabel',
      state === 'online'
        ? 'ONLINE'
        : state === 'offline'
          ? 'OFFLINE'
          : 'CHECKING'
    );
  }

  async function checkStatus({ quiet = false } = {}) {
    if (!quiet) {
      setStatus('checking', 'CHECKING ENGINE');
    }

    try {
      const response = await fetch(`${apiBase()}/status`, {
        method: 'GET',
        cache: 'no-store',
        headers: {
          Accept: 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error(
          `Status endpoint returned HTTP ${response.status}`
        );
      }

      const data = await response.json();
      const isOnline = data.status === 'ONLINE';

      if (isOnline) {
        const device = data.device || 'device unknown';

        setStatus('online', `ENGINE ONLINE · ${device}`);

        if (!quiet && previousEngineState !== 'online') {
          showToast(
            `ICE engine online · vocabulary ${pretty(
              data.vocabulary_capacity
            )}`
          );
        }

        previousEngineState = 'online';
      } else {
        setStatus(
          'offline',
          `ENGINE ${data.status || 'UNAVAILABLE'}`
        );

        if (!quiet && previousEngineState !== 'offline') {
          showToast(
            `Backend responded, but engine status is ${
              data.status || 'unavailable'
            }.`
          );
        }

        previousEngineState = 'offline';
      }

      return data;
    } catch (error) {
      setStatus('offline', 'ENGINE UNREACHABLE');

      if (!quiet && previousEngineState !== 'offline') {
        showToast(
          `Cannot reach ${apiBase()}/status. Check the backend URL.`
        );
      }

      previousEngineState = 'offline';

      console.warn('ICE status check failed:', error);

      return null;
    }
  }

  // ============================================================
  // RUNNER AND TRACE STATUS
  // ============================================================

  function setTrace(label, state = '') {
    const badge = $('traceBadge');

    if (!badge) return;

    badge.textContent = label;
    badge.className = `trace-badge${state ? ` ${state}` : ''}`;
  }

  function setRunnerStatus(which, status) {
    const element = $(
      which === 'ice' ? 'iceStatus' : 'baseStatus'
    );

    if (!element) return;

    element.textContent = status || 'RUNNING';
    element.className = 'stream-state';

    if (status === 'RUNNING') {
      element.classList.add('running');
    } else if (status === 'COMPLETED') {
      element.classList.add('completed');
    } else if (
      status &&
      !['IDLE', 'UNKNOWN'].includes(status)
    ) {
      element.classList.add('error');
    }
  }

  // ============================================================
  // OUTPUT RENDERING
  // Treat generated model output as text, never as HTML.
  // ============================================================

  function appendToken(which, text) {
    if (typeof text !== 'string' || !text) return;

    const id = which === 'ice' ? 'iceOutput' : 'baseOutput';
    const box = $(id);

    if (!box) return;

    const placeholder = box.querySelector('.placeholder-line');

    if (placeholder) {
      placeholder.remove();
    }

    output[which] += text;

    const span = document.createElement('span');

    span.className = 'new-token';
    span.textContent = text;

    box.appendChild(span);
    box.scrollTop = box.scrollHeight;
  }

  function setPlaceholder(id, message) {
    const box = $(id);

    if (!box) return;

    const placeholder = document.createElement('span');

    placeholder.className = 'placeholder-line';
    placeholder.textContent = message;

    box.replaceChildren(placeholder);
  }

  function showError(message) {
    const banner = $('errorBanner');

    if (!banner) {
      console.error(message);
      return;
    }

    banner.textContent = String(
      message || 'An unknown error occurred.'
    );

    banner.hidden = false;
  }

  function clearError() {
    const banner = $('errorBanner');

    if (!banner) return;

    banner.hidden = true;
    banner.textContent = '';
  }

  // ============================================================
  // RESET WORKBENCH
  // ============================================================

  function resetWorkbench() {
    output.ice = '';
    output.baseline = '';

    latestStep = 0;
    terminalSeen = false;

    setPlaceholder(
      'iceOutput',
      'Waiting for the first token…'
    );

    setPlaceholder(
      'baseOutput',
      'Waiting for the first token…'
    );

    [
      'vocabValue',
      'allowedValue',
      'maskedValue',
      'ratioValue',
      'latencyValue'
    ].forEach((id) => setText(id, '—'));

    const maskBar = $('maskBarFill');

    if (maskBar) {
      maskBar.style.width = '0%';
    }

    const allowed = $('allowedTokens');

    if (allowed) {
      const empty = document.createElement('span');

      empty.className = 'token-empty';
      empty.textContent =
        'Waiting for compatible token sample…';

      allowed.replaceChildren(empty);
    }

    setText('grammarState', '—');
    setText('baseJsonState', 'NOT CHECKED');

    if ($('baseJsonState')) {
      $('baseJsonState').style.color = '';
    }

    setText('stepCounter', '0 steps');
    setText('elapsedLabel', 'Preparing request…');

    setTrace('PREPARING', 'running');

    clearError();

    if ($('resultsSection')) {
      $('resultsSection').hidden = true;
    }

    setRunnerStatus('ice', 'RUNNING');
    setRunnerStatus('baseline', 'RUNNING');

    setText('iceResultStatus', 'UNKNOWN');
    setText('baseResultStatus', 'UNKNOWN');

    ['iceChecks', 'baseChecks'].forEach((id) => {
      if ($(id)) {
        $(id).replaceChildren();
      }
    });

    ['iceFinalText', 'baseFinalText'].forEach((id) => {
      setText(id, '(No final output)');
    });

    ['iceResultError', 'baseResultError'].forEach((id) => {
      setText(id, '');
    });
  }

  // ============================================================
  // BASIC CLIENT-SIDE SCHEMA COMPATIBILITY CHECK
  //
  // This catches obvious unsupported schema features before
  // sending a request. The backend remains the authority.
  // ============================================================

  function validateSupportedSchema(schema) {
    const errors = [];

    function visit(node, path) {
      if (!node || typeof node !== 'object' || Array.isArray(node)) {
        errors.push(`${path}: expected a schema object.`);
        return;
      }

      if (node.type != null) {
        if (typeof node.type !== 'string') {
          errors.push(`${path}.type: expected a string type.`);
        } else if (!SUPPORTED_TYPES.has(node.type)) {
          errors.push(
            `${path}.type: unsupported type "${node.type}". ` +
            `Supported types: ${Array.from(SUPPORTED_TYPES).join(', ')}.`
          );
        }
      }

      // The current backend's advertised contract does not
      // establish support for JSON Schema enum constraints.
      // Warn rather than silently claiming they are enforced.
      if (node.enum != null) {
        errors.push(
          `${path}.enum: enum constraints are not enabled by this frontend's ` +
          'known backend contract. Remove enum constraints unless you have ' +
          'verified that your backend grammar enforces them.'
        );
      }

      if (node.properties != null) {
        if (
          typeof node.properties !== 'object' ||
          Array.isArray(node.properties)
        ) {
          errors.push(
            `${path}.properties: expected an object of property schemas.`
          );
        } else {
          for (const [key, child] of Object.entries(node.properties)) {
            visit(child, `${path}.properties.${key}`);
          }
        }
      }

      if (node.items != null) {
        visit(node.items, `${path}.items`);
      }

      if (node.anyOf != null || node.oneOf != null || node.allOf != null) {
        errors.push(
          `${path}: anyOf, oneOf and allOf have not been confirmed as supported.`
        );
      }
    }

    visit(schema, '$');

    return errors;
  }

  // ============================================================
  // LIVE TELEMETRY
  // ============================================================

  function renderTelemetry(data) {
    if (!data || typeof data !== 'object') return;

    if (data.total_vocab != null) {
      setText('vocabValue', pretty(data.total_vocab));
    }

    if (data.valid_count != null) {
      setText('allowedValue', pretty(data.valid_count));
    }

    if (data.masked_count != null) {
      setText('maskedValue', pretty(data.masked_count));
    }

    if (
      data.masking_ratio != null &&
      Number.isFinite(Number(data.masking_ratio))
    ) {
      const ratio = Math.max(
        0,
        Math.min(1, Number(data.masking_ratio))
      );

      setText('ratioValue', `${(ratio * 100).toFixed(2)}%`);

      const bar = $('maskBarFill');

      if (bar) {
        bar.style.width = `${ratio * 100}%`;
      }
    }

    if (
      data.ms_per_token != null &&
      Number.isFinite(Number(data.ms_per_token))
    ) {
      setText(
        'latencyValue',
        `${pretty(data.ms_per_token)} ms`
      );
    }
  }

  function renderAllowedTokens(tokens) {
    const box = $('allowedTokens');

    if (!box) return;

    box.replaceChildren();

    if (!Array.isArray(tokens) || tokens.length === 0) {
      const empty = document.createElement('span');

      empty.className = 'token-empty';
      empty.textContent =
        'No token sample supplied for this step.';

      box.appendChild(empty);
      return;
    }

    tokens.forEach((token) => {
      const element = document.createElement('span');

      element.className = 'allowed-token';

      if (typeof token === 'string') {
        element.textContent = token;
      } else {
        try {
          element.textContent = JSON.stringify(token);
        } catch {
          element.textContent = String(token);
        }
      }

      box.appendChild(element);
    });
  }

  // ============================================================
  // FINAL VALIDATION REPORT
  // ============================================================

  function addCheck(container, label, value) {
    const chip = document.createElement('span');

    chip.className = 'check-chip';

    if (value === true) {
      chip.classList.add('pass');
      chip.textContent = `${label}: YES`;
    } else if (value === false) {
      chip.classList.add('fail');
      chip.textContent = `${label}: NO`;
    } else {
      chip.textContent = `${label}: N/A`;
    }

    container.appendChild(chip);
  }

  function renderFinal(which, report) {
    const prefix = which === 'ice' ? 'ice' : 'base';

    setRunnerStatus(which, report.status || 'UNKNOWN');

    const status = $(`${prefix}ResultStatus`);

    if (status) {
      status.textContent = report.status || 'UNKNOWN';

      status.className = `result-status ${
        report.generation_completed === true
          ? 'pass'
          : report.generation_completed === false
            ? 'fail'
            : ''
      }`;
    }

    const checks = $(`${prefix}Checks`);

    if (checks) {
      checks.replaceChildren();

      addCheck(
        checks,
        'Completed',
        report.generation_completed
      );

      addCheck(checks, 'JSON valid', report.json_valid);
      addCheck(checks, 'Schema valid', report.schema_valid);
    }

    setText(
      `${prefix}FinalText`,
      report.text || '(empty output)'
    );

    setText(
      `${prefix}ResultError`,
      report.error || ''
    );

    if (which === 'baseline') {
      setText(
        'baseJsonState',
        report.json_valid === true
          ? 'JSON VALID'
          : report.json_valid === false
            ? 'JSON INVALID'
            : 'NOT CHECKED'
      );

      if ($('baseJsonState')) {
        $('baseJsonState').style.color =
          report.json_valid === true
            ? 'var(--mint)'
            : report.json_valid === false
              ? 'var(--red)'
              : '';
      }
    }
  }

  // ============================================================
  // SERVER-SENT EVENT HANDLING
  // ============================================================

  function handleEvent(event) {
    if (!event || typeof event !== 'object') return;

    if (event.error) {
      showError(event.error);
      setTrace('STREAM ERROR', 'error');
      return;
    }

    if (event.runner_type === 'ice') {
      latestStep = Math.max(
        latestStep,
        Number(event.step) || 0
      );

      appendToken('ice', event.token_text || '');

      if (event.active_grammar_state != null) {
        setText(
          'grammarState',
          event.active_grammar_state
        );
      }

      renderTelemetry(event.telemetry);
      renderAllowedTokens(event.sample_allowed_tokens);

      if (event.status) {
        setRunnerStatus('ice', event.status);
      }

      if (event.telemetry?.error) {
        showError(
          `ICE step error: ${event.telemetry.error}`
        );
      }

      setTrace('STREAMING', 'running');
    } else if (event.runner_type === 'baseline') {
      latestStep = Math.max(
        latestStep,
        Number(event.step) || 0
      );

      appendToken('baseline', event.token_text || '');

      if (event.status) {
        setRunnerStatus('baseline', event.status);
      }

      setText(
        'baseJsonState',
        event.has_error
          ? 'PARSE ERROR REPORTED'
          : 'NO PARSE ERROR REPORTED'
      );

      if ($('baseJsonState')) {
        $('baseJsonState').style.color = event.has_error
          ? 'var(--amber)'
          : 'var(--muted)';
      }

      setTrace('STREAMING', 'running');
    } else if (event.runner_type === 'terminal') {
      terminalSeen = true;

      renderFinal('ice', event.ice || {});
      renderFinal('baseline', event.baseline || {});

      setTrace('FINISHED', 'completed');

      setText(
        'elapsedLabel',
        `${(
          (performance.now() - startedAt) /
          1000
        ).toFixed(2)} s total`
      );

      const results = $('resultsSection');

      if (results) {
        results.hidden = false;

        results.scrollIntoView({
          behavior: 'smooth',
          block: 'nearest'
        });
      }
    }

    setText(
      'stepCounter',
      `${latestStep} ${latestStep === 1 ? 'step' : 'steps'}`
    );
  }

  function parseSSEFrame(frame) {
    const dataLines = [];
    let eventName = '';

    for (const line of frame.split(/\r?\n/)) {
      if (!line || line.startsWith(':')) continue;

      const colon = line.indexOf(':');
      const field = colon === -1
        ? line
        : line.slice(0, colon);

      let value = colon === -1
        ? ''
        : line.slice(colon + 1);

      if (value.startsWith(' ')) {
        value = value.slice(1);
      }

      if (field === 'data') {
        dataLines.push(value);
      } else if (field === 'event') {
        eventName = value;
      }
    }

    if (!dataLines.length) return;

    const raw = dataLines.join('\n');

    try {
      const payload = JSON.parse(raw);

      if (!payload.runner_type && eventName) {
        if (
          ['ice', 'baseline', 'terminal'].includes(eventName)
        ) {
          payload.runner_type = eventName;
        }
      }

      handleEvent(payload);
    } catch (error) {
      console.warn(
        'Could not parse SSE data:',
        raw,
        error
      );

      showError(
        'Received a stream frame that was not valid JSON. ' +
        'Check the backend SSE format.'
      );
    }
  }

  // ============================================================
  // RUN COMPARISON
  // ============================================================

  async function runComparison() {
    if (controller) return;

    const promptElement = $('prompt');
    const schemaElement = $('schema');
    const maxStepsElement = $('maxSteps');

    if (!promptElement || !schemaElement || !maxStepsElement) {
      showToast(
        'The workbench form is incomplete. Check the HTML element IDs.'
      );
      return;
    }

    let schema;

    try {
      schema = JSON.parse(schemaElement.value);

      if (
        !schema ||
        typeof schema !== 'object' ||
        Array.isArray(schema)
      ) {
        throw new Error('Schema must be a JSON object.');
      }
    } catch (error) {
      showToast(`Schema JSON error: ${error.message}`);
      schemaElement.focus();
      return;
    }

    const schemaErrors = validateSupportedSchema(schema);

    if (schemaErrors.length > 0) {
      showError(
        'SCHEMA COMPATIBILITY CHECK\n\n' +
        schemaErrors.map((error) => `• ${error}`).join('\n')
      );

      setTrace('SCHEMA CHECK FAILED', 'error');

      showToast(
        'The schema contains unsupported or unverified features. See the error message.'
      );

      schemaElement.focus();
      return;
    }

    const prompt = promptElement.value.trim();

    if (!prompt) {
      showToast('Enter a prompt before running.');
      promptElement.focus();
      return;
    }

    const maxSteps = Math.trunc(
      Number(maxStepsElement.value)
    );

    if (
      !Number.isInteger(maxSteps) ||
      maxSteps < 1 ||
      maxSteps > 512
    ) {
      showToast(
        'Token-step limit must be between 1 and 512.'
      );

      maxStepsElement.focus();
      return;
    }

    resetWorkbench();

    const thisRun = ++runId;
    const thisController = new AbortController();

    controller = thisController;
    activeReader = null;
    startedAt = performance.now();

    const runButton = $('runBtn');
    const stopButton = $('stopBtn');

    if (runButton) runButton.disabled = true;
    if (stopButton) stopButton.disabled = false;

    elapsedTimer = setInterval(() => {
      if (controller === thisController && !terminalSeen) {
        setText(
          'elapsedLabel',
          `${(
            (performance.now() - startedAt) /
            1000
          ).toFixed(1)} s elapsed`
        );
      }
    }, 250);

    try {
      const params = new URLSearchParams({
        prompt,
        schema: JSON.stringify(schema),
        max_steps: String(maxSteps)
      });

      const url = `${apiBase()}/stream?${params.toString()}`;

      setText('elapsedLabel', 'Opening stream…');
      setTrace('CONNECTING', 'running');

      const response = await fetch(url, {
        method: 'GET',
        headers: {
          Accept: 'text/event-stream'
        },
        signal: thisController.signal,
        cache: 'no-store'
      });

      // Distinguish a server-side rejection from a network failure.
      if (!response.ok) {
        let detail = `The stream endpoint returned HTTP ${response.status}.`;

        try {
          const body = await response.json();

          if (body.detail != null) {
            detail = Array.isArray(body.detail)
              ? body.detail
                  .map((item) =>
                    typeof item === 'string'
                      ? item
                      : item.msg || JSON.stringify(item)
                  )
                  .join('\n')
              : typeof body.detail === 'string'
                ? body.detail
                : JSON.stringify(body.detail);
          }
        } catch {
          // The HTTP status remains useful if the body is not JSON.
        }

        const error = new Error(detail);

        error.httpStatus = response.status;

        throw error;
      }

      const contentType =
        response.headers.get('content-type') || '';

      if (
        !contentType.toLowerCase().includes('text/event-stream')
      ) {
        console.warn(
          'Unexpected stream content type:',
          contentType
        );
      }

      if (!response.body) {
        throw new Error(
          'The browser did not provide a readable response stream.'
        );
      }

      setTrace('STREAM OPEN', 'running');

      const reader = response.body.getReader();
      activeReader = reader;

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();

        if (thisRun !== runId) break;
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // A network chunk may contain part of a frame or several frames.
        while (true) {
          const boundaryMatch = /\r?\n\r?\n/.exec(buffer);

          if (!boundaryMatch) break;

          const boundary = boundaryMatch.index;
          const frame = buffer.slice(0, boundary);

          buffer = buffer.slice(
            boundary + boundaryMatch[0].length
          );

          parseSSEFrame(frame);
        }
      }

      if (thisRun === runId) {
        buffer += decoder.decode();

        if (buffer.trim()) {
          parseSSEFrame(buffer);
        }

        if (!terminalSeen) {
          setTrace('NO FINAL REPORT', 'error');

          showError(
            'The stream ended without a terminal validation event. ' +
            'Generation may have stopped early. Inspect the backend logs.'
          );
        }
      }
    } catch (error) {
      if (error.name === 'AbortError') {
        // User cancellation is handled by stopComparison().
      } else if (thisRun === runId) {
        console.error(
          'ICE comparison request failed:',
          error
        );

        if (Number.isInteger(error.httpStatus)) {
          // The server responded: this is an API/request error,
          // not a connection failure.
          setTrace(
            `HTTP ${error.httpStatus}`,
            'error'
          );

          const heading =
            error.httpStatus === 422
              ? 'REQUEST REJECTED (HTTP 422)'
              : `BACKEND ERROR (HTTP ${error.httpStatus})`;

          showError(
            `${heading}\n\n${error.message}\n\n` +
            `Requested URL: ${apiBase()}/stream\n\n` +
            'The browser reached the backend, but the backend rejected ' +
            'the request. Check the schema and the backend error details.'
          );

          setText(
            'elapsedLabel',
            `Request rejected · HTTP ${error.httpStatus}`
          );
        } else {
          setTrace('CONNECTION ERROR', 'error');

          showError(
            `${error.message || 'Could not connect to the ICE backend.'}\n\n` +
            `Requested URL: ${apiBase()}/stream\n\n` +
            'No HTTP error response was received. Check that Uvicorn is ' +
            'running and that the frontend uses the correct origin.'
          );

          setText('elapsedLabel', 'Unable to open stream');
        }
      }
    } finally {
      if (thisRun === runId) {
        controller = null;
        activeReader = null;

        if (runButton) runButton.disabled = false;
        if (stopButton) stopButton.disabled = true;

        clearInterval(elapsedTimer);
        elapsedTimer = null;

        if (!terminalSeen) {
          if (
            $('iceStatus') &&
            $('iceStatus').textContent === 'RUNNING'
          ) {
            setRunnerStatus('ice', 'INTERRUPTED');
          }

          if (
            $('baseStatus') &&
            $('baseStatus').textContent === 'RUNNING'
          ) {
            setRunnerStatus('baseline', 'INTERRUPTED');
          }
        }

        checkStatus({ quiet: true });
      }
    }
  }

  // ============================================================
  // STOP GENERATION
  // ============================================================

  async function stopComparison() {
    if (!controller) return;

    const oldController = controller;
    const oldReader = activeReader;

    runId++;
    controller = null;
    activeReader = null;

    oldController.abort();

    if (oldReader) {
      try {
        await oldReader.cancel();
      } catch {
        // The abort may already have closed the stream.
      }
    }

    clearInterval(elapsedTimer);
    elapsedTimer = null;

    if ($('runBtn')) $('runBtn').disabled = false;
    if ($('stopBtn')) $('stopBtn').disabled = true;

    setText('elapsedLabel', 'Stream cancelled by user');

    if (!terminalSeen) {
      setTrace('STOPPED');

      setRunnerStatus('ice', 'INTERRUPTED');
      setRunnerStatus('baseline', 'INTERRUPTED');
    }

    showToast('Stream cancelled.');
  }

  // ============================================================
  // EXAMPLE PRESETS
  // ============================================================

  function applyExample(name) {
    const example = examples[name] || examples.order;

    const promptElement = $('prompt');
    const schemaElement = $('schema');

    if (promptElement) {
      promptElement.value = example.prompt;
    }

    if (schemaElement) {
      schemaElement.value = JSON.stringify(
        example.schema,
        null,
        2
      );
    }

    clearError();
    setTrace('READY');

    setText('elapsedLabel', 'Ready to run');

    showToast(
      `${name === 'custom' ? 'Custom' : 'Example'} loaded. Review the prompt and schema.`
    );
  }

  // ============================================================
  // INITIALISATION
  // ============================================================

  function initialise() {
    const scenario = $('scenario');
    const runButton = $('runBtn');
    const stopButton = $('stopBtn');
    const refreshButton = $('refreshStatus');
    const stepDown = $('stepDown');
    const stepUp = $('stepUp');
    const maxStepsElement = $('maxSteps');

    if (scenario) {
      scenario.addEventListener('change', (event) => {
        applyExample(event.target.value);
      });
    }

    if (runButton) {
      runButton.addEventListener('click', runComparison);
    }

    if (stopButton) {
      stopButton.addEventListener('click', stopComparison);
      stopButton.disabled = true;
    }

    if (refreshButton) {
      refreshButton.addEventListener('click', () => {
        checkStatus();
      });
    }

    if (stepDown && maxStepsElement) {
      stepDown.addEventListener('click', () => {
        const current =
          Number(maxStepsElement.value) || 64;

        maxStepsElement.value = Math.max(
          1,
          current - 8
        );
      });
    }

    if (stepUp && maxStepsElement) {
      stepUp.addEventListener('click', () => {
        const current =
          Number(maxStepsElement.value) || 64;

        maxStepsElement.value = Math.min(
          512,
          current + 8
        );
      });
    }

    if (maxStepsElement) {
      maxStepsElement.addEventListener('change', () => {
        const value = Math.trunc(
          Number(maxStepsElement.value)
        );

        maxStepsElement.value = Math.max(
          1,
          Math.min(
            512,
            Number.isFinite(value) ? value : 64
          )
        );
      });
    }

    // Load the default supported scenario.
    if (scenario) {
      scenario.value = 'order';
    }

    applyExample('order');

    checkStatus();

    statusTimer = setInterval(() => {
      if (!controller) {
        checkStatus({ quiet: true });
      }
    }, 15000);
  }

  window.addEventListener('beforeunload', () => {
    clearInterval(elapsedTimer);
    clearInterval(statusTimer);

    if (controller) {
      controller.abort();
    }
  });

  if (document.readyState === 'loading') {
    document.addEventListener(
      'DOMContentLoaded',
      initialise,
      { once: true }
    );
  } else {
    initialise();
  }
})();