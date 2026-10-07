/**
 * Modern Industrial AI Safety Monitoring - Control Room App Controller
 * Ultra-lightweight Vanilla JS (No heavy libraries, zero unnecessary re-render overhead)
 */

// Application State
const state = {
  currentView: "dashboard",
  settings: null,
  isCameraRunning: false,
  cameraMode: "browser", // "browser" or "server"
  clientWebcamStream: null,
  clientWs: null,
  telemetryWs: null,
  videoTaskId: null,
  videoPollTimer: null,
  recentEvents: [],
  latestDetections: [],
  animFrameId: null,
  streamSendTimer: null
};

// ==========================================================================
// 1. Client-Side Router
// ==========================================================================
function initRouter() {
  function handleHash() {
    const hash = window.location.hash.replace("#", "") || "dashboard";
    switchView(hash);
  }

  window.addEventListener("hashchange", handleHash);
  handleHash();

  document.querySelectorAll(".nav-link").forEach(link => {
    link.addEventListener("click", e => {
      const view = link.getAttribute("data-view");
      if (view) {
        window.location.hash = `#${view}`;
      }
    });
  });
}

function switchView(viewName) {
  state.currentView = viewName;

  document.querySelectorAll(".nav-link").forEach(link => {
    if (link.getAttribute("data-view") === viewName) {
      link.classList.add("active");
    } else {
      link.classList.remove("active");
    }
  });

  document.querySelectorAll(".view-panel").forEach(panel => {
    panel.style.display = "none";
  });

  const activePanel = document.getElementById(`view-${viewName}`);
  if (activePanel) {
    activePanel.style.display = "block";
  }

  // Synchronize camera display on newly switched view
  updateCameraUI(state.isCameraRunning);

  // View specific activations
  if (viewName === "settings") {
    loadSettings();
  }
}

// ==========================================================================
// 2. Toast Notifications
// ==========================================================================
function showToast(message, type = "success") {
  const container = document.getElementById("toast-container");
  if (!container) return;

  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span>${type === 'success' ? '✓' : '!'}</span><span>${message}</span>`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transition = "opacity 0.3s";
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

// ==========================================================================
// 3. Telemetry Stream (Updates 5-10 times/sec)
// ==========================================================================
function initTelemetryWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;

  function connect() {
    state.telemetryWs = new WebSocket(wsUrl);

    state.telemetryWs.onopen = () => {
      setTopStatus(true);
    };

    state.telemetryWs.onmessage = event => {
      try {
        const data = JSON.parse(event.data);
        updateDashboardMetrics(data);
      } catch (err) {}
    };

    state.telemetryWs.onclose = () => {
      setTopStatus(false);
      setTimeout(connect, 2500);
    };

    state.telemetryWs.onerror = () => {
      setTopStatus(false);
    };
  }

  connect();
}

function setTopStatus(isOnline) {
  const el = document.getElementById("top-system-status");
  if (!el) return;
  if (isOnline) {
    el.className = "system-indicator";
    el.innerHTML = '<span class="status-dot"></span> System Online';
  } else {
    el.className = "system-indicator offline";
    el.innerHTML = '<span class="status-dot danger"></span> System Offline';
  }
}

function updateDashboardMetrics(data) {
  // 1. Metric Cards (only driven by telemetry when browser webcam is not locally streaming)
  if (data.counts && (!state.isCameraRunning || state.cameraMode !== "browser")) {
    setText("dash-count-person", data.counts.person || 0);
    setText("dash-count-forklift", data.counts.forklift || 0);
    setText("dash-count-helmet", data.counts.helmet || 0);
    setText("dash-count-vest", data.counts.safety_vest || 0);

    // Live page sidebar counts
    setText("live-cnt-person", data.counts.person || 0);
    setText("live-cnt-forklift", data.counts.forklift || 0);
    setText("live-cnt-helmet", data.counts.helmet || 0);
    setText("live-cnt-vest", data.counts.safety_vest || 0);
  }

  // 2. Safety Status Banner
  const banner = document.getElementById("dash-safety-banner");
  const icon = document.getElementById("dash-safety-icon");
  const title = document.getElementById("dash-safety-title");
  const sub = document.getElementById("dash-safety-sub");

  if (banner && data.safety_status && (!state.isCameraRunning || state.cameraMode !== "browser")) {
    if (data.safety_status === "ALL_CLEAR") {
      banner.className = "safety-status-card clear";
      if (icon) icon.innerText = "✓";
      if (title) title.innerText = "ALL CLEAR";
      if (sub) sub.innerText = "No active safety violations detected";
      setText("dash-safety-compliance", "100%");
    } else {
      banner.className = "safety-status-card warning";
      if (icon) icon.innerText = "!";
      if (title) title.innerText = "SAFETY ALERT: PPE VIOLATION";
      if (sub) sub.innerText = `${data.active_violations} worker(s) missing required helmet/vest`;
      const compliance = data.counts && data.counts.person > 0
        ? Math.max(0, Math.round(((data.counts.person - data.active_violations) / data.counts.person) * 100))
        : 100;
      setText("dash-safety-compliance", `${compliance}%`);
    }
  }

  setText("dash-safety-time", data.timestamp || "--:--:--");

  // 3. Performance Cards
  if (data.fps !== undefined) {
    setText("dash-perf-fps", `${data.fps} FPS`);
    setBarWidth("dash-bar-fps", Math.min(100, (data.fps / 35.0) * 100));
    setText("live-perf-fps", `${data.fps} FPS`);
  }
  if (data.inference_ms !== undefined) {
    setText("dash-perf-inf", `${data.inference_ms} ms`);
    setBarWidth("dash-bar-inf", Math.min(100, (data.inference_ms / 100.0) * 100));
    setText("live-perf-inf", `${data.inference_ms} ms`);
  }
  if (data.cpu_pct !== undefined) {
    setText("dash-perf-cpu", `${data.cpu_pct}%`);
    setBarWidth("dash-bar-cpu", data.cpu_pct);
  }
  if (data.ram_mb !== undefined) {
    setText("dash-perf-ram", `${data.ram_mb} MB`);
  }

  // Dashboard camera status badge & preview sync
  const camBadge = document.getElementById("dash-camera-status");
  const previewImg = document.getElementById("dash-preview-img");
  const previewCanvas = document.getElementById("dash-preview-canvas");
  const previewPlaceholder = document.getElementById("dash-preview-placeholder");
  const quickBtn = document.getElementById("dash-camera-quick-btn");

  const isLive = state.isCameraRunning || data.camera_live;

  if (camBadge) {
    if (isLive) {
      camBadge.className = "hud-badge hud-live";
      camBadge.innerText = "● LIVE";
      if (quickBtn) {
        quickBtn.className = "btn btn-danger btn-sm";
        quickBtn.innerText = "Stop Camera";
      }
      if (previewPlaceholder) previewPlaceholder.style.display = "none";
      if (state.cameraMode === "server") {
        if (previewCanvas) previewCanvas.style.display = "none";
        if (previewImg) {
          previewImg.style.display = "block";
          if (!previewImg.src || previewImg.src.indexOf("/api/camera/stream") === -1) {
            previewImg.src = `/api/camera/stream?t=${Date.now()}`;
          }
        }
      } else {
        if (previewImg) previewImg.style.display = "none";
        if (previewCanvas) previewCanvas.style.display = "block";
      }
    } else {
      camBadge.className = "hud-badge";
      camBadge.innerText = "● Offline";
      if (quickBtn) {
        quickBtn.className = "btn btn-primary btn-sm";
        quickBtn.innerText = "Start Camera";
      }
      if (previewPlaceholder) previewPlaceholder.style.display = "block";
      if (previewImg) {
        previewImg.src = "";
        previewImg.style.display = "none";
      }
      if (previewCanvas) previewCanvas.style.display = "none";
    }
  }
}

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.innerText = text;
}

function setBarWidth(id, pct) {
  const el = document.getElementById(id);
  if (el) el.style.width = `${pct}%`;
}

// ==========================================================================
// 4. Live Monitor Camera Controls (Server Camera & Client Browser Webcam)
// ==========================================================================
function toggleQuickCamera() {
  if (state.isCameraRunning) {
    stopCamera();
  } else {
    startCamera();
  }
}

function initCameraControls() {
  const toggleBtn = document.getElementById("btn-camera-toggle");
  const modeSelect = document.getElementById("camera-mode-select");

  if (modeSelect) {
    modeSelect.addEventListener("change", e => {
      state.cameraMode = e.target.value;
      const idLabel = document.getElementById("camera-id-label");
      const idInput = document.getElementById("camera-id-input");
      if (idLabel && idInput) {
        const isServer = (state.cameraMode === "server");
        idLabel.style.display = isServer ? "inline" : "none";
        idInput.style.display = isServer ? "inline" : "none";
      }
      if (state.isCameraRunning) {
        stopCamera();
      }
    });
  }

  if (toggleBtn) {
    toggleBtn.addEventListener("click", () => {
      if (state.isCameraRunning) {
        stopCamera();
      } else {
        startCamera();
      }
    });
  }
}

async function startCamera() {
  state.isCameraRunning = true;
  updateCameraUI(true);

  if (state.cameraMode === "server") {
    // Start Server-Side Hardware Camera
    const camIdInput = document.getElementById("camera-id-input");
    const camId = camIdInput ? parseInt(camIdInput.value) || 0 : 0;

    try {
      const res = await fetch("/api/camera/start", { method: "POST" });
      const data = await res.json();
      updateCameraUI(true);
      showToast("Server hardware camera streaming active");
    } catch (err) {
      showToast("Failed to start server camera", "danger");
      stopCamera();
    }
  } else {
    // Start Client Local Webcam (Global video element, renders directly to active canvas)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } }
      });
      state.clientWebcamStream = stream;

      const video = document.getElementById("global-webcam-video");
      if (video) {
        video.srcObject = stream;
        await video.play();
        startClientWebcamDetectionLoop(video);
      }
      showToast("Live webcam online");
    } catch (err) {
      console.error(err);
      showToast(`Cannot access webcam: ${err.message}`, "danger");
      stopCamera();
    }
  }
}

function stopCamera() {
  state.isCameraRunning = false;
  state.latestDetections = [];
  updateCameraUI(false);

  if (state.animFrameId) {
    cancelAnimationFrame(state.animFrameId);
    state.animFrameId = null;
  }

  if (state.streamSendTimer) {
    clearInterval(state.streamSendTimer);
    state.streamSendTimer = null;
  }

  // Stop server stream
  if (state.cameraMode === "server") {
    fetch("/api/camera/stop", { method: "POST" }).catch(() => {});
    const streamImg = document.getElementById("live-stream-img");
    const dashImg = document.getElementById("dash-preview-img");
    if (streamImg) streamImg.src = "";
    if (dashImg) dashImg.src = "";
  }

  // Stop browser webcam
  if (state.clientWebcamStream) {
    state.clientWebcamStream.getTracks().forEach(track => track.stop());
    state.clientWebcamStream = null;
  }

  if (state.clientWs) {
    state.clientWs.close();
    state.clientWs = null;
  }

  const video = document.getElementById("global-webcam-video");
  if (video) {
    video.srcObject = null;
  }

  const liveCanvas = document.getElementById("client-webcam-canvas");
  if (liveCanvas) {
    const ctx = liveCanvas.getContext("2d");
    ctx.clearRect(0, 0, liveCanvas.width, liveCanvas.height);
  }

  const dashCanvas = document.getElementById("dash-preview-canvas");
  if (dashCanvas) {
    const ctx = dashCanvas.getContext("2d");
    ctx.clearRect(0, 0, dashCanvas.width, dashCanvas.height);
  }

  showToast("Camera stopped");
}

function updateCameraUI(isRunning) {
  const toggleBtn = document.getElementById("btn-camera-toggle");
  const quickBtn = document.getElementById("dash-camera-quick-btn");
  const badge = document.getElementById("camera-live-badge");
  const dashBadge = document.getElementById("dash-camera-status");
  const dashPlaceholder = document.getElementById("dash-preview-placeholder");
  const dashCanvas = document.getElementById("dash-preview-canvas");
  const dashImg = document.getElementById("dash-preview-img");
  const liveCanvas = document.getElementById("client-webcam-canvas");
  const liveImg = document.getElementById("live-stream-img");
  const offlineMsg = document.getElementById("camera-offline-msg");

  // Live page toggle button
  if (toggleBtn) {
    if (isRunning) {
      toggleBtn.className = "btn btn-danger";
      toggleBtn.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="6" y="6" width="12" height="12"/>
        </svg> Stop Camera
      `;
    } else {
      toggleBtn.className = "btn btn-primary";
      toggleBtn.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polygon points="5 3 19 12 5 21 5 3"/>
        </svg> Start Camera
      `;
    }
  }

  // Dashboard quick button
  if (quickBtn) {
    if (isRunning) {
      quickBtn.className = "btn btn-danger btn-sm";
      quickBtn.innerText = "Stop Camera";
    } else {
      quickBtn.className = "btn btn-primary btn-sm";
      quickBtn.innerText = "Start Camera";
    }
  }

  // Badges
  if (badge) {
    badge.className = isRunning ? "hud-badge hud-live" : "hud-badge";
    badge.innerText = isRunning ? "● LIVE" : "Camera Offline";
  }
  if (dashBadge) {
    dashBadge.className = isRunning ? "hud-badge hud-live" : "hud-badge";
    dashBadge.innerText = isRunning ? "● LIVE" : "● Offline";
  }

  // View elements
  if (isRunning) {
    if (dashPlaceholder) dashPlaceholder.style.display = "none";
    if (offlineMsg) offlineMsg.style.display = "none";

    if (state.cameraMode === "browser") {
      if (dashCanvas) dashCanvas.style.display = "block";
      if (dashImg) dashImg.style.display = "none";
      if (liveCanvas) liveCanvas.style.display = "block";
      if (liveImg) liveImg.style.display = "none";
    } else {
      if (dashCanvas) dashCanvas.style.display = "none";
      if (dashImg) {
        if (!dashImg.src || dashImg.src.indexOf("/api/camera/stream") === -1) {
          dashImg.src = `/api/camera/stream?t=${Date.now()}`;
        }
        dashImg.style.display = "block";
      }
      if (liveCanvas) liveCanvas.style.display = "none";
      if (liveImg) {
        if (!liveImg.src || liveImg.src.indexOf("/api/camera/stream") === -1) {
          liveImg.src = `/api/camera/stream?t=${Date.now()}`;
        }
        liveImg.style.display = "block";
      }
    }
  } else {
    if (dashPlaceholder) dashPlaceholder.style.display = "block";
    if (dashCanvas) dashCanvas.style.display = "none";
    if (dashImg) {
      dashImg.src = "";
      dashImg.style.display = "none";
    }
    if (liveCanvas) liveCanvas.style.display = "none";
    if (liveImg) {
      liveImg.src = "";
      liveImg.style.display = "none";
    }
    if (offlineMsg) offlineMsg.style.display = "block";
  }
}

function startClientWebcamDetectionLoop(video) {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/live`;

  const offscreen = document.createElement("canvas");
  offscreen.width = 640;
  offscreen.height = 480;
  const offCtx = offscreen.getContext("2d");

  state.clientWs = new WebSocket(wsUrl);

  state.clientWs.onmessage = event => {
    try {
      const data = JSON.parse(event.data);
      state.latestDetections = data.detections || [];
      if (data.counts) {
        updateDashboardFromLiveDetections(data);
      }
      logRecentEvents(data.detections);
    } catch (err) {}
  };

  // Continuous animation loop that paints directly onto active view's canvas
  function renderLoop() {
    if (!state.isCameraRunning || state.cameraMode !== "browser") return;

    const w = video.videoWidth || 640;
    const h = video.videoHeight || 480;

    // 1. If user is on Dashboard, render directly onto Dashboard Preview Canvas!
    if (state.currentView === "dashboard") {
      const dashCanvas = document.getElementById("dash-preview-canvas");
      if (dashCanvas) {
        if (dashCanvas.width !== w || dashCanvas.height !== h) {
          dashCanvas.width = w;
          dashCanvas.height = h;
        }
        const dashCtx = dashCanvas.getContext("2d");
        dashCtx.drawImage(video, 0, 0, w, h);
        drawCanvasBBoxes(dashCtx, w, h, state.latestDetections);
      }
    }

    // 2. If user is on Live Monitor, render directly onto Live Monitor Canvas!
    if (state.currentView === "live") {
      const liveCanvas = document.getElementById("client-webcam-canvas");
      if (liveCanvas) {
        if (liveCanvas.width !== w || liveCanvas.height !== h) {
          liveCanvas.width = w;
          liveCanvas.height = h;
        }
        const liveCtx = liveCanvas.getContext("2d");
        liveCtx.drawImage(video, 0, 0, w, h);
        drawCanvasBBoxes(liveCtx, w, h, state.latestDetections);
      }
    }

    state.animFrameId = requestAnimationFrame(renderLoop);
  }
  state.animFrameId = requestAnimationFrame(renderLoop);

  // Network frame sender loop to C++ backend
  let isSending = false;
  state.streamSendTimer = setInterval(() => {
    if (!state.isCameraRunning || !state.clientWs || state.clientWs.readyState !== WebSocket.OPEN) {
      return;
    }
    if (isSending || video.readyState < 2) return;
    isSending = true;

    offCtx.drawImage(video, 0, 0, offscreen.width, offscreen.height);
    offscreen.toBlob(blob => {
      if (blob && state.clientWs && state.clientWs.readyState === WebSocket.OPEN) {
        state.clientWs.send(blob);
      }
      isSending = false;
    }, "image/jpeg", 0.7);
  }, 66);
}

function drawCanvasBBoxes(ctx, width, height, detections) {
  if (!detections || detections.length === 0) return;

  const scaleX = width / 640.0;
  const scaleY = height / 480.0;

  const classColors = {
    0: "#F59E0B", // Forklift: Orange
    1: "#EAB308", // Helmet: Yellow
    2: "#3B82F6", // Person: Blue
    3: "#22C55E", // Safety Vest: Green
  };

  detections.forEach(det => {
    const color = classColors[det.class_id] || "#3B82F6";
    const [bx, by, bw, bh] = det.box;

    const x = bx * scaleX;
    const y = by * scaleY;
    const w = bw * scaleX;
    const h = bh * scaleY;

    // Bounding Box stroke (outline only, no inner fill)
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = color;
    ctx.strokeRect(x, y, w, h);

    // Label header
    const label = `${det.class_name} ${(det.confidence * 100).toFixed(0)}%`;
    ctx.font = "bold 13px -apple-system, BlinkMacSystemFont, sans-serif";
    const textWidth = ctx.measureText(label).width;

    ctx.fillStyle = color;
    ctx.fillRect(x, Math.max(0, y - 22), textWidth + 10, 22);

    ctx.fillStyle = "#000000";
    ctx.fillText(label, x + 5, Math.max(16, y - 6));
  });
}

function updateDashboardFromLiveDetections(data) {
  if (data.counts) {
    setText("dash-count-person", data.counts.person || 0);
    setText("dash-count-forklift", data.counts.forklift || 0);
    setText("dash-count-helmet", data.counts.helmet || 0);
    setText("dash-count-vest", data.counts.safety_vest || 0);

    setText("live-cnt-person", data.counts.person || 0);
    setText("live-cnt-forklift", data.counts.forklift || 0);
    setText("live-cnt-helmet", data.counts.helmet || 0);
    setText("live-cnt-vest", data.counts.safety_vest || 0);
  }

  // Safety Status Banner
  const banner = document.getElementById("dash-safety-banner");
  const icon = document.getElementById("dash-safety-icon");
  const title = document.getElementById("dash-safety-title");
  const sub = document.getElementById("dash-safety-sub");
  const time = document.getElementById("dash-safety-time");
  const compliance = document.getElementById("dash-safety-compliance");

  if (banner && icon && title && sub) {
    if (data.safety_status === "WARNING") {
      banner.className = "safety-status-card warning";
      icon.innerText = "!";
      title.innerText = "WARNING";
      sub.innerText = `PPE violation detected (${data.active_violations} active violations)`;
      if (compliance) compliance.innerText = "VIOLATION";
    } else {
      banner.className = "safety-status-card clear";
      icon.innerText = "✓";
      title.innerText = "ALL CLEAR";
      sub.innerText = "No active safety violations detected";
      if (compliance) compliance.innerText = "100%";
    }
    if (time) time.innerText = new Date().toLocaleTimeString();
  }

  // Latency & FPS
  if (data.fps) {
    setText("live-perf-fps", `${data.fps} FPS`);
    setText("dash-perf-fps", `${data.fps} FPS`);
    setBarWidth("dash-bar-fps", Math.min(100, (data.fps / 30.0) * 100));
  }
  if (data.inference_ms) {
    setText("live-perf-inf", `${data.inference_ms} ms`);
    setText("dash-perf-inf", `${data.inference_ms} ms`);
    setBarWidth("dash-bar-inf", Math.min(100, (data.inference_ms / 100.0) * 100));
  }
  if (data.total_ms) {
    setText("live-perf-total", `${data.total_ms} ms`);
  }
}

function logRecentEvents(detections) {
  if (!detections || detections.length === 0) return;

  const tbody = document.getElementById("recent-events-tbody");
  if (!tbody) return;

  const timeStr = new Date().toLocaleTimeString();

  detections.slice(0, 3).forEach(d => {
    state.recentEvents.unshift({
      time: timeStr,
      name: d.class_name,
      conf: (d.confidence * 100).toFixed(0) + "%",
      status: d.class_name === "person" ? "Monitored" : "Compliant"
    });
  });

  // Keep up to 30 events
  state.recentEvents = state.recentEvents.slice(0, 30);

  let html = "";
  state.recentEvents.forEach(e => {
    html += `
      <tr>
        <td style="font-family:var(--font-mono);">${e.time}</td>
        <td><strong>${e.name}</strong></td>
        <td>${e.conf}</td>
        <td><span class="status-dot"></span>${e.status}</td>
      </tr>
    `;
  });
  tbody.innerHTML = html;
}

// ==========================================================================
// 5. Model Testing (Image & Video Tabs)
// ==========================================================================
function switchTestTab(tab) {
  const btnImg = document.getElementById("tab-btn-image");
  const btnVid = document.getElementById("tab-btn-video");
  const cntImg = document.getElementById("tab-content-image");
  const cntVid = document.getElementById("tab-content-video");

  if (tab === "image") {
    btnImg.classList.add("active");
    btnVid.classList.remove("active");
    cntImg.style.display = "block";
    cntVid.style.display = "none";
  } else {
    btnVid.classList.add("active");
    btnImg.classList.remove("active");
    cntVid.style.display = "block";
    cntImg.style.display = "none";
    restoreLatestVideoTask();
  }
}

// Image Testing Handlers
function initImageTesting() {
  const dropzone = document.getElementById("image-dropzone");
  const input = document.getElementById("image-file-input");

  if (!dropzone || !input) return;

  dropzone.addEventListener("click", () => input.click());

  dropzone.addEventListener("dragover", e => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", e => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      uploadAndRunImage(e.dataTransfer.files[0]);
    }
  });

  input.addEventListener("change", () => {
    if (input.files.length > 0) {
      uploadAndRunImage(input.files[0]);
    }
  });
}

async function uploadAndRunImage(file) {
  const loading = document.getElementById("image-loading");
  const resultSec = document.getElementById("image-result-section");

  if (loading) loading.style.display = "block";
  if (resultSec) resultSec.style.display = "none";

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/api/test/image", {
      method: "POST",
      body: formData
    });
    const data = await res.json();

    if (data.status !== "success") {
      throw new Error(data.detail || "Inference failed");
    }

    // Set preview images
    document.getElementById("img-original-preview").src = data.original_url;
    document.getElementById("img-result-preview").src = data.result_url;

    // Set download link
    const dlBtn = document.getElementById("btn-download-image");
    if (dlBtn) dlBtn.href = data.result_url;

    // Summary Content
    const summary = document.getElementById("image-summary-content");
    if (summary) {
      summary.innerHTML = `
        <div><strong>👤 Person:</strong> ${data.counts.person}</div>
        <div><strong>🚜 Forklift:</strong> ${data.counts.forklift}</div>
        <div><strong>🪖 Helmet:</strong> ${data.counts.helmet}</div>
        <div><strong>🦺 Safety Vest:</strong> ${data.counts.safety_vest}</div>
        <div><strong>⚡ Inference:</strong> <span style="color:var(--color-success); font-weight:bold;">${data.metrics.inference_ms} ms</span></div>
        <div><strong>🛡️ Status:</strong> <span style="font-weight:bold; color:${data.safety_status === 'ALL_CLEAR' ? 'var(--color-success)' : 'var(--color-danger)'};">${data.safety_status}</span></div>
      `;
    }

    if (loading) loading.style.display = "none";
    if (resultSec) resultSec.style.display = "block";
    showToast(`Detection completed in ${data.metrics.total_ms} ms`);
  } catch (err) {
    if (loading) loading.style.display = "none";
    showToast(`Failed: ${err.message}`, "danger");
  }
}

function clearImageTest() {
  const resultSec = document.getElementById("image-result-section");
  const input = document.getElementById("image-file-input");
  if (resultSec) resultSec.style.display = "none";
  if (input) input.value = "";
}

// Video Testing Handlers
let selectedVideoFile = null;

function initVideoTesting() {
  const dropzone = document.getElementById("video-dropzone");
  const input = document.getElementById("video-file-input");

  if (!dropzone || !input) return;

  dropzone.addEventListener("click", () => input.click());

  dropzone.addEventListener("dragover", e => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", e => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      handleVideoSelected(e.dataTransfer.files[0]);
    }
  });

  input.addEventListener("change", () => {
    if (input.files.length > 0) {
      handleVideoSelected(input.files[0]);
    }
  });
}

function handleVideoSelected(file) {
  selectedVideoFile = file;

  const metaCard = document.getElementById("video-meta-card");
  const resultSec = document.getElementById("video-result-section");
  const progSec = document.getElementById("video-progress-section");
  const btn = document.getElementById("btn-run-video-proc");

  if (resultSec) resultSec.style.display = "none";
  if (progSec) progSec.style.display = "none";
  if (btn) {
    btn.disabled = false;
    btn.innerHTML = "Run Detection (Streaming Process)";
  }

  setText("v-meta-name", file.name);
  setText("v-meta-duration", "Reading...");
  setText("v-meta-res", "Reading...");
  setText("v-meta-fps", "Analyzing...");

  // Read actual metadata from video file via HTML5 Video element
  const tempV = document.createElement("video");
  tempV.preload = "metadata";
  tempV.onloadedmetadata = () => {
    URL.revokeObjectURL(tempV.src);
    const sec = tempV.duration;
    if (!isNaN(sec) && sec > 0) {
      const m = Math.floor(sec / 60);
      const s = Math.floor(sec % 60);
      setText("v-meta-duration", `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`);
    }
    setText("v-meta-res", `${tempV.videoWidth || 1280} × ${tempV.videoHeight || 720}`);
    setText("v-meta-fps", "25-30 FPS");
  };
  tempV.onerror = () => {
    setText("v-meta-duration", "--:--");
    setText("v-meta-res", "Standard MP4");
    setText("v-meta-fps", "30 FPS");
  };
  tempV.src = URL.createObjectURL(file);

  if (metaCard) metaCard.style.display = "block";
}

async function startVideoProcessing() {
  if (!selectedVideoFile) return;

  const btn = document.getElementById("btn-run-video-proc");
  const progSec = document.getElementById("video-progress-section");
  const metaCard = document.getElementById("video-meta-card");

  if (btn) {
    btn.disabled = true;
    btn.innerHTML = `<span style="display:inline-block; width:12px; height:12px; border:2px solid #fff; border-top-color:transparent; border-radius:50%; animation:spin 1s linear infinite; margin-right:6px;"></span> Uploading & Starting C++ Engine...`;
  }
  if (progSec) progSec.style.display = "block";
  setText("v-prog-frames", "Uploading file to server...");
  setText("v-prog-pct", "0%");
  setText("v-prog-fps", "Starting C++ Worker...");

  const formData = new FormData();
  formData.append("file", selectedVideoFile);

  try {
    const res = await fetch("/api/test/video", {
      method: "POST",
      body: formData
    });
    if (!res.ok) throw new Error(`Server returned ${res.status}`);
    const data = await res.json();
    state.videoTaskId = data.task_id;
    localStorage.setItem("last_video_task_id", data.task_id);

    if (metaCard) metaCard.style.display = "none";

    // Poll Progress
    if (state.videoPollTimer) clearInterval(state.videoPollTimer);
    state.videoPollTimer = setInterval(pollVideoProgress, 400);
  } catch (err) {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = "Run Detection (Streaming Process)";
    }
    if (progSec) progSec.style.display = "none";
    showToast(`Error: ${err.message}`, "danger");
  }
}

async function pollVideoProgress() {
  if (!state.videoTaskId) return;

  try {
    const res = await fetch(`/api/test/video/progress/${state.videoTaskId}`);
    const data = await res.json();

    const bar = document.getElementById("video-progress-bar");
    if (bar) bar.style.width = `${data.progress}%`;

    setText("v-prog-frames", `Frame ${data.processed_frames} / ${data.total_frames}`);
    setText("v-prog-pct", `${data.progress}%`);
    setText("v-prog-fps", `FPS: ${data.fps || '--'}`);

    if (data.status === "completed") {
      clearInterval(state.videoPollTimer);
      state.videoPollTimer = null;
      localStorage.setItem("last_video_task_id", state.videoTaskId);

      // Show result video
      const progSec = document.getElementById("video-progress-section");
      const resultSec = document.getElementById("video-result-section");
      const player = document.getElementById("video-result-player");
      const dlBtn = document.getElementById("btn-download-video");

      if (progSec) progSec.style.display = "none";
      if (resultSec) resultSec.style.display = "block";
      if (player) {
        player.src = data.result_url;
        player.load();
      }
      if (dlBtn) dlBtn.href = data.result_url;

      const summary = document.getElementById("video-result-summary");
      if (summary) {
        summary.innerHTML = `
          <strong>Filename:</strong> ${data.filename || "Video Result"} | 
          <strong>Total Frames:</strong> ${data.total_frames} | 
          <strong>Average FPS:</strong> ${data.fps} | 
          <strong>Avg Inference:</strong> ${data.avg_inference_ms || 32} ms
        `;
      }
      showToast("Video processing completed!");
    } else if (data.status === "error") {
      clearInterval(state.videoPollTimer);
      state.videoPollTimer = null;
      showToast(`Error processing video: ${data.error}`, "danger");
    }
  } catch (err) {}
}

async function restoreLatestVideoTask() {
  if (state.videoTaskId || selectedVideoFile) return;

  const savedId = localStorage.getItem("last_video_task_id");
  let url = savedId ? `/api/test/video/progress/${savedId}` : `/api/test/video/latest`;

  try {
    const res = await fetch(url);
    if (!res.ok) return;
    const data = await res.json();
    if (!data || data.status === "none" || !data.task_id) return;

    if (data.status === "completed") {
      state.videoTaskId = data.task_id;
      const progSec = document.getElementById("video-progress-section");
      const resultSec = document.getElementById("video-result-section");
      const player = document.getElementById("video-result-player");
      const dlBtn = document.getElementById("btn-download-video");

      if (progSec) progSec.style.display = "none";
      if (resultSec) resultSec.style.display = "block";
      if (player) {
        player.src = data.result_url || `/api/test/video/result/${data.task_id}`;
        player.load();
      }
      if (dlBtn) dlBtn.href = data.result_url || `/api/test/video/result/${data.task_id}`;

      const summary = document.getElementById("video-result-summary");
      if (summary) {
        summary.innerHTML = `
          <strong>Filename:</strong> ${data.filename || "Video Result"} | 
          <strong>Total Frames:</strong> ${data.total_frames || "--"} | 
          <strong>Average FPS:</strong> ${data.fps || "--"} | 
          <strong>Avg Latency:</strong> ${data.avg_inference_ms || 32} ms
        `;
      }
    } else if (data.status === "processing") {
      state.videoTaskId = data.task_id;
      const progSec = document.getElementById("video-progress-section");
      if (progSec) progSec.style.display = "block";
      if (state.videoPollTimer) clearInterval(state.videoPollTimer);
      state.videoPollTimer = setInterval(pollVideoProgress, 400);
    }
  } catch (err) {}
}

function clearVideoTest() {
  selectedVideoFile = null;
  state.videoTaskId = null;
  localStorage.removeItem("last_video_task_id");
  if (state.videoPollTimer) {
    clearInterval(state.videoPollTimer);
    state.videoPollTimer = null;
  }
  const metaCard = document.getElementById("video-meta-card");
  const progSec = document.getElementById("video-progress-section");
  const resultSec = document.getElementById("video-result-section");
  const input = document.getElementById("video-file-input");
  const btn = document.getElementById("btn-run-video-proc");
  const player = document.getElementById("video-result-player");

  if (player) {
    player.pause();
    player.src = "";
  }
  if (btn) {
    btn.disabled = false;
    btn.innerHTML = "Run Detection (Streaming Process)";
  }
  if (metaCard) metaCard.style.display = "none";
  if (progSec) progSec.style.display = "none";
  if (resultSec) resultSec.style.display = "none";
  if (input) input.value = "";
}

// ==========================================================================
// 6. Settings Page Handlers
// ==========================================================================
async function loadSettings() {
  try {
    const res = await fetch("/api/settings");
    state.settings = await res.json();
    populateSettingsUI(state.settings);
  } catch (err) {}
}

function populateSettingsUI(cfg) {
  if (!cfg) return;

  const t = cfg.thresholds || {};

  setSliderVal("forklift", t.forklift || 0.50);
  setSliderVal("helmet", t.helmet || 0.60);
  setSliderVal("person", t.person || 0.50);
  setSliderVal("safety_vest", t.safety_vest || 0.55);
  setSliderVal("iou", t.iou || 0.45);

  const c = cfg.camera || {};
  const srcSelect = document.getElementById("cfg-cam-source");
  const idInput = document.getElementById("cfg-cam-id");
  if (srcSelect && c.source) srcSelect.value = c.source;
  if (idInput && c.camera_id !== undefined) idInput.value = c.camera_id;
}

function setSliderVal(key, val) {
  const slider = document.getElementById(`slider-${key}`);
  const label = document.getElementById(`val-thresh-${key}`);
  if (slider) slider.value = val;
  if (label) label.innerText = parseFloat(val).toFixed(2);

  if (slider) {
    slider.oninput = () => {
      if (label) label.innerText = parseFloat(slider.value).toFixed(2);
    };
  }
}

async function saveSettingsToServer() {
  const settings = {
    ...state.settings,
    thresholds: {
      forklift: parseFloat(document.getElementById("slider-forklift").value),
      helmet: parseFloat(document.getElementById("slider-helmet").value),
      person: parseFloat(document.getElementById("slider-person").value),
      safety_vest: parseFloat(document.getElementById("slider-safety_vest").value),
      iou: parseFloat(document.getElementById("slider-iou").value),
    },
    camera: {
      ...state.settings?.camera,
      source: document.getElementById("cfg-cam-source").value,
      camera_id: parseInt(document.getElementById("cfg-cam-id").value) || 0,
    }
  };

  try {
    const res = await fetch("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(settings)
    });
    const data = await res.json();
    if (data.status === "success") {
      state.settings = settings;
      showToast("✓ Settings saved successfully");
    }
  } catch (err) {
    showToast("Failed to save settings", "danger");
  }
}

function resetSettings() {
  populateSettingsUI({
    thresholds: {
      forklift: 0.50,
      helmet: 0.60,
      person: 0.50,
      safety_vest: 0.55,
      iou: 0.45
    },
    camera: {
      source: "webcam",
      camera_id: 0
    }
  });
  showToast("Reset to default thresholds");
}

// ==========================================================================
// Initialization on Page Load
// ==========================================================================
document.addEventListener("DOMContentLoaded", () => {
  initRouter();
  initTelemetryWebSocket();
  initCameraControls();
  initImageTesting();
  initVideoTesting();
  loadSettings();
});
