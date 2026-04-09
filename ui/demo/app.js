async function loadDashboard() {
  const response = await fetch("/api/dashboard");
  if (!response.ok) {
    throw new Error(`Failed to load dashboard data: ${response.status}`);
  }
  return response.json();
}

function renderLiveMarket(liveMarket) {
  const title = document.getElementById("live-title");
  const note = document.getElementById("live-note");
  const subtitle = document.getElementById("live-market-subtitle");
  const metrics = document.getElementById("live-metrics");
  const table = document.getElementById("live-table");

  if (!liveMarket) {
    title.textContent = "Live Market Snapshot";
    subtitle.textContent = "No MT5 Local source detected.";
    note.textContent = "Start MetaTrader 5 locally and keep the exporter or bridge available if you want a live gold snapshot alongside the local demo artifacts.";
    metrics.innerHTML = "";
    table.innerHTML = "";
    return;
  }

  if (liveMarket.error) {
    title.textContent = "Live Market Snapshot";
    subtitle.textContent = "Live fetch failed.";
    note.textContent = liveMarket.error;
    metrics.innerHTML = "";
    table.innerHTML = "";
    return;
  }

  title.textContent = "Live Market Snapshot";
  subtitle.textContent = `${liveMarket.symbol} | ${liveMarket.interval} | ${liveMarket.type}`;
  note.textContent = liveMarket.note;

  const cards = [
    ["Last Close", liveMarket.last_close.toFixed(2)],
    ["Change", `${liveMarket.change >= 0 ? "+" : ""}${liveMarket.change.toFixed(2)}`],
    ["Change %", `${liveMarket.change_pct >= 0 ? "+" : ""}${liveMarket.change_pct.toFixed(2)}%`],
    ["Lookback High", liveMarket.high_lookback.toFixed(2)],
    ["Lookback Low", liveMarket.low_lookback.toFixed(2)],
    ["Last Bar Time", liveMarket.last_time],
  ];

  metrics.innerHTML = cards
    .map(
      ([label, value]) => `
        <article class="live-card">
          <div class="label">${label}</div>
          <div class="value">${value}</div>
        </article>
      `,
    )
    .join("");

  renderTable("live-table", liveMarket.preview);
}

function renderMetrics(summary) {
  const grid = document.getElementById("metrics-grid");
  grid.innerHTML = summary.metrics
    .map(
      ([label, value]) => `
        <article class="metric-card">
          <div class="label">${label}</div>
          <div class="value">${value}</div>
        </article>
      `,
    )
    .join("");

  document.getElementById("range-text").textContent = summary.range_text;
  document.getElementById("runtime-badge").textContent = summary.runtime_badge;
}

function renderPrediction(data) {
  document.getElementById("prediction-title").textContent = data.context.prediction_label;
  document.getElementById("prediction-note").textContent = data.context.prediction_note;
  document.getElementById("signal-chip").textContent = data.prediction.label;
  document.getElementById("prediction-time").textContent = data.context.prediction_time_label;
  document.getElementById("model-meta").innerHTML = `
    <div>${data.model.num_features} features</div>
    <div>${data.model.num_trees} trees</div>
    <div>Model size: ${data.model.onnx_size}</div>
  `;

  const colors = {
    SHORT: "linear-gradient(90deg, #a54e2f, #d77f3f)",
    HOLD: "linear-gradient(90deg, #8a5b10, #c68b24)",
    LONG: "linear-gradient(90deg, #314d3a, #5d8a63)",
  };

  const stack = document.getElementById("probability-stack");
  stack.innerHTML = data.prediction.probabilities
    .map(
      (item) => `
        <div class="probability-row">
          <div>${item.label}</div>
          <div class="probability-bar">
            <div class="probability-fill" style="width:${(item.value * 100).toFixed(1)}%; background:${colors[item.label]}"></div>
          </div>
          <div>${(item.value * 100).toFixed(1)}%</div>
        </div>
      `,
    )
    .join("");
}

function renderFeatures(data) {
  document.getElementById("feature-note").textContent = data.context.feature_note;
  const grid = document.getElementById("feature-grid");
  grid.innerHTML = data.feature_snapshot.values
    .map(
      (item) => `
        <article class="feature-card">
          <div class="label">${item.label}</div>
          <div class="value">${Number(item.value).toFixed(4)}</div>
        </article>
      `,
    )
    .join("");
}

function renderArtifacts(data) {
  const list = document.getElementById("artifact-list");
  list.innerHTML = data.artifacts
    .map(
      (item) => `
        <article class="artifact-card">
          <div class="label">${item.label}</div>
          <div class="path">${item.path}</div>
          <div class="size">${item.size}</div>
        </article>
      `,
    )
    .join("");
}

function renderTable(containerId, preview) {
  const container = document.getElementById(containerId);
  const header = preview.columns.map((column) => `<th>${column}</th>`).join("");
  const rows = preview.rows
    .map(
      (row) => `
        <tr>
          ${preview.columns.map((column) => `<td>${row[column]}</td>`).join("")}
        </tr>
      `,
    )
    .join("");

  container.innerHTML = `
    <table>
      <thead>
        <tr>${header}</tr>
      </thead>
      <tbody>
        ${rows}
      </tbody>
    </table>
  `;
}

async function bootstrap() {
  try {
    const data = await loadDashboard();
    document.getElementById("page-title").textContent = data.title;
    document.getElementById("page-lede").textContent =
      "A local view of the generated CSV artifacts, MT5 validation snapshot, exported ONNX model, and live market context.";
    renderLiveMarket(data.live_market);
    renderMetrics(data.summary);
    renderPrediction(data);
    renderFeatures(data);
    renderArtifacts(data);
    renderTable("standardized-table", data.previews.standardized);
    renderTable("overlap-table", data.previews.overlap);
    renderTable("validation-table", data.previews.validation);
  } catch (error) {
    document.body.innerHTML = `<pre style="padding:24px;font-family:monospace;">${error.message}</pre>`;
  }
}

bootstrap();
