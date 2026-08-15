"use strict";

const $ = (id) => document.getElementById(id);

const state = {
  providers: [],
  config: null,
  file: null,
  jobId: null,
  eventSource: null,
  running: false,
  // TMDB에서 가져온 줄거리/출연진 정보. 직접 입력한 줄거리가 없을 때만 사용됨
  tmdb: null,
};

// --- 공통 유틸 ------------------------------------------------------------

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const isJson = (response.headers.get("content-type") || "").includes("application/json");
  const body = isJson ? await response.json() : null;
  if (!response.ok) {
    const message = (body && (body.detail || body.message)) || `요청 실패 (HTTP ${response.status})`;
    throw new Error(message);
  }
  return body;
}

function postJson(path, payload) {
  return api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

let toastTimer = null;
function toast(message, kind = "") {
  const element = $("toast");
  element.textContent = message;
  element.className = `toast ${kind}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => element.classList.add("hidden"), 4000);
}

function logLine(level, message, time) {
  const console_ = $("console");
  const atBottom = console_.scrollHeight - console_.scrollTop - console_.clientHeight < 40;
  const line = document.createElement("div");
  line.className = "line";
  const stamp = document.createElement("span");
  stamp.className = "time";
  stamp.textContent = time || new Date().toTimeString().slice(0, 8);
  const text = document.createElement("span");
  text.className = level || "info";
  text.textContent = message;
  line.append(stamp, text);
  console_.appendChild(line);
  if (atBottom) console_.scrollTop = console_.scrollHeight;
}

function numberOrNull(input) {
  const value = input.value.trim();
  if (value === "") return null;
  const parsed = Number(value);
  return Number.isNaN(parsed) ? null : parsed;
}

// --- 초기화 ---------------------------------------------------------------

async function init() {
  bindEvents();
  try {
    const [providerData, config] = await Promise.all([
      api("/api/providers"),
      api("/api/config"),
    ]);
    state.providers = providerData.providers;
    state.config = config;
    renderProviders();
    applyConfig(config);
    $("config-path").textContent = `설정: ${config.config_path}`;
  } catch (error) {
    toast(`초기화 실패: ${error.message}`, "error");
  }
}

function renderProviders() {
  const select = $("provider");
  select.innerHTML = "";
  for (const provider of state.providers) {
    const option = document.createElement("option");
    option.value = provider.id;
    option.textContent = provider.label;
    select.appendChild(option);
  }
}

function currentProvider() {
  return state.providers.find((item) => item.id === $("provider").value) || state.providers[0];
}

function applyConfig(config) {
  if (config.provider) $("provider").value = config.provider;

  const provider = currentProvider();
  if (provider) {
    const link = $("api-key-link");
    link.href = provider.api_key_url || "#";
    link.style.display = provider.api_key_url ? "" : "none";
  }

  updateKeyState();

  const model = (config.models || {})[$("provider").value] || "";
  if (model) {
    const select = $("model");
    select.innerHTML = "";
    const option = document.createElement("option");
    option.value = model;
    option.textContent = model;
    select.appendChild(option);
    select.value = model;
  }

  $("batch-size").value = config.batch_size ?? 300;
  $("language-code").value = config.language_code || "ko";
  $("thinking").checked = config.thinking !== false;
  $("streaming").checked = config.streaming !== false;
  $("thinking-budget").value = config.thinking_budget ?? 2048;
  if (config.temperature !== null && config.temperature !== undefined) {
    $("temperature").value = config.temperature;
  }
  if (config.top_p !== null && config.top_p !== undefined) $("top-p").value = config.top_p;
  if (config.top_k !== null && config.top_k !== undefined) $("top-k").value = config.top_k;
  if (config.story_context) $("story-context").value = config.story_context;
  if (config.extra_instruction) $("extra-instruction").value = config.extra_instruction;
}

function updateKeyState() {
  const providerId = $("provider").value;
  const keysSet = (state.config && state.config.api_keys_set) || {};
  const element = $("api-key-state");
  if (keysSet[providerId]) {
    const masked = (state.config.api_keys || {})[providerId] || "";
    element.textContent = `저장됨 ${masked}`;
    element.className = "state ok";
  } else {
    element.textContent = "미설정";
    element.className = "state bad";
  }

  const tmdbState = $("tmdb-key-state");
  if (state.config && state.config.tmdb_api_key_set) {
    tmdbState.textContent = `저장됨 ${state.config.tmdb_api_key}`;
    tmdbState.className = "state ok";
  } else {
    tmdbState.textContent = "미설정";
    tmdbState.className = "state";
  }
}

// --- 이벤트 바인딩 --------------------------------------------------------

function bindEvents() {
  document.querySelectorAll("[data-toggle]").forEach((head) => {
    head.addEventListener("click", () => {
      const body = $(head.dataset.toggle);
      const hidden = body.classList.toggle("hidden");
      head.setAttribute("aria-expanded", String(!hidden));
      head.querySelector(".chevron").textContent = hidden ? "▸" : "▾";
    });
  });

  $("provider").addEventListener("change", () => {
    applyConfig(state.config || {});
    $("api-key").value = "";
  });

  $("toggle-api-key").addEventListener("click", () => togglePassword("api-key"));
  $("toggle-tmdb-key").addEventListener("click", () => togglePassword("tmdb-key"));
  $("save-api-key").addEventListener("click", saveApiKey);
  $("save-tmdb-key").addEventListener("click", saveTmdbKey);
  $("load-models").addEventListener("click", loadModels);
  $("model").addEventListener("change", saveSelectedModel);

  setupDropzone();
  $("load-local").addEventListener("click", loadLocalFile);
  $("local-path").addEventListener("keydown", (event) => {
    if (event.key === "Enter") loadLocalFile();
  });

  $("tmdb-search").addEventListener("click", searchTmdb);
  $("tmdb-clear").addEventListener("click", clearTmdb);
  $("tmdb-preview").addEventListener("click", () => {
    $("tmdb-preview-body").classList.toggle("hidden");
  });
  $("story-context").addEventListener("input", updateTmdbNote);
  $("language-code").addEventListener("change", updateOutputName);

  $("start").addEventListener("click", startTranslation);
  $("cancel").addEventListener("click", cancelTranslation);
  $("download").addEventListener("click", () => {
    if (state.jobId) window.location.href = `/api/jobs/${state.jobId}/download`;
  });
  $("clear-log").addEventListener("click", () => ($("console").innerHTML = ""));
}

function togglePassword(id) {
  const input = $(id);
  input.type = input.type === "password" ? "text" : "password";
}

// --- 설정 저장 ------------------------------------------------------------

async function saveApiKey() {
  const key = $("api-key").value.trim();
  if (!key) {
    toast("API 키를 입력하세요.", "error");
    return;
  }
  try {
    state.config = await postJson("/api/config", {
      provider: $("provider").value,
      api_keys: { [$("provider").value]: key },
    });
    $("api-key").value = "";
    updateKeyState();
    toast("API 키를 저장했습니다.", "ok");
  } catch (error) {
    toast(error.message, "error");
  }
}

async function saveTmdbKey() {
  const key = $("tmdb-key").value.trim();
  if (!key) {
    toast("TMDB API 키를 입력하세요.", "error");
    return;
  }
  try {
    state.config = await postJson("/api/config", { tmdb_api_key: key });
    $("tmdb-key").value = "";
    updateKeyState();
    toast("TMDB API 키를 저장했습니다.", "ok");
  } catch (error) {
    toast(error.message, "error");
  }
}

async function saveSelectedModel() {
  const model = $("model").value;
  if (!model) return;
  try {
    state.config = await postJson("/api/config", {
      provider: $("provider").value,
      models: { [$("provider").value]: model },
    });
  } catch (error) {
    toast(error.message, "error");
  }
}

async function loadModels() {
  const button = $("load-models");
  const typedKey = $("api-key").value.trim();
  button.disabled = true;
  button.textContent = "불러오는 중...";
  try {
    const data = await postJson("/api/models", {
      provider: $("provider").value,
      api_key: typedKey || null,
    });
    const select = $("model");
    const previous = select.value;
    select.innerHTML = "";
    for (const model of data.models) {
      const option = document.createElement("option");
      option.value = model;
      option.textContent = model;
      select.appendChild(option);
    }
    const saved = (state.config && (state.config.models || {})[$("provider").value]) || previous;
    if (saved && data.models.includes(saved)) select.value = saved;
    toast(`${data.models.length}개 모델을 불러왔습니다.`, "ok");
    await saveSelectedModel();
  } catch (error) {
    toast(error.message, "error");
  } finally {
    button.disabled = false;
    button.textContent = "모델 목록 불러오기";
  }
}

// --- 파일 입력 ------------------------------------------------------------

function setupDropzone() {
  const zone = $("dropzone");
  const input = $("file-input");

  zone.addEventListener("click", () => input.click());
  input.addEventListener("change", () => {
    if (input.files.length) uploadFile(input.files[0]);
    input.value = "";
  });

  ["dragenter", "dragover"].forEach((type) =>
    zone.addEventListener(type, (event) => {
      event.preventDefault();
      zone.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach((type) =>
    zone.addEventListener(type, (event) => {
      event.preventDefault();
      zone.classList.remove("dragover");
    })
  );
  zone.addEventListener("drop", (event) => {
    const files = event.dataTransfer.files;
    if (files && files.length) uploadFile(files[0]);
  });

  // 페이지 전체에서 기본 드롭 동작을 막음
  ["dragover", "drop"].forEach((type) =>
    document.addEventListener(type, (event) => event.preventDefault())
  );
}

async function uploadFile(file) {
  if (!file.name.toLowerCase().endsWith(".srt")) {
    toast("SRT 파일만 사용할 수 있습니다.", "error");
    return;
  }
  const form = new FormData();
  form.append("file", file);
  try {
    const info = await api("/api/upload", { method: "POST", body: form });
    setFile(info);
    toast(`${info.name} 을(를) 불러왔습니다.`, "ok");
  } catch (error) {
    toast(error.message, "error");
  }
}

async function loadLocalFile() {
  const path = $("local-path").value.trim();
  if (!path) {
    toast("경로를 입력하세요.", "error");
    return;
  }
  try {
    const info = await postJson("/api/local-file", { path });
    setFile(info);
    toast(`${info.name} 을(를) 불러왔습니다.`, "ok");
  } catch (error) {
    toast(error.message, "error");
  }
}

function setFile(info) {
  state.file = info;
  $("file-info").classList.remove("hidden");
  $("fi-name").textContent = info.name;
  $("fi-count").textContent = info.subtitle_count;
  $("fi-path").textContent = info.local_path || "업로드 파일 (다운로드로만 저장 가능)";
  if (!$("title").value.trim() && info.title) $("title").value = info.title;
  if (info.is_series) $("is-series").checked = true;
  $("start-index").max = String(info.subtitle_count);
  updateOutputName();
}

function updateOutputName() {
  if (!state.file) return;
  const code = ($("language-code").value.trim() || "ko").replace(/^\.+|\.+$/g, "");
  const stem = state.file.name.replace(/\.[^.]+$/, "").replace(/\.[a-z]{2,3}$/i, "");
  $("fi-output").textContent = `${stem}.${code}.srt`;
}

// --- TMDB -----------------------------------------------------------------

async function searchTmdb() {
  const query = $("title").value.trim();
  if (!query) {
    toast("검색할 제목을 입력하세요.", "error");
    return;
  }
  const button = $("tmdb-search");
  button.disabled = true;
  try {
    const data = await postJson("/api/tmdb/search", {
      query,
      is_series: $("is-series").checked,
      year: state.file ? state.file.year || null : null,
    });
    renderTmdbResults(data.results);
  } catch (error) {
    toast(error.message, "error");
  } finally {
    button.disabled = false;
  }
}

function renderTmdbResults(results) {
  const container = $("tmdb-results");
  container.innerHTML = "";
  if (!results.length) {
    container.classList.add("hidden");
    toast("검색 결과가 없습니다.", "error");
    return;
  }
  container.classList.remove("hidden");

  for (const item of results) {
    const row = document.createElement("div");
    row.className = "tmdb-item";

    const poster = document.createElement("img");
    poster.src = item.poster_url || "data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==";
    poster.alt = "";

    const meta = document.createElement("div");
    meta.className = "meta";
    const title = document.createElement("b");
    title.textContent = `${item.title}${item.year ? ` (${item.year})` : ""}`;
    const overview = document.createElement("p");
    overview.textContent = item.overview || "줄거리 정보 없음";
    meta.append(title, overview);

    row.append(poster, meta);
    row.addEventListener("click", () => selectTmdbItem(item));
    container.appendChild(row);
  }
}

async function selectTmdbItem(item) {
  $("title").value = item.title;
  $("is-series").checked = item.is_series;
  $("tmdb-results").classList.add("hidden");

  // 출연진까지 포함된 상세 정보를 가져와 번역 컨텍스트 후보로 보관함
  let detail = item;
  try {
    detail = await postJson("/api/tmdb/details", {
      tmdb_id: item.id,
      is_series: item.is_series,
    });
  } catch (error) {
    logLine("warning", `TMDB 상세 조회 실패: ${error.message}`);
  }

  state.tmdb = {
    title: detail.title,
    year: detail.year,
    text: buildTmdbContext(detail),
  };
  renderTmdbSelected();
  toast(`작품 정보를 "${item.title}"(으)로 설정했습니다.`, "ok");
}

function buildTmdbContext(detail) {
  const lines = [];
  const heading = `${detail.title}${detail.year ? ` (${detail.year})` : ""}`;
  lines.push(`[작품] ${heading}`);
  if (detail.original_title && detail.original_title !== detail.title) {
    lines.push(`[원제] ${detail.original_title}`);
  }
  if (detail.genres && detail.genres.length) lines.push(`[장르] ${detail.genres.join(", ")}`);
  if (detail.overview) lines.push(`\n[줄거리]\n${detail.overview}`);
  if (detail.cast && detail.cast.length) {
    const cast = detail.cast
      .filter((member) => member.character)
      .map((member) => `- ${member.character} (배우: ${member.name})`)
      .join("\n");
    if (cast) lines.push(`\n[등장인물]\n${cast}`);
  }
  return lines.join("\n");
}

function renderTmdbSelected() {
  const box = $("tmdb-selected");
  if (!state.tmdb) {
    box.classList.add("hidden");
    return;
  }
  box.classList.remove("hidden");
  $("tmdb-selected-title").textContent = `${state.tmdb.title}${
    state.tmdb.year ? ` (${state.tmdb.year})` : ""
  }`;
  $("tmdb-preview-body").textContent = state.tmdb.text;
  $("tmdb-preview-body").classList.add("hidden");
  updateTmdbNote();
}

function updateTmdbNote() {
  if (!state.tmdb) return;
  const manual = $("story-context").value.trim();
  $("tmdb-selected-note").textContent = manual
    ? "줄거리를 직접 입력했으므로 이 TMDB 정보는 번역에 사용되지 않습니다."
    : "줄거리 입력란이 비어 있으므로 이 TMDB 정보가 번역 컨텍스트로 사용됩니다.";
}

function clearTmdb() {
  state.tmdb = null;
  $("tmdb-selected").classList.add("hidden");
}

// --- 번역 실행 ------------------------------------------------------------

function collectRequest() {
  const startIndex = Math.max(1, Number($("start-index").value || 1)) - 1;
  return {
    file_id: state.file.file_id,
    provider: $("provider").value,
    model: $("model").value,
    title: $("title").value.trim(),
    is_series: $("is-series").checked,
    story_context: $("story-context").value,
    // 서버는 story_context가 비어 있을 때만 이 값을 사용함
    tmdb_context: state.tmdb ? state.tmdb.text : "",
    extra_instruction: $("extra-instruction").value,
    batch_size: Number($("batch-size").value || 300),
    temperature: numberOrNull($("temperature")),
    top_p: numberOrNull($("top-p")),
    top_k: numberOrNull($("top-k")),
    thinking: $("thinking").checked,
    thinking_budget: Number($("thinking-budget").value || 2048),
    streaming: $("streaming").checked,
    language_code: $("language-code").value.trim() || "ko",
    start_index: startIndex,
    save_to_source_dir: $("save-to-source").checked,
  };
}

async function persistSettings(request) {
  try {
    state.config = await postJson("/api/config", {
      provider: request.provider,
      models: { [request.provider]: request.model },
      batch_size: request.batch_size,
      temperature: request.temperature,
      top_p: request.top_p,
      top_k: request.top_k,
      thinking: request.thinking,
      thinking_budget: request.thinking_budget,
      streaming: request.streaming,
      language_code: request.language_code,
      story_context: request.story_context,
      extra_instruction: request.extra_instruction,
    });
  } catch (error) {
    // 설정 저장 실패가 번역을 막지는 않음
    logLine("warning", `설정 저장 실패: ${error.message}`);
  }
}

async function startTranslation() {
  if (state.running) return;
  if (!state.file) {
    toast("먼저 자막 파일을 불러오세요.", "error");
    return;
  }
  if (!$("model").value) {
    toast("모델을 선택하세요. (API 설정 > 모델 목록 불러오기)", "error");
    return;
  }
  const keysSet = (state.config && state.config.api_keys_set) || {};
  if (!keysSet[$("provider").value]) {
    toast("API 키를 먼저 저장하세요.", "error");
    return;
  }
  if (!$("story-context").value.trim() && !state.tmdb) {
    const proceed = window.confirm(
      "상세 줄거리 및 등장인물 정보가 비어 있고, 선택된 TMDB 작품 정보도 없습니다.\n" +
        "이 정보가 없으면 호칭·어투 일관성이 떨어질 수 있습니다.\n\n그대로 진행할까요?"
    );
    if (!proceed) return;
  }

  const request = collectRequest();
  await persistSettings(request);

  try {
    const result = await postJson("/api/translate", request);
    state.jobId = result.job_id;
    setRunning(true);
    $("download").classList.add("hidden");
    $("console").innerHTML = "";
    logLine("info", `번역을 시작합니다. 총 ${result.total}줄 → ${result.output_name}`);
    connectEvents(result.job_id);
  } catch (error) {
    toast(error.message, "error");
  }
}

async function cancelTranslation() {
  if (!state.jobId) return;
  try {
    await postJson(`/api/jobs/${state.jobId}/cancel`, {});
    $("cancel").disabled = true;
  } catch (error) {
    toast(error.message, "error");
  }
}

function connectEvents(jobId) {
  if (state.eventSource) state.eventSource.close();
  const source = new EventSource(`/api/jobs/${jobId}/events`);
  state.eventSource = source;

  source.addEventListener("snapshot", (event) => {
    const data = JSON.parse(event.data);
    updateProgress(data.done, data.total);
    for (const entry of data.logs || []) logLine(entry.level, entry.message, entry.time);
  });

  source.addEventListener("log", (event) => {
    const data = JSON.parse(event.data);
    logLine(data.level, data.message, data.time);
  });

  source.addEventListener("progress", (event) => {
    const data = JSON.parse(event.data);
    updateProgress(data.done, data.total);
  });

  source.addEventListener("status", (event) => {
    const data = JSON.parse(event.data);
    handleStatus(data);
  });

  source.addEventListener("end", () => {
    source.close();
    state.eventSource = null;
  });

  source.onerror = () => {
    if (state.running) logLine("warning", "이벤트 연결이 끊겼습니다. 재연결을 시도합니다.");
  };
}

function updateProgress(done, total) {
  const ratio = total > 0 ? Math.min(100, (done / total) * 100) : 0;
  $("progress-bar").style.width = `${ratio}%`;
  $("progress-text").textContent = total > 0 ? `${done} / ${total} (${ratio.toFixed(1)}%)` : "대기 중";
}

function handleStatus(data) {
  if (data.status === "running") {
    setRunning(true);
    return;
  }
  if (["completed", "failed", "cancelled"].includes(data.status)) {
    setRunning(false);
    if (data.status === "completed") {
      toast(
        data.output_path ? `저장 완료: ${data.output_path}` : "번역이 완료되었습니다.",
        "ok"
      );
      if (data.has_result) $("download").classList.remove("hidden");
    } else if (data.status === "failed") {
      toast(`번역 실패: ${data.error || "알 수 없는 오류"}`, "error");
      if (data.has_result) $("download").classList.remove("hidden");
    } else {
      toast("번역이 취소되었습니다.");
    }
  }
}

function setRunning(running) {
  state.running = running;
  $("start").classList.toggle("hidden", running);
  $("cancel").classList.toggle("hidden", !running);
  $("cancel").disabled = false;
}

init();
