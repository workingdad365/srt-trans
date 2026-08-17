"use strict";

const $ = (id) => document.getElementById(id);

// 추론 강도 표시 라벨 (낮은 것부터)
const EFFORT_LABELS = {
  none: "사용 안 함 (none)",
  minimal: "최소 (minimal)",
  low: "낮음 (low)",
  medium: "보통 (medium)",
  high: "높음 (high)",
  xhigh: "매우 높음 (xhigh)",
  max: "최대 (max)",
};

const state = {
  providers: [],
  config: null,
  file: null,
  jobId: null,
  eventSource: null,
  running: false,
  // TMDB에서 가져온 줄거리/출연진 정보. 직접 입력한 줄거리가 없을 때만 사용됨
  tmdb: null,
  // 현재 선택한 모델이 지원하는 파라미터 정보
  capabilities: null,
  // 마지막으로 불러온 전체 모델 목록 (검색 필터용)
  models: [],
  modelsLoaded: false,
  // OpenRouter 제공자 목록과 선택 순서
  endpoints: [],
  selectedProviders: [],
  lastModel: "",
  // 사용자가 직접 고른 추론 강도 (자동 대체된 값과 구분하기 위함)
  userEffort: "",
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
  element.className = "toast hidden";
  // 같은 종류가 연달아 떠도 등장 애니메이션이 다시 재생되도록 리플로우를 유발함
  void element.offsetWidth;
  element.className = `toast ${kind}`;
  clearTimeout(toastTimer);
  // 오류는 놓치기 쉬우므로 더 오래 표시함
  toastTimer = setTimeout(() => element.classList.add("hidden"), kind === "error" ? 8000 : 4000);
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

/**
 * 현재 선택된 프로바이더에 맞춰 화면을 갱신함.
 * 주의: 여기서 provider select의 값을 바꾸면 사용자의 선택을 되돌리게 되므로 건드리지 않음.
 */
function applyProviderView() {
  const provider = currentProvider();
  if (provider) {
    const link = $("api-key-link");
    link.href = provider.api_key_url || "#";
    link.style.display = provider.api_key_url ? "" : "none";
  }

  updateKeyState();

  // 프로바이더가 바뀌면 이전 프로바이더의 모델 목록이 남지 않도록 초기화함
  const providerId = $("provider").value;
  const model = ((state.config && state.config.models) || {})[providerId] || "";
  state.models = model ? [model] : [];
  state.modelsLoaded = false;
  $("model-filter").value = "";
  renderModelOptions(model);

  refreshCapabilities();
}

/** 사용자가 프로바이더를 바꿨을 때 처리 */
async function onProviderChanged() {
  const providerId = $("provider").value;
  $("api-key").value = "";
  applyProviderView();

  // 선택한 프로바이더를 바로 저장해 다음 실행에도 유지되게 함
  try {
    state.config = await postJson("/api/config", { provider: providerId });
    applyProviderView();
  } catch (error) {
    logLine("warning", `프로바이더 저장 실패: ${error.message}`);
  }
}

function applyConfig(config) {
  if (config.provider) $("provider").value = config.provider;
  applyProviderView();

  $("batch-size").value = config.batch_size ?? 300;
  $("language-code").value = config.language_code || "ko";
  $("thinking").checked = config.thinking !== false;
  $("streaming").checked = config.streaming !== false;
  $("timeout").value = config.timeout ?? 600;
  $("strip-period").checked = config.strip_trailing_period !== false;
  $("thinking-budget").value = config.thinking_budget ?? 2048;
  if (config.temperature !== null && config.temperature !== undefined) {
    $("temperature").value = config.temperature;
  }
  if (config.top_p !== null && config.top_p !== undefined) $("top-p").value = config.top_p;
  if (config.top_k !== null && config.top_k !== undefined) $("top-k").value = config.top_k;
  // 줄거리/등장인물 정보는 작품마다 다르므로 저장/복원하지 않음
  if (config.extra_instruction) $("extra-instruction").value = config.extra_instruction;
  applyRouting(config.routing);
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

  $("provider").addEventListener("change", onProviderChanged);

  $("toggle-api-key").addEventListener("click", () => togglePassword("api-key"));
  $("toggle-tmdb-key").addEventListener("click", () => togglePassword("tmdb-key"));
  $("save-api-key").addEventListener("click", saveApiKey);
  $("save-tmdb-key").addEventListener("click", saveTmdbKey);
  $("load-models").addEventListener("click", loadModels);
  $("model").addEventListener("change", saveSelectedModel);
  $("model-filter").addEventListener("input", () => {
    renderModelOptions();
    saveSelectedModel();
  });
  $("reasoning-effort").addEventListener("change", () => {
    state.userEffort = $("reasoning-effort").value;
  });
  $("load-endpoints").addEventListener("click", loadEndpoints);
  $("clear-providers").addEventListener("click", () => {
    state.selectedProviders = [];
    renderEndpoints();
  });

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
  // 어느 프로바이더에 저장되는지 명시해 잘못 저장하는 실수를 막음
  const providerId = $("provider").value;
  const label = (currentProvider() || {}).label || providerId;

  try {
    state.config = await postJson("/api/config", {
      provider: providerId,
      api_keys: { [providerId]: key },
    });
    $("api-key").value = "";
    updateKeyState();
    toast(`${label} API 키를 저장했습니다.`, "ok");
    logLine("success", `${label}(${providerId}) API 키를 저장했습니다.`);
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

  // 모델이 바뀌면 이전 모델의 제공자 목록은 의미가 없음
  if (state.lastModel && state.lastModel !== model) {
    state.endpoints = [];
    state.selectedProviders = [];
    renderEndpoints();
  }
  state.lastModel = model;

  try {
    state.config = await postJson("/api/config", {
      provider: $("provider").value,
      models: { [$("provider").value]: model },
    });
  } catch (error) {
    toast(error.message, "error");
  }
  await refreshCapabilities();
}

/**
 * 검색어에 맞는 모델만 드롭다운에 채움.
 * OpenRouter처럼 모델이 수백 개인 경우를 위해 필요함.
 */
function renderModelOptions(preferred) {
  const keyword = $("model-filter").value.trim().toLowerCase();
  const words = keyword.split(/\s+/).filter(Boolean);
  const matched = words.length
    ? state.models.filter((model) => {
        const lowered = model.toLowerCase();
        return words.every((word) => lowered.includes(word));
      })
    : state.models;

  const select = $("model");
  const current = preferred || select.value;
  select.innerHTML = "";

  if (!matched.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = state.models.length ? "검색 결과 없음" : "모델 목록을 불러오세요";
    select.appendChild(option);
  } else {
    for (const model of matched) {
      const option = document.createElement("option");
      option.value = model;
      option.textContent = model;
      select.appendChild(option);
    }
    select.value = current && matched.includes(current) ? current : matched[0];
  }

  const count = $("model-count");
  if (!state.modelsLoaded) {
    count.textContent = "";
  } else if (words.length) {
    count.textContent = `${matched.length} / ${state.models.length}개`;
  } else {
    count.textContent = `${state.models.length}개`;
  }
}

// --- OpenRouter 라우팅 ----------------------------------------------------

function renderRouteVariants(variants) {
  const select = $("route-variant");
  const previous = select.value;
  select.innerHTML = "";
  const base = document.createElement("option");
  base.value = "";
  base.textContent = "기본 (가격·속도 균형)";
  select.appendChild(base);
  for (const variant of variants || []) {
    const option = document.createElement("option");
    option.value = variant.value;
    option.textContent = variant.label;
    select.appendChild(option);
  }
  const saved = savedRouting().route_variant || previous || "";
  select.value = Array.from(select.options).some((o) => o.value === saved) ? saved : "";
}

function savedRouting() {
  return (state.config && state.config.routing) || {};
}

function formatPrice(value) {
  const price = Number(value);
  if (!Number.isFinite(price) || price <= 0) return "무료";
  // 응답은 토큰당 단가라 100만 토큰 기준으로 환산함
  return `$${(price * 1e6).toFixed(2)}/M`;
}

async function loadEndpoints() {
  const model = $("model").value;
  if (!model) {
    toast("먼저 모델을 선택하세요.", "error");
    return;
  }
  const button = $("load-endpoints");
  button.disabled = true;
  button.textContent = "불러오는 중...";
  try {
    const data = await postJson("/api/model-endpoints", {
      provider: $("provider").value,
      model,
    });
    state.endpoints = data.endpoints || [];
    // 목록에 없는 제공자는 선택에서 제거함
    const available = new Set(state.endpoints.map((e) => e.tag));
    state.selectedProviders = state.selectedProviders.filter((tag) => available.has(tag));
    renderEndpoints();
    if (!state.endpoints.length) {
      toast("이 모델의 제공자 정보를 찾지 못했습니다.", "error");
    } else {
      toast(`제공자 ${state.endpoints.length}곳을 불러왔습니다.`, "ok");
    }
  } catch (error) {
    toast(error.message, "error");
  } finally {
    button.disabled = false;
    button.textContent = "제공자 목록 불러오기";
  }
}

function renderEndpoints() {
  const container = $("provider-list");
  container.innerHTML = "";
  if (!state.endpoints.length) {
    container.classList.add("hidden");
    return;
  }
  container.classList.remove("hidden");

  for (const endpoint of state.endpoints) {
    const order = state.selectedProviders.indexOf(endpoint.tag);
    const row = document.createElement("label");
    row.className = `provider-item${order >= 0 ? " selected" : ""}`;

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = order >= 0;
    checkbox.addEventListener("change", () => toggleProvider(endpoint.tag));

    const meta = document.createElement("div");
    meta.className = "meta";
    const title = document.createElement("b");
    title.textContent = `${endpoint.provider_name} · ${endpoint.tag}`;

    const sub = document.createElement("div");
    sub.className = "sub";
    const parts = [
      `입력 ${formatPrice(endpoint.prompt_price)}`,
      `출력 ${formatPrice(endpoint.completion_price)}`,
    ];
    if (endpoint.context_length) parts.push(`컨텍스트 ${endpoint.context_length.toLocaleString()}`);
    if (endpoint.max_completion_tokens) {
      parts.push(`출력한도 ${endpoint.max_completion_tokens.toLocaleString()}`);
    }
    if (typeof endpoint.uptime_last_30m === "number") {
      parts.push(`가동률 ${endpoint.uptime_last_30m.toFixed(1)}%`);
    }
    for (const text of parts) {
      const span = document.createElement("span");
      span.textContent = text;
      sub.appendChild(span);
    }
    if (!endpoint.supports_structured_outputs) {
      const warn = document.createElement("span");
      warn.className = "badge-warn";
      warn.textContent = "구조화 출력 미지원";
      sub.appendChild(warn);
    }

    meta.append(title, sub);
    row.append(checkbox, meta);

    if (order >= 0) {
      const badge = document.createElement("span");
      badge.className = "order-badge";
      badge.textContent = String(order + 1);
      row.appendChild(badge);
    }
    container.appendChild(row);
  }
}

function toggleProvider(tag) {
  const index = state.selectedProviders.indexOf(tag);
  if (index >= 0) state.selectedProviders.splice(index, 1);
  else state.selectedProviders.push(tag);
  renderEndpoints();
}

function collectRouting() {
  return {
    route_variant: $("route-variant").value || "",
    providers: state.selectedProviders.slice(),
    allow_fallbacks: $("allow-fallbacks").checked,
    deny_data_collection: $("deny-data-collection").checked,
  };
}

function applyRouting(routing) {
  const saved = routing || {};
  state.selectedProviders = Array.isArray(saved.providers) ? saved.providers.slice() : [];
  $("allow-fallbacks").checked = saved.allow_fallbacks !== false;
  $("deny-data-collection").checked = Boolean(saved.deny_data_collection);
  renderEndpoints();
}

// --- 모델별 지원 파라미터 -------------------------------------------------

async function refreshCapabilities() {
  const model = $("model").value;
  if (!model) {
    state.capabilities = null;
    applyCapabilities(null);
    return;
  }
  try {
    state.capabilities = await postJson("/api/model-info", {
      provider: $("provider").value,
      model,
    });
  } catch (error) {
    state.capabilities = null;
    logLine("warning", `모델 정보 조회 실패: ${error.message}`);
  }
  applyCapabilities(state.capabilities);
}

function setFieldEnabled(wrapperId, inputId, enabled, reason) {
  const wrapper = $(wrapperId);
  const input = $(inputId);
  if (!wrapper || !input) return;
  input.disabled = !enabled;
  wrapper.style.opacity = enabled ? "" : "0.45";
  wrapper.title = enabled ? "" : reason || "이 모델에서는 사용할 수 없습니다";
}

function applyCapabilities(caps) {
  // 정보를 못 받았으면 모든 입력란을 열어 둠
  const unknown = !caps;
  const reason = "선택한 모델이 지원하지 않는 설정입니다";

  // OpenRouter에서만 라우팅 설정을 보여 줌
  const routingOn = Boolean(caps && caps.supports_routing);
  $("routing-box").classList.toggle("hidden", !routingOn);
  if (routingOn) renderRouteVariants(caps.route_variants);

  setFieldEnabled("wrap-temperature", "temperature", unknown || caps.temperature, reason);
  setFieldEnabled("wrap-top-p", "top-p", unknown || caps.top_p, reason);
  setFieldEnabled("wrap-top-k", "top-k", unknown || caps.top_k, reason);

  const control = unknown ? null : caps.thinking_control;
  const budgetOn = unknown || control === "budget";
  const effortOn = !unknown && control === "effort";

  setFieldEnabled("wrap-thinking-budget", "thinking-budget", budgetOn, reason);
  $("wrap-thinking-budget").classList.toggle("hidden", effortOn);

  $("wrap-reasoning-effort").classList.toggle("hidden", !effortOn);
  if (effortOn) {
    const select = $("reasoning-effort");
    // 사용자가 직접 고른 값만 유지함. 모델이 지원하지 않아 자동 대체된 값은
    // 다음 모델에서 다시 기본값(최저 단계)으로 돌아가야 함
    const preferred = state.userEffort || (state.config && state.config.reasoning_effort) || "";
    select.innerHTML = "";
    for (const choice of caps.effort_choices) {
      const option = document.createElement("option");
      option.value = choice;
      option.textContent = EFFORT_LABELS[choice] || choice;
      select.appendChild(option);
    }
    // 선택지는 낮은 단계부터 정렬되어 오므로 첫 번째가 최저 단계임.
    // 추론을 끌 수 없는 모델이면 자연히 그다음 낮은 단계가 선택됨
    const lowest = caps.effort_choices[0];
    select.value = caps.effort_choices.includes(preferred) ? preferred : lowest;
    select.disabled = false;
  }

  // Thinking 사용 체크박스는 켜고 끌 수 있는 모델에서만 의미가 있음
  const thinkingToggle = unknown || control === "budget" || control === "on_off";
  $("thinking").disabled = !thinkingToggle;
  $("wrap-thinking").style.opacity = thinkingToggle ? "" : "0.45";

  $("model-notes").textContent = unknown ? "" : (caps.notes || []).join(" ");
}

async function loadModels() {
  const button = $("load-models");
  const typedKey = $("api-key").value.trim();
  const providerId = $("provider").value;
  const label = (currentProvider() || {}).label || providerId;
  button.disabled = true;
  button.textContent = "불러오는 중...";
  try {
    const data = await postJson("/api/models", {
      provider: providerId,
      api_key: typedKey || null,
    });
    if (!data.models.length) {
      toast(`${label}에서 사용 가능한 모델을 찾지 못했습니다.`, "error");
      logLine("warning", `${label}: 모델 목록이 비어 있습니다.`);
      return;
    }
    state.models = data.models;
    state.modelsLoaded = true;
    const previous = $("model").value;
    const saved = (state.config && (state.config.models || {})[providerId]) || previous;
    renderModelOptions(saved);
    toast(`${label} 모델 ${data.models.length}개를 불러왔습니다.`, "ok");
    logLine("success", `${label}: 모델 ${data.models.length}개 조회 완료`);
    await saveSelectedModel();
  } catch (error) {
    toast(`${label} 모델 조회 실패: ${error.message}`, "error");
    logLine("error", `${label} 모델 조회 실패: ${error.message}`);
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
  // 다른 파일로 바뀌면 이전 작품의 정보가 남지 않도록 초기화함
  const changed = !state.file || state.file.name !== info.name;

  state.file = info;
  $("file-info").classList.remove("hidden");
  $("fi-name").textContent = info.name;
  $("fi-count").textContent = info.subtitle_count;
  $("fi-path").textContent = info.local_path || "업로드 파일 (다운로드로만 저장 가능)";

  if (changed) {
    const hadContext = $("story-context").value.trim().length > 0;
    $("story-context").value = "";
    $("title").value = info.title || "";
    $("is-series").checked = Boolean(info.is_series);
    $("tmdb-results").classList.add("hidden");
    clearTmdb();
    updateTmdbNote();
    if (hadContext) {
      logLine("info", "다른 자막을 불러와 줄거리 및 등장인물 정보를 비웠습니다.");
    }
  }

  $("start-index").max = String(info.subtitle_count);
  $("start-index").value = "1";
  updateOutputName();
}

// 서버의 normalize_output_code 와 동일한 규칙
const OUTPUT_CODE_ALIASES = { ko: "kor", korean: "kor" };

function normalizeOutputCode(value) {
  const code = (value || "").trim().replace(/^\.+|\.+$/g, "").toLowerCase() || "kor";
  return OUTPUT_CODE_ALIASES[code] || code;
}

function updateOutputName() {
  if (!state.file) return;
  const code = normalizeOutputCode($("language-code").value);
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
    reasoning_effort: $("reasoning-effort").value || null,
    streaming: $("streaming").checked,
    timeout: Number($("timeout").value || 600),
    strip_trailing_period: $("strip-period").checked,
    language_code: $("language-code").value.trim() || "ko",
    start_index: startIndex,
    save_to_source_dir: $("save-to-source").checked,
    routing: collectRouting(),
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
      reasoning_effort: request.reasoning_effort,
      streaming: request.streaming,
      timeout: request.timeout,
      strip_trailing_period: request.strip_trailing_period,
      language_code: request.language_code,
      extra_instruction: request.extra_instruction,
      routing: request.routing,
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
      // 중단 지점까지 번역된 부분이 있으면 받아 갈 수 있게 함
      if (data.has_result) $("download").classList.remove("hidden");
    } else {
      toast("번역이 취소되었습니다.");
      if (data.has_result) $("download").classList.remove("hidden");
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
