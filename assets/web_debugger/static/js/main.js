(() => {
  const VIEW_IDS = ["overview", "analysis", "inspector"];
  const MODE_OPTIONS = [
    { mode: 1, key: "auto_aim", label: "自瞄" },
    { mode: 2, key: "small_buff", label: "小符" },
    { mode: 3, key: "big_buff", label: "大符" },
  ];
  const POLL_STATE_MS = 250;
  const POLL_LOG_MS = 400;

  const STREAM_LAYOUT = {
    overview: [
      {
        imageId: "overview-main-frame",
        placeholderId: "overview-main-placeholder",
        path: "/stream/main.mjpg",
      },
    ],
    analysis: [
      {
        imageId: "analysis-main-frame",
        placeholderId: "analysis-main-placeholder",
        path: "/stream/main.mjpg",
      },
      {
        imageId: "analysis-ballistic-frame",
        placeholderId: "analysis-ballistic-placeholder",
        path: "/stream/ballistic.mjpg",
      },
    ],
    inspector: [],
  };

  const OVERLAY_CONTROL_ITEMS = [
    {
      key: "state_layers",
      label: "状态分层",
      description: "按搜索、跟踪、锁定和可击发阶段裁剪信息层级",
    },
    {
      key: "stabilize",
      label: "抗抖",
      description: "稳定装甲板标签和转向说明的位置",
    },
    {
      key: "armors",
      label: "装甲板",
      description: "显示重投影装甲板轮廓",
    },
    {
      key: "labels",
      label: "标签",
      description: "显示 A0 与 d_yaw 等局部标注",
    },
    {
      key: "target_motion",
      label: "目标转向",
      description: "显示目标中心与 V_yaw 指示",
    },
    {
      key: "aim",
      label: "瞄准提示",
      description: "显示瞄准圈、控制箭头和击发提示",
    },
    {
      key: "decision_hud",
      label: "决策 HUD",
      description: "显示切板、模型与开火理由",
    },
    {
      key: "decision_track",
      label: "切板轴",
      description: "显示各装甲板 delta 角度的决策轴",
    },
    {
      key: "footer",
      label: "底部控制条",
      description: "显示 yaw 与 pitch 的控制输出摘要",
    },
  ];

  let currentView = "overview";
  let lastStateUpdatedAt = 0;
  let overlaySyncPending = false;
  let modeSyncPending = false;
  let runtimeParamSnapshot = null;
  let currentModeState = MODE_OPTIONS[0];

  const streamControllers = new Map();
  const viewScrollPositions = new Map();

  const hasOwn = (obj, key) => Object.prototype.hasOwnProperty.call(obj, key);
  const isFiniteNumber = (value) => typeof value === "number" && Number.isFinite(value);

  const getByPath = (source, path, fallback = undefined) => {
    const value = path.split(".").reduce((acc, key) => {
      if (acc && typeof acc === "object" && hasOwn(acc, key)) return acc[key];
      return undefined;
    }, source);
    return value === undefined || value === null ? fallback : value;
  };

  const formatNumber = (value, digits = 2, suffix = "") =>
    isFiniteNumber(value) ? `${value.toFixed(digits)}${suffix}` : "--";

  const formatSigned = (value, digits = 2, suffix = "") =>
    isFiniteNumber(value) ? `${value >= 0 ? "+" : ""}${value.toFixed(digits)}${suffix}` : "--";

  const formatBool = (value, truthy = "ON", falsy = "OFF") => (value ? truthy : falsy);

  const formatArmorId = (value) =>
    Number.isInteger(value) && value >= 0 ? `A${value}` : "A-";

  const formatClock = (unixMs) => {
    if (!isFiniteNumber(unixMs)) return "--";
    return new Date(unixMs).toLocaleTimeString("zh-CN", { hour12: false });
  };

  const formatDateTime = (unixMs) => {
    if (!isFiniteNumber(unixMs)) return "--";
    return new Date(unixMs).toLocaleString("zh-CN", { hour12: false });
  };

  const formatDeltaList = (values) => {
    if (!Array.isArray(values) || !values.length) return "--";
    return values
      .map((value, index) =>
        `${formatArmorId(index)}:${isFiniteNumber(value) ? `${value >= 0 ? "+" : ""}${value.toFixed(1)} deg` : "--"}`,
      )
      .join("  ");
  };

  const formatParamValue = (value, digits = null) => {
    if (Array.isArray(value)) {
      return value
        .map((item) => formatParamValue(item, digits))
        .join(", ");
    }
    if (typeof value === "boolean") return value ? "true" : "false";
    if (value === null || value === undefined) return "--";
    if (typeof value === "number" && Number.isFinite(value) && Number.isInteger(digits) && digits >= 0) {
      return value.toFixed(digits);
    }
    return String(value);
  };

  const formatRangeText = (item) => {
    const hasMin = Number.isFinite(item?.min);
    const hasMax = Number.isFinite(item?.max);
    if (!hasMin && !hasMax) return null;
    const digits = Number.isInteger(item?.display_precision) ? item.display_precision : null;
    const minText = hasMin ? formatParamValue(item.min, digits) : "-inf";
    const maxText = hasMax ? formatParamValue(item.max, digits) : "+inf";
    return `${minText} ~ ${maxText}`;
  };

  const formatControlValue = (item, value) => {
    const digits = Number.isInteger(item?.display_precision) ? item.display_precision : null;
    return formatParamValue(value, digits);
  };

  const formatPathTail = (value) => {
    if (!value) return "--";
    const normalized = String(value).replace(/\\/g, "/");
    const tail = normalized.split("/").filter(Boolean).pop();
    return tail || normalized;
  };

  const isBuffModeKey = (modeKey) => modeKey === "small_buff" || modeKey === "big_buff";

  const normalizeModePayload = (payload = {}) => {
    const requestedMode = Number(
      getByPath(payload, "mode", getByPath(payload, "current", currentModeState?.mode || 1)),
    );
    const fallback = MODE_OPTIONS.find((item) => item.mode === requestedMode) || MODE_OPTIONS[0];
    return {
      mode: fallback.mode,
      mode_key: getByPath(
        payload,
        "mode_key",
        getByPath(payload, "current_key", fallback.key),
      ),
      mode_label: getByPath(
        payload,
        "mode_label",
        getByPath(payload, "current_label", fallback.label),
      ),
      source: getByPath(payload, "source", "web"),
      serial_mode_key: getByPath(payload, "serial_mode_key", "idle"),
      serial_mode_label: getByPath(payload, "serial_mode_label", "--"),
    };
  };

  const setText = (id, value) => {
    const node = document.getElementById(id);
    if (node) node.textContent = value;
  };

  const applyModeState = (payload) => {
    currentModeState = normalizeModePayload(payload);
    document.querySelectorAll(".mode-btn").forEach((button) => {
      button.classList.toggle("active", Number(button.dataset.mode) === currentModeState.mode);
    });
    setText("status-mode", currentModeState.mode_label);
  };

  const setFireBadge = (active) => {
    const node = document.getElementById("status-fire");
    if (!node) return;
    node.textContent = active ? "FIRE" : "SAFE";
    node.classList.toggle("fire", !!active);
  };

  const renderRows = (containerId, items) => {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = "";

    const visibleItems = items.filter((item) => item && item.value !== undefined);
    if (!visibleItems.length) {
      const empty = document.createElement("div");
      empty.className = "empty-hint";
      empty.textContent = "等待状态数据";
      container.appendChild(empty);
      return;
    }

    visibleItems.forEach((item) => {
      const row = document.createElement("div");
      row.className = "row-item";

      const label = document.createElement("span");
      label.textContent = item.label;

      const value = document.createElement("strong");
      value.textContent = item.value;

      row.appendChild(label);
      row.appendChild(value);
      container.appendChild(row);
    });
  };

  const setOverlayMeta = (text, isError = false) => {
    const node = document.getElementById("overlay-config-meta");
    if (!node) return;
    node.textContent = text;
    node.classList.toggle("overlay-sync-bad", isError);
  };

  const setRuntimeParamStatus = (text, isError = false) => {
    const node = document.getElementById("param-status-banner");
    if (!node) return;
    node.textContent = text;
    node.classList.toggle("param-status-bad", isError);
  };

  const setRuntimeParamMeta = (text) => {
    const node = document.getElementById("param-session-meta");
    if (!node) return;
    node.textContent = text;
  };

  const getViewNode = (viewId) => document.getElementById(`view-${viewId}`);

  const rememberViewScroll = (viewId) => {
    const node = getViewNode(viewId);
    if (!node) return;
    viewScrollPositions.set(viewId, node.scrollTop);
  };

  const restoreViewScroll = (viewId) => {
    const node = getViewNode(viewId);
    if (!node) return;
    node.scrollTop = viewScrollPositions.get(viewId) || 0;
  };

  const createParamBadge = (label, value, extraClass = "") => {
    const badge = document.createElement("div");
    badge.className = "param-badge";
    if (extraClass) badge.classList.add(extraClass);

    const badgeLabel = document.createElement("span");
    badgeLabel.textContent = label;

    const badgeValue = document.createElement("strong");
    badgeValue.textContent = value;

    badge.appendChild(badgeLabel);
    badge.appendChild(badgeValue);
    return badge;
  };

  const parseRuntimeParamInput = (item, control) => {
    if (!item || !control) throw new Error("参数控件不存在");

    if (item.type === "boolean") return !!control.checked;
    if (item.type === "enum") return control.value;

    if (item.type === "integer") {
      const value = Number(control.value);
      if (!Number.isFinite(value)) throw new Error(`${item.label} 需要整数`);
      if (Number.isFinite(item.min) && value < item.min) {
        throw new Error(`${item.label} 不能小于 ${item.min}`);
      }
      if (Number.isFinite(item.max) && value > item.max) {
        throw new Error(`${item.label} 不能大于 ${item.max}`);
      }
      return Math.round(value);
    }

    if (item.type === "number") {
      const value = Number(control.value);
      if (!Number.isFinite(value)) throw new Error(`${item.label} 需要数字`);
      if (Number.isFinite(item.min) && value < item.min) {
        throw new Error(`${item.label} 不能小于 ${item.min}`);
      }
      if (Number.isFinite(item.max) && value > item.max) {
        throw new Error(`${item.label} 不能大于 ${item.max}`);
      }
      return value;
    }

    if (item.type === "number_array") {
      const values = String(control.value)
        .split(/[\s,]+/)
        .map((token) => token.trim())
        .filter(Boolean)
        .map((token) => Number(token));
      if (!values.length || values.some((value) => !Number.isFinite(value))) {
        throw new Error(`${item.label} 需要逗号分隔的数字列表`);
      }
      return values;
    }

    throw new Error(`未知参数类型: ${item.type}`);
  };

  const renderRuntimeParams = (payload) => {
    runtimeParamSnapshot = payload;
    const groupsHost = document.getElementById("param-groups");
    const navHost = document.getElementById("param-group-nav");
    const exportNode = document.getElementById("param-export-text");
    if (!groupsHost || !exportNode || !navHost) return;

    groupsHost.innerHTML = "";
    navHost.innerHTML = "";
    exportNode.value = payload?.export_yaml || "# 当前还没有网页改过的参数";
    setText("param-config-brief", formatPathTail(payload?.config_path));
    setText("param-override-count", `${payload?.override_count || 0} 项`);
    setText("param-last-update", formatDateTime(payload?.last_update_unix_ms));
    setText("param-version", `v${payload?.version || 0}`);

    if (!payload?.enabled) {
      setRuntimeParamStatus(payload?.error || "当前入口没有启用运行时参数热调", true);
      setRuntimeParamMeta(payload?.config_path || "runtime parameter session unavailable");
      renderRows("param-session-card", [
        { label: "会话状态", value: "未绑定" },
        { label: "当前配置", value: formatPathTail(payload?.config_path) },
      ]);
      const empty = document.createElement("div");
      empty.className = "empty-hint";
      empty.textContent = "当前进程没有绑定运行时参数会话";
      groupsHost.appendChild(empty);
      return;
    }

    setRuntimeParamStatus(
      `当前覆盖 ${payload.override_count || 0} 项参数，改动已实时写入会话日志与最新快照。`,
    );
    setRuntimeParamMeta(
      `配置文件\n${payload.config_path}\n\n会话日志\n${payload.session_log_path}\n\n最新快照\n${payload.snapshot_path}`,
    );
    renderRows("param-session-card", [
      { label: "配置", value: formatPathTail(payload.config_path) },
      { label: "日志", value: formatPathTail(payload.session_log_path) },
      { label: "快照", value: formatPathTail(payload.snapshot_path) },
      {
        label: "回填片段",
        value:
          payload.export_yaml && payload.export_yaml.trim()
            ? `${payload.export_yaml.split("\n").length} 行`
            : "暂无覆盖",
      },
    ]);

    (payload.groups || []).forEach((group, groupIndex) => {
      const overriddenCount = (group.items || []).filter((item) => !!item.overridden).length;
      const jumpButton = document.createElement("button");
      jumpButton.type = "button";
      jumpButton.className = "param-group-jump";
      jumpButton.classList.toggle("has-overrides", overriddenCount > 0);
      jumpButton.addEventListener("click", () => {
        document.getElementById(`param-group-${groupIndex}`)?.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      });

      const jumpMeta = document.createElement("span");
      jumpMeta.textContent = `${(group.items || []).length} 项${overriddenCount ? ` · 覆盖 ${overriddenCount}` : ""}`;
      const jumpLabel = document.createElement("strong");
      jumpLabel.textContent = group.label;
      jumpButton.appendChild(jumpMeta);
      jumpButton.appendChild(jumpLabel);
      navHost.appendChild(jumpButton);

      const section = document.createElement("section");
      section.className = "param-group";
      section.id = `param-group-${groupIndex}`;

      const head = document.createElement("div");
      head.className = "param-group-head";

      const title = document.createElement("h3");
      title.textContent = group.label;
      const meta = document.createElement("span");
      meta.className = "panel-meta";
      meta.textContent =
        overriddenCount > 0 ?
          `${(group.items || []).length} 项 · 覆盖 ${overriddenCount}` :
          `${(group.items || []).length} 项`;

      head.appendChild(title);
      head.appendChild(meta);
      section.appendChild(head);

      const list = document.createElement("div");
      list.className = "param-grid";

      (group.items || []).forEach((item) => {
        const row = document.createElement("div");
        row.className = "param-row";
        row.classList.toggle("is-overridden", !!item.overridden);

        const rowHead = document.createElement("div");
        rowHead.className = "param-row-head";

        const titleWrap = document.createElement("div");
        titleWrap.className = "param-row-title";
        const strong = document.createElement("strong");
        strong.textContent = item.label;
        const desc = document.createElement("span");
        desc.textContent = item.description || item.key;
        titleWrap.appendChild(strong);
        titleWrap.appendChild(desc);

        const status = document.createElement("div");
        status.className = "param-row-meta";
        status.textContent = item.overridden ? "运行时覆盖" : "YAML 基线";

        rowHead.appendChild(titleWrap);
        rowHead.appendChild(status);

        const badges = document.createElement("div");
        badges.className = "param-badges";
        badges.appendChild(createParamBadge("Key", item.key));
        badges.appendChild(
          createParamBadge(
            "当前",
            `${formatParamValue(item.value, item.display_precision)}${item.unit ? ` ${item.unit}` : ""}`,
          ),
        );
        badges.appendChild(
          createParamBadge(
            "基线",
            `${formatParamValue(item.base_value, item.display_precision)}${item.unit ? ` ${item.unit}` : ""}`,
          ),
        );
        const rangeText = formatRangeText(item);
        if (rangeText) badges.appendChild(createParamBadge("范围", rangeText));
        if (item.overridden) {
          badges.appendChild(createParamBadge("状态", "网页覆盖", "is-hot"));
        }

        const actions = document.createElement("div");
        actions.className = "param-actions";
        const controlWrap = document.createElement("div");
        controlWrap.className = "param-control-wrap";

        let control = null;
        if (item.type === "boolean") {
          actions.classList.add("is-boolean");
          control = document.createElement("input");
          control.type = "checkbox";
          control.className = "param-checkbox";
          control.checked = !!item.value;
          const toggle = document.createElement("label");
          toggle.className = "param-toggle";
          const toggleText = document.createElement("span");
          const syncToggleText = () => {
            toggleText.textContent = control.checked ? "当前: 开启" : "当前: 关闭";
          };
          control.addEventListener("change", syncToggleText);
          syncToggleText();
          toggle.appendChild(control);
          toggle.appendChild(toggleText);
          controlWrap.appendChild(toggle);
        } else if (item.type === "enum") {
          actions.classList.add("is-scalar");
          control = document.createElement("select");
          (item.choices || []).forEach((choice) => {
            const option = document.createElement("option");
            option.value = choice;
            option.textContent = choice;
            option.selected = choice === item.value;
            control.appendChild(option);
          });
          controlWrap.appendChild(control);
        } else if (item.type === "number_array") {
          actions.classList.add("is-array");
          control = document.createElement("textarea");
          control.rows = 2;
          control.value = formatParamValue(item.value, item.display_precision);
          control.placeholder = "例如: 3e6, 0.3";
          controlWrap.appendChild(control);
        } else {
          actions.classList.add("is-scalar");
          control = document.createElement("input");
          control.type = "number";
          control.step =
            Number.isFinite(item.step) ? String(item.step) : item.type === "integer" ? "1" : "any";
          if (Number.isFinite(item.min)) control.min = String(item.min);
          if (Number.isFinite(item.max)) control.max = String(item.max);
          control.value = formatControlValue(item, item.value);
          controlWrap.appendChild(control);
        }

        const applyBtn = document.createElement("button");
        applyBtn.type = "button";
        applyBtn.textContent = "应用";
        applyBtn.addEventListener("click", async () => {
          try {
            setRuntimeParamStatus(`正在应用 ${item.label}`);
            const value = parseRuntimeParamInput(item, control);
            const response = await fetch("/api/params", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ updates: { [item.key]: value } }),
            });
            const payload = await response.json();
            if (!response.ok) throw new Error(payload?.error || response.statusText);
            renderRuntimeParams(payload);
            setRuntimeParamStatus(`${item.label} 已应用到当前运行链路`);
          } catch (error) {
            setRuntimeParamStatus(`${item.label} 应用失败: ${error.message}`, true);
            console.warn("apply runtime param failed", item.key, error);
          }
        });

        const resetBtn = document.createElement("button");
        resetBtn.type = "button";
        resetBtn.className = "ghost-btn";
        resetBtn.textContent = "恢复";
        resetBtn.addEventListener("click", async () => {
          try {
            setRuntimeParamStatus(`正在恢复 ${item.label}`);
            const response = await fetch("/api/params/reset", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ keys: [item.key] }),
            });
            const payload = await response.json();
            if (!response.ok) throw new Error(payload?.error || response.statusText);
            renderRuntimeParams(payload);
            setRuntimeParamStatus(`${item.label} 已恢复为 YAML 基线值`);
          } catch (error) {
            setRuntimeParamStatus(`${item.label} 恢复失败: ${error.message}`, true);
            console.warn("reset runtime param failed", item.key, error);
          }
        });

        actions.appendChild(controlWrap);
        actions.appendChild(applyBtn);
        actions.appendChild(resetBtn);

        row.appendChild(rowHead);
        row.appendChild(badges);
        row.appendChild(actions);
        list.appendChild(row);
      });

      section.appendChild(list);
      groupsHost.appendChild(section);
    });
  };

  const fetchRuntimeParams = async (quiet = false) => {
    if (!quiet) setRuntimeParamStatus("正在同步运行时参数");
    try {
      const response = await fetch(`/api/params?ts=${Date.now()}`, { cache: "no-store" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload?.error || response.statusText);
      renderRuntimeParams(payload);
      if (!quiet) setRuntimeParamStatus("运行时参数已同步");
    } catch (error) {
      setRuntimeParamStatus(`运行时参数同步失败: ${error.message}`, true);
      console.warn("fetch /api/params failed", error);
    }
  };

  const bindRuntimeParamToolbar = () => {
    document.getElementById("refresh-params-btn")?.addEventListener("click", () => {
      fetchRuntimeParams().catch((error) => console.warn(error));
    });

    document.getElementById("reset-all-params-btn")?.addEventListener("click", async () => {
      try {
        setRuntimeParamStatus("正在恢复全部运行时覆盖");
        const response = await fetch("/api/params/reset", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ keys: [] }),
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload?.error || response.statusText);
        renderRuntimeParams(payload);
        setRuntimeParamStatus("所有参数已恢复为 YAML 基线值");
      } catch (error) {
        setRuntimeParamStatus(`恢复全部参数失败: ${error.message}`, true);
        console.warn("reset all runtime params failed", error);
      }
    });
  };

  const fetchMode = async (quiet = false) => {
    try {
      const response = await fetch(`/api/mode?ts=${Date.now()}`, { cache: "no-store" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload?.error || response.statusText);
      applyModeState(payload);
      return payload;
    } catch (error) {
      if (!quiet) console.warn("fetch /api/mode failed", error);
      throw error;
    }
  };

  const pushMode = async (mode) => {
    modeSyncPending = true;
    try {
      const response = await fetch("/api/mode", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload?.error || response.statusText);
      applyModeState(payload);
    } catch (error) {
      console.warn("post /api/mode failed", error);
    } finally {
      modeSyncPending = false;
    }
  };

  const bindModeSwitch = () => {
    document.querySelectorAll(".mode-btn").forEach((button) => {
      button.addEventListener("click", () => {
        const nextMode = Number(button.dataset.mode);
        if (!Number.isFinite(nextMode) || nextMode === currentModeState.mode) return;
        pushMode(nextMode).catch((error) => console.warn(error));
      });
    });
  };

  const initOverlayControls = () => {
    const container = document.getElementById("overlay-control-grid");
    if (!container) return;
    container.innerHTML = "";

    OVERLAY_CONTROL_ITEMS.forEach((item) => {
      const label = document.createElement("label");
      label.className = "overlay-toggle";

      const input = document.createElement("input");
      input.type = "checkbox";
      input.id = `overlay-${item.key}`;
      input.dataset.overlayKey = item.key;
      input.checked = true;

      const textWrap = document.createElement("div");
      const title = document.createElement("strong");
      title.textContent = item.label;
      const description = document.createElement("span");
      description.textContent = item.description;

      textWrap.appendChild(title);
      textWrap.appendChild(description);
      label.appendChild(input);
      label.appendChild(textWrap);
      container.appendChild(label);
    });
  };

  const readOverlayConfigFromDom = () => {
    const config = {};
    OVERLAY_CONTROL_ITEMS.forEach((item) => {
      const input = document.getElementById(`overlay-${item.key}`);
      if (input) config[item.key] = !!input.checked;
    });
    return config;
  };

  const syncOverlayControls = (controls) => {
    OVERLAY_CONTROL_ITEMS.forEach((item) => {
      const input = document.getElementById(`overlay-${item.key}`);
      if (!input) return;
      input.checked = !!getByPath(controls, item.key, true);
    });
  };

  const pushOverlayConfig = async () => {
    overlaySyncPending = true;
    setOverlayMeta("正在同步网页图层设置");
    try {
      const response = await fetch("/api/overlay", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(readOverlayConfigFromDom()),
      });
      if (!response.ok) throw new Error(response.statusText);
      const payload = await response.json();
      syncOverlayControls(payload);
      setOverlayMeta("图层设置已同步到当前可视化");
    } catch (error) {
      setOverlayMeta("图层设置同步失败，请检查服务端日志", true);
      console.warn("post /api/overlay failed", error);
    } finally {
      overlaySyncPending = false;
    }
  };

  const bindOverlayControls = () => {
    OVERLAY_CONTROL_ITEMS.forEach((item) => {
      const input = document.getElementById(`overlay-${item.key}`);
      input?.addEventListener("change", () => {
        pushOverlayConfig().catch((error) => console.warn(error));
      });
    });
  };

  const createStreamController = ({ imageId, placeholderId, path }) => {
    const image = document.getElementById(imageId);
    const placeholder = document.getElementById(placeholderId);
    const frameShell = image?.closest(".frame-shell");
    let active = false;
    let retryTimer = 0;

    const showPlaceholder = (text = "") => {
      if (!placeholder) return;
      if (text) placeholder.textContent = text;
      placeholder.style.display = "block";
    };

    const hidePlaceholder = () => {
      if (placeholder) placeholder.style.display = "none";
    };

    const attach = () => {
      if (!active || !image) return;
      image.classList.remove("is-ready");
      showPlaceholder("连接图像流中");
      image.src = `${path}?ts=${Date.now()}`;
    };

    const detach = () => {
      window.clearTimeout(retryTimer);
      retryTimer = 0;
      if (!image) return;
      image.classList.remove("is-ready");
      image.removeAttribute("src");
      image.src = "";
      showPlaceholder("切换到该视图后开始拉流");
    };

    if (image) {
      image.addEventListener("load", () => {
        hidePlaceholder();
        image.classList.add("is-ready");
        if (!frameShell) return;
        const width = image.naturalWidth;
        const height = image.naturalHeight;
        if (width > 0 && height > 0) {
          frameShell.style.setProperty("--frame-ratio", `${width} / ${height}`);
        }
      });
      image.addEventListener("error", () => {
        image.classList.remove("is-ready");
        showPlaceholder("图像流暂时不可用，正在重连");
        if (!active || retryTimer) return;
        retryTimer = window.setTimeout(() => {
          retryTimer = 0;
          attach();
        }, 600);
      });
    }

    return {
      start() {
        if (active) return;
        active = true;
        attach();
      },
      stop() {
        active = false;
        detach();
      },
    };
  };

  const stopAllStreams = () => {
    streamControllers.forEach((controller) => controller.stop());
  };

  const syncStreamsForView = (viewId) => {
    stopAllStreams();
    (STREAM_LAYOUT[viewId] || []).forEach(({ imageId }) => {
      streamControllers.get(imageId)?.start();
    });
    setText(
      "overview-stream-meta",
      viewId === "overview"
        ? "stream: /stream/main.mjpg"
        : viewId === "analysis"
          ? "stream: moved to analysis view"
          : "stream: inactive on inspector view",
    );
  };

  const syncChartView = () => {
    if (!window.DebugCharts || typeof window.DebugCharts.setActive !== "function") return;
    const analysisActive = currentView === "analysis";
    window.DebugCharts.setActive(analysisActive);
    if (analysisActive && typeof window.DebugCharts.renderAll === "function") {
      window.requestAnimationFrame(() => window.DebugCharts.renderAll());
      window.setTimeout(() => window.DebugCharts.renderAll(), 120);
    }
  };

  const activateView = (viewId) => {
    const nextView = VIEW_IDS.includes(viewId) ? viewId : "overview";
    rememberViewScroll(currentView);
    currentView = nextView;

    document.querySelectorAll(".view").forEach((section) => {
      section.classList.toggle("active", section.id === `view-${nextView}`);
    });
    document.querySelectorAll(".view-btn").forEach((button) => {
      button.classList.toggle("active", button.dataset.view === nextView);
    });

    syncStreamsForView(nextView);
    syncChartView();
    window.requestAnimationFrame(() => restoreViewScroll(nextView));

    if (nextView === "inspector") {
      fetchAndDisplayJsonWithTree("json-log", "/log");
      fetchRuntimeParams(true).catch((error) => console.warn(error));
    }
  };

  const resolveViewFromHash = () => {
    const candidate = window.location.hash.replace("#", "").trim();
    return VIEW_IDS.includes(candidate) ? candidate : "overview";
  };

  const renderStatus = (state) => {
    const frame = state.frame || {};
    const preview = state.preview || {};
    const planner = state.planner || {};
    const command = state.command || {};
    const ballistic = state.ballistic || {};
    const overlay = state.overlay || {};
    const buff = state.buff || {};
    const modeFromState = normalizeModePayload(getByPath(state, "mode", currentModeState));
    if (!modeSyncPending) {
      applyModeState(modeFromState);
    }
    const mode = modeSyncPending ? currentModeState : modeFromState;
    const buffMode = isBuffModeKey(mode.mode_key);
    const serialBulletSpeed = getByPath(
      frame,
      "bullet_speed_mps",
      getByPath(command, "bullet_speed_mps", getByPath(ballistic, "bullet_speed_raw_mps", getByPath(ballistic, "bullet_speed_mps"))),
    );
    const effectiveBulletSpeed = getByPath(
      command,
      "bullet_speed_effective_mps",
      getByPath(ballistic, "bullet_speed_effective_mps", getByPath(ballistic, "bullet_speed_mps")),
    );
    const bulletSpeedFallback = !!getByPath(
      command,
      "bullet_speed_fallback",
      getByPath(ballistic, "bullet_speed_fallback", false),
    );
    const bulletSpeedSource = getByPath(
      frame,
      "bullet_speed_source",
      getByPath(command, "bullet_speed_source", "runtime"),
    );

    const hasTarget = !!getByPath(preview, "has_target", false);
    const fire = !!getByPath(preview, "fire", false);
    const latencyMs = getByPath(frame, "latency_ms");
    const selectedArmor = getByPath(planner, "selected_armor");
    const buffDetected = !!getByPath(buff, "has_detection", false);
    const buffSolved = !!getByPath(buff, "target_solved", false);

    const linkState =
      lastStateUpdatedAt && Date.now() - lastStateUpdatedAt < POLL_STATE_MS * 3 ? "ONLINE" : "STALE";

    setText("status-link", linkState);
    setFireBadge(fire);
    setText(
      "status-target",
      buffMode
        ? hasTarget
          ? `${mode.mode_label} hit`
          : buffDetected
            ? `${mode.mode_label} detect`
            : "none"
        : hasTarget
          ? getByPath(preview, "target_name", "target")
          : "none",
    );
    setText("status-latency", formatNumber(latencyMs, 1, " ms"));
    setText("status-turn", getByPath(planner, "turn_direction", "STEADY"));
    setText(
      "status-armor",
      buffMode ? (buffSolved ? "SOLVED" : buffDetected ? "DETECT" : "WAIT") : formatArmorId(selectedArmor),
    );
    setText("overlay-stage", getByPath(overlay, "stage", "--"));
    if (!overlaySyncPending) {
      syncOverlayControls(getByPath(overlay, "controls", {}));
    }
    setOverlayMeta(`图层同步: ${getByPath(overlay, "stage", "--")} · 实时生效`);

    if (buffMode) {
      renderRows("overview-summary", [
        { label: "网页模式", value: `${mode.mode_label} · ${mode.source}` },
        { label: "串口模式", value: mode.serial_mode_label || "--" },
        { label: "图像尺寸", value: isFiniteNumber(getByPath(frame, "image_width")) && isFiniteNumber(getByPath(frame, "image_height")) ? `${frame.image_width} x ${frame.image_height}` : "--" },
        { label: "识别链路", value: buffSolved ? "DETECT + SOLVE" : buffDetected ? "DETECT ONLY" : "WAITING" },
        {
          label: "击打点角度",
          value: `${formatSigned(getByPath(preview, "target_yaw_deg"), 2, " deg")} / ${formatSigned(getByPath(preview, "target_pitch_deg"), 2, " deg")}`,
        },
        {
          label: "规划输出",
          value: `${formatSigned(getByPath(preview, "plan_yaw_deg"), 2, " deg")} / ${formatSigned(getByPath(preview, "plan_pitch_deg"), 2, " deg")}`,
        },
        {
          label: "串口弹速",
          value: formatNumber(serialBulletSpeed, 2, " m/s"),
        },
        { label: "图层阶段", value: getByPath(overlay, "stage", "--") },
      ]);

      renderRows("planner-card", [
        { label: "机关方向", value: getByPath(planner, "turn_direction", "STEADY") },
        {
          label: "中心 yaw / pitch",
          value: `${formatSigned(getByPath(buff, "rune_yaw_deg"), 2, " deg")} / ${formatSigned(getByPath(buff, "rune_pitch_deg"), 2, " deg")}`,
        },
        { label: "中心距离", value: formatNumber(getByPath(buff, "rune_dist_m"), 3, " m") },
        {
          label: "击打点 yaw / pitch",
          value: `${formatSigned(getByPath(buff, "blade_yaw_deg"), 2, " deg")} / ${formatSigned(getByPath(buff, "blade_pitch_deg"), 2, " deg")}`,
        },
        { label: "击打点距离", value: formatNumber(getByPath(buff, "blade_dist_m"), 3, " m") },
        {
          label: "世界姿态",
          value: `${formatSigned(getByPath(buff, "buff_yaw_deg"), 2, " deg")} / ${formatSigned(getByPath(buff, "buff_pitch_deg"), 2, " deg")} / ${formatSigned(getByPath(buff, "buff_roll_deg"), 2, " deg")}`,
        },
        {
          label: "EKF angle / spd",
          value: `${formatSigned(getByPath(buff, "angle_deg"), 2, " deg")} / ${formatSigned(getByPath(buff, "spd_deg_s"), 2, " deg/s")}`,
        },
        {
          label: "大符参数",
          value: mode.mode_key === "big_buff"
            ? `${formatSigned(getByPath(buff, "fit_a_deg_s"), 2, " deg/s")} / ${formatNumber(getByPath(buff, "fit_w_rad_s"), 3, " rad/s")} / ${formatSigned(getByPath(buff, "fit_fi_deg"), 2, " deg")}`
            : "small buff",
        },
      ]);

      renderRows("ballistic-card", [
        { label: "控制有效", value: formatBool(getByPath(ballistic, "valid", false), "YES", "NO") },
        { label: "命中判定", value: formatBool(getByPath(ballistic, "hit", false), "FIRE", "SAFE") },
        {
          label: "串口 / 算法弹速",
          value:
            `${formatNumber(getByPath(ballistic, "bullet_speed_raw_mps", serialBulletSpeed), 2, " m/s")} / ` +
            `${formatNumber(getByPath(ballistic, "bullet_speed_effective_mps", effectiveBulletSpeed), 2, " m/s")}` +
            (bulletSpeedFallback ? " · FALLBACK" : ""),
        },
        {
          label: "击打距离 XY / 3D",
          value: `${formatNumber(getByPath(ballistic, "target_dist_xy_m"), 2, " m")} / ${formatNumber(getByPath(ballistic, "target_dist_3d_m"), 2, " m")}`,
        },
        {
          label: "规划角度",
          value: `${formatSigned(getByPath(command, "plan_yaw_deg"), 2, " deg")} / ${formatSigned(getByPath(command, "plan_pitch_deg"), 2, " deg")}`,
        },
        {
          label: "规划速度",
          value: `${formatSigned(getByPath(command, "plan_yaw_vel_deg"), 2, " deg/s")} / ${formatSigned(getByPath(command, "plan_pitch_vel_deg"), 2, " deg/s")}`,
        },
        {
          label: "规划加速度",
          value: `${formatSigned(getByPath(command, "plan_yaw_acc_deg"), 2, " deg/s2")} / ${formatSigned(getByPath(command, "plan_pitch_acc_deg"), 2, " deg/s2")}`,
        },
      ]);

      renderRows("analysis-command-card", [
        { label: "当前模式", value: `${mode.mode_label} · web` },
        { label: "控制状态", value: fire ? "ARMED" : "SAFE" },
        {
          label: "云台角度",
          value: `${formatSigned(getByPath(command, "gimbal_yaw_deg"), 2, " deg")} / ${formatSigned(getByPath(command, "gimbal_pitch_deg"), 2, " deg")}`,
        },
        {
          label: "规划角度",
          value: `${formatSigned(getByPath(command, "plan_yaw_deg"), 2, " deg")} / ${formatSigned(getByPath(command, "plan_pitch_deg"), 2, " deg")}`,
        },
        {
          label: "规划速度",
          value: `${formatSigned(getByPath(command, "plan_yaw_vel_deg"), 2, " deg/s")} / ${formatSigned(getByPath(command, "plan_pitch_vel_deg"), 2, " deg/s")}`,
        },
        {
          label: "串口弹速",
          value: `${formatNumber(serialBulletSpeed, 2, " m/s")} · ${bulletSpeedSource}`,
        },
        bulletSpeedFallback
          ? { label: "算法弹速", value: `${formatNumber(effectiveBulletSpeed, 2, " m/s")} · fallback` }
          : null,
      ]);
    } else {
      renderRows("overview-summary", [
        { label: "帧号", value: getByPath(frame, "frame_index", "--") },
        {
          label: "图像尺寸",
          value:
            isFiniteNumber(getByPath(frame, "image_width")) && isFiniteNumber(getByPath(frame, "image_height"))
              ? `${frame.image_width} x ${frame.image_height}`
              : "--",
        },
        {
          label: "回放时间",
          value: getByPath(frame, "playback_t_s") !== undefined ? formatNumber(frame.playback_t_s, 3, " s") : "--",
        },
        {
          label: "目标姿态",
          value: `${formatSigned(getByPath(preview, "target_yaw_deg"), 2, " deg")} / ${formatSigned(getByPath(preview, "target_pitch_deg"), 2, " deg")}`,
        },
        {
          label: "规划输出",
          value: `${formatSigned(getByPath(preview, "plan_yaw_deg"), 2, " deg")} / ${formatSigned(getByPath(preview, "plan_pitch_deg"), 2, " deg")}`,
        },
        {
          label: "串口弹速",
          value: formatNumber(serialBulletSpeed, 2, " m/s"),
        },
        {
          label: "图层阶段",
          value: getByPath(overlay, "stage", "--"),
        },
      ]);

      renderRows("planner-card", [
        { label: "转向判断", value: getByPath(planner, "turn_direction", "STEADY") },
        { label: "选中装甲板", value: formatArmorId(selectedArmor) },
        { label: "物理板号", value: formatArmorId(getByPath(planner, "physical_armor")) },
        { label: "中心偏航", value: formatSigned(getByPath(planner, "center_yaw_deg"), 2, " deg") },
        { label: "Spin Gate", value: formatBool(getByPath(planner, "spin_gate", false), "ON", "OFF") },
        { label: "规划延迟", value: formatNumber(getByPath(planner, "delay_ms"), 1, " ms") },
        { label: "Delta 列表", value: formatDeltaList(getByPath(planner, "delta_angle_deg_list", [])) },
        {
          label: "模型模式",
          value: formatBool(getByPath(planner, "fixed_center_rotation_model", false), "FIXED", "FOLLOW"),
        },
        {
          label: "高度/偏置",
          value: `${formatNumber(getByPath(planner, "h_m"), 3, " m")} / ${formatSigned(getByPath(planner, "selected_z_offset_m"), 3, " m")}`,
        },
        {
          label: "击打补偿",
          value: formatSigned(getByPath(planner, "selected_aim_z_compensation_m"), 3, " m"),
        },
      ]);

      renderRows("ballistic-card", [
        { label: "弹道有效", value: formatBool(getByPath(ballistic, "valid", false), "YES", "NO") },
        { label: "轨迹可解", value: formatBool(!getByPath(ballistic, "unsolvable", false), "YES", "NO") },
        { label: "命中判定", value: formatBool(getByPath(ballistic, "hit", false), "HIT", "MISS") },
        {
          label: "串口 / 算法弹速",
          value:
            `${formatNumber(getByPath(ballistic, "bullet_speed_raw_mps", serialBulletSpeed), 2, " m/s")} / ` +
            `${formatNumber(getByPath(ballistic, "bullet_speed_effective_mps", effectiveBulletSpeed), 2, " m/s")}` +
            (bulletSpeedFallback ? " · FALLBACK" : ""),
        },
        { label: "总误差", value: formatNumber(getByPath(ballistic, "total_error_mm"), 1, " mm") },
        {
          label: "Yaw / Pitch 残差",
          value: `${formatSigned(getByPath(ballistic, "yaw_residual_deg"), 2, " deg")} / ${formatSigned(getByPath(ballistic, "pitch_residual_deg"), 2, " deg")}`,
        },
        {
          label: "目标距离",
          value: `${formatNumber(getByPath(ballistic, "target_dist_xy_m"), 2, " m")} / ${formatNumber(getByPath(ballistic, "target_dist_3d_m"), 2, " m")}`,
        },
      ]);

      renderRows("analysis-command-card", [
        { label: "控制状态", value: fire ? "ARMED" : "SAFE" },
        {
          label: "云台角度",
          value: `${formatSigned(getByPath(command, "gimbal_yaw_deg"), 2, " deg")} / ${formatSigned(getByPath(command, "gimbal_pitch_deg"), 2, " deg")}`,
        },
        {
          label: "规划角度",
          value: `${formatSigned(getByPath(command, "plan_yaw_deg"), 2, " deg")} / ${formatSigned(getByPath(command, "plan_pitch_deg"), 2, " deg")}`,
        },
        {
          label: "规划速度",
          value: `${formatSigned(getByPath(command, "plan_yaw_vel_deg"), 2, " deg/s")} / ${formatSigned(getByPath(command, "plan_pitch_vel_deg"), 2, " deg/s")}`,
        },
        {
          label: "规划加速度",
          value: `${formatSigned(getByPath(command, "plan_yaw_acc_deg"), 2, " deg/s2")} / ${formatSigned(getByPath(command, "plan_pitch_acc_deg"), 2, " deg/s2")}`,
        },
        {
          label: "串口弹速",
          value: `${formatNumber(serialBulletSpeed, 2, " m/s")} · ${bulletSpeedSource}`,
        },
        bulletSpeedFallback ?
          {
            label: "算法弹速",
            value: `${formatNumber(effectiveBulletSpeed, 2, " m/s")} · safety fallback`,
          } :
          null,
      ]);
    }

    renderRows("inspector-summary", [
      { label: "服务器时间", value: formatClock(getByPath(state, "server.unix_ms")) },
      { label: "网页模式", value: `${mode.mode_label} · ${mode.source}` },
      { label: "串口模式", value: mode.serial_mode_label || "--" },
      { label: "帧号", value: getByPath(frame, "frame_index", "--") },
      { label: "回放时间", value: formatNumber(getByPath(frame, "playback_t_s"), 3, " s") },
      { label: "原始时间", value: formatNumber(getByPath(frame, "raw_t_s"), 3, " s") },
      { label: "链路延迟", value: formatNumber(latencyMs, 2, " ms") },
      { label: "目标存在", value: formatBool(hasTarget, "YES", "NO") },
      { label: "图层阶段", value: getByPath(overlay, "stage", "--") },
    ]);

    renderRows("command-card", [
      { label: "单位来源", value: getByPath(command, "gimbal_source_unit", "--") },
      { label: "开火建议", value: formatBool(getByPath(command, "fire", fire), "YES", "NO") },
      { label: "实际发射", value: formatBool(getByPath(command, "fired", false), "YES", "NO") },
      {
        label: "Yaw 原始/deg/rad",
        value: `${formatNumber(getByPath(command, "gimbal_yaw_raw"), 2)} / ${formatSigned(getByPath(command, "gimbal_yaw_deg"), 2)} / ${formatSigned(getByPath(command, "gimbal_yaw_rad"), 3)}`,
      },
      {
        label: "Pitch 原始/deg/rad",
        value: `${formatNumber(getByPath(command, "gimbal_pitch_raw"), 2)} / ${formatSigned(getByPath(command, "gimbal_pitch_deg"), 2)} / ${formatSigned(getByPath(command, "gimbal_pitch_rad"), 3)}`,
      },
      {
        label: "计划角度 deg",
        value: `${formatSigned(getByPath(command, "plan_yaw_deg"), 2, " deg")} / ${formatSigned(getByPath(command, "plan_pitch_deg"), 2, " deg")}`,
      },
      {
        label: "目标角度 deg",
        value: `${formatSigned(getByPath(command, "target_yaw_deg"), 2, " deg")} / ${formatSigned(getByPath(command, "target_pitch_deg"), 2, " deg")}`,
      },
      {
        label: "弹速来源",
        value: `${bulletSpeedSource} · ${formatNumber(serialBulletSpeed, 2, " m/s")}`,
      },
      bulletSpeedFallback ?
        {
          label: "算法兜底弹速",
          value: formatNumber(effectiveBulletSpeed, 2, " m/s"),
        } :
        null,
    ]);

    if (buffMode) {
      renderRows("preview-card", [
        { label: "当前模式", value: mode.mode_label },
        { label: "目标存在", value: formatBool(hasTarget, "YES", "NO") },
        {
          label: "击打点位置",
          value: `${formatNumber(getByPath(buff, "aim_x_m"), 3, " m")} / ${formatNumber(getByPath(buff, "aim_y_m"), 3, " m")} / ${formatNumber(getByPath(buff, "aim_z_m"), 3, " m")}`,
        },
        {
          label: "预测击打点",
          value: `${formatNumber(getByPath(buff, "predicted_aim_x_m"), 3, " m")} / ${formatNumber(getByPath(buff, "predicted_aim_y_m"), 3, " m")} / ${formatNumber(getByPath(buff, "predicted_aim_z_m"), 3, " m")}`,
        },
        {
          label: "击打点角度",
          value: `${formatSigned(getByPath(preview, "target_yaw_deg"), 2, " deg")} / ${formatSigned(getByPath(preview, "target_pitch_deg"), 2, " deg")}`,
        },
        {
          label: "规划角度",
          value: `${formatSigned(getByPath(preview, "plan_yaw_deg"), 2, " deg")} / ${formatSigned(getByPath(preview, "plan_pitch_deg"), 2, " deg")}`,
        },
      ]);

      renderRows("inspector-planner-card", [
        { label: "机关方向", value: getByPath(planner, "turn_direction", "STEADY") },
        {
          label: "中心 yaw / pitch",
          value: `${formatSigned(getByPath(buff, "rune_yaw_deg"), 2, " deg")} / ${formatSigned(getByPath(buff, "rune_pitch_deg"), 2, " deg")}`,
        },
        { label: "中心距离", value: formatNumber(getByPath(buff, "rune_dist_m"), 3, " m") },
        {
          label: "EKF angle / spd",
          value: `${formatSigned(getByPath(buff, "angle_deg"), 2, " deg")} / ${formatSigned(getByPath(buff, "spd_deg_s"), 2, " deg/s")}`,
        },
        {
          label: "rad/s 速度",
          value: formatSigned(getByPath(buff, "spd_rad_s"), 3, " rad/s"),
        },
        {
          label: "大符参数",
          value: mode.mode_key === "big_buff"
            ? `${formatSigned(getByPath(buff, "fit_a_deg_s"), 2, " deg/s")} / ${formatNumber(getByPath(buff, "fit_w_rad_s"), 3, " rad/s")} / ${formatSigned(getByPath(buff, "fit_fi_deg"), 2, " deg")}`
            : "small buff",
        },
      ]);
    } else {
      renderRows("preview-card", [
        { label: "目标名称", value: getByPath(preview, "target_name", "none") },
        { label: "装甲类型", value: getByPath(preview, "armor_type", "none") },
        {
          label: "目标位置",
          value: `${formatNumber(getByPath(preview, "target_x_m"), 3, " m")} / ${formatNumber(getByPath(preview, "target_y_m"), 3, " m")} / ${formatNumber(getByPath(preview, "target_z_m"), 3, " m")}`,
        },
        {
          label: "目标角度",
          value: `${formatSigned(getByPath(preview, "target_yaw_deg"), 2, " deg")} / ${formatSigned(getByPath(preview, "target_pitch_deg"), 2, " deg")}`,
        },
        {
          label: "规划角度",
          value: `${formatSigned(getByPath(preview, "plan_yaw_deg"), 2, " deg")} / ${formatSigned(getByPath(preview, "plan_pitch_deg"), 2, " deg")}`,
        },
      ]);

      renderRows("inspector-planner-card", [
        { label: "选中装甲板", value: formatArmorId(selectedArmor) },
        { label: "物理板号", value: formatArmorId(getByPath(planner, "physical_armor")) },
        { label: "转向符号", value: getByPath(planner, "turn_sign", "--") },
        { label: "转向角速度", value: formatSigned(getByPath(planner, "w_rad_s"), 3, " rad/s") },
        { label: "中心偏航", value: formatSigned(getByPath(planner, "center_yaw_deg"), 2, " deg") },
        { label: "切板角列表", value: formatDeltaList(getByPath(planner, "delta_angle_deg_list", [])) },
        { label: "装甲高度", value: formatNumber(getByPath(planner, "h_m"), 3, " m") },
        { label: "选中 Z 偏置", value: formatSigned(getByPath(planner, "selected_z_offset_m"), 3, " m") },
        {
          label: "击打 Z 补偿",
          value: formatSigned(getByPath(planner, "selected_aim_z_compensation_m"), 3, " m"),
        },
        {
          label: "中心模型",
          value: formatBool(getByPath(planner, "fixed_center_rotation_model", false), "FIXED", "FOLLOW"),
        },
      ]);
    }

    setText("log-meta", `latest snapshot · ${formatClock(getByPath(state, "server.unix_ms"))}`);
  };

  const markOffline = (message = "OFFLINE") => {
    setText("status-link", message);
  };

  const refreshState = async () => {
    try {
      const response = await fetch(`/api/state?ts=${Date.now()}`, { cache: "no-store" });
      if (!response.ok) throw new Error(response.statusText);
      const state = await response.json();
      lastStateUpdatedAt = Date.now();
      renderStatus(state);
    } catch (error) {
      markOffline("OFFLINE");
      console.warn("fetch /api/state failed", error);
    }
  };

  const toggleFullscreen = async (panelSelector) => {
    const frameShell = document.querySelector(panelSelector);
    if (!frameShell) return;
    if (!document.fullscreenElement) {
      await frameShell.requestFullscreen();
      document.body.classList.add("fullscreen-mode");
      syncChartView();
      return;
    }
    await document.exitFullscreen();
  };

  const bindViewSwitch = () => {
    document.querySelectorAll(".view-btn").forEach((button) => {
      button.addEventListener("click", () => {
        const next = button.dataset.view;
        if (!next) return;
        if (window.location.hash !== `#${next}`) {
          window.location.hash = next;
          return;
        }
        activateView(next);
      });
    });

    window.addEventListener("hashchange", () => activateView(resolveViewFromHash()));
  };

  const bindFullscreen = () => {
    document
      .getElementById("overview-fullscreen-btn")
      ?.addEventListener("click", () => toggleFullscreen(".hero-frame").catch((err) => console.warn(err)));

    document
      .getElementById("analysis-fullscreen-btn")
      ?.addEventListener("click", () => toggleFullscreen(".analysis-frame").catch((err) => console.warn(err)));

    document.addEventListener("fullscreenchange", () => {
      document.body.classList.toggle("fullscreen-mode", !!document.fullscreenElement);
      syncChartView();
    });
  };

  const bindViewScrollMemory = () => {
    VIEW_IDS.forEach((viewId) => {
      const node = getViewNode(viewId);
      if (!node) return;
      viewScrollPositions.set(viewId, 0);
      node.addEventListener(
        "scroll",
        () => {
          viewScrollPositions.set(viewId, node.scrollTop);
        },
        { passive: true },
      );
    });
  };

  const initStreams = () => {
    Object.values(STREAM_LAYOUT)
      .flat()
      .forEach((config) => {
        if (!streamControllers.has(config.imageId)) {
          streamControllers.set(config.imageId, createStreamController(config));
        }
      });
  };

  const initPolling = () => {
    refreshState();
    window.setInterval(refreshState, POLL_STATE_MS);
    window.setInterval(() => {
      if (currentView === "inspector") {
        fetchAndDisplayJsonWithTree("json-log", "/log");
      }
    }, POLL_LOG_MS);
  };

  document.addEventListener("DOMContentLoaded", () => {
    if ("scrollRestoration" in window.history) {
      window.history.scrollRestoration = "manual";
    }
    setText("server-url", `访问地址: ${window.location.origin}`);
    initStreams();
    initOverlayControls();
    bindOverlayControls();
    bindModeSwitch();
    bindRuntimeParamToolbar();
    setOverlayMeta("等待状态同步后载入图层设置");
    setRuntimeParamStatus("等待运行时参数会话同步");
    bindViewSwitch();
    bindViewScrollMemory();
    bindFullscreen();

    if (window.DebugCharts && typeof window.DebugCharts.init === "function") {
      window.DebugCharts.init();
    }

    activateView(resolveViewFromHash());
    fetchMode(true).catch((error) => console.warn(error));
    fetchRuntimeParams(true).catch((error) => console.warn(error));
    initPolling();
  });
})();
