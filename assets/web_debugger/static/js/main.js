(() => {
  const VIEW_IDS = ["overview", "analysis", "inspector"];
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
  let runtimeParamSnapshot = null;

  const streamControllers = new Map();

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

  const formatDeltaList = (values) => {
    if (!Array.isArray(values) || !values.length) return "--";
    return values
      .map((value, index) =>
        `${formatArmorId(index)}:${isFiniteNumber(value) ? `${value >= 0 ? "+" : ""}${value.toFixed(1)} deg` : "--"}`,
      )
      .join("  ");
  };

  const formatParamValue = (value) => {
    if (Array.isArray(value)) return value.join(", ");
    if (typeof value === "boolean") return value ? "true" : "false";
    if (value === null || value === undefined) return "--";
    return String(value);
  };

  const setText = (id, value) => {
    const node = document.getElementById(id);
    if (node) node.textContent = value;
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

  const parseRuntimeParamInput = (item, control) => {
    if (!item || !control) throw new Error("参数控件不存在");

    if (item.type === "boolean") return !!control.checked;
    if (item.type === "enum") return control.value;

    if (item.type === "integer") {
      const value = Number(control.value);
      if (!Number.isFinite(value)) throw new Error(`${item.label} 需要整数`);
      return Math.round(value);
    }

    if (item.type === "number") {
      const value = Number(control.value);
      if (!Number.isFinite(value)) throw new Error(`${item.label} 需要数字`);
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
    const exportNode = document.getElementById("param-export-text");
    if (!groupsHost || !exportNode) return;

    groupsHost.innerHTML = "";
    exportNode.value = payload?.export_yaml || "# 当前还没有网页改过的参数";

    if (!payload?.enabled) {
      setRuntimeParamStatus(payload?.error || "当前入口没有启用运行时参数热调", true);
      setRuntimeParamMeta(payload?.config_path || "runtime parameter session unavailable");
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
      `配置: ${payload.config_path} · 会话日志: ${payload.session_log_path} · 快照: ${payload.snapshot_path}`,
    );

    (payload.groups || []).forEach((group) => {
      const section = document.createElement("section");
      section.className = "param-group";

      const head = document.createElement("div");
      head.className = "param-group-head";

      const title = document.createElement("h3");
      title.textContent = group.label;
      const meta = document.createElement("span");
      meta.className = "panel-meta";
      meta.textContent = `${(group.items || []).length} 项`;

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

        const info = document.createElement("div");
        info.className = "param-row-meta";
        info.textContent = `Key: ${item.key} · 基线: ${formatParamValue(item.base_value)}${item.unit ? ` ${item.unit}` : ""}${item.overridden ? " · 当前为运行时覆盖" : ""}`;

        rowHead.appendChild(titleWrap);
        rowHead.appendChild(info);

        const actions = document.createElement("div");
        actions.className = "param-actions";

        let control = null;
        if (item.type === "boolean") {
          control = document.createElement("input");
          control.type = "checkbox";
          control.className = "param-checkbox";
          control.checked = !!item.value;
        } else if (item.type === "enum") {
          control = document.createElement("select");
          (item.choices || []).forEach((choice) => {
            const option = document.createElement("option");
            option.value = choice;
            option.textContent = choice;
            option.selected = choice === item.value;
            control.appendChild(option);
          });
        } else if (item.type === "number_array") {
          control = document.createElement("textarea");
          control.rows = 2;
          control.value = formatParamValue(item.value);
          control.placeholder = "例如: 3e6, 0.3";
        } else {
          control = document.createElement("input");
          control.type = "number";
          control.step = item.type === "integer" ? "1" : "any";
          control.value = item.value;
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

        actions.appendChild(control);
        actions.appendChild(applyBtn);
        actions.appendChild(resetBtn);

        row.appendChild(rowHead);
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
    currentView = nextView;

    document.querySelectorAll(".view").forEach((section) => {
      section.classList.toggle("active", section.id === `view-${nextView}`);
    });
    document.querySelectorAll(".view-btn").forEach((button) => {
      button.classList.toggle("active", button.dataset.view === nextView);
    });

    syncStreamsForView(nextView);
    syncChartView();

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

    const hasTarget = !!getByPath(preview, "has_target", false);
    const fire = !!getByPath(preview, "fire", false);
    const latencyMs = getByPath(frame, "latency_ms");
    const selectedArmor = getByPath(planner, "selected_armor");

    const linkState =
      lastStateUpdatedAt && Date.now() - lastStateUpdatedAt < POLL_STATE_MS * 3 ? "ONLINE" : "STALE";

    setText("status-link", linkState);
    setFireBadge(fire);
    setText("status-target", hasTarget ? getByPath(preview, "target_name", "target") : "none");
    setText("status-latency", formatNumber(latencyMs, 1, " ms"));
    setText("status-turn", getByPath(planner, "turn_direction", "STEADY"));
    setText("status-armor", formatArmorId(selectedArmor));
    setText("overlay-stage", getByPath(overlay, "stage", "--"));
    if (!overlaySyncPending) {
      syncOverlayControls(getByPath(overlay, "controls", {}));
    }
    setOverlayMeta(`图层同步: ${getByPath(overlay, "stage", "--")} · 实时生效`);

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
        label: "发射速度",
        value: formatNumber(getByPath(command, "bullet_speed_mps", getByPath(ballistic, "bullet_speed_mps")), 2, " m/s"),
      },
      {
        label: "图层阶段",
        value: getByPath(overlay, "stage", "--"),
      },
    ]);

    renderRows("planner-card", [
      { label: "转向判断", value: getByPath(planner, "turn_direction", "STEADY") },
      { label: "选中装甲板", value: formatArmorId(selectedArmor) },
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
    ]);

    renderRows("ballistic-card", [
      { label: "弹道有效", value: formatBool(getByPath(ballistic, "valid", false), "YES", "NO") },
      { label: "轨迹可解", value: formatBool(!getByPath(ballistic, "unsolvable", false), "YES", "NO") },
      { label: "命中判定", value: formatBool(getByPath(ballistic, "hit", false), "HIT", "MISS") },
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
      { label: "弹速", value: formatNumber(getByPath(command, "bullet_speed_mps"), 2, " m/s") },
    ]);

    renderRows("inspector-summary", [
      { label: "服务器时间", value: formatClock(getByPath(state, "server.unix_ms")) },
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
    ]);

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
      { label: "转向符号", value: getByPath(planner, "turn_sign", "--") },
      { label: "转向角速度", value: formatSigned(getByPath(planner, "w_rad_s"), 3, " rad/s") },
      { label: "中心偏航", value: formatSigned(getByPath(planner, "center_yaw_deg"), 2, " deg") },
      { label: "切板角列表", value: formatDeltaList(getByPath(planner, "delta_angle_deg_list", [])) },
      { label: "装甲高度", value: formatNumber(getByPath(planner, "h_m"), 3, " m") },
      { label: "选中 Z 偏置", value: formatSigned(getByPath(planner, "selected_z_offset_m"), 3, " m") },
      {
        label: "中心模型",
        value: formatBool(getByPath(planner, "fixed_center_rotation_model", false), "FIXED", "FOLLOW"),
      },
    ]);

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
    setText("server-url", `访问地址: ${window.location.origin}`);
    initStreams();
    initOverlayControls();
    bindOverlayControls();
    bindRuntimeParamToolbar();
    setOverlayMeta("等待状态同步后载入图层设置");
    setRuntimeParamStatus("等待运行时参数会话同步");
    bindViewSwitch();
    bindFullscreen();

    if (window.DebugCharts && typeof window.DebugCharts.init === "function") {
      window.DebugCharts.init();
    }

    activateView(resolveViewFromHash());
    fetchRuntimeParams(true).catch((error) => console.warn(error));
    initPolling();
  });
})();
