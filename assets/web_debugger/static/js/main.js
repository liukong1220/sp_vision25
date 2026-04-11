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

  let currentView = "overview";
  let lastStateUpdatedAt = 0;

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

  const createStreamController = ({ imageId, placeholderId, path }) => {
    const image = document.getElementById(imageId);
    const placeholder = document.getElementById(placeholderId);
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
      showPlaceholder("连接图像流中");
      image.src = `${path}?ts=${Date.now()}`;
    };

    const detach = () => {
      window.clearTimeout(retryTimer);
      retryTimer = 0;
      if (!image) return;
      image.removeAttribute("src");
      image.src = "";
      showPlaceholder("切换到该视图后开始拉流");
    };

    if (image) {
      image.addEventListener("load", hidePlaceholder);
      image.addEventListener("error", () => {
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
    const panel = document.querySelector(panelSelector);
    if (!panel) return;
    if (!document.fullscreenElement) {
      await panel.requestFullscreen();
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
      ?.addEventListener("click", () => toggleFullscreen(".hero-panel").catch((err) => console.warn(err)));

    document
      .getElementById("analysis-fullscreen-btn")
      ?.addEventListener("click", () => toggleFullscreen(".analysis-video-panel").catch((err) => console.warn(err)));

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
    bindViewSwitch();
    bindFullscreen();

    if (window.DebugCharts && typeof window.DebugCharts.init === "function") {
      window.DebugCharts.init();
    }

    activateView(resolveViewFromHash());
    initPolling();
  });
})();
