const DebugCharts = (() => {
  const chartMap = {
    gimbal_yaw: { label: "云台 Yaw", color: "#7de3ff", unit: "deg", group: "angles" },
    gimbal_pitch: { label: "云台 Pitch", color: "#ffcf70", unit: "deg", group: "angles" },
    target_yaw: { label: "目标 Yaw", color: "#9df3bf", unit: "deg", group: "angles" },
    target_pitch: { label: "目标 Pitch", color: "#ff9db4", unit: "deg", group: "angles" },
    plan_yaw: { label: "规划 Yaw", color: "#38f0b8", unit: "deg", group: "angles" },
    plan_pitch: { label: "规划 Pitch", color: "#ff758f", unit: "deg", group: "angles" },
    cmd_yaw: { label: "控制 Yaw", color: "#52ffe0", unit: "deg", group: "angles" },
    cmd_pitch: { label: "控制 Pitch", color: "#ff8c94", unit: "deg", group: "angles" },

    plan_yaw_vel: { label: "规划 Yaw 速度", color: "#78b7ff", unit: "deg/s", group: "motion" },
    plan_pitch_vel: { label: "规划 Pitch 速度", color: "#f7a34b", unit: "deg/s", group: "motion" },
    plan_yaw_acc: {
      label: "规划 Yaw 加速度",
      color: "#45d6c2",
      unit: "deg/s^2",
      group: "motion",
    },
    plan_pitch_acc: {
      label: "规划 Pitch 加速度",
      color: "#ff89b0",
      unit: "deg/s^2",
      group: "motion",
    },
    w: { label: "目标角速度", color: "#8bd0ff", unit: "rad/s", group: "motion" },
    target_vz: { label: "目标 Z 速度", color: "#ffcb77", unit: "m/s", group: "motion" },

    target_z: { label: "目标高度 Z", color: "#acf58e", unit: "m", group: "target" },
    target_h: { label: "目标尺寸 H", color: "#d7a6ff", unit: "m", group: "target" },
    planner_selected_z_offset: {
      label: "选中板 Z 偏置",
      color: "#8af6cf",
      unit: "m",
      group: "target",
    },

    R_yaw: { label: "机关中心 Yaw", color: "#7dd6ff", unit: "deg", group: "buff" },
    R_pitch: { label: "机关中心 Pitch", color: "#ffe082", unit: "deg", group: "buff" },
    R_dis: { label: "机关中心距离", color: "#9df3bf", unit: "m", group: "buff" },
    blade_yaw: { label: "击打点 Yaw", color: "#52ffe0", unit: "deg", group: "buff" },
    blade_pitch: { label: "击打点 Pitch", color: "#ff9db4", unit: "deg", group: "buff" },
    blade_dis: { label: "击打点距离", color: "#c4ff9f", unit: "m", group: "buff" },
    buff_yaw: { label: "机关姿态 Yaw", color: "#99c9ff", unit: "deg", group: "buff" },
    buff_pitch: { label: "机关姿态 Pitch", color: "#ffd166", unit: "deg", group: "buff" },
    buff_roll: { label: "机关姿态 Roll", color: "#ff8ab0", unit: "deg", group: "buff" },
    angle: { label: "扇叶角度", color: "#8af6cf", unit: "deg", group: "buff" },
    spd: { label: "扇叶角速度", color: "#7ce3ff", unit: "deg/s", group: "buff" },
    a: { label: "大符振幅 a", color: "#d7a6ff", unit: "deg/s", group: "buff" },
    fi: { label: "大符相位 fi", color: "#ff95d6", unit: "deg", group: "buff" },

    planner_delay_ms: { label: "规划延迟", color: "#7de3ff", unit: "ms", group: "planner" },
    planner_spin_gate: { label: "Spin Gate", color: "#ffe082", unit: "bool", group: "planner" },
    planner_center_yaw: { label: "中心偏航", color: "#9fc6ff", unit: "deg", group: "planner" },
    planner_turn_sign: { label: "转向符号", color: "#87ffad", unit: "sign", group: "planner" },
    planner_selected_armor: {
      label: "选中装甲板",
      color: "#ff95d6",
      unit: "idx",
      group: "planner",
    },

    bullet_speed: { label: "串口弹速", color: "#53d6ff", unit: "m/s", group: "fire" },
    fire: { label: "建议开火", color: "#ff7388", unit: "bool", group: "fire" },
    fired: { label: "实际发射", color: "#ff8d55", unit: "bool", group: "fire" },
    shoot: { label: "击发判定", color: "#ff7388", unit: "bool", group: "fire" },

    residual_yaw: { label: "Yaw 残差", color: "#8df1a2", unit: "deg", group: "diagnostics" },
    residual_pitch: {
      label: "Pitch 残差",
      color: "#ffd166",
      unit: "deg",
      group: "diagnostics",
    },
    residual_distance: {
      label: "距离残差",
      color: "#6ef3ff",
      unit: "mm",
      group: "diagnostics",
    },
    nis: { label: "NIS", color: "#d087ff", unit: "score", group: "diagnostics" },
    nees: { label: "NEES", color: "#ff7dc0", unit: "score", group: "diagnostics" },
  };

  const groupMeta = {
    angles: { label: "角度", description: "云台、目标与规划角度" },
    motion: { label: "速度/加速度", description: "目标转动与规划速度" },
    target: { label: "目标高度/尺寸", description: "高度、尺寸和 Z 偏置" },
    buff: { label: "机关", description: "机关中心、击打点和大符拟合参数" },
    planner: { label: "切板/MPC", description: "切板状态和规划诊断" },
    fire: { label: "击发", description: "开火建议与弹速" },
    diagnostics: { label: "残差/滤波", description: "残差、NIS 和 NEES" },
  };

  const defaultKeys = ["gimbal_yaw", "target_yaw", "plan_yaw"];

  const presetDefs = [
    { id: "default", label: "默认", keys: defaultKeys },
    {
      id: "angles",
      label: "角度",
      keys: Object.keys(chartMap).filter((key) => chartMap[key].group === "angles"),
    },
    {
      id: "motion",
      label: "速度",
      keys: Object.keys(chartMap).filter((key) => chartMap[key].group === "motion"),
    },
    {
      id: "planner",
      label: "规划",
      keys: Object.keys(chartMap).filter((key) => chartMap[key].group === "planner"),
    },
    {
      id: "buff",
      label: "机关",
      keys: Object.keys(chartMap).filter((key) => chartMap[key].group === "buff"),
    },
    {
      id: "fire",
      label: "击发",
      keys: Object.keys(chartMap).filter((key) => chartMap[key].group === "fire"),
    },
    {
      id: "diagnostics",
      label: "残差",
      keys: Object.keys(chartMap).filter((key) => chartMap[key].group === "diagnostics"),
    },
    { id: "clear", label: "清空", keys: [] },
  ];

  let latestData = { time: [] };
  let mainMaxPoints = 120;
  const rangeState = {};
  let active = false;

  const fitCanvas = (canvas) => {
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(280, Math.floor(canvas.clientWidth * ratio));
    const height = Math.max(220, Math.floor(canvas.clientHeight * ratio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    return ratio;
  };

  const selectedKeys = () =>
    Array.from(document.querySelectorAll(".chart-metric:checked")).map((input) => input.dataset.key);

  const sameKeySet = (left, right) => {
    if (left.length !== right.length) return false;
    const leftSet = new Set(left);
    return right.every((key) => leftSet.has(key));
  };

  const rangeStepForKey = (key) => {
    const unit = chartMap[key]?.unit || "";
    if (unit === "bool" || unit === "idx" || unit === "sign") return 1;
    if (unit === "deg") return 0.5;
    if (unit === "deg/s") return 0.5;
    if (unit === "deg/s^2") return 1;
    if (unit === "rad/s") return 0.05;
    if (unit === "m") return 0.01;
    if (unit === "m/s") return 0.05;
    if (unit === "ms") return 1;
    if (unit === "mm") return 1;
    return 0.1;
  };

  const summarizeUnits = (keys) => {
    const units = [...new Set(keys.map((key) => chartMap[key]?.unit).filter(Boolean))];
    if (!units.length) return "未选择曲线";
    if (units.length === 1) return `主图单位统一为 ${units[0]}`;
    return `主图混合单位: ${units.join(" / ")}`;
  };

  const updateSelectionSummary = (keys) => {
    const node = document.getElementById("chart-selection-summary");
    if (!node) return;

    if (!keys.length) {
      node.textContent = "当前未选择曲线，可用上方预设一键切到角度、速度、击发等常用组合。";
      node.classList.add("is-empty");
      return;
    }

    const unitSummary = summarizeUnits(keys);
    const advice =
      unitSummary.startsWith("主图混合单位") ? " 建议重点看下方单图，避免同轴误判。" : "";
    node.textContent = `已选 ${keys.length} 条曲线，${unitSummary}.${advice}`;
    node.classList.remove("is-empty");
  };

  const updatePresetState = (keys) => {
    document.querySelectorAll(".chart-preset-btn").forEach((button) => {
      const preset = presetDefs.find((item) => item.id === button.dataset.preset);
      button.classList.toggle("active", !!preset && sameKeySet(keys, preset.keys));
    });
  };

  const setMetricSelection = (keys) => {
    const wanted = new Set(keys);
    document.querySelectorAll(".chart-metric").forEach((input) => {
      input.checked = wanted.has(input.dataset.key);
    });
    updateSelectionSummary(keys);
    updatePresetState(keys);
    renderAll();
  };

  const ensureControls = () => {
    const container = document.getElementById("chart-select-controls");
    const presetContainer = document.getElementById("chart-preset-controls");
    if (!container || !presetContainer) return;

    container.innerHTML = "";
    presetContainer.innerHTML = "";

    const multiChip = document.createElement("label");
    multiChip.className = "checkbox-chip chart-toggle-chip";
    multiChip.innerHTML = '<input type="checkbox" id="multiLineChart" checked /> <div><strong>主图叠加</strong><span>在总图中同时观察已选曲线</span></div>';
    presetContainer.appendChild(multiChip);

    presetDefs.forEach((preset) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "chart-preset-btn";
      button.dataset.preset = preset.id;
      button.textContent = preset.label;
      button.addEventListener("click", () => setMetricSelection(preset.keys));
      presetContainer.appendChild(button);
    });

    Object.entries(groupMeta).forEach(([groupId, meta]) => {
      const section = document.createElement("section");
      section.className = "chart-select-group";

      const head = document.createElement("div");
      head.className = "chart-select-title";
      head.innerHTML = `<strong>${meta.label}</strong><span>${meta.description}</span>`;
      section.appendChild(head);

      const cluster = document.createElement("div");
      cluster.className = "chart-select-cluster";

      Object.entries(chartMap)
        .filter(([, item]) => item.group === groupId)
        .forEach(([key, item]) => {
          const chip = document.createElement("label");
          chip.className = "checkbox-chip metric-chip";
          chip.innerHTML = `
            <input type="checkbox" class="chart-metric" data-key="${key}" ${
              defaultKeys.includes(key) ? "checked" : ""
            } />
            <div>
              <strong>${item.label}</strong>
              <span>${item.unit || "raw"}</span>
            </div>
          `;
          cluster.appendChild(chip);
        });

      section.appendChild(cluster);
      container.appendChild(section);
    });

    container.addEventListener("change", () => {
      const keys = selectedKeys();
      updateSelectionSummary(keys);
      updatePresetState(keys);
      renderAll();
    });

    document.getElementById("apply-main-range")?.addEventListener("click", () => {
      mainMaxPoints = Number(document.getElementById("mainMaxPts")?.value) || 120;
      renderAll();
    });

    updateSelectionSummary(selectedKeys());
    updatePresetState(selectedKeys());
  };

  const seriesSlice = (key) => {
    const source = Array.isArray(latestData[key]) ? latestData[key] : [];
    return source.slice(Math.max(0, source.length - mainMaxPoints));
  };

  const drawChart = (canvas, seriesList, fixedRange = null) => {
    fitCanvas(canvas);
    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;
    const ratio = window.devicePixelRatio || 1;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#0b1722";
    ctx.fillRect(0, 0, width, height);

    const pad = {
      top: 28 * ratio,
      right: 20 * ratio,
      bottom: 28 * ratio,
      left: 50 * ratio,
    };
    const plotWidth = width - pad.left - pad.right;
    const plotHeight = height - pad.top - pad.bottom;

    const allValues = [];
    seriesList.forEach((series) => {
      series.values.forEach((value) => {
        if (typeof value === "number" && Number.isFinite(value)) allValues.push(value);
      });
    });

    if (!allValues.length) {
      ctx.fillStyle = "#90b6c8";
      ctx.font = `${13 * ratio}px Segoe UI`;
      ctx.fillText("等待数据", pad.left, height / 2);
      return;
    }

    let minValue = fixedRange && fixedRange.enabled ? fixedRange.min : Math.min(...allValues);
    let maxValue = fixedRange && fixedRange.enabled ? fixedRange.max : Math.max(...allValues);
    if (Math.abs(maxValue - minValue) < 1e-6) {
      minValue -= 1;
      maxValue += 1;
    } else if (!(fixedRange && fixedRange.enabled)) {
      const margin = (maxValue - minValue) * 0.12;
      minValue -= margin;
      maxValue += margin;
    }

    ctx.strokeStyle = "rgba(125, 194, 220, 0.16)";
    ctx.lineWidth = 1 * ratio;
    for (let i = 0; i <= 4; i += 1) {
      const y = pad.top + (plotHeight * i) / 4;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + plotWidth, y);
      ctx.stroke();
    }

    const pointCount = Math.max(...seriesList.map((series) => series.values.length));
    const mapX = (index) =>
      pad.left + (pointCount <= 1 ? 0 : (index / (pointCount - 1)) * plotWidth);
    const mapY = (value) =>
      pad.top + plotHeight - ((value - minValue) / (maxValue - minValue)) * plotHeight;

    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    seriesList.forEach((series) => {
      ctx.strokeStyle = series.color;
      ctx.lineWidth = 2.2 * ratio;
      ctx.beginPath();
      let first = true;
      series.values.forEach((value, index) => {
        if (typeof value !== "number" || !Number.isFinite(value)) {
          first = true;
          return;
        }
        const x = mapX(index);
        const y = mapY(value);
        if (first) {
          ctx.moveTo(x, y);
          first = false;
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();
    });

    ctx.fillStyle = "#d8f3ff";
    ctx.font = `${11 * ratio}px Segoe UI`;
    ctx.textAlign = "left";
    ctx.fillText(maxValue.toFixed(2), 8 * ratio, pad.top + 4 * ratio);
    ctx.fillText(minValue.toFixed(2), 8 * ratio, height - pad.bottom + 8 * ratio);

    const units = [...new Set(seriesList.map((series) => series.unit).filter(Boolean))];
    if (units.length) {
      const unitText = units.length === 1 ? `单位 ${units[0]}` : `混合 ${units.join(" / ")}`;
      ctx.textAlign = "right";
      ctx.fillStyle = units.length === 1 ? "#90b6c8" : "#ffcf70";
      ctx.fillText(unitText, width - pad.right, 16 * ratio);
      ctx.textAlign = "left";
    }
  };

  const renderMainChart = (keys) => {
    const canvas = document.getElementById("mainChart");
    const showMulti = document.getElementById("multiLineChart")?.checked;
    if (!canvas) return;
    if (!showMulti) {
      drawChart(canvas, []);
      return;
    }

    const seriesList = keys
      .map((key) => ({
        key,
        color: chartMap[key]?.color || "#7de3ff",
        label: chartMap[key]?.label || key,
        unit: chartMap[key]?.unit || "",
        values: seriesSlice(key),
      }))
      .filter((series) => series.values.length);

    drawChart(canvas, seriesList);

    const ctx = canvas.getContext("2d");
    const ratio = window.devicePixelRatio || 1;
    let legendX = 60 * ratio;
    const legendY = 18 * ratio;
    ctx.font = `${11 * ratio}px Segoe UI`;
    seriesList.forEach((series) => {
      ctx.fillStyle = series.color;
      ctx.fillRect(legendX, legendY - 8 * ratio, 12 * ratio, 12 * ratio);
      ctx.fillStyle = "#d8f3ff";
      ctx.fillText(series.label, legendX + 18 * ratio, legendY + 2 * ratio);
      legendX += 24 * ratio + ctx.measureText(series.label).width;
    });
  };

  const renderIndividualCharts = (keys) => {
    const container = document.getElementById("individualCharts");
    if (!container) return;
    container.innerHTML = "";

    if (!keys.length) {
      container.innerHTML = '<div class="empty-hint">勾选指标后会在这里生成单图，也可以直接点上方预设。</div>';
      return;
    }

    keys.forEach((key) => {
      if (!rangeState[key]) rangeState[key] = { enabled: false, min: 0, max: 1 };

      const meta = chartMap[key] || { label: key, unit: "raw", color: "#7de3ff" };
      const rangeStep = rangeStepForKey(key);

      const box = document.createElement("div");
      box.className = "chart-box";
      box.innerHTML = `
        <div class="chart-box-head">
          <h4>${meta.label}</h4>
          <span class="chart-unit-chip">${meta.unit || "raw"}</span>
        </div>
        <div class="range-controls">
          <label><input type="checkbox" class="child-enable" ${
            rangeState[key].enabled ? "checked" : ""
          } /> 固定范围</label>
          <span>min</span>
          <input type="number" class="child-min" value="${rangeState[key].min}" step="${rangeStep}" />
          <span>max</span>
          <input type="number" class="child-max" value="${rangeState[key].max}" step="${rangeStep}" />
          <button type="button" class="apply-range">应用</button>
        </div>
      `;

      const canvas = document.createElement("canvas");
      canvas.className = "child-chart";
      canvas.style.height = "240px";
      box.appendChild(canvas);
      container.appendChild(box);

      box.querySelector(".apply-range")?.addEventListener("click", () => {
        rangeState[key] = {
          enabled: !!box.querySelector(".child-enable")?.checked,
          min: Number(box.querySelector(".child-min")?.value),
          max: Number(box.querySelector(".child-max")?.value),
        };
        renderAll();
      });

      drawChart(
        canvas,
        [
          {
            key,
            color: meta.color,
            label: meta.label,
            unit: meta.unit,
            values: seriesSlice(key),
          },
        ],
        rangeState[key],
      );
    });
  };

  async function refreshData(force = false) {
    if (!force && !active) return;
    try {
      const res = await fetch(`/data?ts=${Date.now()}`, { cache: "no-store" });
      if (!res.ok) return;
      latestData = await res.json();
      renderAll();
    } catch (err) {
      console.warn("fetch /data failed", err);
    }
  }

  function renderAll() {
    const keys = selectedKeys();
    updateSelectionSummary(keys);
    updatePresetState(keys);
    renderMainChart(keys);
    renderIndividualCharts(keys);
  }

  function init() {
    ensureControls();
    refreshData(true);
    window.addEventListener("resize", renderAll);
    window.setInterval(refreshData, 200);
  }

  function setActive(nextActive) {
    active = !!nextActive;
    if (active) {
      refreshData(true);
      window.requestAnimationFrame(renderAll);
    }
  }

  return {
    init,
    renderAll,
    setActive,
  };
})();

window.DebugCharts = DebugCharts;
