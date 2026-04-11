const DebugCharts = (() => {
  const chartMap = {
    gimbal_yaw: { label: "Gimbal Yaw", color: "#7de3ff" },
    gimbal_pitch: { label: "Gimbal Pitch", color: "#ffcf70" },
    target_yaw: { label: "Target Yaw", color: "#9df3bf" },
    target_pitch: { label: "Target Pitch", color: "#ff9db4" },
    plan_yaw: { label: "Plan Yaw", color: "#38f0b8" },
    plan_pitch: { label: "Plan Pitch", color: "#ff758f" },
    plan_yaw_vel: { label: "Plan Yaw Vel", color: "#78b7ff" },
    plan_pitch_vel: { label: "Plan Pitch Vel", color: "#f7a34b" },
    plan_yaw_acc: { label: "Plan Yaw Acc", color: "#45d6c2" },
    plan_pitch_acc: { label: "Plan Pitch Acc", color: "#ff89b0" },
    cmd_yaw: { label: "Cmd Yaw", color: "#52ffe0" },
    cmd_pitch: { label: "Cmd Pitch", color: "#ff8c94" },
    w: { label: "Spin W", color: "#8bd0ff" },
    target_z: { label: "Target Z", color: "#acf58e" },
    target_vz: { label: "Target VZ", color: "#ffcb77" },
    target_h: { label: "Target H", color: "#d7a6ff" },
    planner_delay_ms: { label: "Planner Delay", color: "#7de3ff" },
    planner_spin_gate: { label: "Spin Gate", color: "#ffe082" },
    planner_center_yaw: { label: "Center Yaw", color: "#9fc6ff" },
    planner_turn_sign: { label: "Turn Sign", color: "#87ffad" },
    planner_selected_armor: { label: "Selected Armor", color: "#ff95d6" },
    planner_selected_z_offset: { label: "Selected DZ", color: "#8af6cf" },
    bullet_speed: { label: "Bullet Speed", color: "#53d6ff" },
    fire: { label: "Fire", color: "#ff7388" },
    fired: { label: "Fired", color: "#ff8d55" },
    shoot: { label: "Shoot", color: "#ff7388" },
    residual_yaw: { label: "Residual Yaw", color: "#8df1a2" },
    residual_pitch: { label: "Residual Pitch", color: "#ffd166" },
    residual_distance: { label: "Residual Dist", color: "#6ef3ff" },
    nis: { label: "NIS", color: "#d087ff" },
    nees: { label: "NEES", color: "#ff7dc0" },
  };

  const defaultKeys = [
    "gimbal_yaw",
    "target_yaw",
    "plan_yaw",
    "gimbal_pitch",
    "target_pitch",
    "plan_pitch",
    "w",
    "planner_delay_ms",
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

  const ensureControls = () => {
    const container = document.getElementById("chart-select-controls");
    container.innerHTML = "";

    const multiChip = document.createElement("label");
    multiChip.className = "checkbox-chip";
    multiChip.innerHTML = '<input type="checkbox" id="multiLineChart" checked /> 显示总图';
    container.appendChild(multiChip);

    Object.entries(chartMap).forEach(([key, meta]) => {
      const chip = document.createElement("label");
      chip.className = "checkbox-chip";
      chip.innerHTML = `<input type="checkbox" class="chart-metric" data-key="${key}" ${
        defaultKeys.includes(key) ? "checked" : ""
      } /> ${meta.label}`;
      container.appendChild(chip);
    });

    container.addEventListener("change", renderAll);
    document.getElementById("apply-main-range").addEventListener("click", () => {
      mainMaxPoints = Number(document.getElementById("mainMaxPts").value) || 120;
      renderAll();
    });
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
      top: 26 * ratio,
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
    ctx.fillText(maxValue.toFixed(2), 8 * ratio, pad.top + 4 * ratio);
    ctx.fillText(minValue.toFixed(2), 8 * ratio, height - pad.bottom + 8 * ratio);
  };

  const renderMainChart = (keys) => {
    const canvas = document.getElementById("mainChart");
    const showMulti = document.getElementById("multiLineChart").checked;
    if (!showMulti) {
      drawChart(canvas, []);
      return;
    }

    const seriesList = keys
      .map((key) => ({
        key,
        color: chartMap[key]?.color || "#7de3ff",
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
      const label = chartMap[series.key]?.label || series.key;
      ctx.fillText(label, legendX + 18 * ratio, legendY + 2 * ratio);
      legendX += 24 * ratio + ctx.measureText(label).width;
    });
  };

  const renderIndividualCharts = (keys) => {
    const container = document.getElementById("individualCharts");
    container.innerHTML = "";

    if (!keys.length) {
      container.innerHTML = '<div class="empty-hint">勾选指标后会在这里生成单图。</div>';
      return;
    }

    keys.forEach((key) => {
      if (!rangeState[key]) rangeState[key] = { enabled: false, min: 0, max: 1 };

      const box = document.createElement("div");
      box.className = "chart-box";
      box.innerHTML = `
        <h4>${chartMap[key]?.label || key}</h4>
        <div class="range-controls">
          <label><input type="checkbox" class="child-enable" ${
            rangeState[key].enabled ? "checked" : ""
          } /> 固定范围</label>
          <span>min</span>
          <input type="number" class="child-min" value="${rangeState[key].min}" step="0.1" />
          <span>max</span>
          <input type="number" class="child-max" value="${rangeState[key].max}" step="0.1" />
          <button type="button" class="apply-range">应用</button>
        </div>
      `;

      const canvas = document.createElement("canvas");
      canvas.className = "child-chart";
      canvas.style.height = "240px";
      box.appendChild(canvas);
      container.appendChild(box);

      box.querySelector(".apply-range").addEventListener("click", () => {
        rangeState[key] = {
          enabled: box.querySelector(".child-enable").checked,
          min: Number(box.querySelector(".child-min").value),
          max: Number(box.querySelector(".child-max").value),
        };
        renderAll();
      });

      drawChart(
        canvas,
        [
          {
            key,
            color: chartMap[key]?.color || "#7de3ff",
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
