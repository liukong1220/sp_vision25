function jsonToHtml(data, parent) {
  parent.innerHTML = "";

  const buildTree = (value, container) => {
    if (typeof value !== "object" || value === null) {
      const line = document.createElement("div");
      line.className = "json-value";
      line.textContent = String(value);
      container.appendChild(line);
      return;
    }

    const list = document.createElement("ul");
    list.className = "json-tree";
    const entries = Array.isArray(value) ? value.map((item, index) => [index, item]) : Object.entries(value);

    entries.forEach(([key, child]) => {
      const item = document.createElement("li");
      if (typeof child === "object" && child !== null) {
        const details = document.createElement("details");
        details.open = true;
        const summary = document.createElement("summary");
        summary.innerHTML = `<span class="json-key">${key}</span>`;
        details.appendChild(summary);
        buildTree(child, details);
        item.appendChild(details);
      } else {
        item.innerHTML = `<span class="json-key">${key}</span>: <span class="json-value">${child}</span>`;
      }
      list.appendChild(item);
    });

    container.appendChild(list);
  };

  buildTree(data, parent);
}

let lastLogText = "";

async function fetchAndDisplayJsonWithTree(id, url) {
  const container = document.getElementById(id);
  const parent = document.getElementById(`${id}-container`) || container;
  try {
    parent.classList.add("json-updating");
    const res = await fetch(`${url}?ts=${Date.now()}`, { cache: "no-store" });
    if (!res.ok) {
      throw new Error(res.statusText);
    }

    const data = await res.json();
    const text = JSON.stringify(data);
    if (text !== lastLogText) {
      jsonToHtml(data, container);
      lastLogText = text;
    }
  } catch (err) {
    container.innerHTML = `<div class="empty-hint">日志请求失败: ${err.message}</div>`;
  } finally {
    parent.classList.remove("json-updating");
  }
}
