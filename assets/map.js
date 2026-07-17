"use strict";

const state = {
  points: [],
  visiblePoints: [],
  screenPoints: [],
  scale: 1,
  offsetX: 0,
  offsetY: 0,
  dragging: false,
  dragX: 0,
  dragY: 0,
  framePending: false,
  updatedLabel: "",
  spatialIndex: new Map(),
  hoverFrame: 0,
  hoverEvent: null,
};

const canvas = document.querySelector("#research-map");
const context = canvas.getContext("2d", { alpha: false });
const titleElement = document.querySelector("#map-title");
const statusElement = document.querySelector("#map-status");
const tooltip = document.querySelector("#tooltip");
const searchInput = document.querySelector("#search");
const groupFilter = document.querySelector("#group-filter");
const resetButton = document.querySelector("#reset-view");
const detailPanel = document.querySelector("#detail-panel");
const detailTitle = document.querySelector("#detail-title");
const detailMeta = document.querySelector("#detail-meta");
const detailAuthors = document.querySelector("#detail-authors");
const detailFaculty = document.querySelector("#detail-faculty");
const detailLink = document.querySelector("#detail-link");
const closeDetail = document.querySelector("#close-detail");

function colorFor(point) {
  const selectedGroup = groupFilter.value;
  if (selectedGroup && point.groups.includes(selectedGroup)) return "#62d8ff";
  return "#aab5c0";
}

function resizeCanvas() {
  const ratio = window.devicePixelRatio || 1;
  const bounds = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(bounds.width * ratio));
  canvas.height = Math.max(1, Math.round(bounds.height * ratio));
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  scheduleDraw();
}

function mapToScreen(point, width, height) {
  const margin = 28;
  const plotWidth = Math.max(1, width - margin * 2);
  const plotHeight = Math.max(1, height - margin * 2);
  const baseScale = Math.min(plotWidth, plotHeight) / 2;
  return {
    x: width / 2 + point.x * baseScale * state.scale + state.offsetX,
    y: height / 2 - point.y * baseScale * state.scale + state.offsetY,
    point,
  };
}

function draw() {
  state.framePending = false;
  const bounds = canvas.getBoundingClientRect();
  context.fillStyle = "#0d1014";
  context.fillRect(0, 0, bounds.width, bounds.height);
  state.screenPoints = state.visiblePoints.map((point) =>
    mapToScreen(point, bounds.width, bounds.height),
  );
  state.spatialIndex = new Map();

  context.globalAlpha = 0.72;
  for (const screenPoint of state.screenPoints) {
    if (
      screenPoint.x < -4 ||
      screenPoint.y < -4 ||
      screenPoint.x > bounds.width + 4 ||
      screenPoint.y > bounds.height + 4
    ) {
      continue;
    }
    const cellX = Math.floor(screenPoint.x / 20);
    const cellY = Math.floor(screenPoint.y / 20);
    const cellKey = `${cellX}:${cellY}`;
    const cell = state.spatialIndex.get(cellKey) || [];
    cell.push(screenPoint);
    state.spatialIndex.set(cellKey, cell);
    context.beginPath();
    context.arc(screenPoint.x, screenPoint.y, 2.1, 0, Math.PI * 2);
    context.fillStyle = colorFor(screenPoint.point);
    context.fill();
  }
  context.globalAlpha = 1;
}

function scheduleDraw() {
  if (!state.framePending) {
    state.framePending = true;
    window.requestAnimationFrame(draw);
  }
}

function applyFilters() {
  const query = searchInput.value.trim().toLocaleLowerCase();
  const group = groupFilter.value;
  state.visiblePoints = state.points.filter((point) => {
    if (group && !point.groups.includes(group)) return false;
    if (!query) return true;
    return `${point.title}\n${point.authors}\n${point.faculty.join(" ")}\n${point.venue}`
      .toLocaleLowerCase()
      .includes(query);
  });
  statusElement.dataset.filteredCount = String(state.visiblePoints.length);
  const total = state.points.length.toLocaleString();
  const visible = state.visiblePoints.length.toLocaleString();
  const countLabel =
    state.visiblePoints.length === state.points.length
      ? `${total} publications`
      : `${visible} of ${total} publications`;
  statusElement.textContent = state.updatedLabel
    ? `${countLabel} · ${state.updatedLabel}`
    : countLabel;
  tooltip.hidden = true;
  scheduleDraw();
}

function populateGroups(points) {
  const groups = [...new Set(points.flatMap((point) => point.groups))].sort((a, b) =>
    a.localeCompare(b),
  );
  for (const group of groups) {
    const option = document.createElement("option");
    option.value = group;
    option.textContent = group;
    groupFilter.append(option);
  }
}

function nearestPoint(clientX, clientY) {
  const bounds = canvas.getBoundingClientRect();
  const x = clientX - bounds.left;
  const y = clientY - bounds.top;
  let nearest = null;
  let distanceSquared = 100;
  const cellX = Math.floor(x / 20);
  const cellY = Math.floor(y / 20);
  for (let offsetX = -1; offsetX <= 1; offsetX += 1) {
    for (let offsetY = -1; offsetY <= 1; offsetY += 1) {
      const cell = state.spatialIndex.get(`${cellX + offsetX}:${cellY + offsetY}`) || [];
      for (const screenPoint of cell) {
        const dx = screenPoint.x - x;
        const dy = screenPoint.y - y;
        const candidate = dx * dx + dy * dy;
        if (candidate < distanceSquared) {
          distanceSquared = candidate;
          nearest = screenPoint;
        }
      }
    }
  }
  return nearest ? { ...nearest, pointerX: x, pointerY: y } : null;
}

function showTooltip(event) {
  if (state.dragging) return;
  const nearest = nearestPoint(event.clientX, event.clientY);
  if (!nearest) {
    tooltip.hidden = true;
    return;
  }
  const point = nearest.point;
  tooltip.replaceChildren();
  const heading = document.createElement("strong");
  heading.textContent = point.title;
  const details = document.createElement("span");
  const year = point.year ? ` · ${point.year}` : "";
  details.textContent = `${point.faculty.join(", ")}${year} · ${point.citation_count.toLocaleString()} citations`;
  tooltip.append(heading, details);
  const bounds = canvas.getBoundingClientRect();
  const tooltipWidth = Math.min(384, bounds.width - 32);
  tooltip.style.left = `${Math.min(nearest.pointerX + 14, bounds.width - tooltipWidth - 12)}px`;
  tooltip.style.top = `${Math.max(12, nearest.pointerY - 72)}px`;
  tooltip.hidden = false;
}

function showDetails(point) {
  detailTitle.textContent = point.title;
  const year = point.year || "Year unavailable";
  const venue = point.venue || "Venue unavailable";
  detailMeta.textContent = `${year} · ${venue} · ${point.citation_count.toLocaleString()} citations`;
  detailAuthors.textContent = point.authors ? `Authors: ${point.authors}` : "";
  detailFaculty.textContent = `CMU faculty: ${point.faculty.join(", ")}`;
  if (point.source_url) {
    detailLink.href = point.source_url;
    detailLink.hidden = false;
  } else {
    detailLink.removeAttribute("href");
    detailLink.hidden = true;
  }
  detailPanel.hidden = false;
}

canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  const bounds = canvas.getBoundingClientRect();
  const pointerX = event.clientX - bounds.left - bounds.width / 2;
  const pointerY = event.clientY - bounds.top - bounds.height / 2;
  const oldScale = state.scale;
  const multiplier = Math.exp(-event.deltaY * 0.0012);
  state.scale = Math.min(25, Math.max(0.7, oldScale * multiplier));
  const ratio = state.scale / oldScale;
  state.offsetX = pointerX - (pointerX - state.offsetX) * ratio;
  state.offsetY = pointerY - (pointerY - state.offsetY) * ratio;
  tooltip.hidden = true;
  scheduleDraw();
}, { passive: false });

canvas.addEventListener("pointerdown", (event) => {
  state.dragging = true;
  state.dragX = event.clientX;
  state.dragY = event.clientY;
  canvas.classList.add("dragging");
  canvas.setPointerCapture(event.pointerId);
  tooltip.hidden = true;
});

canvas.addEventListener("pointermove", (event) => {
  if (state.dragging) {
    state.offsetX += event.clientX - state.dragX;
    state.offsetY += event.clientY - state.dragY;
    state.dragX = event.clientX;
    state.dragY = event.clientY;
    scheduleDraw();
  } else {
    state.hoverEvent = event;
    if (!state.hoverFrame) {
      state.hoverFrame = window.requestAnimationFrame(() => {
        state.hoverFrame = 0;
        if (state.hoverEvent) showTooltip(state.hoverEvent);
      });
    }
  }
});

canvas.addEventListener("pointerup", (event) => {
  state.dragging = false;
  canvas.classList.remove("dragging");
  canvas.releasePointerCapture(event.pointerId);
});

canvas.addEventListener("click", (event) => {
  const nearest = nearestPoint(event.clientX, event.clientY);
  if (nearest) showDetails(nearest.point);
});

canvas.addEventListener("keydown", (event) => {
  const amount = event.shiftKey ? 80 : 30;
  if (event.key === "+" || event.key === "=") state.scale = Math.min(25, state.scale * 1.2);
  else if (event.key === "-") state.scale = Math.max(0.7, state.scale / 1.2);
  else if (event.key === "ArrowLeft") state.offsetX += amount;
  else if (event.key === "ArrowRight") state.offsetX -= amount;
  else if (event.key === "ArrowUp") state.offsetY += amount;
  else if (event.key === "ArrowDown") state.offsetY -= amount;
  else if (event.key === "Escape") detailPanel.hidden = true;
  else return;
  event.preventDefault();
  scheduleDraw();
});

canvas.addEventListener("pointerleave", () => {
  if (!state.dragging) tooltip.hidden = true;
});

searchInput.addEventListener("input", applyFilters);
groupFilter.addEventListener("change", applyFilters);
resetButton.addEventListener("click", () => {
  state.scale = 1;
  state.offsetX = 0;
  state.offsetY = 0;
  scheduleDraw();
});
closeDetail.addEventListener("click", () => {
  detailPanel.hidden = true;
  canvas.focus();
});
window.addEventListener("resize", resizeCanvas);

async function loadMap() {
  try {
    const configResponse = await fetch("map-config.json", { cache: "no-cache" });
    if (!configResponse.ok) throw new Error("Could not load map configuration");
    const config = await configResponse.json();
    titleElement.textContent = config.title;
    document.title = config.title;
    const artifactUrl =
      config.artifact_url ||
      `https://huggingface.co/datasets/${config.dataset_id}/resolve/${config.dataset_revision || "main"}/maps/${config.map_slug}.json`;
    const artifactResponse = await fetch(artifactUrl, { cache: "no-cache" });
    if (!artifactResponse.ok) throw new Error("Could not load the publication artifact");
    const artifact = await artifactResponse.json();
    if (artifact.schema_version !== 2 || !Array.isArray(artifact.points)) {
      throw new Error("The publication artifact has an unsupported schema");
    }
    state.points = artifact.points;
    state.visiblePoints = artifact.points;
    populateGroups(artifact.points);
    if (artifact.source_data_newest_at_utc) {
      const updated = new Date(artifact.source_data_newest_at_utc).toLocaleDateString();
      state.updatedLabel = `Newest profile refresh ${updated}`;
    } else {
      state.updatedLabel = "No verified Scholar profiles yet";
    }
    applyFilters();
    resizeCanvas();
  } catch (error) {
    console.error(error);
    statusElement.textContent = "The publication map is temporarily unavailable. The dataset link remains available.";
    statusElement.classList.add("error");
  }
}

loadMap();
