"use strict";

const TITLE_COLOR_CAPACITY = 32;
const SEQUENTIAL_COLOR_STEPS = 48;
const core = globalThis.ResearchMapCore;

const state = {
  points: [],
  matchedPoints: [],
  drawablePoints: [],
  screenPoints: [],
  departments: [],
  faculty: [],
  layouts: [],
  departmentById: new Map(),
  facultyById: new Map(),
  layoutById: new Map(),
  departmentColors: new Map(),
  facultyColors: new Map(),
  titleColors: new Map(),
  titlePalette: [],
  yearPalette: [],
  citationPalette: [],
  yearRange: null,
  citationMaximum: 0,
  citationMaximumClipped: false,
  activeTitleTerms: new Map(),
  activeDepartments: new Set(),
  activeFaculty: new Set(),
  colorMode: "department",
  displayMode: "highlight",
  layoutId: "",
  filtersActive: false,
  selectedWorkId: "",
  scale: 1,
  offsetX: 0,
  offsetY: 0,
  dragging: false,
  dragX: 0,
  dragY: 0,
  dragDistance: 0,
  framePending: false,
  sourceLabel: "",
  omittedPointCount: 0,
  canvasWidth: 0,
  canvasHeight: 0,
  spatialIndex: new Map(),
  hoverFrame: 0,
  hoverEvent: null,
};

const canvas = document.querySelector("#research-map");
const context = canvas.getContext("2d", { alpha: false });

function resolvedTheme() {
  return document.documentElement.dataset.resolvedTheme === "light"
    ? "light"
    : "dark";
}

function themeColor(property, fallback) {
  return getComputedStyle(canvas).getPropertyValue(property).trim() || fallback;
}

function colorAt(palette, amount) {
  if (!palette.length) return themeColor("--canvas-default", "#79b7cf");
  const position = Math.min(1, Math.max(0, Number(amount) || 0));
  return palette[Math.round(position * (palette.length - 1))];
}

function yearColor(point) {
  if (!Number.isInteger(point.year) || !state.yearRange) {
    return themeColor("--canvas-unassigned", "#657789");
  }
  const { minimum, maximum } = state.yearRange;
  const amount =
    maximum === minimum ? 0.5 : (point.year - minimum) / (maximum - minimum);
  return colorAt(state.yearPalette, amount);
}

function citationColor(point) {
  const citations = Math.max(0, Number(point.citation_count) || 0);
  const amount = state.citationMaximum
    ? Math.log1p(citations) / Math.log1p(state.citationMaximum)
    : 0.5;
  return colorAt(state.citationPalette, amount);
}

const titleElement = document.querySelector("#map-title");
const statusElement = document.querySelector("#map-status");
const retryLoadButton = document.querySelector("#retry-load");
const filterPanel = document.querySelector(".filter-panel");
const mapColumn = document.querySelector(".map-column");
const tooltip = document.querySelector("#tooltip");
const titleSearch = document.querySelector("#title-search");
const selectedTitles = document.querySelector("#selected-titles");
const authorSearch = document.querySelector("#author-search");
const authorSuggestions = document.querySelector("#author-suggestions");
const selectedAuthors = document.querySelector("#selected-authors");
const departmentSearch = document.querySelector("#department-search");
const departmentSuggestions = document.querySelector(
  "#department-suggestions",
);
const selectedDepartments = document.querySelector("#selected-departments");
const layoutOptions = document.querySelector("#layout-options");
const layoutNote = document.querySelector("#layout-note");
const clearFiltersButton = document.querySelector("#clear-filters");
const zoomResultsButton = document.querySelector("#zoom-results");
const resetViewButton = document.querySelector("#reset-view");
const matchCount = document.querySelector("#match-count");
const matchLabel = document.querySelector("#match-label");
const mapLegend = document.querySelector("#map-legend");
const emptyState = document.querySelector("#empty-state");
const emptyTitle = document.querySelector("#empty-title");
const emptyCopy = document.querySelector("#empty-copy");
const detailPanel = document.querySelector("#detail-panel");
const detailTitle = document.querySelector("#detail-title");
const detailMeta = document.querySelector("#detail-meta");
const detailAuthors = document.querySelector("#detail-authors");
const detailFaculty = document.querySelector("#detail-faculty");
const detailDepartments = document.querySelector("#detail-departments");
const detailLink = document.querySelector("#detail-link");
const closeDetail = document.querySelector("#close-detail");
const colorKeyDialog = document.querySelector("#color-key-dialog");
const colorKeyKicker = document.querySelector("#color-key-kicker");
const colorKeyTitle = document.querySelector("#color-key-title");
const colorKeySearch = document.querySelector("#color-key-search");
const colorKeySearchLabel = document.querySelector("#color-key-search-label");
const colorKeyCount = document.querySelector("#color-key-count");
const colorKeyList = document.querySelector("#color-key-list");
const closeColorKey = document.querySelector("#close-color-key");
let colorKeyMode = "faculty";

function normalizedText(value) {
  return core.normalizedText(value);
}

function facultyNames(point) {
  return point.faculty_ids
    .map((personId) => state.facultyById.get(personId)?.display_name)
    .filter(Boolean);
}

function departmentNames(point) {
  return point.department_ids
    .map((departmentId) => state.departmentById.get(departmentId)?.title)
    .filter(Boolean);
}

function preparePoint(point) {
  const coordinates = new Map(Object.entries(point._coordinates));
  const activeCoordinates = coordinates.get(state.layoutId);
  return {
    ...point,
    x: activeCoordinates?.x,
    y: activeCoordinates?.y,
    department_ids: Array.isArray(point.department_ids)
      ? point.department_ids
      : [],
    faculty_ids: Array.isArray(point.faculty_ids) ? point.faculty_ids : [],
    _coordinates: coordinates,
    _title: point._title,
  };
}

function layoutDescription(layout) {
  return layout ? `${layout.method}: ${layout.description}` : "";
}

function populateLayouts() {
  layoutOptions.replaceChildren();
  for (const layout of state.layouts) {
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "radio";
    input.name = "layout-mode";
    input.value = layout.layout_id;
    input.checked = layout.layout_id === state.layoutId;
    input.setAttribute("aria-describedby", "layout-note");
    const text = document.createElement("span");
    text.textContent = layout.label;
    label.append(input, text);
    layoutOptions.append(label);
  }
  const layout = state.layoutById.get(state.layoutId);
  layoutNote.textContent = layoutDescription(layout);
}

function activateLayout(layoutId, { fit = true } = {}) {
  const layout = state.layoutById.get(layoutId);
  if (!layout || layoutId === state.layoutId) return;
  state.layoutId = layoutId;
  for (const point of state.points) {
    const coordinates = point._coordinates.get(layoutId);
    point.x = coordinates.x;
    point.y = coordinates.y;
  }
  layoutOptions
    .querySelectorAll('input[name="layout-mode"]')
    .forEach((input) => {
      input.checked = input.value === layoutId;
    });
  layoutNote.textContent = layoutDescription(layout);
  tooltip.hidden = true;
  if (fit) resetView();
  else scheduleDraw();
}

function resizeCanvas() {
  const ratio = Math.min(2, window.devicePixelRatio || 1);
  const bounds = canvas.getBoundingClientRect();
  if (bounds.width <= 0 || bounds.height <= 0 || !context) return;
  const hadSize = state.canvasWidth > 0 && state.canvasHeight > 0;
  const oldScale = core.baseScale(state.canvasWidth, state.canvasHeight);
  const centerX = state.canvasWidth
    ? -state.offsetX / (oldScale * state.scale)
    : 0;
  const centerY = state.canvasHeight
    ? state.offsetY / (oldScale * state.scale)
    : 0;
  state.canvasWidth = bounds.width;
  state.canvasHeight = bounds.height;
  canvas.width = Math.max(1, Math.round(bounds.width * ratio));
  canvas.height = Math.max(1, Math.round(bounds.height * ratio));
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  if (hadSize) {
    const newScale = core.baseScale(state.canvasWidth, state.canvasHeight);
    state.offsetX = -centerX * newScale * state.scale;
    state.offsetY = centerY * newScale * state.scale;
  }
  scheduleDraw();
}

function baseScale(width, height) {
  return core.baseScale(width, height);
}

function mapToScreen(point, width, height) {
  const unitScale = baseScale(width, height);
  return {
    x: width / 2 + point.x * unitScale * state.scale + state.offsetX,
    y: height / 2 - point.y * unitScale * state.scale + state.offsetY,
    point,
    matched: point._matched,
  };
}

function drawGrid(width, height) {
  const spacing = 48;
  context.save();
  context.strokeStyle = themeColor("--canvas-grid", "#ffffff08");
  context.lineWidth = 1;
  context.beginPath();
  for (let x = 0.5; x < width; x += spacing) {
    context.moveTo(x, 0);
    context.lineTo(x, height);
  }
  for (let y = 0.5; y < height; y += spacing) {
    context.moveTo(0, y);
    context.lineTo(width, y);
  }
  context.stroke();
  context.restore();
}

function drawBatch(screenPoints, radius, fillStyle, alpha) {
  if (!screenPoints.length) return;
  context.save();
  context.fillStyle = fillStyle;
  context.globalAlpha = alpha;
  context.beginPath();
  for (const screenPoint of screenPoints) {
    context.moveTo(screenPoint.x + radius, screenPoint.y);
    context.arc(screenPoint.x, screenPoint.y, radius, 0, Math.PI * 2);
  }
  context.fill();
  context.restore();
}

function colorFor(point) {
  if (state.colorMode === "title" && state.activeTitleTerms.size) {
    for (const query of state.activeTitleTerms.keys()) {
      if (point._title?.includes(query)) return state.titleColors.get(query);
    }
  }
  if (state.colorMode === "year") return yearColor(point);
  if (state.colorMode === "citations") return citationColor(point);
  if (state.colorMode === "faculty") {
    const personId = core.preferredId(
      point.faculty_ids,
      state.activeFaculty,
      state.facultyColors,
    );
    if (personId) return state.facultyColors.get(personId);
    return themeColor("--canvas-unassigned", "#657789");
  }
  if (state.colorMode === "department") {
    const departmentId = core.preferredId(
      point.department_ids,
      state.activeDepartments,
      state.departmentColors,
    );
    if (departmentId) return state.departmentColors.get(departmentId);
  }
  if (
    state.activeTitleTerms.size ||
    state.activeFaculty.size ||
    state.activeDepartments.size
  ) {
    return themeColor("--canvas-highlight", "#73c5df");
  }
  return themeColor("--canvas-default", "#79b7cf");
}

function drawMatched(screenPoints) {
  if (!screenPoints.length) return;
  const groups = new Map();
  for (const screenPoint of screenPoints) {
    const color = colorFor(screenPoint.point);
    const group = groups.get(color) || [];
    group.push(screenPoint);
    groups.set(color, group);
  }
  let radius = 0.72;
  if (state.filtersActive) {
    const matchTotal = state.matchedPoints.length;
    if (matchTotal > 7500) radius = 0.9;
    else if (matchTotal > 3000) radius = 1.05;
    else if (matchTotal > 1000) radius = 1.25;
    else radius = 1.7;
  }
  const quantitativeMode = ["year", "citations"].includes(state.colorMode);
  const alpha = state.filtersActive ? 0.9 : quantitativeMode ? 0.68 : 0.54;
  for (const [color, group] of groups) {
    drawBatch(group, radius, color, alpha);
  }
  if (state.filtersActive && state.matchedPoints.length <= 1500) {
    context.save();
    context.strokeStyle = themeColor("--canvas-selected", "#f4fbff");
    context.globalAlpha = 0.58;
    context.lineWidth = 0.65;
    context.beginPath();
    for (const screenPoint of screenPoints) {
      context.moveTo(screenPoint.x + radius + 0.75, screenPoint.y);
      context.arc(
        screenPoint.x,
        screenPoint.y,
        radius + 0.75,
        0,
        Math.PI * 2,
      );
    }
    context.stroke();
    context.restore();
  }
}

function indexPoint(screenPoint) {
  const cellX = Math.floor(screenPoint.x / 20);
  const cellY = Math.floor(screenPoint.y / 20);
  const cellKey = `${cellX}:${cellY}`;
  const cell = state.spatialIndex.get(cellKey) || [];
  cell.push(screenPoint);
  state.spatialIndex.set(cellKey, cell);
}

function drawSelected(screenPoints) {
  if (!state.selectedWorkId) return;
  const selected = screenPoints.find(
    (screenPoint) => screenPoint.point.work_id === state.selectedWorkId,
  );
  if (!selected) return;
  context.save();
  context.strokeStyle = themeColor("--canvas-selected", "#ffffff");
  context.globalAlpha = 0.95;
  context.lineWidth = 1.5;
  context.beginPath();
  context.arc(selected.x, selected.y, 7, 0, Math.PI * 2);
  context.stroke();
  context.restore();
}

function draw() {
  state.framePending = false;
  const bounds = canvas.getBoundingClientRect();
  context.fillStyle = themeColor("--canvas", "#071019");
  context.fillRect(0, 0, bounds.width, bounds.height);
  drawGrid(bounds.width, bounds.height);

  state.screenPoints = [];
  state.spatialIndex = new Map();
  for (const point of state.drawablePoints) {
    const screenPoint = mapToScreen(point, bounds.width, bounds.height);
    if (
      screenPoint.x < -8 ||
      screenPoint.y < -8 ||
      screenPoint.x > bounds.width + 8 ||
      screenPoint.y > bounds.height + 8
    ) {
      continue;
    }
    state.screenPoints.push(screenPoint);
    indexPoint(screenPoint);
  }

  const contextPoints = state.screenPoints.filter((point) => !point.matched);
  const matchedPoints = state.screenPoints.filter((point) => point.matched);
  drawBatch(
    contextPoints,
    0.55,
    themeColor("--canvas-context", "#748696"),
    0.12,
  );
  drawMatched(matchedPoints);
  drawSelected(state.screenPoints);
}

function scheduleDraw() {
  if (!state.framePending) {
    state.framePending = true;
    window.requestAnimationFrame(draw);
  }
}

function updateStatus() {
  const total = state.points.length;
  const matched = state.matchedPoints.length;
  matchCount.textContent = state.filtersActive
    ? `${matched.toLocaleString()} of ${total.toLocaleString()}`
    : total.toLocaleString();
  matchLabel.textContent = state.filtersActive
    ? "matching publications"
    : "publications";
  const countLabel = state.filtersActive
    ? `${matched.toLocaleString()} of ${total.toLocaleString()} publications match`
    : `${total.toLocaleString()} publications`;
  const statusParts = [countLabel];
  if (state.sourceLabel) statusParts.push(state.sourceLabel);
  if (state.omittedPointCount) {
    statusParts.push(
      `${state.omittedPointCount.toLocaleString()} invalid record${state.omittedPointCount === 1 ? "" : "s"} omitted`,
    );
  }
  statusElement.textContent = statusParts.join(" · ");
  statusElement.classList.toggle("warning", state.omittedPointCount > 0);
  emptyState.hidden = matched !== 0;
  if (!matched && total) {
    emptyTitle.textContent = "No publications match this combination.";
    emptyCopy.textContent =
      "Try a broader title, author, or department selection.";
  } else if (!total) {
    emptyTitle.textContent = "No publications are available yet.";
    emptyCopy.textContent = "The data source loaded successfully but is empty.";
  }
  clearFiltersButton.disabled = !state.filtersActive;
  zoomResultsButton.disabled = !state.filtersActive || matched === 0;
  canvas.setAttribute(
    "aria-label",
    `Semantic scatter plot showing ${matched.toLocaleString()} ${
      state.filtersActive ? "matching " : ""
    }publications`,
  );
}

function appendLegendItem(label, color, className = "legend-dot-match") {
  const item = document.createElement("span");
  const dot = document.createElement("i");
  dot.className = `legend-dot ${className}`;
  if (color) dot.style.backgroundColor = color;
  item.append(dot, document.createTextNode(label));
  mapLegend.append(item);
}

function compactNumber(value) {
  return new Intl.NumberFormat("en", {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(value);
}

function appendContinuousLegend({ title, labels, palette, accessibleLabel }) {
  const item = document.createElement("span");
  item.className = "continuous-legend";
  item.setAttribute("role", "img");
  item.setAttribute("aria-label", accessibleLabel);

  const heading = document.createElement("strong");
  heading.textContent = title;
  const gradient = document.createElement("i");
  gradient.className = "continuous-gradient";
  gradient.setAttribute("aria-hidden", "true");
  gradient.style.backgroundImage = `linear-gradient(90deg, ${palette
    .map(
      (color, index) =>
        `${color} ${((index / Math.max(1, palette.length - 1)) * 100).toFixed(2)}%`,
    )
    .join(", ")})`;
  const ticks = document.createElement("span");
  ticks.className = "continuous-ticks";
  for (const label of labels) {
    const tick = document.createElement("span");
    tick.textContent = label;
    ticks.append(tick);
  }
  item.append(heading, gradient, ticks);
  mapLegend.append(item);
}

function renderYearLegend() {
  if (!state.yearRange) {
    appendLegendItem(
      "Year unavailable",
      themeColor("--canvas-unassigned", "#657789"),
    );
    return;
  }
  const { minimum, maximum } = state.yearRange;
  const midpoint = Math.round((minimum + maximum) / 2);
  const minimumLabel = state.yearRange.clippedMinimum
    ? `≤${minimum}`
    : String(minimum);
  const maximumLabel = state.yearRange.clippedMaximum
    ? `≥${maximum}`
    : String(maximum);
  appendContinuousLegend({
    title: "Publication year",
    labels:
      minimum === maximum
        ? [String(minimum)]
        : [minimumLabel, String(midpoint), maximumLabel],
    palette: state.yearPalette,
    accessibleLabel:
      minimum === maximum
        ? `All dated publications are from ${minimum}`
        : `Publication year color scale from ${minimum}${
            state.yearRange.clippedMinimum ? " and earlier" : ""
          } to ${maximum}${
            state.yearRange.clippedMaximum ? " and later" : ""
          }`,
  });
  if (state.matchedPoints.some((point) => !Number.isInteger(point.year))) {
    appendLegendItem(
      "Year unavailable",
      themeColor("--canvas-unassigned", "#657789"),
    );
  }
}

function renderCitationLegend() {
  const maximum = state.citationMaximum;
  const midpoint = Math.round(Math.expm1(Math.log1p(maximum) / 2));
  const maximumLabel = state.citationMaximumClipped
    ? `≥${compactNumber(maximum)}`
    : compactNumber(maximum);
  appendContinuousLegend({
    title: "Citations · log scale",
    labels: maximum
      ? ["0", compactNumber(midpoint), maximumLabel]
      : ["0"],
    palette: state.citationPalette,
    accessibleLabel: `Citation count color scale from 0 to ${maximum.toLocaleString()}${
      state.citationMaximumClipped ? " and higher" : ""
    }, logarithmic when nonzero`,
  });
}

function colorKeyItems(mode = state.colorMode) {
  const isFaculty = mode === "faculty";
  const idField = isFaculty ? "person_id" : "department_id";
  const labelField = isFaculty ? "display_name" : "title";
  const pointField = isFaculty ? "faculty_ids" : "department_ids";
  const catalog = isFaculty ? state.faculty : state.departments;
  const colors = isFaculty ? state.facultyColors : state.departmentColors;
  const activeIds = isFaculty ? state.activeFaculty : state.activeDepartments;
  const counts = new Map();
  for (const point of state.matchedPoints) {
    for (const id of point[pointField]) {
      if (colors.has(id)) counts.set(id, (counts.get(id) || 0) + 1);
    }
  }
  return catalog
    .filter((item) => counts.has(item[idField]))
    .map((item) => ({
      id: item[idField],
      label: item[labelField],
      color: colors.get(item[idField]),
      count: counts.get(item[idField]),
      selected: activeIds.has(item[idField]),
    }));
}

function appendColorKeyButton(items, mode) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "faculty-legend-button";
  button.dataset.action = "open-color-key";
  button.dataset.colorMode = mode;
  button.setAttribute("aria-haspopup", "dialog");
  const spectrum = document.createElement("span");
  spectrum.className = "legend-spectrum";
  spectrum.setAttribute("aria-hidden", "true");
  const colorCount = items.length;
  const swatchCount = Math.min(10, colorCount);
  for (let index = 0; index < swatchCount; index += 1) {
    const swatch = document.createElement("i");
    const paletteIndex = Math.floor((index * colorCount) / swatchCount);
    swatch.style.backgroundColor = items[paletteIndex].color;
    spectrum.append(swatch);
  }
  const label = document.createElement("span");
  label.textContent = `View ${colorCount.toLocaleString()} ${
    mode === "faculty" ? "faculty" : "department"
  } colors`;
  button.append(spectrum, label);
  mapLegend.append(button);
}

function renderLegend() {
  mapLegend.replaceChildren();
  if (state.colorMode === "year") {
    renderYearLegend();
  } else if (state.colorMode === "citations") {
    renderCitationLegend();
  } else if (state.colorMode === "faculty") {
    const items = colorKeyItems("faculty");
    const selectedItems = items.filter((item) => item.selected);
    for (const item of selectedItems.slice(0, 3)) {
      appendLegendItem(item.label, item.color);
    }
    if (selectedItems.length > 3) {
      appendLegendItem(
        `+${selectedItems.length - 3} selected`,
        themeColor("--quiet", "#9dabb9"),
      );
    }
    const hasUnmappedMatches = state.matchedPoints.some(
      (point) =>
        !point.faculty_ids.some((personId) =>
          state.facultyColors.has(personId),
        ),
    );
    if (items.length) {
      appendColorKeyButton(items, "faculty");
      if (hasUnmappedMatches) {
        appendLegendItem(
          "No mapped faculty",
          themeColor("--canvas-unassigned", "#657789"),
        );
      }
    } else if (state.matchedPoints.length && hasUnmappedMatches) {
      appendLegendItem(
        "No mapped faculty",
        themeColor("--canvas-unassigned", "#657789"),
      );
    } else {
      appendLegendItem(
        "No faculty matches",
        themeColor("--canvas-unassigned", "#657789"),
      );
    }
  } else if (state.colorMode === "title") {
    const activeTerms = [...state.activeTitleTerms].map(([query, label]) => ({
      label,
      color: state.titleColors.get(query),
    }));
    if (activeTerms.length) {
      for (const term of activeTerms.slice(0, 4)) {
        appendLegendItem(term.label, term.color);
      }
      if (activeTerms.length > 4) {
        appendLegendItem(
          `+${activeTerms.length - 4} more`,
          themeColor("--quiet", "#9dabb9"),
        );
      }
    } else {
      appendLegendItem(
        state.filtersActive ? "Match" : "Publication",
        colorFor({ _title: "", department_ids: [], faculty_ids: [] }),
      );
    }
  } else if (state.colorMode === "department") {
    const items = colorKeyItems("department");
    const selectedItems = items.filter((item) => item.selected);
    if (selectedItems.length) {
      for (const item of selectedItems.slice(0, 4)) {
        appendLegendItem(item.label, item.color);
      }
      if (selectedItems.length > 4) {
        appendLegendItem(
          `+${selectedItems.length - 4} more`,
          themeColor("--quiet", "#9dabb9"),
        );
      }
    }
    if (items.length) {
      appendColorKeyButton(items, "department");
    } else {
      appendLegendItem(
        "No department matches",
        themeColor("--canvas-unassigned", "#657789"),
      );
    }
  }
  if (state.filtersActive && state.displayMode === "highlight") {
    appendLegendItem(
      "Context",
      themeColor("--canvas-context", "#748696"),
      "legend-dot-context",
    );
  }
}

function renderColorKey() {
  const query = normalizedText(colorKeySearch.value);
  const allItems = colorKeyItems(colorKeyMode);
  const items = allItems.filter((item) =>
    normalizedText(item.label).includes(query),
  );
  const noun = colorKeyMode === "faculty" ? "faculty" : "departments";
  colorKeyCount.textContent = query
    ? `${items.length.toLocaleString()} of ${allItems.length.toLocaleString()} ${noun} with matches`
    : `${allItems.length.toLocaleString()} ${noun} with matches · select any name to filter the map`;
  colorKeyList.replaceChildren();
  for (const item of items) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "faculty-legend-item";
    button.dataset.itemId = item.id;
    button.setAttribute("aria-pressed", String(item.selected));
    const swatch = document.createElement("i");
    swatch.style.backgroundColor = item.color;
    swatch.setAttribute("aria-hidden", "true");
    const name = document.createElement("span");
    name.textContent = item.label;
    const count = document.createElement("small");
    count.textContent = `${item.count.toLocaleString()} match${item.count === 1 ? "" : "es"}`;
    button.append(swatch, name, count);
    colorKeyList.append(button);
  }
  if (!items.length) {
    const message = document.createElement("p");
    message.className = "color-key-empty";
    message.textContent = "No colors match that search.";
    colorKeyList.append(message);
  }
}

function openColorKey(mode) {
  colorKeyMode = mode === "department" ? "department" : "faculty";
  const facultyMode = colorKeyMode === "faculty";
  colorKeyKicker.textContent = facultyMode
    ? "Faculty color key"
    : "Department color key";
  colorKeyTitle.textContent = facultyMode
    ? "Faculty represented in matches"
    : "Departments represented in matches";
  colorKeySearchLabel.textContent = facultyMode
    ? "Find a faculty member"
    : "Find a department";
  colorKeySearch.placeholder = facultyMode
    ? "Search faculty names"
    : "Search department names";
  colorKeySearch.value = "";
  renderColorKey();
  colorKeyDialog.showModal();
  colorKeySearch.focus();
}

function applyFilters() {
  const titleQueries = [...state.activeTitleTerms.keys()];
  state.filtersActive = Boolean(
    state.activeTitleTerms.size ||
      state.activeDepartments.size ||
      state.activeFaculty.size,
  );
  state.matchedPoints = state.points.filter((point) =>
    core.pointMatches(point, {
      titleQueries,
      departmentIds: state.activeDepartments,
      facultyIds: state.activeFaculty,
    }),
  );
  const matchedIds = new Set(
    state.matchedPoints.map((point) => point.work_id),
  );
  for (const point of state.points) {
    point._matched = matchedIds.has(point.work_id);
  }
  state.drawablePoints =
    state.filtersActive && state.displayMode === "show"
      ? state.matchedPoints
      : state.points;
  if (
    state.selectedWorkId &&
    state.displayMode === "show" &&
    !matchedIds.has(state.selectedWorkId)
  ) {
    detailPanel.hidden = true;
    state.selectedWorkId = "";
  }
  tooltip.hidden = true;
  updateStatus();
  renderLegend();
  if (colorKeyDialog.open) renderColorKey();
  scheduleDraw();
}

function initializeDepartmentColors() {
  const generator = globalThis.ResearchMapColors?.generatePerceptualPalette;
  if (!generator) throw new Error("Department color generator is unavailable");
  const palette = generator(state.departments.length, resolvedTheme());
  state.departments.forEach((department, index) => {
    state.departmentColors.set(department.department_id, palette[index]);
  });
}

function initializeSequentialColors() {
  const generator = globalThis.ResearchMapColors?.generateSequentialPalette;
  if (!generator) throw new Error("Sequential color generator is unavailable");
  state.yearPalette = generator(
    SEQUENTIAL_COLOR_STEPS,
    resolvedTheme(),
    "year",
  );
  state.citationPalette = generator(
    SEQUENTIAL_COLOR_STEPS,
    resolvedTheme(),
    "citations",
  );
  const years = state.points
    .map((point) => point.year)
    .filter(Number.isInteger)
    .sort((left, right) => left - right);
  if (years.length) {
    const actualMinimum = years[0];
    const actualMaximum = years.at(-1);
    const useRobustRange = years.length >= 100;
    const minimum = useRobustRange
      ? years[Math.floor((years.length - 1) * 0.01)]
      : actualMinimum;
    const maximum = useRobustRange
      ? years[Math.floor((years.length - 1) * 0.99)]
      : actualMaximum;
    state.yearRange = {
      minimum,
      maximum,
      clippedMinimum: actualMinimum < minimum,
      clippedMaximum: actualMaximum > maximum,
    };
  } else {
    state.yearRange = null;
  }
  const citations = state.points
    .map((point) => Math.max(0, Number(point.citation_count) || 0))
    .sort((left, right) => left - right);
  const actualCitationMaximum = citations.at(-1) || 0;
  const robustCitationMaximum =
    citations.length >= 100
      ? citations[Math.floor((citations.length - 1) * 0.99)]
      : actualCitationMaximum;
  state.citationMaximum = robustCitationMaximum || actualCitationMaximum;
  state.citationMaximumClipped =
    actualCitationMaximum > state.citationMaximum;
}

function createMultiSelect({
  input,
  suggestionsList,
  selectedList,
  optionPrefix,
  items,
  activeIds,
  itemId,
  itemLabel,
  itemCount,
  showAllOnEmpty = false,
  suggestionLimit = 8,
}) {
  let suggestions = [];
  let suggestionIndex = -1;
  let blurTimer = 0;

  function hideSuggestions() {
    suggestions = [];
    suggestionIndex = -1;
    suggestionsList.hidden = true;
    suggestionsList.replaceChildren();
    input.setAttribute("aria-expanded", "false");
    input.removeAttribute("aria-activedescendant");
  }

  function updateSuggestionSelection() {
    const buttons = suggestionsList.querySelectorAll("button");
    buttons.forEach((button, index) => {
      button.setAttribute("aria-selected", String(index === suggestionIndex));
    });
    if (suggestionIndex < 0) return;
    const activeId = `${optionPrefix}-option-${suggestionIndex}`;
    input.setAttribute("aria-activedescendant", activeId);
    document.querySelector(`#${activeId}`)?.scrollIntoView({ block: "nearest" });
  }

  function renderSuggestions() {
    window.clearTimeout(blurTimer);
    const query = normalizedText(input.value);
    if (!query && !showAllOnEmpty) {
      hideSuggestions();
      return;
    }
    suggestions = items()
      .filter(
        (item) =>
          !activeIds.has(itemId(item)) &&
          normalizedText(itemLabel(item)).includes(query),
      )
      .sort((left, right) => {
        if (!query) return 0;
        const leftLabel = normalizedText(itemLabel(left));
        const rightLabel = normalizedText(itemLabel(right));
        const leftScore = leftLabel.startsWith(query) ? 0 : 1;
        const rightScore = rightLabel.startsWith(query) ? 0 : 1;
        return leftScore - rightScore || leftLabel.localeCompare(rightLabel);
      })
      .slice(0, suggestionLimit);
    if (!suggestions.length) {
      suggestionIndex = -1;
      suggestionsList.replaceChildren();
      const message = document.createElement("li");
      message.className = "suggestion-empty";
      message.setAttribute("role", "option");
      message.setAttribute("aria-disabled", "true");
      message.textContent = query ? "No matches" : "No more options";
      suggestionsList.append(message);
      suggestionsList.hidden = false;
      input.setAttribute("aria-expanded", "true");
      input.removeAttribute("aria-activedescendant");
      return;
    }

    suggestionIndex = 0;
    suggestionsList.replaceChildren();
    suggestions.forEach((item, index) => {
      const listItem = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.id = `${optionPrefix}-option-${index}`;
      button.setAttribute("role", "option");
      button.setAttribute("aria-selected", String(index === suggestionIndex));
      button.dataset.itemId = itemId(item);
      const name = document.createElement("span");
      name.textContent = itemLabel(item);
      const count = document.createElement("small");
      count.textContent = `${Number(itemCount(item) || 0).toLocaleString()} works`;
      button.append(name, count);
      listItem.append(button);
      suggestionsList.append(listItem);
    });
    suggestionsList.hidden = false;
    input.setAttribute("aria-expanded", "true");
    input.setAttribute("aria-activedescendant", `${optionPrefix}-option-0`);
  }

  function renderSelected() {
    selectedList.replaceChildren();
    const selectedItems = items().filter((item) => activeIds.has(itemId(item)));
    for (const item of selectedItems) {
      const id = itemId(item);
      const label = itemLabel(item);
      const button = document.createElement("button");
      button.type = "button";
      button.className = "filter-chip";
      button.dataset.itemId = id;
      button.title = `Remove ${label}`;
      button.setAttribute("aria-label", `Remove ${label}`);
      const name = document.createElement("span");
      name.textContent = label;
      const close = document.createElement("i");
      close.textContent = "×";
      close.setAttribute("aria-hidden", "true");
      button.append(name, close);
      selectedList.append(button);
    }
  }

  function addItem(id) {
    if (!items().some((item) => itemId(item) === id)) return;
    activeIds.add(id);
    input.value = "";
    hideSuggestions();
    renderSelected();
    applyFilters();
    input.focus();
  }

  input.addEventListener("input", renderSuggestions);
  input.addEventListener("focus", renderSuggestions);
  input.addEventListener("blur", () => {
    blurTimer = window.setTimeout(hideSuggestions, 120);
  });
  input.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !suggestionsList.hidden) {
      window.clearTimeout(blurTimer);
      hideSuggestions();
      event.preventDefault();
      return;
    }
    if (suggestionsList.hidden || !suggestions.length) return;
    if (event.key === "ArrowDown") {
      suggestionIndex = (suggestionIndex + 1) % suggestions.length;
    } else if (event.key === "ArrowUp") {
      suggestionIndex =
        (suggestionIndex - 1 + suggestions.length) % suggestions.length;
    } else if (event.key === "Enter") {
      addItem(itemId(suggestions[suggestionIndex]));
      event.preventDefault();
      return;
    } else {
      return;
    }
    event.preventDefault();
    updateSuggestionSelection();
  });

  suggestionsList.addEventListener("pointerdown", (event) => {
    event.preventDefault();
  });
  suggestionsList.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-item-id]");
    if (button) addItem(button.dataset.itemId);
  });

  selectedList.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-item-id]");
    if (!button) return;
    activeIds.delete(button.dataset.itemId);
    renderSelected();
    applyFilters();
  });

  return {
    clear() {
      input.value = "";
      activeIds.clear();
      hideSuggestions();
      renderSelected();
    },
    renderSelected,
  };
}

function initializeFacultyColors() {
  const generator = globalThis.ResearchMapColors?.generatePerceptualPalette;
  if (!generator) throw new Error("Faculty color generator is unavailable");
  const palette = generator(state.faculty.length, resolvedTheme());
  state.facultyColors = new Map(
    state.faculty.map((person, index) => [person.person_id, palette[index]]),
  );
}

function refreshThemeColors() {
  if (state.departments.length || state.faculty.length) {
    state.departmentColors = new Map();
    state.facultyColors = new Map();
    state.titleColors = new Map();
    state.titlePalette = [];
    initializeFacultyColors();
    initializeDepartmentColors();
    initializeSequentialColors();
    for (const query of state.activeTitleTerms.keys()) ensureTitleColor(query);
    renderLegend();
    if (colorKeyDialog.open) renderColorKey();
  }
  tooltip.hidden = true;
  scheduleDraw();
}

function ensureTitleColor(query) {
  if (state.titleColors.has(query)) return;
  const generator = globalThis.ResearchMapColors?.generatePerceptualPalette;
  if (!generator) throw new Error("Title color generator is unavailable");
  const colorIndex = state.titleColors.size;
  if (colorIndex >= state.titlePalette.length) {
    const nextCapacity = Math.max(
      TITLE_COLOR_CAPACITY,
      state.titlePalette.length * 2,
    );
    state.titlePalette = generator(nextCapacity, resolvedTheme());
  }
  state.titleColors.set(query, state.titlePalette[colorIndex]);
}

function renderSelectedTitles() {
  selectedTitles.replaceChildren();
  for (const [query, label] of state.activeTitleTerms) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "filter-chip";
    button.dataset.titleQuery = query;
    button.title = `Remove ${label}`;
    button.setAttribute("aria-label", `Remove title term ${label}`);
    const name = document.createElement("span");
    name.textContent = label;
    const close = document.createElement("i");
    close.textContent = "×";
    close.setAttribute("aria-hidden", "true");
    button.append(name, close);
    selectedTitles.append(button);
  }
}

function addTitleTerm() {
  const label = titleSearch.value.trim().replace(/\s+/g, " ");
  const query = normalizedText(label);
  if (!query) return;
  ensureTitleColor(query);
  if (!state.activeTitleTerms.has(query)) {
    state.activeTitleTerms.set(query, label);
  }
  titleSearch.value = "";
  renderSelectedTitles();
  applyFilters();
  titleSearch.focus();
}

const authorFilter = createMultiSelect({
  input: authorSearch,
  suggestionsList: authorSuggestions,
  selectedList: selectedAuthors,
  optionPrefix: "author",
  items: () => state.faculty,
  activeIds: state.activeFaculty,
  itemId: (person) => person.person_id,
  itemLabel: (person) => person.display_name,
  itemCount: (person) => person.publication_count,
});

const departmentFilter = createMultiSelect({
  input: departmentSearch,
  suggestionsList: departmentSuggestions,
  selectedList: selectedDepartments,
  optionPrefix: "department",
  items: () => state.departments,
  activeIds: state.activeDepartments,
  itemId: (department) => department.department_id,
  itemLabel: (department) => department.title,
  itemCount: (department) => department.publication_count,
  showAllOnEmpty: true,
  suggestionLimit: Number.POSITIVE_INFINITY,
});

function nearestPoint(clientX, clientY) {
  const bounds = canvas.getBoundingClientRect();
  const x = clientX - bounds.left;
  const y = clientY - bounds.top;
  const cellX = Math.floor(x / 20);
  const cellY = Math.floor(y / 20);
  let nearestMatch = null;
  let nearestContext = null;
  let matchDistance = 100;
  let contextDistance = 64;
  for (let offsetX = -1; offsetX <= 1; offsetX += 1) {
    for (let offsetY = -1; offsetY <= 1; offsetY += 1) {
      const key = `${cellX + offsetX}:${cellY + offsetY}`;
      for (const screenPoint of state.spatialIndex.get(key) || []) {
        const dx = screenPoint.x - x;
        const dy = screenPoint.y - y;
        const distance = dx * dx + dy * dy;
        if (screenPoint.matched && distance < matchDistance) {
          matchDistance = distance;
          nearestMatch = screenPoint;
        } else if (!screenPoint.matched && distance < contextDistance) {
          contextDistance = distance;
          nearestContext = screenPoint;
        }
      }
    }
  }
  const nearest = nearestMatch || nearestContext;
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
  const faculty = facultyNames(point).join(", ") || "CMU faculty unavailable";
  const year = point.year ? ` · ${point.year}` : "";
  details.textContent = `${faculty}${year} · ${Number(point.citation_count || 0).toLocaleString()} citations`;
  tooltip.append(heading, details);
  const bounds = canvas.getBoundingClientRect();
  const tooltipWidth = Math.min(368, bounds.width - 32);
  tooltip.style.left = `${Math.min(nearest.pointerX + 14, bounds.width - tooltipWidth - 12)}px`;
  tooltip.style.top = `${Math.max(12, nearest.pointerY - 78)}px`;
  tooltip.hidden = false;
}

function showDetails(point) {
  state.selectedWorkId = point.work_id;
  detailTitle.textContent = point.title;
  const year = point.year || "Year unavailable";
  const venue = point.venue || "Venue unavailable";
  const citations = Number(point.citation_count || 0).toLocaleString();
  const observations = Number(point.observation_count || 1).toLocaleString();
  detailMeta.textContent = `${year} · ${venue} · ${citations} citations · ${observations} source observation${Number(point.observation_count || 1) === 1 ? "" : "s"}`;
  detailAuthors.textContent = point.authors || "Author list unavailable";
  detailFaculty.textContent = facultyNames(point).join(", ") || "Unavailable";
  detailDepartments.textContent = departmentNames(point).join(" · ");
  if (point.source_url) {
    detailLink.href = point.source_url;
    detailLink.hidden = false;
  } else if (point.doi) {
    detailLink.href = `https://doi.org/${point.doi}`;
    detailLink.hidden = false;
  } else {
    detailLink.removeAttribute("href");
    detailLink.hidden = true;
  }
  detailPanel.hidden = false;
  scheduleDraw();
}

function resetView() {
  fitPoints(state.points);
}

function fitPoints(points) {
  if (!points.length) return false;
  const bounds = canvas.getBoundingClientRect();
  const view = core.fitView(points, bounds.width, bounds.height);
  if (!view) return false;
  Object.assign(state, view);
  tooltip.hidden = true;
  scheduleDraw();
  return true;
}

function pointNearestCanvasCenter() {
  const candidates = state.screenPoints.filter(
    (screenPoint) => !state.filtersActive || screenPoint.matched,
  );
  if (!candidates.length) return null;
  const centerX = state.canvasWidth / 2;
  const centerY = state.canvasHeight / 2;
  return candidates.reduce((nearest, candidate) => {
    const distance =
      (candidate.x - centerX) ** 2 + (candidate.y - centerY) ** 2;
    return !nearest || distance < nearest.distance
      ? { point: candidate.point, distance }
      : nearest;
  }, null)?.point;
}

canvas.addEventListener(
  "wheel",
  (event) => {
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
  },
  { passive: false },
);

canvas.addEventListener("pointerdown", (event) => {
  state.dragging = true;
  state.dragX = event.clientX;
  state.dragY = event.clientY;
  state.dragDistance = 0;
  canvas.classList.add("dragging");
  canvas.setPointerCapture(event.pointerId);
  tooltip.hidden = true;
});

canvas.addEventListener("pointermove", (event) => {
  if (state.dragging) {
    const deltaX = event.clientX - state.dragX;
    const deltaY = event.clientY - state.dragY;
    state.offsetX += deltaX;
    state.offsetY += deltaY;
    state.dragDistance += Math.abs(deltaX) + Math.abs(deltaY);
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

function stopDragging(event) {
  state.dragging = false;
  canvas.classList.remove("dragging");
  if (canvas.hasPointerCapture(event.pointerId)) {
    canvas.releasePointerCapture(event.pointerId);
  }
}

canvas.addEventListener("pointerup", stopDragging);
canvas.addEventListener("pointercancel", stopDragging);

canvas.addEventListener("click", (event) => {
  if (state.dragDistance > 5) return;
  const nearest = nearestPoint(event.clientX, event.clientY);
  if (nearest) showDetails(nearest.point);
});

canvas.addEventListener("keydown", (event) => {
  const amount = event.shiftKey ? 80 : 30;
  if (event.key === "+" || event.key === "=") {
    state.scale = Math.min(25, state.scale * 1.2);
  } else if (event.key === "-") {
    state.scale = Math.max(0.7, state.scale / 1.2);
  } else if (event.key === "ArrowLeft") {
    state.offsetX += amount;
  } else if (event.key === "ArrowRight") {
    state.offsetX -= amount;
  } else if (event.key === "ArrowUp") {
    state.offsetY += amount;
  } else if (event.key === "ArrowDown") {
    state.offsetY -= amount;
  } else if (event.key === "Escape") {
    detailPanel.hidden = true;
    state.selectedWorkId = "";
  } else if (event.key === "Home") {
    resetView();
  } else if (event.key === "Enter") {
    const point = pointNearestCanvasCenter();
    if (point) {
      showDetails(point);
      detailPanel.focus();
    }
  } else {
    return;
  }
  event.preventDefault();
  tooltip.hidden = true;
  scheduleDraw();
});

canvas.addEventListener("pointerleave", () => {
  if (!state.dragging) tooltip.hidden = true;
});

titleSearch.addEventListener("keydown", (event) => {
  if (event.key !== "Enter") return;
  event.preventDefault();
  addTitleTerm();
});

selectedTitles.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-title-query]");
  if (!button) return;
  state.activeTitleTerms.delete(button.dataset.titleQuery);
  renderSelectedTitles();
  applyFilters();
});

layoutOptions.addEventListener("change", (event) => {
  if (!event.target.matches('input[name="layout-mode"]')) return;
  activateLayout(event.target.value);
});

document.querySelectorAll('input[name="display-mode"]').forEach((input) => {
  input.addEventListener("change", () => {
    if (!input.checked) return;
    state.displayMode = input.value;
    applyFilters();
  });
});

document.querySelectorAll('input[name="color-mode"]').forEach((input) => {
  input.addEventListener("change", () => {
    if (!input.checked) return;
    state.colorMode = input.value;
    if (colorKeyDialog.open) colorKeyDialog.close();
    renderLegend();
    scheduleDraw();
  });
});

mapLegend.addEventListener("click", (event) => {
  const button = event.target.closest('[data-action="open-color-key"]');
  if (button) openColorKey(button.dataset.colorMode);
});

colorKeySearch.addEventListener("input", renderColorKey);
colorKeyList.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-item-id]");
  if (!button) return;
  const { itemId } = button.dataset;
  const activeIds =
    colorKeyMode === "faculty"
      ? state.activeFaculty
      : state.activeDepartments;
  if (activeIds.has(itemId)) activeIds.delete(itemId);
  else activeIds.add(itemId);
  if (colorKeyMode === "faculty") authorFilter.renderSelected();
  else departmentFilter.renderSelected();
  applyFilters();
  renderColorKey();
});
closeColorKey.addEventListener("click", () => {
  colorKeyDialog.close();
});
colorKeyDialog.addEventListener("click", (event) => {
  if (event.target === colorKeyDialog) colorKeyDialog.close();
});

clearFiltersButton.addEventListener("click", () => {
  titleSearch.value = "";
  state.activeTitleTerms.clear();
  renderSelectedTitles();
  authorFilter.clear();
  departmentFilter.clear();
  applyFilters();
  if (colorKeyDialog.open) renderColorKey();
});

zoomResultsButton.addEventListener("click", () => {
  fitPoints(state.matchedPoints);
});
resetViewButton.addEventListener("click", resetView);
closeDetail.addEventListener("click", () => {
  detailPanel.hidden = true;
  state.selectedWorkId = "";
  canvas.focus();
  scheduleDraw();
});
window.addEventListener("research-map-theme-change", refreshThemeColors);
if ("ResizeObserver" in window) {
  const canvasResizeObserver = new ResizeObserver(resizeCanvas);
  canvasResizeObserver.observe(canvas);
} else {
  window.addEventListener("resize", resizeCanvas);
}

function wait(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

async function fetchJson(url, label, { attempts = 2, timeout = 15000 } = {}) {
  let lastError;
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), timeout);
    try {
      const response = await fetch(url, {
        signal: controller.signal,
      });
      if (!response.ok) {
        const error = new Error(`${label} returned HTTP ${response.status}`);
        error.retryable =
          response.status === 408 ||
          response.status === 429 ||
          response.status >= 500;
        throw error;
      }
      try {
        return await response.json();
      } catch (error) {
        throw new Error(`${label} did not contain valid JSON`, { cause: error });
      }
    } catch (error) {
      lastError = error;
      const retryable =
        error.name === "AbortError" ||
        error instanceof TypeError ||
        error.retryable === true;
      if (!retryable || attempt === attempts) break;
      await wait(400 * attempt);
    } finally {
      window.clearTimeout(timeoutId);
    }
  }
  throw lastError;
}

function setLoadingState() {
  statusElement.textContent = "Loading the publication landscape…";
  statusElement.classList.remove("error", "warning");
  retryLoadButton.hidden = true;
  filterPanel.inert = true;
  filterPanel.setAttribute("aria-busy", "true");
  mapColumn.setAttribute("aria-busy", "true");
  canvas.removeAttribute("aria-disabled");
  emptyState.hidden = true;
}

function setReadyState() {
  retryLoadButton.hidden = true;
  filterPanel.inert = false;
  filterPanel.setAttribute("aria-busy", "false");
  mapColumn.setAttribute("aria-busy", "false");
}

function setUnavailableState(error) {
  console.error(error);
  state.points = [];
  state.matchedPoints = [];
  state.drawablePoints = [];
  statusElement.textContent =
    "The publication map is temporarily unavailable. The dataset and provenance remain available.";
  statusElement.classList.add("error");
  statusElement.classList.remove("warning");
  matchCount.textContent = "—";
  matchLabel.textContent = "map unavailable";
  retryLoadButton.hidden = false;
  filterPanel.inert = true;
  filterPanel.setAttribute("aria-busy", "false");
  mapColumn.setAttribute("aria-busy", "false");
  canvas.setAttribute("aria-disabled", "true");
  emptyTitle.textContent = "The map could not be loaded.";
  emptyCopy.textContent =
    "Check your connection and try again. The full dataset is still available above.";
  emptyState.hidden = false;
  scheduleDraw();
}

function resetFilters() {
  titleSearch.value = "";
  state.activeTitleTerms.clear();
  state.activeDepartments.clear();
  state.activeFaculty.clear();
  renderSelectedTitles();
  authorFilter.renderSelected();
  departmentFilter.renderSelected();
}

async function loadMap() {
  setLoadingState();
  try {
    if (!core || !context) {
      throw new Error("Required browser map features are unavailable");
    }
    const config = core.parseConfig(
      await fetchJson("map-config.json", "Map configuration"),
    );
    if (config.heading) titleElement.textContent = config.heading;
    document.title = `${config.title} Map`;
    const artifact = core.parseArtifact(
      await fetchJson(core.artifactUrl(config), "Publication artifact"),
    );

    state.departments = artifact.catalogs.departments;
    state.faculty = artifact.catalogs.faculty;
    state.layouts = artifact.layouts;
    state.layoutById = new Map(
      state.layouts.map((layout) => [layout.layout_id, layout]),
    );
    state.layoutId = artifact.default_layout_id;
    state.departmentById = new Map(
      state.departments.map((item) => [item.department_id, item]),
    );
    state.facultyById = new Map(
      state.faculty.map((item) => [item.person_id, item]),
    );
    state.points = artifact.points.map(preparePoint);
    state.omittedPointCount = artifact.omitted_point_count;
    state.departmentColors = new Map();
    state.facultyColors = new Map();
    state.titleColors = new Map();
    state.titlePalette = [];
    state.yearPalette = [];
    state.citationPalette = [];
    state.yearRange = null;
    state.citationMaximum = 0;
    state.citationMaximumClipped = false;
    state.selectedWorkId = "";
    detailPanel.hidden = true;
    if (artifact.warnings.length) console.warn(...artifact.warnings);

    populateLayouts();
    initializeFacultyColors();
    initializeDepartmentColors();
    initializeSequentialColors();
    resetFilters();
    const updated = core.formatUtcDate(artifact.source_data_newest_at_utc);
    state.sourceLabel = updated
      ? `newest Scholar profile refresh ${updated}`
      : "no verified Scholar profile refresh date";
    setReadyState();
    applyFilters();
    resizeCanvas();
    resetView();
  } catch (error) {
    setUnavailableState(error);
  }
}

retryLoadButton.addEventListener("click", loadMap);
loadMap();
