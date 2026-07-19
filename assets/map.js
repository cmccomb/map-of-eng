"use strict";

const TITLE_COLOR_CAPACITY = 32;
const SEQUENTIAL_COLOR_STEPS = 48;
const SIZE_RADIUS_STEPS = 8;
const KEYWORD_LABEL_MIN_MATCHES = 3;
const ARTIFACT_FETCH_TIMEOUT_MS = 60_000;
const core = globalThis.ResearchMapCore;

const state = {
  points: [],
  matchedPoints: [],
  drawablePoints: [],
  screenPoints: [],
  departments: [],
  faculty: [],
  keywords: [],
  layouts: [],
  departmentById: new Map(),
  facultyById: new Map(),
  keywordById: new Map(),
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
  sizeMode: "none",
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

function normalizedYear(point) {
  if (!Number.isInteger(point.year) || !state.yearRange) return null;
  const { minimum, maximum } = state.yearRange;
  if (minimum === maximum) return 0.5;
  return Math.min(1, Math.max(0, (point.year - minimum) / (maximum - minimum)));
}

function sizeAmount(point) {
  if (state.sizeMode === "citations") {
    const citations = Math.max(0, Number(point.citation_count) || 0);
    return state.citationMaximum
      ? Math.min(1, Math.log1p(citations) / Math.log1p(state.citationMaximum))
      : 0;
  }
  const year = normalizedYear(point);
  if (year === null) return null;
  if (state.sizeMode === "oldest") return 1 - year;
  if (state.sizeMode === "newest") return year;
  return null;
}

function pointRadius(point, baseRadius) {
  if (state.sizeMode === "none") return baseRadius;
  const amount = sizeAmount(point);
  if (amount === null) return Math.max(0.45, baseRadius * 0.55);
  const quantized =
    Math.round(amount * (SIZE_RADIUS_STEPS - 1)) / (SIZE_RADIUS_STEPS - 1);
  return Math.max(0.45, baseRadius * (0.55 + 2.25 * Math.sqrt(quantized)));
}

function matchedBaseRadius() {
  if (!state.filtersActive) return 0.72;
  const matchTotal = state.matchedPoints.length;
  if (matchTotal > 7500) return 0.9;
  if (matchTotal > 3000) return 1.05;
  if (matchTotal > 1000) return 1.25;
  return 1.7;
}

const titleElement = document.querySelector("#map-title");
const statusElement = document.querySelector("#map-status");
const retryLoadButton = document.querySelector("#retry-load");
const filterPanel = document.querySelector(".filter-panel");
const filterControls = document.querySelector(".filter-controls");
const legendPanel = document.querySelector(".legend-panel");
const mapColumn = document.querySelector(".map-column");
const toggleFiltersButton = document.querySelector("#toggle-filters");
const toggleLegendButton = document.querySelector("#toggle-legend");
const mapControls = document.querySelector(".map-controls");
const mapZoomInButton = document.querySelector("#map-zoom-in");
const mapZoomOutButton = document.querySelector("#map-zoom-out");
const mapFitButton = document.querySelector("#map-fit");
const loadingOverlay = document.querySelector("#loading-overlay");
const loadingProgress = document.querySelector("#loading-progress");
const loadingProgressValue = document.querySelector("#loading-progress-value");
const loadingPercent = document.querySelector("#loading-percent");
const loadingDetail = document.querySelector("#loading-detail");
const tooltip = document.querySelector("#tooltip");
const titleSearch = document.querySelector("#title-search");
const selectedTitles = document.querySelector("#selected-titles");
const authorSearch = document.querySelector("#author-search");
const authorSuggestions = document.querySelector("#author-suggestions");
const selectedAuthors = document.querySelector("#selected-authors");
const departmentSearch = document.querySelector("#department-search");
const departmentSuggestions = document.querySelector("#department-suggestions");
const selectedDepartments = document.querySelector("#selected-departments");
const layoutOptions = document.querySelector("#layout-options");
const layoutNote = document.querySelector("#layout-note");
const clearFiltersButton = document.querySelector("#clear-filters");
const zoomResultsButton = document.querySelector("#zoom-results");
const resetViewButton = document.querySelector("#reset-view");
const mapLegend = document.querySelector("#map-legend");
const legendSummary = document.querySelector("#legend-summary");
const sizeLegend = document.querySelector("#size-legend");
const sizeNote = document.querySelector("#size-note");
const emptyState = document.querySelector("#empty-state");
const emptyTitle = document.querySelector("#empty-title");
const emptyCopy = document.querySelector("#empty-copy");
const detailPanel = document.querySelector("#detail-panel");
const detailTitle = document.querySelector("#detail-title");
const detailMeta = document.querySelector("#detail-meta");
const detailAuthors = document.querySelector("#detail-authors");
const detailFaculty = document.querySelector("#detail-faculty");
const detailDepartments = document.querySelector("#detail-departments");
const detailKeyword = document.querySelector("#detail-keyword");
const detailLink = document.querySelector("#detail-link");
const closeDetail = document.querySelector("#close-detail");
const compactLayoutQuery = window.matchMedia("(max-width: 680px)");

function setCompactPanel(openPanel = "") {
  const compact = compactLayoutQuery.matches;
  const filtersOpen = compact && openPanel === "filters";
  const legendOpen = compact && openPanel === "legend";

  filterPanel.classList.toggle("mobile-open", filtersOpen);
  legendPanel.classList.toggle("mobile-open", legendOpen);
  toggleFiltersButton.setAttribute("aria-expanded", String(filtersOpen));
  toggleLegendButton.setAttribute("aria-expanded", String(legendOpen));

  if (!compact) {
    filterPanel.inert = false;
    legendPanel.inert = false;
    filterPanel.removeAttribute("aria-hidden");
    legendPanel.removeAttribute("aria-hidden");
    return;
  }

  filterPanel.inert = !filtersOpen;
  legendPanel.inert = !legendOpen;
  filterPanel.setAttribute("aria-hidden", String(!filtersOpen));
  legendPanel.setAttribute("aria-hidden", String(!legendOpen));
}

function toggleCompactPanel(panelName) {
  if (!compactLayoutQuery.matches) return;
  const panel = panelName === "filters" ? filterPanel : legendPanel;
  const shouldOpen = !panel.classList.contains("mobile-open");
  setCompactPanel(shouldOpen ? panelName : "");
  if (shouldOpen) {
    detailPanel.hidden = true;
    state.selectedWorkId = "";
    tooltip.hidden = true;
    scheduleDraw();
  }
}

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

function prepareKeyword(keyword) {
  return {
    ...keyword,
    _coordinates: new Map(Object.entries(keyword.coordinates)),
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
    const key = `${color}:${screenPoint.radius.toFixed(3)}`;
    const group = groups.get(key) || {
      color,
      radius: screenPoint.radius,
      points: [],
    };
    group.points.push(screenPoint);
    groups.set(key, group);
  }
  const quantitativeMode = ["year", "citations"].includes(state.colorMode);
  const alpha = state.filtersActive ? 0.9 : quantitativeMode ? 0.68 : 0.54;
  for (const group of groups.values()) {
    drawBatch(group.points, group.radius, group.color, alpha);
  }
  if (state.filtersActive && state.matchedPoints.length <= 1500) {
    context.save();
    context.strokeStyle = themeColor("--canvas-selected", "#f4fbff");
    context.globalAlpha = 0.58;
    context.lineWidth = 0.65;
    context.beginPath();
    for (const screenPoint of screenPoints) {
      const outlineRadius = screenPoint.radius + 0.75;
      context.moveTo(screenPoint.x + outlineRadius, screenPoint.y);
      context.arc(screenPoint.x, screenPoint.y, outlineRadius, 0, Math.PI * 2);
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
  context.arc(
    selected.x,
    selected.y,
    Math.max(7, selected.radius + 3),
    0,
    Math.PI * 2,
  );
  context.stroke();
  context.restore();
}

function roundedRectangle(x, y, width, height, radius) {
  const corner = Math.min(radius, width / 2, height / 2);
  context.beginPath();
  context.moveTo(x + corner, y);
  context.lineTo(x + width - corner, y);
  context.quadraticCurveTo(x + width, y, x + width, y + corner);
  context.lineTo(x + width, y + height - corner);
  context.quadraticCurveTo(
    x + width,
    y + height,
    x + width - corner,
    y + height,
  );
  context.lineTo(x + corner, y + height);
  context.quadraticCurveTo(x, y + height, x, y + height - corner);
  context.lineTo(x, y + corner);
  context.quadraticCurveTo(x, y, x + corner, y);
  context.closePath();
}

function rectanglesOverlap(left, right) {
  return !(
    left.right + 4 < right.left ||
    right.right + 4 < left.left ||
    left.bottom + 4 < right.top ||
    right.bottom + 4 < left.top
  );
}

function drawKeywordLabels(width, height) {
  if (!state.keywords.length) return;
  const matchedCounts = new Map();
  if (state.filtersActive) {
    for (const point of state.matchedPoints) {
      matchedCounts.set(
        point.keyword_id,
        (matchedCounts.get(point.keyword_id) || 0) + 1,
      );
    }
  }
  const unitScale = baseScale(width, height);
  const fontSize = width < 700 ? 9.5 : 11;
  const candidates = state.keywords
    .filter(
      (keyword) =>
        !state.filtersActive ||
        (matchedCounts.get(keyword.keyword_id) || 0) >= KEYWORD_LABEL_MIN_MATCHES,
    )
    .map((keyword) => {
      const coordinates = keyword._coordinates.get(state.layoutId);
      return {
        ...keyword,
        x: width / 2 + coordinates.x * unitScale * state.scale + state.offsetX,
        y: height / 2 - coordinates.y * unitScale * state.scale + state.offsetY,
      };
    })
    .filter(
      (keyword) =>
        keyword.x >= -80 &&
        keyword.x <= width + 80 &&
        keyword.y >= -30 &&
        keyword.y <= height + 30,
    )
    .sort(
      (left, right) =>
        right.publication_count - left.publication_count ||
        left.label.localeCompare(right.label),
    );

  context.save();
  context.font = `650 ${fontSize}px Inter, ui-sans-serif, system-ui, sans-serif`;
  context.textAlign = "center";
  context.textBaseline = "middle";
  const placed = [];
  for (const keyword of candidates) {
    const textWidth = context.measureText(keyword.label).width;
    const box = {
      left: keyword.x - textWidth / 2 - 7,
      right: keyword.x + textWidth / 2 + 7,
      top: keyword.y - fontSize / 2 - 5,
      bottom: keyword.y + fontSize / 2 + 5,
    };
    if (placed.some((existing) => rectanglesOverlap(box, existing))) continue;
    placed.push(box);
    roundedRectangle(
      box.left,
      box.top,
      box.right - box.left,
      box.bottom - box.top,
      7,
    );
    context.fillStyle = themeColor("--keyword-background", "#091621dc");
    context.fill();
    context.strokeStyle = themeColor("--keyword-border", "#8fa5b540");
    context.lineWidth = 0.75;
    context.stroke();
    context.fillStyle = themeColor("--keyword-text", "#eef5f8");
    context.fillText(keyword.label, keyword.x, keyword.y + 0.25);
  }
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
  const baseRadius = matchedBaseRadius();
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
    screenPoint.radius = screenPoint.matched
      ? pointRadius(point, baseRadius)
      : 0.55;
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
  drawKeywordLabels(bounds.width, bounds.height);
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
    }publications across ${state.keywords.length.toLocaleString()} topic keywords`,
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
          } to ${maximum}${state.yearRange.clippedMaximum ? " and later" : ""}`,
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
    labels: maximum ? ["0", compactNumber(midpoint), maximumLabel] : ["0"],
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
    }))
    .sort((left, right) =>
      left.label.localeCompare(right.label, "en", { sensitivity: "base" }),
    );
}

function appendColorKeyItems(items, mode) {
  const facultyMode = mode === "faculty";
  const singular = facultyMode ? "faculty member" : "department";
  const plural = facultyMode ? "faculty members" : "departments";
  legendSummary.textContent = `${items.length.toLocaleString()} ${
    items.length === 1 ? singular : plural
  } represented`;

  for (const item of items) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "color-key-item";
    button.dataset.itemId = item.id;
    button.dataset.colorMode = mode;
    button.setAttribute("aria-pressed", String(item.selected));
    const swatch = document.createElement("i");
    swatch.style.backgroundColor = item.color;
    swatch.setAttribute("aria-hidden", "true");
    const name = document.createElement("span");
    name.textContent = item.label;
    const count = document.createElement("small");
    count.textContent = `${item.count.toLocaleString()} match${
      item.count === 1 ? "" : "es"
    }`;
    button.append(swatch, name, count);
    mapLegend.append(button);
  }
}

function renderSizeLegend() {
  const descriptions = {
    none: {
      note: "Every publication uses the same dot size.",
      legend: "",
    },
    oldest: {
      note: "Older publications appear larger; unknown years stay small.",
      legend: "Larger dots are older",
    },
    newest: {
      note: "Newer publications appear larger; unknown years stay small.",
      legend: "Larger dots are newer",
    },
    citations: {
      note: "More widely cited publications appear larger on a log scale.",
      legend: "Larger dots are more cited",
    },
  };
  const description = descriptions[state.sizeMode] || descriptions.none;
  sizeNote.textContent = description.note;
  sizeLegend.replaceChildren();
  sizeLegend.hidden = state.sizeMode === "none";
  if (sizeLegend.hidden) return;

  const examples = document.createElement("div");
  examples.className = "size-examples";
  for (const diameter of [5, 9, 14]) {
    const dot = document.createElement("i");
    dot.style.width = `${diameter}px`;
    dot.style.height = `${diameter}px`;
    dot.setAttribute("aria-hidden", "true");
    examples.append(dot);
  }
  const label = document.createElement("span");
  label.textContent = description.legend;
  sizeLegend.append(examples, label);
}

function renderLegend() {
  mapLegend.replaceChildren();
  legendSummary.textContent = "";
  if (state.colorMode === "year") {
    legendSummary.textContent = "Publications by year";
    renderYearLegend();
  } else if (state.colorMode === "citations") {
    legendSummary.textContent = "Publications by citation count";
    renderCitationLegend();
  } else if (state.colorMode === "faculty") {
    const items = colorKeyItems("faculty");
    const hasUnmappedMatches = state.matchedPoints.some(
      (point) =>
        !point.faculty_ids.some((personId) =>
          state.facultyColors.has(personId),
        ),
    );
    if (items.length) {
      appendColorKeyItems(items, "faculty");
      if (hasUnmappedMatches) {
        appendLegendItem(
          "No mapped faculty",
          themeColor("--canvas-unassigned", "#657789"),
        );
      }
    } else if (state.matchedPoints.length && hasUnmappedMatches) {
      legendSummary.textContent = "No mapped faculty";
      appendLegendItem(
        "No mapped faculty",
        themeColor("--canvas-unassigned", "#657789"),
      );
    } else {
      appendLegendItem(
        "No faculty matches",
        themeColor("--canvas-unassigned", "#657789"),
      );
      legendSummary.textContent = "No faculty represented";
    }
  } else if (state.colorMode === "title") {
    const activeTerms = [...state.activeTitleTerms]
      .map(([query, label]) => ({
        label,
        color: state.titleColors.get(query),
      }))
      .sort((left, right) =>
        left.label.localeCompare(right.label, "en", { sensitivity: "base" }),
      );
    if (activeTerms.length) {
      legendSummary.textContent = `${activeTerms.length.toLocaleString()} title ${
        activeTerms.length === 1 ? "term" : "terms"
      }`;
      for (const term of activeTerms) {
        appendLegendItem(term.label, term.color);
      }
    } else {
      legendSummary.textContent = "Publications";
      appendLegendItem(
        state.filtersActive ? "Match" : "Publication",
        colorFor({ _title: "", department_ids: [], faculty_ids: [] }),
      );
    }
  } else if (state.colorMode === "department") {
    const items = colorKeyItems("department");
    if (items.length) {
      appendColorKeyItems(items, "department");
    } else {
      appendLegendItem(
        "No department matches",
        themeColor("--canvas-unassigned", "#657789"),
      );
      legendSummary.textContent = "No departments represented";
    }
  }
  if (state.filtersActive && state.displayMode === "highlight") {
    appendLegendItem(
      "Context",
      themeColor("--canvas-context", "#748696"),
      "legend-dot-context",
    );
  }
  renderSizeLegend();
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
  const matchedIds = new Set(state.matchedPoints.map((point) => point.work_id));
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
  state.citationMaximumClipped = actualCitationMaximum > state.citationMaximum;
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
    document
      .querySelector(`#${activeId}`)
      ?.scrollIntoView({ block: "nearest" });
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
  detailKeyword.textContent =
    state.keywordById.get(point.keyword_id)?.label || "Unavailable";
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

function zoomView(multiplier) {
  const previousScale = state.scale;
  const nextScale = Math.min(25, Math.max(0.7, previousScale * multiplier));
  if (nextScale === previousScale) return;
  const ratio = nextScale / previousScale;
  state.scale = nextScale;
  state.offsetX *= ratio;
  state.offsetY *= ratio;
  tooltip.hidden = true;
  scheduleDraw();
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
  if (compactLayoutQuery.matches) setCompactPanel();
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
    renderLegend();
    scheduleDraw();
  });
});

document.querySelectorAll('input[name="size-mode"]').forEach((input) => {
  input.addEventListener("change", () => {
    if (!input.checked) return;
    state.sizeMode = input.value;
    renderSizeLegend();
    tooltip.hidden = true;
    scheduleDraw();
  });
});

mapLegend.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-item-id][data-color-mode]");
  if (!button) return;
  const { itemId, colorMode } = button.dataset;
  const activeIds =
    colorMode === "faculty" ? state.activeFaculty : state.activeDepartments;
  if (activeIds.has(itemId)) activeIds.delete(itemId);
  else activeIds.add(itemId);
  if (colorMode === "faculty") authorFilter.renderSelected();
  else departmentFilter.renderSelected();
  applyFilters();
});

clearFiltersButton.addEventListener("click", () => {
  titleSearch.value = "";
  state.activeTitleTerms.clear();
  renderSelectedTitles();
  authorFilter.clear();
  departmentFilter.clear();
  applyFilters();
});

zoomResultsButton.addEventListener("click", () => {
  fitPoints(state.matchedPoints);
});
resetViewButton.addEventListener("click", resetView);
mapZoomInButton.addEventListener("click", () => {
  zoomView(1.25);
  canvas.focus({ preventScroll: true });
});
mapZoomOutButton.addEventListener("click", () => {
  zoomView(0.8);
  canvas.focus({ preventScroll: true });
});
mapFitButton.addEventListener("click", () => {
  resetView();
  canvas.focus({ preventScroll: true });
});
toggleFiltersButton.addEventListener("click", () => {
  toggleCompactPanel("filters");
});
toggleLegendButton.addEventListener("click", () => {
  toggleCompactPanel("legend");
});
compactLayoutQuery.addEventListener("change", () => setCompactPanel());
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

function formatByteCount(bytes) {
  if (bytes < 1024 * 1024) {
    return `${Math.max(0, bytes / 1024).toFixed(bytes < 102400 ? 1 : 0)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function resetLoadingProgress() {
  loadingOverlay.hidden = false;
  loadingProgress.removeAttribute("aria-valuenow");
  loadingProgress.setAttribute("aria-valuetext", "Connecting");
  loadingProgressValue.style.removeProperty("width");
  loadingPercent.hidden = true;
  loadingPercent.textContent = "0%";
  loadingDetail.textContent = "Connecting to the publication archive…";
}

function updateLoadingProgress({ received, total, attempt }) {
  const retryLabel = attempt > 1 ? " · retrying" : "";
  if (!total) {
    loadingProgress.removeAttribute("aria-valuenow");
    loadingProgress.setAttribute(
      "aria-valuetext",
      received ? `${formatByteCount(received)} downloaded` : "Downloading",
    );
    loadingProgressValue.style.removeProperty("width");
    loadingPercent.hidden = true;
    loadingDetail.textContent = received
      ? `${formatByteCount(received)} downloaded${retryLabel}`
      : `Downloading publication data${retryLabel}…`;
    return;
  }

  const percentage = Math.min(100, Math.floor((received / total) * 100));
  loadingProgress.setAttribute("aria-valuenow", String(percentage));
  loadingProgress.setAttribute("aria-valuetext", `${percentage}% downloaded`);
  loadingProgressValue.style.width = `${percentage}%`;
  loadingPercent.hidden = false;
  loadingPercent.textContent = `${percentage}%`;
  loadingDetail.textContent = `${formatByteCount(received)} of ${formatByteCount(
    total,
  )}${retryLabel}`;
}

function setLoadingProcessingState() {
  loadingProgress.setAttribute("aria-valuenow", "100");
  loadingProgress.setAttribute(
    "aria-valuetext",
    "Download complete; building the map",
  );
  loadingProgressValue.style.width = "100%";
  loadingPercent.hidden = false;
  loadingPercent.textContent = "100%";
  loadingDetail.textContent = "Building the interactive map…";
}

function responseBodyLength(response, expectedBytes) {
  if (Number.isSafeInteger(expectedBytes) && expectedBytes > 0) {
    return expectedBytes;
  }
  if (response.headers.get("content-encoding")) return 0;
  const declaredBytes = Number(response.headers.get("content-length"));
  return Number.isSafeInteger(declaredBytes) && declaredBytes > 0
    ? declaredBytes
    : 0;
}

async function readResponseText(
  response,
  { expectedBytes = 0, onProgress, attempt = 1 } = {},
) {
  const total = responseBodyLength(response, expectedBytes);
  if (
    typeof onProgress !== "function" ||
    !response.body?.getReader ||
    typeof TextDecoder !== "function"
  ) {
    const text = await response.text();
    if (typeof onProgress === "function") {
      const received = new Blob([text]).size;
      onProgress({ received, total: total || received, attempt });
    }
    return text;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let received = 0;
  let text = "";
  onProgress({ received, total, attempt });
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    received += value.byteLength;
    text += decoder.decode(value, { stream: true });
    onProgress({ received, total, attempt });
  }
  text += decoder.decode();
  return text;
}

async function fetchJson(
  url,
  label,
  {
    attempts = 2,
    timeout = 15000,
    expectedBytes = 0,
    onDownloaded,
    onProgress,
  } = {},
) {
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
        const text = await readResponseText(response, {
          expectedBytes,
          onProgress,
          attempt,
        });
        if (typeof onDownloaded === "function") onDownloaded();
        return JSON.parse(text);
      } catch (error) {
        if (error.name === "AbortError" || error instanceof TypeError) {
          throw error;
        }
        throw new Error(`${label} did not contain valid JSON`, {
          cause: error,
        });
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
  resetLoadingProgress();
  statusElement.textContent = "Loading the publication landscape…";
  statusElement.classList.remove("error", "warning");
  retryLoadButton.hidden = true;
  filterControls.inert = true;
  mapControls.inert = true;
  toggleFiltersButton.disabled = true;
  toggleLegendButton.disabled = true;
  filterPanel.setAttribute("aria-busy", "true");
  mapColumn.setAttribute("aria-busy", "true");
  canvas.removeAttribute("aria-disabled");
  emptyState.hidden = true;
}

function setReadyState() {
  loadingOverlay.hidden = true;
  retryLoadButton.hidden = true;
  filterControls.inert = false;
  mapControls.inert = false;
  toggleFiltersButton.disabled = false;
  toggleLegendButton.disabled = false;
  filterPanel.setAttribute("aria-busy", "false");
  mapColumn.setAttribute("aria-busy", "false");
}

function setUnavailableState(error) {
  console.error(error);
  loadingOverlay.hidden = true;
  state.points = [];
  state.matchedPoints = [];
  state.drawablePoints = [];
  state.keywords = [];
  state.keywordById = new Map();
  statusElement.textContent =
    "The publication map is temporarily unavailable. The dataset and provenance remain available.";
  statusElement.classList.add("error");
  statusElement.classList.remove("warning");
  retryLoadButton.hidden = false;
  filterControls.inert = true;
  mapControls.inert = true;
  toggleFiltersButton.disabled = false;
  toggleLegendButton.disabled = false;
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
    const artifactDocument = await fetchJson(
      core.artifactUrl(config),
      "Publication artifact",
      {
        timeout: ARTIFACT_FETCH_TIMEOUT_MS,
        expectedBytes: config.artifact_bytes,
        onDownloaded: setLoadingProcessingState,
        onProgress: updateLoadingProgress,
      },
    );
    const artifact = core.parseArtifact(artifactDocument);

    state.departments = artifact.catalogs.departments;
    state.faculty = artifact.catalogs.faculty;
    state.keywords = artifact.keywords.map(prepareKeyword);
    state.layouts = artifact.layouts;
    state.layoutById = new Map(
      state.layouts.map((layout) => [layout.layout_id, layout]),
    );
    state.layoutId = state.layoutById.has(config.default_layout_id)
      ? config.default_layout_id
      : artifact.default_layout_id;
    state.departmentById = new Map(
      state.departments.map((item) => [item.department_id, item]),
    );
    state.facultyById = new Map(
      state.faculty.map((item) => [item.person_id, item]),
    );
    state.keywordById = new Map(
      state.keywords.map((item) => [item.keyword_id, item]),
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
    applyFilters();
    resizeCanvas();
    resetView();
    setReadyState();
  } catch (error) {
    setUnavailableState(error);
  }
}

retryLoadButton.addEventListener("click", loadMap);
setCompactPanel();
loadMap();
