"use strict";

const DEPARTMENT_COLORS = [
  "#67d5ff",
  "#ffbd68",
  "#a9df79",
  "#e89bff",
  "#ff8d7e",
  "#78e4c3",
  "#9aa9ff",
  "#f8df72",
  "#ff9ec8",
  "#7eb6ff",
  "#d5a6ff",
];

const UNASSIGNED_FACULTY_COLOR = "#657789";

const state = {
  points: [],
  matchedPoints: [],
  drawablePoints: [],
  screenPoints: [],
  departments: [],
  faculty: [],
  departmentById: new Map(),
  facultyById: new Map(),
  departmentColors: new Map(),
  facultyColors: new Map(),
  activeDepartments: new Set(),
  activeFaculty: new Set(),
  colorMode: "department",
  displayMode: "highlight",
  filtersActive: false,
  selectedWorkId: "",
  suggestions: [],
  suggestionIndex: -1,
  scale: 1,
  offsetX: 0,
  offsetY: 0,
  dragging: false,
  dragX: 0,
  dragY: 0,
  dragDistance: 0,
  framePending: false,
  sourceLabel: "",
  spatialIndex: new Map(),
  hoverFrame: 0,
  hoverEvent: null,
  titleTimer: 0,
};

const canvas = document.querySelector("#research-map");
const context = canvas.getContext("2d", { alpha: false });
const titleElement = document.querySelector("#map-title");
const statusElement = document.querySelector("#map-status");
const tooltip = document.querySelector("#tooltip");
const titleSearch = document.querySelector("#title-search");
const authorSearch = document.querySelector("#author-search");
const authorSuggestions = document.querySelector("#author-suggestions");
const selectedAuthors = document.querySelector("#selected-authors");
const departmentOptions = document.querySelector("#department-options");
const clearFiltersButton = document.querySelector("#clear-filters");
const zoomResultsButton = document.querySelector("#zoom-results");
const resetViewButton = document.querySelector("#reset-view");
const matchCount = document.querySelector("#match-count");
const matchLabel = document.querySelector("#match-label");
const mapLegend = document.querySelector("#map-legend");
const emptyState = document.querySelector("#empty-state");
const detailPanel = document.querySelector("#detail-panel");
const detailTitle = document.querySelector("#detail-title");
const detailMeta = document.querySelector("#detail-meta");
const detailAuthors = document.querySelector("#detail-authors");
const detailFaculty = document.querySelector("#detail-faculty");
const detailDepartments = document.querySelector("#detail-departments");
const detailLink = document.querySelector("#detail-link");
const closeDetail = document.querySelector("#close-detail");
const facultyLegendDialog = document.querySelector("#faculty-legend-dialog");
const facultyLegendSearch = document.querySelector("#faculty-legend-search");
const facultyLegendCount = document.querySelector("#faculty-legend-count");
const facultyLegendList = document.querySelector("#faculty-legend-list");
const closeFacultyLegend = document.querySelector("#close-faculty-legend");

function normalizedText(value) {
  return String(value || "").trim().toLocaleLowerCase();
}

function facultyWithMatches() {
  const matchedFacultyIds = new Set();
  for (const point of state.matchedPoints) {
    for (const personId of point.faculty_ids) {
      if (state.facultyById.has(personId)) matchedFacultyIds.add(personId);
    }
  }
  return state.faculty.filter((person) =>
    matchedFacultyIds.has(person.person_id),
  );
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
  return {
    ...point,
    x: Number(point.x),
    y: Number(point.y),
    department_ids: Array.isArray(point.department_ids)
      ? point.department_ids
      : [],
    faculty_ids: Array.isArray(point.faculty_ids) ? point.faculty_ids : [],
    _title: normalizedText(point.title),
  };
}

function resizeCanvas() {
  const ratio = window.devicePixelRatio || 1;
  const bounds = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(bounds.width * ratio));
  canvas.height = Math.max(1, Math.round(bounds.height * ratio));
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  scheduleDraw();
}

function baseScale(width, height) {
  const margin = 34;
  return Math.max(1, Math.min(width - margin * 2, height - margin * 2) / 2);
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
  context.strokeStyle = "#ffffff08";
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
  if (state.colorMode === "faculty") {
    const selectedPersonId = point.faculty_ids.find(
      (candidate) =>
        state.activeFaculty.has(candidate) &&
        state.facultyColors.has(candidate),
    );
    const personId =
      selectedPersonId ||
      point.faculty_ids.find((candidate) =>
        state.facultyColors.has(candidate),
      );
    if (personId) return state.facultyColors.get(personId);
    return UNASSIGNED_FACULTY_COLOR;
  }
  if (state.colorMode === "department" && state.activeDepartments.size) {
    const departmentId = point.department_ids.find((candidate) =>
      state.activeDepartments.has(candidate),
    );
    if (departmentId) return state.departmentColors.get(departmentId);
  }
  if (state.activeFaculty.size || state.activeDepartments.size) return "#73c5df";
  if (titleSearch.value.trim()) return "#73c5df";
  return "#79b7cf";
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
  for (const [color, group] of groups) {
    drawBatch(group, radius, color, state.filtersActive ? 0.9 : 0.54);
  }
  if (state.filtersActive && state.matchedPoints.length <= 1500) {
    context.save();
    context.strokeStyle = "#f4fbff";
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
  context.strokeStyle = "#ffffff";
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
  context.fillStyle = "#071019";
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
  drawBatch(contextPoints, 0.55, "#748696", 0.12);
  drawMatched(matchedPoints);
  drawSelected(state.screenPoints);
}

function scheduleDraw() {
  if (!state.framePending) {
    state.framePending = true;
    window.requestAnimationFrame(draw);
  }
}

function pointMatches(point, titleQuery) {
  if (titleQuery && !point._title.includes(titleQuery)) return false;
  if (
    state.activeDepartments.size &&
    !point.department_ids.some((id) => state.activeDepartments.has(id))
  ) {
    return false;
  }
  if (
    state.activeFaculty.size &&
    !point.faculty_ids.some((id) => state.activeFaculty.has(id))
  ) {
    return false;
  }
  return true;
}

function updateStatus() {
  const total = state.points.length;
  const matched = state.matchedPoints.length;
  matchCount.textContent = state.filtersActive
    ? `${matched.toLocaleString()} of ${total.toLocaleString()}`
    : total.toLocaleString();
  matchLabel.textContent = state.filtersActive
    ? "matching publications"
    : "publications in one shared layout";
  const countLabel = state.filtersActive
    ? `${matched.toLocaleString()} of ${total.toLocaleString()} publications match`
    : `${total.toLocaleString()} publications`;
  statusElement.textContent = state.sourceLabel
    ? `${countLabel} · ${state.sourceLabel}`
    : countLabel;
  emptyState.hidden = matched !== 0;
  zoomResultsButton.disabled = matched === 0;
}

function appendLegendItem(label, color, className = "legend-dot-match") {
  const item = document.createElement("span");
  const dot = document.createElement("i");
  dot.className = `legend-dot ${className}`;
  if (color) dot.style.backgroundColor = color;
  item.append(dot, document.createTextNode(label));
  mapLegend.append(item);
}

function appendFacultyLegendButton(people) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "faculty-legend-button";
  button.dataset.action = "open-faculty-legend";
  button.setAttribute("aria-haspopup", "dialog");
  const spectrum = document.createElement("span");
  spectrum.className = "legend-spectrum";
  spectrum.setAttribute("aria-hidden", "true");
  const colorCount = people.length;
  const swatchCount = Math.min(10, colorCount);
  for (let index = 0; index < swatchCount; index += 1) {
    const swatch = document.createElement("i");
    const paletteIndex = Math.floor((index * colorCount) / swatchCount);
    const personId = people[paletteIndex].person_id;
    swatch.style.backgroundColor = state.facultyColors.get(personId);
    spectrum.append(swatch);
  }
  const label = document.createElement("span");
  label.textContent = `View ${colorCount.toLocaleString()} faculty colors`;
  button.append(spectrum, label);
  mapLegend.append(button);
}

function renderLegend() {
  mapLegend.replaceChildren();
  if (state.colorMode === "faculty") {
    const people = facultyWithMatches();
    const selectedPeople = people.filter((person) =>
      state.activeFaculty.has(person.person_id),
    );
    for (const person of selectedPeople.slice(0, 3)) {
      appendLegendItem(
        person.display_name,
        state.facultyColors.get(person.person_id),
      );
    }
    if (selectedPeople.length > 3) {
      appendLegendItem(`+${selectedPeople.length - 3} selected`, "#9dabb9");
    }
    const hasUnmappedMatches = state.matchedPoints.some(
      (point) =>
        !point.faculty_ids.some((personId) =>
          state.facultyColors.has(personId),
        ),
    );
    if (people.length) {
      appendFacultyLegendButton(people);
      if (hasUnmappedMatches) {
        appendLegendItem("No mapped faculty", UNASSIGNED_FACULTY_COLOR);
      }
    } else if (state.matchedPoints.length && hasUnmappedMatches) {
      appendLegendItem("No mapped faculty", UNASSIGNED_FACULTY_COLOR);
    } else {
      appendLegendItem("No faculty matches", UNASSIGNED_FACULTY_COLOR);
    }
  } else {
    const activeGroups = state.departments
      .filter((department) =>
        state.activeDepartments.has(department.department_id),
      )
      .map((department) => ({
        label: department.title,
        color: state.departmentColors.get(department.department_id),
      }));
    if (activeGroups.length) {
      for (const group of activeGroups.slice(0, 4)) {
        appendLegendItem(group.label, group.color);
      }
      if (activeGroups.length > 4) {
        appendLegendItem(`+${activeGroups.length - 4} more`, "#9dabb9");
      }
    } else {
      appendLegendItem(
        state.filtersActive ? "Match" : "Publication",
        colorFor({ department_ids: [], faculty_ids: [] }),
      );
    }
  }
  if (state.filtersActive && state.displayMode === "highlight") {
    appendLegendItem("Context", "#748696", "legend-dot-context");
  }
}

function renderFacultyLegend() {
  const query = normalizedText(facultyLegendSearch.value);
  const matchingPeople = facultyWithMatches();
  const people = matchingPeople.filter((person) =>
    normalizedText(person.display_name).includes(query),
  );
  facultyLegendCount.textContent = query
    ? `${people.length.toLocaleString()} of ${matchingPeople.length.toLocaleString()} faculty with matches`
    : `${matchingPeople.length.toLocaleString()} faculty with matches · select any name to filter the map`;
  facultyLegendList.replaceChildren();
  for (const person of people) {
    const selected = state.activeFaculty.has(person.person_id);
    const button = document.createElement("button");
    button.type = "button";
    button.className = "faculty-legend-item";
    button.dataset.personId = person.person_id;
    button.setAttribute("aria-pressed", String(selected));
    const swatch = document.createElement("i");
    swatch.style.backgroundColor = state.facultyColors.get(person.person_id);
    swatch.setAttribute("aria-hidden", "true");
    const name = document.createElement("span");
    name.textContent = person.display_name;
    const count = document.createElement("small");
    count.textContent = `${Number(person.publication_count || 0).toLocaleString()} works`;
    button.append(swatch, name, count);
    facultyLegendList.append(button);
  }
}

function openFacultyLegend() {
  renderFacultyLegend();
  facultyLegendDialog.showModal();
  facultyLegendSearch.focus();
}

function applyFilters() {
  const titleQuery = normalizedText(titleSearch.value);
  state.filtersActive = Boolean(
    titleQuery || state.activeDepartments.size || state.activeFaculty.size,
  );
  state.matchedPoints = state.points.filter((point) =>
    pointMatches(point, titleQuery),
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
  scheduleDraw();
}

function populateDepartments() {
  departmentOptions.replaceChildren();
  state.departments.forEach((department, index) => {
    state.departmentColors.set(
      department.department_id,
      DEPARTMENT_COLORS[index % DEPARTMENT_COLORS.length],
    );
    const label = document.createElement("label");
    label.className = "department-option";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = department.department_id;
    const name = document.createElement("span");
    name.textContent = department.title;
    const count = document.createElement("span");
    count.className = "department-count";
    count.textContent = Number(department.publication_count || 0).toLocaleString();
    label.append(input, name, count);
    departmentOptions.append(label);
  });
}

function hideSuggestions() {
  state.suggestions = [];
  state.suggestionIndex = -1;
  authorSuggestions.hidden = true;
  authorSuggestions.replaceChildren();
  authorSearch.setAttribute("aria-expanded", "false");
  authorSearch.removeAttribute("aria-activedescendant");
}

function renderSuggestions() {
  const query = normalizedText(authorSearch.value);
  if (!query) {
    hideSuggestions();
    return;
  }
  state.suggestions = state.faculty
    .filter(
      (person) =>
        !state.activeFaculty.has(person.person_id) &&
        normalizedText(person.display_name).includes(query),
    )
    .sort((left, right) => {
      const leftName = normalizedText(left.display_name);
      const rightName = normalizedText(right.display_name);
      const leftScore = leftName.startsWith(query) ? 0 : 1;
      const rightScore = rightName.startsWith(query) ? 0 : 1;
      return leftScore - rightScore || leftName.localeCompare(rightName);
    })
    .slice(0, 8);
  if (!state.suggestions.length) {
    hideSuggestions();
    return;
  }
  state.suggestionIndex = 0;
  authorSuggestions.replaceChildren();
  state.suggestions.forEach((person, index) => {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.id = `author-option-${index}`;
    button.setAttribute("role", "option");
    button.setAttribute("aria-selected", String(index === state.suggestionIndex));
    button.dataset.personId = person.person_id;
    const name = document.createElement("span");
    name.textContent = person.display_name;
    const count = document.createElement("small");
    count.textContent = `${Number(person.publication_count || 0).toLocaleString()} works`;
    button.append(name, count);
    item.append(button);
    authorSuggestions.append(item);
  });
  authorSuggestions.hidden = false;
  authorSearch.setAttribute("aria-expanded", "true");
  authorSearch.setAttribute("aria-activedescendant", "author-option-0");
}

function updateSuggestionSelection() {
  const buttons = authorSuggestions.querySelectorAll("button");
  buttons.forEach((button, index) => {
    button.setAttribute("aria-selected", String(index === state.suggestionIndex));
  });
  if (state.suggestionIndex >= 0) {
    const activeId = `author-option-${state.suggestionIndex}`;
    authorSearch.setAttribute("aria-activedescendant", activeId);
    document.querySelector(`#${activeId}`)?.scrollIntoView({ block: "nearest" });
  }
}

function renderSelectedAuthors() {
  selectedAuthors.replaceChildren();
  const people = state.faculty.filter((person) =>
    state.activeFaculty.has(person.person_id),
  );
  for (const person of people) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "author-chip";
    button.dataset.personId = person.person_id;
    button.title = `Remove ${person.display_name}`;
    const name = document.createElement("span");
    name.textContent = person.display_name;
    const close = document.createElement("i");
    close.textContent = "×";
    close.setAttribute("aria-hidden", "true");
    button.append(name, close);
    selectedAuthors.append(button);
  }
}

function initializeFacultyColors() {
  const generator = globalThis.ResearchMapColors?.generatePerceptualPalette;
  if (!generator) throw new Error("Faculty color generator is unavailable");
  const palette = generator(state.faculty.length);
  state.facultyColors = new Map(
    state.faculty.map((person, index) => [person.person_id, palette[index]]),
  );
}

function addAuthor(personId) {
  if (!state.facultyById.has(personId)) return;
  state.activeFaculty.add(personId);
  authorSearch.value = "";
  hideSuggestions();
  renderSelectedAuthors();
  applyFilters();
  authorSearch.focus();
}

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
  if (!points.length) return;
  const bounds = canvas.getBoundingClientRect();
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const point of points) {
    minX = Math.min(minX, point.x);
    maxX = Math.max(maxX, point.x);
    minY = Math.min(minY, point.y);
    maxY = Math.max(maxY, point.y);
  }
  const rangeX = Math.max(maxX - minX, 0.04);
  const rangeY = Math.max(maxY - minY, 0.04);
  const padding = 28;
  const unitScale = baseScale(bounds.width, bounds.height);
  state.scale = Math.min(
    25,
    Math.max(
      0.7,
      Math.min(
        (bounds.width - padding * 2) / (rangeX * unitScale),
        (bounds.height - padding * 2) / (rangeY * unitScale),
      ),
    ),
  );
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  state.offsetX = -centerX * unitScale * state.scale;
  state.offsetY = centerY * unitScale * state.scale;
  tooltip.hidden = true;
  scheduleDraw();
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

titleSearch.addEventListener("input", () => {
  window.clearTimeout(state.titleTimer);
  state.titleTimer = window.setTimeout(applyFilters, 100);
});

authorSearch.addEventListener("input", renderSuggestions);
authorSearch.addEventListener("focus", renderSuggestions);
authorSearch.addEventListener("blur", () => {
  window.setTimeout(hideSuggestions, 120);
});
authorSearch.addEventListener("keydown", (event) => {
  if (authorSuggestions.hidden || !state.suggestions.length) return;
  if (event.key === "ArrowDown") {
    state.suggestionIndex =
      (state.suggestionIndex + 1) % state.suggestions.length;
  } else if (event.key === "ArrowUp") {
    state.suggestionIndex =
      (state.suggestionIndex - 1 + state.suggestions.length) %
      state.suggestions.length;
  } else if (event.key === "Enter") {
    addAuthor(state.suggestions[state.suggestionIndex].person_id);
    event.preventDefault();
    return;
  } else if (event.key === "Escape") {
    hideSuggestions();
    return;
  } else {
    return;
  }
  event.preventDefault();
  updateSuggestionSelection();
});

authorSuggestions.addEventListener("pointerdown", (event) => {
  event.preventDefault();
});
authorSuggestions.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-person-id]");
  if (button) addAuthor(button.dataset.personId);
});

selectedAuthors.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-person-id]");
  if (!button) return;
  state.activeFaculty.delete(button.dataset.personId);
  renderSelectedAuthors();
  applyFilters();
});

departmentOptions.addEventListener("change", (event) => {
  if (!event.target.matches('input[type="checkbox"]')) return;
  if (event.target.checked) state.activeDepartments.add(event.target.value);
  else state.activeDepartments.delete(event.target.value);
  applyFilters();
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
    if (state.colorMode !== "faculty" && facultyLegendDialog.open) {
      facultyLegendDialog.close();
    }
    renderLegend();
    scheduleDraw();
  });
});

mapLegend.addEventListener("click", (event) => {
  if (event.target.closest('[data-action="open-faculty-legend"]')) {
    openFacultyLegend();
  }
});

facultyLegendSearch.addEventListener("input", renderFacultyLegend);
facultyLegendList.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-person-id]");
  if (!button) return;
  const { personId } = button.dataset;
  if (state.activeFaculty.has(personId)) state.activeFaculty.delete(personId);
  else state.activeFaculty.add(personId);
  renderSelectedAuthors();
  applyFilters();
  renderFacultyLegend();
});
closeFacultyLegend.addEventListener("click", () => {
  facultyLegendDialog.close();
});
facultyLegendDialog.addEventListener("click", (event) => {
  if (event.target === facultyLegendDialog) facultyLegendDialog.close();
});

clearFiltersButton.addEventListener("click", () => {
  titleSearch.value = "";
  authorSearch.value = "";
  state.activeDepartments.clear();
  state.activeFaculty.clear();
  departmentOptions
    .querySelectorAll('input[type="checkbox"]')
    .forEach((input) => {
      input.checked = false;
    });
  hideSuggestions();
  renderSelectedAuthors();
  applyFilters();
  if (facultyLegendDialog.open) renderFacultyLegend();
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
window.addEventListener("resize", resizeCanvas);

async function loadMap() {
  try {
    const configResponse = await fetch("map-config.json", { cache: "no-cache" });
    if (!configResponse.ok) throw new Error("Could not load map configuration");
    const config = await configResponse.json();
    if (config.heading) titleElement.textContent = config.heading;
    document.title = `${config.title} Map`;
    const artifactPath = config.artifact_path || "maps/publications.json";
    const artifactUrl =
      config.artifact_url ||
      `https://huggingface.co/datasets/${config.dataset_id}/resolve/${config.dataset_revision || "main"}/${artifactPath}`;
    const artifactResponse = await fetch(artifactUrl, { cache: "no-cache" });
    if (!artifactResponse.ok) {
      throw new Error("Could not load the publication artifact");
    }
    const artifact = await artifactResponse.json();
    if (
      artifact.schema_version !== 3 ||
      !Array.isArray(artifact.points) ||
      !Array.isArray(artifact.catalogs?.departments) ||
      !Array.isArray(artifact.catalogs?.faculty)
    ) {
      throw new Error("The publication artifact has an unsupported schema");
    }
    state.departments = artifact.catalogs.departments;
    state.faculty = artifact.catalogs.faculty;
    state.departmentById = new Map(
      state.departments.map((item) => [item.department_id, item]),
    );
    state.facultyById = new Map(
      state.faculty.map((item) => [item.person_id, item]),
    );
    state.points = artifact.points
      .map(preparePoint)
      .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
    initializeFacultyColors();
    populateDepartments();
    renderSelectedAuthors();
    if (artifact.source_data_newest_at_utc) {
      const updated = new Date(
        artifact.source_data_newest_at_utc,
      ).toLocaleDateString(undefined, {
        year: "numeric",
        month: "short",
        day: "numeric",
      });
      state.sourceLabel = `newest Scholar profile refresh ${updated}`;
    } else {
      state.sourceLabel = "no verified Scholar profiles yet";
    }
    applyFilters();
    resizeCanvas();
    resetView();
  } catch (error) {
    console.error(error);
    statusElement.textContent =
      "The publication map is temporarily unavailable. The dataset and provenance remain available.";
    statusElement.classList.add("error");
    matchCount.textContent = "—";
    matchLabel.textContent = "map unavailable";
  }
}

loadMap();
