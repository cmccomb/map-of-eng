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
const TITLE_COLOR_CAPACITY = 32;

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
  spatialIndex: new Map(),
  hoverFrame: 0,
  hoverEvent: null,
};

const canvas = document.querySelector("#research-map");
const context = canvas.getContext("2d", { alpha: false });
const titleElement = document.querySelector("#map-title");
const statusElement = document.querySelector("#map-status");
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
  const coordinates = new Map(
    state.layouts.map((layout) => [
      layout.layout_id,
      {
        x: Number(point[layout.x_field]),
        y: Number(point[layout.y_field]),
      },
    ]),
  );
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
    _title: normalizedText(point.title),
  };
}

function hasEveryLayout(point) {
  return state.layouts.every((layout) => {
    const coordinates = point._coordinates.get(layout.layout_id);
    return Number.isFinite(coordinates?.x) && Number.isFinite(coordinates?.y);
  });
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
  if (state.colorMode === "title" && state.activeTitleTerms.size) {
    for (const query of state.activeTitleTerms.keys()) {
      if (point._title?.includes(query)) return state.titleColors.get(query);
    }
  }
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
  if (
    state.activeTitleTerms.size ||
    state.activeFaculty.size ||
    state.activeDepartments.size
  ) {
    return "#73c5df";
  }
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

function pointMatches(point, titleQueries) {
  if (
    titleQueries.length &&
    !titleQueries.some((query) => point._title.includes(query))
  ) {
    return false;
  }
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
    : "publications";
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
        appendLegendItem(`+${activeTerms.length - 4} more`, "#9dabb9");
      }
    } else {
      appendLegendItem(
        state.filtersActive ? "Match" : "Publication",
        colorFor({ _title: "", department_ids: [], faculty_ids: [] }),
      );
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
  const titleQueries = [...state.activeTitleTerms.keys()];
  state.filtersActive = Boolean(
    state.activeTitleTerms.size ||
      state.activeDepartments.size ||
      state.activeFaculty.size,
  );
  state.matchedPoints = state.points.filter((point) =>
    pointMatches(point, titleQueries),
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

function initializeDepartmentColors() {
  state.departments.forEach((department, index) => {
    state.departmentColors.set(
      department.department_id,
      DEPARTMENT_COLORS[index % DEPARTMENT_COLORS.length],
    );
  });
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
      hideSuggestions();
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
    window.setTimeout(hideSuggestions, 120);
  });
  input.addEventListener("keydown", (event) => {
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
    } else if (event.key === "Escape") {
      hideSuggestions();
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
  const palette = generator(state.faculty.length);
  state.facultyColors = new Map(
    state.faculty.map((person, index) => [person.person_id, palette[index]]),
  );
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
    state.titlePalette = generator(nextCapacity);
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
  authorFilter.renderSelected();
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
  state.activeTitleTerms.clear();
  renderSelectedTitles();
  authorFilter.clear();
  departmentFilter.clear();
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
      !Number.isInteger(artifact.schema_version) ||
      artifact.schema_version < 4 ||
      !Array.isArray(artifact.points) ||
      !Array.isArray(artifact.layouts) ||
      !Array.isArray(artifact.catalogs?.departments) ||
      !Array.isArray(artifact.catalogs?.faculty)
    ) {
      throw new Error("The publication artifact has an unsupported schema");
    }
    state.departments = artifact.catalogs.departments;
    state.faculty = artifact.catalogs.faculty;
    state.layouts = artifact.layouts.filter(
      (layout) =>
        layout &&
        typeof layout.layout_id === "string" &&
        typeof layout.label === "string" &&
        typeof layout.method === "string" &&
        typeof layout.description === "string" &&
        typeof layout.x_field === "string" &&
        typeof layout.y_field === "string",
    );
    state.layoutById = new Map(
      state.layouts.map((layout) => [layout.layout_id, layout]),
    );
    state.layoutId = artifact.default_layout_id;
    if (
      state.layouts.length < 2 ||
      state.layoutById.size !== state.layouts.length ||
      !state.layoutById.has(state.layoutId)
    ) {
      throw new Error("The publication artifact has invalid layout metadata");
    }
    state.departmentById = new Map(
      state.departments.map((item) => [item.department_id, item]),
    );
    state.facultyById = new Map(
      state.faculty.map((item) => [item.person_id, item]),
    );
    state.points = artifact.points
      .map(preparePoint)
      .filter(hasEveryLayout);
    if (state.points.length !== artifact.points.length) {
      throw new Error("The publication artifact has incomplete layout coordinates");
    }
    populateLayouts();
    initializeFacultyColors();
    initializeDepartmentColors();
    renderSelectedTitles();
    authorFilter.renderSelected();
    departmentFilter.renderSelected();
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
