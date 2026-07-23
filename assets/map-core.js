(function initializeResearchMapCore(root, factory) {
  const api = Object.freeze(factory());
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.ResearchMapCore = api;
})(typeof globalThis === "object" ? globalThis : this, function buildCore() {
  "use strict";

  const MAX_LAYOUTS = 8;
  const MAX_POINTS = 500000;
  const MAX_ARTIFACT_BYTES = 64 * 1024 * 1024;
  const OVERVIEW_TOPIC_MIN_ISOLATION = 3.5;
  const OVERVIEW_TOPIC_LIMIT = 3;
  const ACTIVE_TOPIC_MIN_SHARE = 0.05;
  const ACTIVE_TOPIC_PARENT_DOMINANCE = 0.65;
  const ACTIVE_TOPIC_LIMIT = 4;
  const MAX_URL_STATE_VALUES = 64;
  const MAX_URL_STATE_LENGTH = 8192;
  const MAX_URL_STATE_VALUE_LENGTH = 500;
  const FIELD_NAME_PATTERN = /^[A-Za-z][A-Za-z0-9_]{0,63}$/;
  const DATASET_ID_PATTERN = /^[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+$/;

  class ArtifactError extends Error {
    constructor(message) {
      super(message);
      this.name = "ArtifactError";
    }
  }

  function isObject(value) {
    return Boolean(value) && typeof value === "object" && !Array.isArray(value);
  }

  function normalizedText(value) {
    return String(value || "")
      .normalize("NFKC")
      .trim()
      .toLocaleLowerCase();
  }

  function requiredString(value, label, { maxLength = 1000 } = {}) {
    if (typeof value !== "string" || !value.trim()) {
      throw new ArtifactError(`${label} must be a non-empty string`);
    }
    const trimmed = value.trim();
    if (trimmed.length > maxLength) {
      throw new ArtifactError(`${label} is longer than ${maxLength} characters`);
    }
    return trimmed;
  }

  function optionalString(value, { maxLength = 10000 } = {}) {
    if (typeof value !== "string") return "";
    const trimmed = value.trim();
    return trimmed.length <= maxLength ? trimmed : trimmed.slice(0, maxLength);
  }

  function safeHttpUrl(value) {
    const candidate = optionalString(value, { maxLength: 4000 });
    if (!candidate) return "";
    try {
      const url = new URL(candidate);
      if (url.protocol !== "http:" && url.protocol !== "https:") return "";
      return url.href;
    } catch {
      return "";
    }
  }

  function safeRelativePath(value, label) {
    const path = requiredString(value, label, { maxLength: 500 });
    if (
      path.startsWith("/") ||
      path.includes("\\") ||
      path.split("/").some((part) => !part || part === "." || part === "..")
    ) {
      throw new ArtifactError(`${label} must be a safe relative path`);
    }
    return path;
  }

  function parseConfig(config) {
    if (!isObject(config)) throw new ArtifactError("Map configuration is invalid");
    const title = requiredString(config.title, "config.title", { maxLength: 200 });
    const heading = optionalString(config.heading, { maxLength: 200 });
    const defaultLayoutId = optionalString(config.default_layout_id, {
      maxLength: 64,
    });
    if (defaultLayoutId && !FIELD_NAME_PATTERN.test(defaultLayoutId)) {
      throw new ArtifactError("config.default_layout_id is invalid");
    }
    const artifactBytes =
      config.artifact_bytes === undefined ? 0 : Number(config.artifact_bytes);
    if (
      !Number.isSafeInteger(artifactBytes) ||
      artifactBytes < 0 ||
      artifactBytes > MAX_ARTIFACT_BYTES
    ) {
      throw new ArtifactError("config.artifact_bytes is invalid");
    }
    const artifactUrl = optionalString(config.artifact_url, { maxLength: 4000 });
    if (artifactUrl) {
      const absoluteUrl = safeHttpUrl(artifactUrl);
      if (absoluteUrl) {
        return {
          title,
          heading,
          default_layout_id: defaultLayoutId,
          artifact_url: absoluteUrl,
          artifact_bytes: artifactBytes,
        };
      }
      if (
        artifactUrl.startsWith("//") ||
        /^[A-Za-z][A-Za-z0-9+.-]*:/.test(artifactUrl)
      ) {
        throw new ArtifactError(
          "config.artifact_url must use HTTP, HTTPS, or a relative path",
        );
      }
      return {
        title,
        heading,
        default_layout_id: defaultLayoutId,
        artifact_url: safeRelativePath(artifactUrl, "config.artifact_url"),
        artifact_bytes: artifactBytes,
      };
    }
    const datasetId = requiredString(config.dataset_id, "config.dataset_id", {
      maxLength: 200,
    });
    if (!DATASET_ID_PATTERN.test(datasetId)) {
      throw new ArtifactError("config.dataset_id has an invalid format");
    }
    const revision = requiredString(
      config.dataset_revision || "main",
      "config.dataset_revision",
      { maxLength: 200 },
    );
    const artifactPath = safeRelativePath(
      config.artifact_path || "maps/publications.json",
      "config.artifact_path",
    );
    return {
      title,
      heading,
      default_layout_id: defaultLayoutId,
      dataset_id: datasetId,
      dataset_revision: revision,
      artifact_path: artifactPath,
      artifact_bytes: artifactBytes,
    };
  }

  function artifactUrl(config) {
    if (config.artifact_url) return config.artifact_url;
    const revision = encodeURIComponent(config.dataset_revision);
    const path = config.artifact_path
      .split("/")
      .map((part) => encodeURIComponent(part))
      .join("/");
    return `https://huggingface.co/datasets/${config.dataset_id}/resolve/${revision}/${path}`;
  }

  function parseLayouts(layouts, defaultLayoutId) {
    if (!Array.isArray(layouts) || layouts.length < 2) {
      throw new ArtifactError("At least two map layouts are required");
    }
    if (layouts.length > MAX_LAYOUTS) {
      throw new ArtifactError(`At most ${MAX_LAYOUTS} map layouts are supported`);
    }
    const seen = new Set();
    const parsed = layouts.map((layout, index) => {
      if (!isObject(layout)) {
        throw new ArtifactError(`layouts[${index}] must be an object`);
      }
      const layoutId = requiredString(
        layout.layout_id,
        `layouts[${index}].layout_id`,
        { maxLength: 64 },
      );
      if (seen.has(layoutId)) {
        throw new ArtifactError(`Duplicate layout id: ${layoutId}`);
      }
      seen.add(layoutId);
      const xField = requiredString(layout.x_field, `${layoutId}.x_field`, {
        maxLength: 64,
      });
      const yField = requiredString(layout.y_field, `${layoutId}.y_field`, {
        maxLength: 64,
      });
      if (
        !FIELD_NAME_PATTERN.test(xField) ||
        !FIELD_NAME_PATTERN.test(yField) ||
        xField === yField
      ) {
        throw new ArtifactError(`Layout ${layoutId} has invalid coordinate fields`);
      }
      return {
        ...layout,
        layout_id: layoutId,
        label: requiredString(layout.label, `${layoutId}.label`, {
          maxLength: 80,
        }),
        method: requiredString(layout.method, `${layoutId}.method`, {
          maxLength: 80,
        }),
        description: requiredString(
          layout.description,
          `${layoutId}.description`,
          { maxLength: 500 },
        ),
        x_field: xField,
        y_field: yField,
      };
    });
    const selectedId = requiredString(defaultLayoutId, "default_layout_id", {
      maxLength: 64,
    });
    if (!seen.has(selectedId)) {
      throw new ArtifactError("The default layout is not in the layout catalog");
    }
    return { layouts: parsed, defaultLayoutId: selectedId };
  }

  function parseCatalog(items, { name, idField, labelField }) {
    if (!Array.isArray(items) || !items.length) {
      throw new ArtifactError(`${name} catalog must not be empty`);
    }
    const seen = new Set();
    const parsed = items.map((item, index) => {
      if (!isObject(item)) {
        throw new ArtifactError(`${name}[${index}] must be an object`);
      }
      const id = requiredString(item[idField], `${name}[${index}].${idField}`, {
        maxLength: 200,
      });
      if (seen.has(id)) throw new ArtifactError(`Duplicate ${name} id: ${id}`);
      seen.add(id);
      const publicationCount = Number(item.publication_count);
      if (!Number.isInteger(publicationCount) || publicationCount < 0) {
        throw new ArtifactError(`${name}[${index}].publication_count is invalid`);
      }
      return {
        ...item,
        [idField]: id,
        [labelField]: requiredString(
          item[labelField],
          `${name}[${index}].${labelField}`,
          { maxLength: 500 },
        ),
        publication_count: publicationCount,
      };
    });
    return { items: parsed, ids: seen };
  }

  function knownIdArray(value, knownIds, label, { allowEmpty = true } = {}) {
    if (!Array.isArray(value)) throw new ArtifactError(`${label} must be an array`);
    const parsed = [];
    const seen = new Set();
    for (const candidate of value) {
      if (typeof candidate !== "string" || !candidate || !knownIds.has(candidate)) {
        throw new ArtifactError(`${label} contains an unknown id`);
      }
      if (seen.has(candidate)) throw new ArtifactError(`${label} contains duplicates`);
      seen.add(candidate);
      parsed.push(candidate);
    }
    if (!allowEmpty && !parsed.length) {
      throw new ArtifactError(`${label} must not be empty`);
    }
    return parsed;
  }

  function numberOr(value, fallback, { minimum = -Infinity } = {}) {
    const parsed = Number(value);
    return Number.isFinite(parsed) && parsed >= minimum ? parsed : fallback;
  }

  function parseKeywords(
    items,
    layouts,
    { required = false, allowEmpty = false } = {},
  ) {
    if (!Array.isArray(items)) {
      if (!required && items === undefined) {
        return { items: [], ids: new Set(), byId: new Map() };
      }
      throw new ArtifactError("Artifact keywords must be an array");
    }
    if (required && !allowEmpty && !items.length) {
      throw new ArtifactError("Artifact keywords must not be empty");
    }
    const seen = new Set();
    const parsed = items.map((item, index) => {
      if (!isObject(item)) {
        throw new ArtifactError(`keywords[${index}] must be an object`);
      }
      const keywordId = requiredString(
        item.keyword_id,
        `keywords[${index}].keyword_id`,
        { maxLength: 200 },
      );
      if (seen.has(keywordId)) {
        throw new ArtifactError(`Duplicate keyword id: ${keywordId}`);
      }
      seen.add(keywordId);
      const publicationCount = Number(item.publication_count);
      if (!Number.isInteger(publicationCount) || publicationCount < 1) {
        throw new ArtifactError(
          `keywords[${index}].publication_count is invalid`,
        );
      }
      if (!isObject(item.coordinates)) {
        throw new ArtifactError(`keywords[${index}].coordinates is invalid`);
      }
      const coordinates = Object.create(null);
      for (const layout of layouts) {
        const pair = item.coordinates[layout.layout_id];
        if (
          !isObject(pair) ||
          typeof pair.x !== "number" ||
          typeof pair.y !== "number" ||
          !Number.isFinite(pair.x) ||
          !Number.isFinite(pair.y)
        ) {
          throw new ArtifactError(
            `keywords[${index}] has invalid ${layout.layout_id} coordinates`,
          );
        }
        coordinates[layout.layout_id] = { x: pair.x, y: pair.y };
      }
      const level = !required && item.level === undefined ? 0 : Number(item.level);
      if (!Number.isInteger(level) || level < 0 || level > 8) {
        throw new ArtifactError(`keywords[${index}].level is invalid`);
      }
      return {
        ...item,
        keyword_id: keywordId,
        label: requiredString(item.label, `keywords[${index}].label`, {
          maxLength: 160,
        }),
        level,
        parent_keyword_id: optionalString(item.parent_keyword_id, {
          maxLength: 200,
        }),
        publication_count: publicationCount,
        coordinates,
      };
    });
    const byId = new Map(parsed.map((item) => [item.keyword_id, item]));
    for (const item of parsed) {
      if (item.level === 0 && item.parent_keyword_id) {
        throw new ArtifactError(`Keyword ${item.keyword_id} must not have a parent`);
      }
      if (item.level > 0) {
        const parent = byId.get(item.parent_keyword_id);
        if (!parent || parent.level !== item.level - 1) {
          throw new ArtifactError(`Keyword ${item.keyword_id} has an invalid parent`);
        }
      }
    }
    return { items: parsed, ids: seen, byId };
  }

  function parseKeywordLevels(items, keywords, { required = false } = {}) {
    if (!Array.isArray(items)) {
      if (!required && items === undefined) {
        return keywords.length
          ? [
              {
                level: 0,
                label: "Topic regions",
                keyword_count: keywords.length,
              },
            ]
          : [];
      }
      throw new ArtifactError("Artifact keyword_levels must be an array");
    }
    if (required && !items.length) {
      throw new ArtifactError("Artifact keyword_levels must not be empty");
    }
    return items.map((item, index) => {
      if (!isObject(item) || Number(item.level) !== index) {
        throw new ArtifactError(`keyword_levels[${index}] is invalid`);
      }
      const keywordCount = Number(item.keyword_count);
      const actualCount = keywords.filter((keyword) => keyword.level === index).length;
      if (!Number.isInteger(keywordCount) || keywordCount !== actualCount) {
        throw new ArtifactError(`keyword_levels[${index}].keyword_count is invalid`);
      }
      return {
        ...item,
        level: index,
        label: requiredString(item.label, `keyword_levels[${index}].label`, {
          maxLength: 160,
        }),
        keyword_count: keywordCount,
      };
    });
  }

  function parsePoint(
    point,
    index,
    layouts,
    departmentIds,
    facultyIds,
    keywords,
    keywordLevels,
    requireKeyword,
    seenWorkIds,
  ) {
    if (!isObject(point)) throw new ArtifactError(`points[${index}] is invalid`);
    const workId = requiredString(point.work_id, `points[${index}].work_id`, {
      maxLength: 500,
    });
    if (seenWorkIds.has(workId)) {
      throw new ArtifactError(`points[${index}] duplicates work_id ${workId}`);
    }
    const coordinates = Object.create(null);
    for (const layout of layouts) {
      const x = point[layout.x_field];
      const y = point[layout.y_field];
      if (
        typeof x !== "number" ||
        typeof y !== "number" ||
        !Number.isFinite(x) ||
        !Number.isFinite(y)
      ) {
        throw new ArtifactError(`points[${index}] has invalid ${layout.layout_id} coordinates`);
      }
      coordinates[layout.layout_id] = { x, y };
    }
    const year = Number(point.year);
    let keywordIds = [];
    if (requireKeyword) {
      keywordIds = knownIdArray(
        point.keyword_ids,
        keywords.ids,
        `points[${index}].keyword_ids`,
        { allowEmpty: false },
      );
      if (keywordIds.length !== keywordLevels.length) {
        throw new ArtifactError(`points[${index}].keyword_ids is incomplete`);
      }
      const selectedByLevel = new Map();
      for (const keywordId of keywordIds) {
        const keyword = keywords.byId.get(keywordId);
        if (selectedByLevel.has(keyword.level)) {
          throw new ArtifactError(`points[${index}].keyword_ids repeats a level`);
        }
        selectedByLevel.set(keyword.level, keywordId);
      }
      for (const keywordId of keywordIds) {
        const keyword = keywords.byId.get(keywordId);
        if (
          keyword.level > 0 &&
          selectedByLevel.get(keyword.level - 1) !== keyword.parent_keyword_id
        ) {
          throw new ArtifactError(`points[${index}].keyword_ids breaks the hierarchy`);
        }
      }
    } else {
      const legacyKeywordId = optionalString(point.keyword_id, {
        maxLength: 200,
      });
      if (legacyKeywordId) {
        if (!keywords.ids.has(legacyKeywordId)) {
          throw new ArtifactError(`points[${index}].keyword_id is unknown`);
        }
        keywordIds = [legacyKeywordId];
      }
    }
    const parsed = {
      ...point,
      work_id: workId,
      title: requiredString(point.title, `points[${index}].title`, {
        maxLength: 5000,
      }),
      authors: optionalString(point.authors, { maxLength: 10000 }),
      department_ids: knownIdArray(
        point.department_ids,
        departmentIds,
        `points[${index}].department_ids`,
        { allowEmpty: false },
      ),
      faculty_ids: knownIdArray(
        point.faculty_ids,
        facultyIds,
        `points[${index}].faculty_ids`,
      ),
      year: Number.isInteger(year) && year >= 1000 && year <= 3000 ? year : null,
      venue: optionalString(point.venue, { maxLength: 5000 }),
      citation_count: Math.floor(numberOr(point.citation_count, 0, { minimum: 0 })),
      observation_count: Math.max(
        1,
        Math.floor(numberOr(point.observation_count, 1, { minimum: 1 })),
      ),
      doi: optionalString(point.doi, { maxLength: 1000 }),
      source_url: safeHttpUrl(point.source_url),
      keyword_ids: keywordIds,
      _title: normalizedText(point.title),
      _coordinates: coordinates,
    };
    seenWorkIds.add(workId);
    return parsed;
  }

  function parseArtifact(artifact) {
    if (!isObject(artifact)) throw new ArtifactError("Artifact must be an object");
    if (!Number.isInteger(artifact.schema_version) || artifact.schema_version < 4) {
      throw new ArtifactError("The publication artifact has an unsupported schema");
    }
    if (!Array.isArray(artifact.points) || artifact.points.length > MAX_POINTS) {
      throw new ArtifactError(`Artifact points must contain at most ${MAX_POINTS} rows`);
    }
    if (!isObject(artifact.catalogs)) {
      throw new ArtifactError("Artifact catalogs are missing");
    }
    const { layouts, defaultLayoutId } = parseLayouts(
      artifact.layouts,
      artifact.default_layout_id,
    );
    const departments = parseCatalog(artifact.catalogs.departments, {
      name: "departments",
      idField: "department_id",
      labelField: "title",
    });
    const faculty = parseCatalog(artifact.catalogs.faculty, {
      name: "faculty",
      idField: "person_id",
      labelField: "display_name",
    });
    const requireKeywords = artifact.schema_version >= 6;
    const keywords = parseKeywords(artifact.keywords, layouts, {
      required: requireKeywords,
      allowEmpty: artifact.points.length === 0,
    });
    const keywordLevels = parseKeywordLevels(
      artifact.keyword_levels,
      keywords.items,
      { required: requireKeywords },
    );
    const seenWorkIds = new Set();
    const points = [];
    const omissionReasons = new Map();
    for (let index = 0; index < artifact.points.length; index += 1) {
      try {
        points.push(
          parsePoint(
            artifact.points[index],
            index,
            layouts,
            departments.ids,
            faculty.ids,
            keywords,
            keywordLevels,
            requireKeywords,
            seenWorkIds,
          ),
        );
      } catch (error) {
        const reason = error instanceof Error ? error.message : "Invalid point";
        omissionReasons.set(reason, (omissionReasons.get(reason) || 0) + 1);
      }
    }
    const omittedPointCount = artifact.points.length - points.length;
    const warnings = [];
    if (
      Number.isInteger(artifact.point_count) &&
      artifact.point_count !== artifact.points.length
    ) {
      warnings.push(
        `Artifact point_count ${artifact.point_count} does not match ${artifact.points.length} rows`,
      );
    }
    if (omittedPointCount) {
      warnings.push(`${omittedPointCount} invalid publication records were omitted`);
    }
    return {
      ...artifact,
      layouts,
      default_layout_id: defaultLayoutId,
      catalogs: {
        ...artifact.catalogs,
        departments: departments.items,
        faculty: faculty.items,
      },
      keywords: keywords.items,
      keyword_levels: keywordLevels,
      points,
      omitted_point_count: omittedPointCount,
      omission_reasons: omissionReasons,
      warnings,
    };
  }

  function pointMatches(
    point,
    {
      titleQueries = [],
      departmentIds = new Set(),
      facultyIds = new Set(),
      keywordIds = new Set(),
      yearMin = null,
      yearMax = null,
    } = {},
  ) {
    if (
      titleQueries.length &&
      !titleQueries.some((query) => point._title.includes(query))
    ) {
      return false;
    }
    if (
      departmentIds.size &&
      !point.department_ids.some((id) => departmentIds.has(id))
    ) {
      return false;
    }
    if (facultyIds.size && !point.faculty_ids.some((id) => facultyIds.has(id))) {
      return false;
    }
    if (
      keywordIds.size &&
      !point.keyword_ids.some((id) => keywordIds.has(id))
    ) {
      return false;
    }
    const yearFiltered = Number.isFinite(yearMin) || Number.isFinite(yearMax);
    if (yearFiltered && !Number.isFinite(point.year)) return false;
    if (Number.isFinite(yearMin) && point.year < yearMin) return false;
    if (Number.isFinite(yearMax) && point.year > yearMax) return false;
    return true;
  }

  function incrementFacetCounts(counts, ids) {
    if (!Array.isArray(ids)) return;
    for (const id of new Set(ids)) {
      counts.set(id, (counts.get(id) || 0) + 1);
    }
  }

  function contextualFacetCounts(points, filters = {}) {
    const facultyCounts = new Map();
    const departmentCounts = new Map();
    const keywordCounts = new Map();
    if (!Array.isArray(points)) {
      return { facultyCounts, departmentCounts, keywordCounts };
    }
    const withoutFaculty = { ...filters, facultyIds: new Set() };
    const withoutDepartments = { ...filters, departmentIds: new Set() };
    const withoutKeywords = { ...filters, keywordIds: new Set() };
    for (const point of points) {
      if (pointMatches(point, withoutFaculty)) {
        incrementFacetCounts(facultyCounts, point.faculty_ids);
      }
      if (pointMatches(point, withoutDepartments)) {
        incrementFacetCounts(departmentCounts, point.department_ids);
      }
      if (pointMatches(point, withoutKeywords)) {
        incrementFacetCounts(keywordCounts, point.keyword_ids);
      }
    }
    return { facultyCounts, departmentCounts, keywordCounts };
  }

  function stateValues(value, { normalize = false } = {}) {
    const candidates =
      value instanceof Set ? [...value] : Array.isArray(value) ? value : [];
    const values = [];
    const seen = new Set();
    for (const candidate of candidates) {
      if (typeof candidate !== "string") continue;
      const parsed = normalize
        ? normalizedText(candidate)
        : candidate.trim().slice(0, MAX_URL_STATE_VALUE_LENGTH);
      if (!parsed || parsed.length > MAX_URL_STATE_VALUE_LENGTH || seen.has(parsed)) {
        continue;
      }
      seen.add(parsed);
      values.push(parsed);
      if (values.length >= MAX_URL_STATE_VALUES) break;
    }
    return values;
  }

  function validStateYear(value) {
    const year = Number(value);
    return Number.isInteger(year) && year >= 1000 && year <= 3000 ? year : null;
  }

  function serializeMapState(state = {}) {
    const params = new URLSearchParams();
    const collections = [
      ["q", stateValues(state.titleQueries, { normalize: true })],
      ["d", stateValues(state.departmentIds)],
      ["f", stateValues(state.facultyIds)],
      ["t", stateValues(state.keywordIds)],
    ];
    for (const [key, values] of collections) {
      for (const value of values) params.append(key, value);
    }
    const yearMin = validStateYear(state.yearMin);
    const yearMax = validStateYear(state.yearMax);
    if (yearMin !== null) params.set("ymin", String(yearMin));
    if (yearMax !== null) params.set("ymax", String(yearMax));
    for (const [key, value] of [
      ["layout", state.layoutId],
      ["color", state.colorMode],
      ["size", state.sizeMode],
      ["display", state.displayMode],
    ]) {
      const parsed = stateValues([value])[0];
      if (parsed) params.set(key, parsed);
    }
    return params.toString();
  }

  function parseMapState(search = "") {
    const source = typeof search === "string" ? search : String(search || "");
    const params =
      source.length <= MAX_URL_STATE_LENGTH
        ? new URLSearchParams(source.startsWith("?") ? source.slice(1) : source)
        : new URLSearchParams();
    const values = (key, options) => stateValues(params.getAll(key), options);
    const scalar = (key) => stateValues([params.get(key)])[0] || "";
    return {
      titleQueries: values("q", { normalize: true }),
      departmentIds: values("d"),
      facultyIds: values("f"),
      keywordIds: values("t"),
      yearMin: validStateYear(params.get("ymin")),
      yearMax: validStateYear(params.get("ymax")),
      layoutId: scalar("layout"),
      colorMode: scalar("color"),
      sizeMode: scalar("size"),
      displayMode: scalar("display"),
    };
  }

  function preferredId(ids, activeIds, availableIds) {
    return (
      ids.find((id) => activeIds.has(id) && availableIds.has(id)) ||
      ids.find((id) => availableIds.has(id)) ||
      ""
    );
  }

  function keywordCoordinates(keyword, layoutId) {
    const coordinates =
      keyword?._coordinates instanceof Map
        ? keyword._coordinates.get(layoutId)
        : keyword?.coordinates?.[layoutId];
    return coordinates &&
      Number.isFinite(coordinates.x) &&
      Number.isFinite(coordinates.y)
      ? coordinates
      : null;
  }

  function buildKeywordLabelPlan(
    keywords,
    points,
    { layoutId = "", filtersActive = false } = {},
  ) {
    if (!Array.isArray(keywords) || !Array.isArray(points)) return [];
    const byId = new Map(keywords.map((keyword) => [keyword.keyword_id, keyword]));
    const activity = new Map();
    for (const point of points) {
      if (!Array.isArray(point.keyword_ids)) continue;
      for (const keywordId of point.keyword_ids) {
        if (!byId.has(keywordId)) continue;
        const existing = activity.get(keywordId) || {
          count: 0,
          positionCount: 0,
          spreadSquared: 0,
          x: 0,
          y: 0,
        };
        existing.count += 1;
        if (Number.isFinite(point.x) && Number.isFinite(point.y)) {
          existing.positionCount += 1;
          existing.x += point.x;
          existing.y += point.y;
          const coordinates = keywordCoordinates(byId.get(keywordId), layoutId);
          if (coordinates) {
            existing.spreadSquared +=
              (point.x - coordinates.x) ** 2 +
              (point.y - coordinates.y) ** 2;
          }
        }
        activity.set(keywordId, existing);
      }
    }

    const promoted = new Map();
    const demoted = new Set();

    if (filtersActive && points.length) {
      const minimumActiveSupport = Math.max(
        4,
        Math.ceil(points.length * ACTIVE_TOPIC_MIN_SHARE),
      );
      const activeCandidates = [];
      for (const parent of keywords.filter((keyword) => keyword.level === 0)) {
        const parentCount = activity.get(parent.keyword_id)?.count || 0;
        if (!parentCount) continue;
        const children = keywords
          .filter((keyword) => keyword.parent_keyword_id === parent.keyword_id)
          .sort(
            (left, right) =>
              (activity.get(right.keyword_id)?.count || 0) -
                (activity.get(left.keyword_id)?.count || 0) ||
              left.label.localeCompare(right.label),
          );
        const dominant = children[0];
        const dominantCount = activity.get(dominant?.keyword_id)?.count || 0;
        if (
          dominant &&
          dominantCount >= minimumActiveSupport &&
          dominantCount / parentCount >= ACTIVE_TOPIC_PARENT_DOMINANCE
        ) {
          activeCandidates.push({
            child: dominant,
            count: dominantCount,
            dominance: dominantCount / parentCount,
            parent,
          });
        }
      }
      activeCandidates
        .sort(
          (left, right) =>
            right.count - left.count ||
            right.dominance - left.dominance ||
            left.child.label.localeCompare(right.child.label),
        )
        .slice(0, ACTIVE_TOPIC_LIMIT)
        .forEach(({ child, parent }) => {
          promoted.set(child.keyword_id, "active");
          demoted.add(parent.keyword_id);
        });
    } else if (layoutId === "tsne") {
      const minimumOverviewSupport = Math.max(50, Math.ceil(points.length * 0.003));
      const details = keywords
        .filter((keyword) => keyword.level === 1)
        .map((keyword) => ({
          keyword,
          coordinates: keywordCoordinates(keyword, layoutId),
        }))
        .filter((item) => item.coordinates);
      details
        .map(({ keyword, coordinates }) => {
          const totals = activity.get(keyword.keyword_id);
          let nearestDistance = Infinity;
          for (const candidate of details) {
            if (candidate.keyword === keyword) continue;
            nearestDistance = Math.min(
              nearestDistance,
              Math.hypot(
                coordinates.x - candidate.coordinates.x,
                coordinates.y - candidate.coordinates.y,
              ),
            );
          }
          if (!Number.isFinite(nearestDistance)) nearestDistance = 0;
          const radius = totals?.positionCount
            ? Math.sqrt(totals.spreadSquared / totals.positionCount)
            : Infinity;
          return {
            keyword,
            count: totals?.count || 0,
            isolation: nearestDistance / Math.max(radius, 0.001),
          };
        })
        .filter(
          (candidate) =>
            candidate.count >= minimumOverviewSupport &&
            candidate.isolation >= OVERVIEW_TOPIC_MIN_ISOLATION,
        )
        .sort(
          (left, right) =>
            right.isolation - left.isolation ||
            right.count - left.count ||
            left.keyword.label.localeCompare(right.keyword.label),
        )
        .slice(0, OVERVIEW_TOPIC_LIMIT)
        .forEach(({ keyword }) => {
          promoted.set(keyword.keyword_id, "isolation");
        });
    }

    return keywords.map((keyword) => {
      const totals = activity.get(keyword.keyword_id) || {
        count: 0,
        positionCount: 0,
        spreadSquared: 0,
        x: 0,
        y: 0,
      };
      const coordinates = keywordCoordinates(keyword, layoutId) || { x: 0, y: 0 };
      const useActiveCentroid = filtersActive && totals.positionCount > 0;
      const promotion = promoted.get(keyword.keyword_id) || "";
      return {
        ...keyword,
        activity_count: totals.count,
        effective_level: promotion ? 0 : demoted.has(keyword.keyword_id) ? 1 : keyword.level,
        promotion,
        x: useActiveCentroid ? totals.x / totals.positionCount : coordinates.x,
        y: useActiveCentroid ? totals.y / totals.positionCount : coordinates.y,
      };
    });
  }

  function baseScale(width, height) {
    const margin = 34;
    return Math.max(1, Math.min(width - margin * 2, height - margin * 2) / 2);
  }

  function median(values) {
    const sorted = [...values].sort((left, right) => left - right);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2
      ? sorted[middle]
      : (sorted[middle - 1] + sorted[middle]) / 2;
  }

  function pointBounds(points, { centralFraction = 1 } = {}) {
    if (!Array.isArray(points) || !points.length) return null;
    if (
      !Number.isFinite(centralFraction) ||
      centralFraction <= 0 ||
      centralFraction > 1
    ) {
      throw new RangeError("centralFraction must be greater than 0 and at most 1");
    }
    const finitePoints = points.filter(
      (point) => Number.isFinite(point.x) && Number.isFinite(point.y),
    );
    if (!finitePoints.length) return null;
    const includedCount = Math.max(
      1,
      Math.ceil(finitePoints.length * centralFraction),
    );
    let includedPoints = finitePoints;
    if (includedCount < finitePoints.length) {
      const centerX = median(finitePoints.map((point) => point.x));
      const centerY = median(finitePoints.map((point) => point.y));
      includedPoints = finitePoints
        .map((point, index) => ({
          index,
          point,
          distanceSquared:
            (point.x - centerX) ** 2 + (point.y - centerY) ** 2,
        }))
        .sort(
          (left, right) =>
            left.distanceSquared - right.distanceSquared ||
            left.index - right.index,
        )
        .slice(0, includedCount)
        .map((item) => item.point);
    }
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (const point of includedPoints) {
      minX = Math.min(minX, point.x);
      maxX = Math.max(maxX, point.x);
      minY = Math.min(minY, point.y);
      maxY = Math.max(maxY, point.y);
    }
    return {
      minX,
      maxX,
      minY,
      maxY,
      centralFraction,
      pointCount: finitePoints.length,
      includedCount,
      excludedCount: finitePoints.length - includedCount,
    };
  }

  function fitView(points, width, height, { centralFraction = 1 } = {}) {
    if (!Array.isArray(points) || !points.length) return null;
    if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
      throw new TypeError("Canvas dimensions must be positive numbers");
    }
    const bounds = pointBounds(points, { centralFraction });
    if (!bounds) return null;
    const { minX, maxX, minY, maxY } = bounds;
    const rangeX = Math.max(maxX - minX, 0.04);
    const rangeY = Math.max(maxY - minY, 0.04);
    const padding = 28;
    const unitScale = baseScale(width, height);
    const scale = Math.min(
      25,
      Math.max(
        0.7,
        Math.min(
          Math.max(1, width - padding * 2) / (rangeX * unitScale),
          Math.max(1, height - padding * 2) / (rangeY * unitScale),
        ),
      ),
    );
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    return {
      scale,
      offsetX: -centerX * unitScale * scale,
      offsetY: centerY * unitScale * scale,
      centralFraction: bounds.centralFraction,
      includedCount: bounds.includedCount,
      excludedCount: bounds.excludedCount,
    };
  }

  function formatUtcDate(value, locales) {
    if (typeof value !== "string" || !value.trim()) return "";
    const date = new Date(value);
    if (!Number.isFinite(date.getTime())) return "";
    return new Intl.DateTimeFormat(locales, {
      year: "numeric",
      month: "short",
      day: "numeric",
      timeZone: "UTC",
    }).format(date);
  }

  return {
    ArtifactError,
    artifactUrl,
    baseScale,
    buildKeywordLabelPlan,
    contextualFacetCounts,
    fitView,
    formatUtcDate,
    normalizedText,
    parseArtifact,
    parseConfig,
    parseMapState,
    pointBounds,
    pointMatches,
    preferredId,
    safeHttpUrl,
    serializeMapState,
  };
});
