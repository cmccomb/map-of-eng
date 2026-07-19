(function initializeResearchMapCore(root, factory) {
  const api = Object.freeze(factory());
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.ResearchMapCore = api;
})(typeof globalThis === "object" ? globalThis : this, function buildCore() {
  "use strict";

  const MAX_LAYOUTS = 8;
  const MAX_POINTS = 500000;
  const MAX_ARTIFACT_BYTES = 64 * 1024 * 1024;
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
      if (!required && items === undefined) return { items: [], ids: new Set() };
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
      return {
        ...item,
        keyword_id: keywordId,
        label: requiredString(item.label, `keywords[${index}].label`, {
          maxLength: 160,
        }),
        publication_count: publicationCount,
        coordinates,
      };
    });
    return { items: parsed, ids: seen };
  }

  function parsePoint(
    point,
    index,
    layouts,
    departmentIds,
    facultyIds,
    keywordIds,
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
    const keywordId = optionalString(point.keyword_id, { maxLength: 200 });
    if (requireKeyword && !keywordId) {
      throw new ArtifactError(`points[${index}].keyword_id is required`);
    }
    if (keywordId && !keywordIds.has(keywordId)) {
      throw new ArtifactError(`points[${index}].keyword_id is unknown`);
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
      keyword_id: keywordId,
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
            keywords.ids,
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
      points,
      omitted_point_count: omittedPointCount,
      omission_reasons: omissionReasons,
      warnings,
    };
  }

  function pointMatches(
    point,
    { titleQueries = [], departmentIds = new Set(), facultyIds = new Set() } = {},
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
    return true;
  }

  function preferredId(ids, activeIds, availableIds) {
    return (
      ids.find((id) => activeIds.has(id) && availableIds.has(id)) ||
      ids.find((id) => availableIds.has(id)) ||
      ""
    );
  }

  function baseScale(width, height) {
    const margin = 34;
    return Math.max(1, Math.min(width - margin * 2, height - margin * 2) / 2);
  }

  function fitView(points, width, height) {
    if (!Array.isArray(points) || !points.length) return null;
    if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
      throw new TypeError("Canvas dimensions must be positive numbers");
    }
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (const point of points) {
      if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) continue;
      minX = Math.min(minX, point.x);
      maxX = Math.max(maxX, point.x);
      minY = Math.min(minY, point.y);
      maxY = Math.max(maxY, point.y);
    }
    if (!Number.isFinite(minX)) return null;
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
    fitView,
    formatUtcDate,
    normalizedText,
    parseArtifact,
    parseConfig,
    pointMatches,
    preferredId,
    safeHttpUrl,
  };
});
