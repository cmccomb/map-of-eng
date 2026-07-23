"use strict";

const crypto = require("node:crypto");
const fs = require("node:fs/promises");
const path = require("node:path");
const core = require("../assets/map-core.js");

const DEFAULT_OUTPUT_DIRECTORY = "_site";
const MAX_API_BYTES = 1024 * 1024;
const MAX_ARTIFACT_BYTES = 64 * 1024 * 1024;
const OVERVIEW_PROMOTION_LIMIT = 3;
const ACTIVE_PROMOTION_LIMIT = 4;
const CALIBRATION_LAYOUT_ID = "tsne";
const REVISION_PATTERN = /^[0-9a-f]{40}$/;

function datasetRevisionUrl(config) {
  const [owner, repository] = config.dataset_id.split("/");
  const revision = encodeURIComponent(config.dataset_revision);
  return `https://huggingface.co/api/datasets/${encodeURIComponent(owner)}/${encodeURIComponent(repository)}/revision/${revision}`;
}

async function fetchJsonDocument(fetchImpl, url, label, maximumBytes) {
  const response = await fetchImpl(url, {
    headers: { Accept: "application/json" },
    redirect: "follow",
  });
  if (!response.ok) {
    throw new Error(`${label} returned HTTP ${response.status}`);
  }
  const declaredLength = Number(response.headers.get("content-length"));
  if (Number.isFinite(declaredLength) && declaredLength > maximumBytes) {
    throw new Error(`${label} exceeds the ${maximumBytes}-byte limit`);
  }
  const text = await response.text();
  if (Buffer.byteLength(text, "utf8") > maximumBytes) {
    throw new Error(`${label} exceeds the ${maximumBytes}-byte limit`);
  }
  try {
    return { text, value: JSON.parse(text) };
  } catch (error) {
    throw new Error(`${label} did not contain valid JSON`, { cause: error });
  }
}

async function resolveDatasetRevision(config, fetchImpl) {
  const { value } = await fetchJsonDocument(
    fetchImpl,
    datasetRevisionUrl(config),
    "Hugging Face dataset metadata",
    MAX_API_BYTES,
  );
  const revision = String(value.sha || "").toLowerCase();
  if (!REVISION_PATTERN.test(revision)) {
    throw new Error("Hugging Face did not return an immutable dataset revision");
  }
  return revision;
}

async function configuredDatasetRevision({
  sourceDirectory = path.resolve(__dirname, ".."),
  fetchImpl = globalThis.fetch,
} = {}) {
  if (typeof fetchImpl !== "function") {
    throw new Error("This build requires a Fetch-compatible runtime");
  }
  const sourceConfig = JSON.parse(
    await fs.readFile(path.join(sourceDirectory, "map-config.json"), "utf8"),
  );
  const config = core.parseConfig(sourceConfig);
  if (!config.dataset_id) {
    throw new Error("The source map configuration must identify a dataset");
  }
  return resolveDatasetRevision(config, fetchImpl);
}

function stampAssetUrls(html, version) {
  let stamped = html;
  for (const asset of [
    "assets/map.css",
    "assets/theme.js",
    "assets/colors.js",
    "assets/map-core.js",
    "assets/map.js",
  ]) {
    if (!stamped.includes(asset)) {
      throw new Error(`index.html does not reference ${asset}`);
    }
    stamped = stamped.replace(asset, `${asset}?v=${version}`);
  }
  return stamped;
}

function assertSafeOutputDirectory(sourceDirectory, outputDirectory) {
  const root = path.parse(outputDirectory).root;
  if (outputDirectory === root || outputDirectory === sourceDirectory) {
    throw new Error("Refusing to replace an unsafe site output directory");
  }
}

function positionedPoints(artifact, layoutId) {
  return artifact.points.map((point) => {
    const coordinates = point._coordinates?.[layoutId];
    if (
      !coordinates ||
      !Number.isFinite(coordinates.x) ||
      !Number.isFinite(coordinates.y)
    ) {
      throw new Error(`Publication artifact is missing ${layoutId} coordinates`);
    }
    return { ...point, x: coordinates.x, y: coordinates.y };
  });
}

function percentile(values, fraction) {
  if (!values.length) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.max(0, Math.ceil(sorted.length * fraction) - 1)];
}

function topicIsolationMetrics(keywords, points, layoutId) {
  const details = keywords
    .filter((keyword) => keyword.level === 1)
    .map((keyword) => ({
      keyword,
      coordinates: keyword.coordinates?.[layoutId],
    }))
    .filter(
      ({ coordinates }) =>
        Number.isFinite(coordinates?.x) && Number.isFinite(coordinates?.y),
    );
  const detailsById = new Map(
    details.map((item) => [item.keyword.keyword_id, item]),
  );
  const activity = new Map();
  for (const point of points) {
    for (const keywordId of point.keyword_ids) {
      const existing = activity.get(keywordId) || {
        count: 0,
        spreadSquared: 0,
      };
      const keyword = detailsById.get(keywordId);
      if (keyword) {
        existing.count += 1;
        existing.spreadSquared +=
          (point.x - keyword.coordinates.x) ** 2 +
          (point.y - keyword.coordinates.y) ** 2;
        activity.set(keywordId, existing);
      }
    }
  }
  return details.map(({ keyword, coordinates }) => {
    let nearestDistance = Infinity;
    for (const candidate of details) {
      if (candidate.keyword.keyword_id === keyword.keyword_id) continue;
      nearestDistance = Math.min(
        nearestDistance,
        Math.hypot(
          coordinates.x - candidate.coordinates.x,
          coordinates.y - candidate.coordinates.y,
        ),
      );
    }
    const totals = activity.get(keyword.keyword_id) || {
      count: 0,
      spreadSquared: 0,
    };
    const radius = totals.count
      ? Math.sqrt(totals.spreadSquared / totals.count)
      : Infinity;
    const isolation = Number.isFinite(nearestDistance)
      ? nearestDistance / Math.max(radius, 0.001)
      : 0;
    return {
      activity_count: totals.count,
      cluster_radius: Number.isFinite(radius) ? radius : null,
      isolation,
      keyword_id: keyword.keyword_id,
      label: keyword.label,
      nearest_topic_distance: Number.isFinite(nearestDistance)
        ? nearestDistance
        : null,
    };
  });
}

function compactTopicMetric(metric, pointCount) {
  if (!metric) return null;
  return {
    keyword_id: metric.keyword_id,
    label: metric.label,
    activity_count: metric.activity_count,
    corpus_share: Number((metric.activity_count / Math.max(1, pointCount)).toFixed(6)),
    isolation: Number(metric.isolation.toFixed(3)),
    cluster_radius:
      metric.cluster_radius === null
        ? null
        : Number(metric.cluster_radius.toFixed(6)),
    nearest_topic_distance:
      metric.nearest_topic_distance === null
        ? null
        : Number(metric.nearest_topic_distance.toFixed(6)),
  };
}

function labelCalibrationReport(artifact, { layoutId = CALIBRATION_LAYOUT_ID } = {}) {
  if (!artifact.layouts.some((layout) => layout.layout_id === layoutId)) {
    throw new Error(`Publication artifact has no ${layoutId} calibration layout`);
  }
  const points = positionedPoints(artifact, layoutId);
  const overviewPlan = core.buildKeywordLabelPlan(artifact.keywords, points, {
    layoutId,
  });
  const promotedIds = new Set(
    overviewPlan
      .filter((keyword) => keyword.promotion === "isolation")
      .map((keyword) => keyword.keyword_id),
  );
  if (promotedIds.size > OVERVIEW_PROMOTION_LIMIT) {
    throw new Error(
      `Overview label plan exceeds its ${OVERVIEW_PROMOTION_LIMIT}-topic release budget`,
    );
  }

  const minimumOverviewSupport = Math.max(
    50,
    Math.ceil(points.length * 0.003),
  );
  const metrics = topicIsolationMetrics(artifact.keywords, points, layoutId).sort(
    (left, right) =>
      right.isolation - left.isolation ||
      right.activity_count - left.activity_count ||
      left.label.localeCompare(right.label),
  );
  const promotedTopics = metrics
    .filter((metric) => promotedIds.has(metric.keyword_id))
    .map((metric) => compactTopicMetric(metric, points.length));
  const nextCandidate = metrics.find(
    (metric) =>
      !promotedIds.has(metric.keyword_id) &&
      metric.activity_count >= minimumOverviewSupport,
  );

  const pointsByFaculty = new Map(
    artifact.catalogs.faculty.map((person) => [person.person_id, []]),
  );
  for (const point of points) {
    for (const personId of point.faculty_ids) {
      pointsByFaculty.get(personId)?.push(point);
    }
  }
  const activePromotionCounts = [];
  for (const facultyPoints of pointsByFaculty.values()) {
    if (!facultyPoints.length) continue;
    const promotedCount = core
      .buildKeywordLabelPlan(artifact.keywords, facultyPoints, {
        filtersActive: true,
        layoutId,
      })
      .filter((keyword) => keyword.promotion === "active").length;
    if (promotedCount > ACTIVE_PROMOTION_LIMIT) {
      throw new Error(
        `Active label plan exceeds its ${ACTIVE_PROMOTION_LIMIT}-topic release budget`,
      );
    }
    activePromotionCounts.push(promotedCount);
  }
  const maximumActivePromotions = activePromotionCounts.length
    ? Math.max(...activePromotionCounts)
    : 0;

  return {
    schema_version: 1,
    layout_id: layoutId,
    point_count: points.length,
    broad_topic_count: artifact.keywords.filter((keyword) => keyword.level === 0)
      .length,
    detailed_topic_count: artifact.keywords.filter(
      (keyword) => keyword.level === 1,
    ).length,
    overview: {
      promotion_limit: OVERVIEW_PROMOTION_LIMIT,
      minimum_support: minimumOverviewSupport,
      promoted_count: promotedTopics.length,
      promoted_topics: promotedTopics,
      next_supported_candidate: compactTopicMetric(nextCandidate, points.length),
    },
    active_faculty_views: {
      promotion_limit: ACTIVE_PROMOTION_LIMIT,
      view_count: activePromotionCounts.length,
      zero_promotion_views: activePromotionCounts.filter((count) => count === 0)
        .length,
      median_promotions: percentile(activePromotionCounts, 0.5),
      p90_promotions: percentile(activePromotionCounts, 0.9),
      maximum_promotions: maximumActivePromotions,
      views_at_limit: activePromotionCounts.filter(
        (count) => count === ACTIVE_PROMOTION_LIMIT,
      ).length,
    },
  };
}

function formatLabelCalibrationReport(report, reportPath) {
  const promotedLabels = report.overview.promoted_topics
    .map((topic) => topic.label)
    .join(", ");
  const active = report.active_faculty_views;
  return [
    `Labels: ${report.overview.promoted_count}/${report.overview.promotion_limit} overview promotions on ${report.layout_id}: ${promotedLabels || "none"}.`,
    `Active-label drift: ${active.view_count} faculty views; median ${active.median_promotions}, p90 ${active.p90_promotions}, max ${active.maximum_promotions}/${active.promotion_limit}; ${active.views_at_limit} at the limit.`,
    `Calibration report: ${reportPath}`,
  ];
}

async function buildSite({
  sourceDirectory = path.resolve(__dirname, ".."),
  outputDirectory = path.resolve(sourceDirectory, DEFAULT_OUTPUT_DIRECTORY),
  fetchImpl = globalThis.fetch,
  assetVersion = process.env.GITHUB_SHA || "",
} = {}) {
  if (typeof fetchImpl !== "function") {
    throw new Error("This build requires a Fetch-compatible runtime");
  }
  const sourceRoot = path.resolve(sourceDirectory);
  const outputRoot = path.resolve(outputDirectory);
  assertSafeOutputDirectory(sourceRoot, outputRoot);

  const sourceConfig = JSON.parse(
    await fs.readFile(path.join(sourceRoot, "map-config.json"), "utf8"),
  );
  const config = core.parseConfig(sourceConfig);
  if (!config.dataset_id) {
    throw new Error("The source map configuration must identify a dataset");
  }

  const revision = await resolveDatasetRevision(config, fetchImpl);
  const pinnedConfig = { ...config, dataset_revision: revision };
  const artifactUrl = core.artifactUrl(pinnedConfig);
  const artifactDocument = await fetchJsonDocument(
    fetchImpl,
    artifactUrl,
    "Pinned publication artifact",
    MAX_ARTIFACT_BYTES,
  );
  const parsedArtifact = core.parseArtifact(artifactDocument.value);
  if (
    parsedArtifact.omitted_point_count !== 0 ||
    parsedArtifact.points.length !== artifactDocument.value.point_count
  ) {
    throw new Error("Pinned publication artifact failed strict row validation");
  }

  const artifactDigest = crypto
    .createHash("sha256")
    .update(artifactDocument.text, "utf8")
    .digest("hex");
  const artifactBytes = Buffer.byteLength(artifactDocument.text, "utf8");
  const artifactName = `publications.${revision}.json`;
  const calibrationName = `label-calibration.${revision}.json`;
  const calibrationPath = `data/${calibrationName}`;
  const calibration = labelCalibrationReport(parsedArtifact);
  const deployedConfig = {
    ...sourceConfig,
    dataset_revision: revision,
    artifact_url: `data/${artifactName}`,
    artifact_bytes: artifactBytes,
    artifact_sha256: artifactDigest,
    label_calibration_url: calibrationPath,
  };
  const version = (assetVersion || revision).slice(0, 12);

  await fs.rm(outputRoot, { recursive: true, force: true });
  await fs.mkdir(path.join(outputRoot, "data"), { recursive: true });
  await fs.cp(path.join(sourceRoot, "assets"), path.join(outputRoot, "assets"), {
    recursive: true,
  });
  const sourceHtml = await fs.readFile(path.join(sourceRoot, "index.html"), "utf8");
  await Promise.all([
    fs.writeFile(
      path.join(outputRoot, "index.html"),
      stampAssetUrls(sourceHtml, version),
      "utf8",
    ),
    fs.writeFile(
      path.join(outputRoot, "map-config.json"),
      `${JSON.stringify(deployedConfig, null, 2)}\n`,
      "utf8",
    ),
    fs.writeFile(
      path.join(outputRoot, "data", artifactName),
      artifactDocument.text,
      "utf8",
    ),
    fs.writeFile(
      path.join(outputRoot, "data", calibrationName),
      `${JSON.stringify(calibration, null, 2)}\n`,
      "utf8",
    ),
  ]);

  return {
    artifact_bytes: artifactBytes,
    artifact_path: deployedConfig.artifact_url,
    artifact_sha256: artifactDigest,
    label_calibration: calibration,
    label_calibration_path: calibrationPath,
    dataset_revision: revision,
    point_count: parsedArtifact.points.length,
  };
}

async function main() {
  if (process.argv.includes("--revision-only")) {
    process.stdout.write(`${await configuredDatasetRevision()}\n`);
    return;
  }
  const result = await buildSite();
  process.stdout.write(
    `Staged ${result.point_count.toLocaleString("en-US")} publications from Hugging Face revision ${result.dataset_revision}\n`,
  );
  process.stdout.write(
    `Artifact: ${result.artifact_path} (${result.artifact_bytes.toLocaleString("en-US")} bytes, sha256 ${result.artifact_sha256})\n`,
  );
  for (const line of formatLabelCalibrationReport(
    result.label_calibration,
    result.label_calibration_path,
  )) {
    process.stdout.write(`${line}\n`);
  }
}

if (require.main === module) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
  });
}

module.exports = {
  buildSite,
  configuredDatasetRevision,
  datasetRevisionUrl,
  formatLabelCalibrationReport,
  labelCalibrationReport,
  resolveDatasetRevision,
  stampAssetUrls,
};
