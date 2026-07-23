"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");
const core = require("../assets/map-core.js");
const {
  buildSite,
  configuredDatasetRevision,
  datasetRevisionUrl,
  formatLabelCalibrationReport,
  labelCalibrationReport,
} = require("../scripts/build-site.js");
const { makeArtifact } = require("./fixtures/artifact.js");

const REVISION = "0123456789abcdef0123456789abcdef01234567";

function jsonResponse(value) {
  const body = JSON.stringify(value);
  return {
    ok: true,
    status: 200,
    headers: new Headers({ "content-length": String(Buffer.byteLength(body)) }),
    async text() {
      return body;
    },
  };
}

test("site build pins, validates, and stages the map artifact locally", async (t) => {
  const temporaryRoot = await fs.mkdtemp(path.join(os.tmpdir(), "map-site-test-"));
  t.after(() => fs.rm(temporaryRoot, { recursive: true, force: true }));
  const outputDirectory = path.join(temporaryRoot, "site");
  const requestedUrls = [];
  const artifact = makeArtifact();
  const fetchImpl = async (url) => {
    requestedUrls.push(url);
    if (url.includes("/api/datasets/")) return jsonResponse({ sha: REVISION });
    if (url.includes(`/resolve/${REVISION}/maps/publications.json`)) {
      return jsonResponse(artifact);
    }
    return { ok: false, status: 404, headers: new Headers() };
  };

  const result = await buildSite({
    sourceDirectory: path.resolve(__dirname, ".."),
    outputDirectory,
    fetchImpl,
    assetVersion: "fedcba9876543210",
  });

  assert.equal(result.dataset_revision, REVISION);
  assert.equal(result.point_count, artifact.point_count);
  assert.equal(result.artifact_bytes, Buffer.byteLength(JSON.stringify(artifact)));
  assert.deepEqual(requestedUrls, [
    datasetRevisionUrl({
      dataset_id: "ccm/cmu-engineering-publications",
      dataset_revision: "main",
    }),
    `https://huggingface.co/datasets/ccm/cmu-engineering-publications/resolve/${REVISION}/maps/publications.json`,
  ]);
  const deployedConfig = JSON.parse(
    await fs.readFile(path.join(outputDirectory, "map-config.json"), "utf8"),
  );
  assert.equal(deployedConfig.dataset_revision, REVISION);
  assert.equal(deployedConfig.default_layout_id, "tsne");
  assert.equal(
    deployedConfig.artifact_url,
    `data/publications.${REVISION}.json`,
  );
  assert.match(deployedConfig.artifact_sha256, /^[0-9a-f]{64}$/);
  assert.equal(deployedConfig.artifact_bytes, result.artifact_bytes);
  assert.equal(
    deployedConfig.label_calibration_url,
    `data/label-calibration.${REVISION}.json`,
  );
  const deployedArtifact = JSON.parse(
    await fs.readFile(
      path.join(outputDirectory, deployedConfig.artifact_url),
      "utf8",
    ),
  );
  assert.equal(core.parseArtifact(deployedArtifact).points.length, artifact.point_count);
  const deployedHtml = await fs.readFile(
    path.join(outputDirectory, "index.html"),
    "utf8",
  );
  assert.match(deployedHtml, /assets\/theme\.js\?v=fedcba987654/);
  assert.match(deployedHtml, /assets\/map\.js\?v=fedcba987654/);
  const deployedCalibration = JSON.parse(
    await fs.readFile(
      path.join(outputDirectory, deployedConfig.label_calibration_url),
      "utf8",
    ),
  );
  assert.deepEqual(deployedCalibration, result.label_calibration);
  assert.equal(deployedCalibration.layout_id, "tsne");
  assert.equal(deployedCalibration.point_count, artifact.point_count);
  assert.ok(
    deployedCalibration.overview.promoted_count <=
      deployedCalibration.overview.promotion_limit,
  );
  assert.ok(
    deployedCalibration.active_faculty_views.maximum_promotions <=
      deployedCalibration.active_faculty_views.promotion_limit,
  );
});

test("site build refuses mutable or malformed revision metadata", async () => {
  await assert.rejects(
    buildSite({
      sourceDirectory: path.resolve(__dirname, ".."),
      outputDirectory: path.join(os.tmpdir(), "map-site-invalid-revision"),
      fetchImpl: async () => jsonResponse({ sha: "main" }),
    }),
    /immutable dataset revision/,
  );
});

test("revision check resolves metadata without downloading the artifact", async () => {
  const requestedUrls = [];
  const revision = await configuredDatasetRevision({
    sourceDirectory: path.resolve(__dirname, ".."),
    fetchImpl: async (url) => {
      requestedUrls.push(url);
      return jsonResponse({ sha: REVISION });
    },
  });
  assert.equal(revision, REVISION);
  assert.equal(requestedUrls.length, 1);
  assert.match(requestedUrls[0], /\/api\/datasets\//);
});

test("label calibration exposes bounded overview and active-view drift diagnostics", () => {
  const artifact = makeArtifact();
  artifact.catalogs.faculty.push({
    person_id: "f-idle",
    display_name: "Idle Faculty",
    publication_count: 0,
  });
  const report = labelCalibrationReport(core.parseArtifact(artifact));
  assert.deepEqual(
    {
      broad_topic_count: report.broad_topic_count,
      detailed_topic_count: report.detailed_topic_count,
      layout_id: report.layout_id,
      point_count: report.point_count,
      schema_version: report.schema_version,
    },
    {
      broad_topic_count: 3,
      detailed_topic_count: 6,
      layout_id: "tsne",
      point_count: 8,
      schema_version: 1,
    },
  );
  assert.deepEqual(report.overview, {
    promotion_limit: 3,
    minimum_support: 50,
    promoted_count: 0,
    promoted_topics: [],
    next_supported_candidate: null,
  });
  assert.deepEqual(report.active_faculty_views, {
    promotion_limit: 4,
    view_count: 4,
    zero_promotion_views: 4,
    median_promotions: 0,
    p90_promotions: 0,
    maximum_promotions: 0,
    views_at_limit: 0,
  });
  assert.deepEqual(
    formatLabelCalibrationReport(report, "data/calibration.json"),
    [
      "Labels: 0/3 overview promotions on tsne: none.",
      "Active-label drift: 4 faculty views; median 0, p90 0, max 0/4; 0 at the limit.",
      "Calibration report: data/calibration.json",
    ],
  );
});
