"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const core = require("../assets/map-core.js");
const { makeArtifact } = require("./fixtures/artifact.js");

test("configuration parsing builds an encoded Hugging Face artifact URL", () => {
  const config = core.parseConfig({
    title: "Research",
    heading: "A map",
    dataset_id: "owner/data.set",
    dataset_revision: "feature/layout v4",
    artifact_path: "maps/publications.json",
  });
  assert.equal(
    core.artifactUrl(config),
    "https://huggingface.co/datasets/owner/data.set/resolve/feature%2Flayout%20v4/maps/publications.json",
  );
});

test("configuration parsing accepts a safe same-origin artifact path", () => {
  const config = core.parseConfig({
    title: "Research",
    default_layout_id: "tsne",
    artifact_url: "data/publications.0123456789abcdef.json",
    artifact_bytes: 123456,
  });
  assert.equal(
    core.artifactUrl(config),
    "data/publications.0123456789abcdef.json",
  );
  assert.equal(config.artifact_bytes, 123456);
  assert.equal(config.default_layout_id, "tsne");
});

test("configuration parsing rejects unsafe URLs and paths", () => {
  assert.throws(
    () => core.parseConfig({ title: "Map", artifact_url: "javascript:alert(1)" }),
    core.ArtifactError,
  );
  assert.throws(
    () =>
      core.parseConfig({
        title: "Map",
        dataset_id: "owner/data",
        artifact_path: "../secret.json",
      }),
    core.ArtifactError,
  );
  assert.throws(
    () =>
      core.parseConfig({
        title: "Map",
        artifact_url: "data/publications.json",
        artifact_bytes: -1,
      }),
    core.ArtifactError,
  );
  assert.throws(
    () =>
      core.parseConfig({
        title: "Map",
        artifact_url: "data/publications.json",
        default_layout_id: "../tsne",
      }),
    core.ArtifactError,
  );
});

test("artifact parser accepts additive fields and sanitizes publication links", () => {
  const source = makeArtifact();
  source.points[0].source_url = "javascript:alert(1)";
  source.points[0].future_field = { retained: true };
  const artifact = core.parseArtifact(source);
  assert.equal(artifact.points.length, 8);
  assert.equal(artifact.points[0].source_url, "");
  assert.equal(artifact.points[0].keyword_id, "keyword-1");
  assert.equal(artifact.keywords[0].label, "robotic design");
  assert.deepEqual(artifact.keywords[0].coordinates.tsne, {
    x: 0.1,
    y: -0.05,
  });
  assert.deepEqual(artifact.points[0].future_field, { retained: true });
  assert.equal(artifact.additive_metadata.accepted, true);
});

test("bad rows are omitted without allowing them to reserve a work id", () => {
  const source = makeArtifact();
  const validCopy = { ...source.points[0] };
  source.points[0].pca_x = Number.NaN;
  source.points.splice(1, 0, validCopy);
  source.point_count = source.points.length;
  const artifact = core.parseArtifact(source);
  assert.equal(artifact.points.length, 8);
  assert.equal(artifact.omitted_point_count, 1);
  assert.equal(artifact.points.filter((point) => point.work_id === "p1").length, 1);
});

test("structural catalog and layout defects remain fatal", () => {
  const duplicateFaculty = makeArtifact();
  duplicateFaculty.catalogs.faculty.push({
    ...duplicateFaculty.catalogs.faculty[0],
  });
  assert.throws(() => core.parseArtifact(duplicateFaculty), /Duplicate faculty id/);

  const missingLayout = makeArtifact();
  missingLayout.layouts = missingLayout.layouts.slice(0, 1);
  assert.throws(() => core.parseArtifact(missingLayout), /At least two map layouts/);
});

test("schema-six keyword metadata is required and validated", () => {
  const cases = [
    [
      (artifact) => {
        delete artifact.keywords;
      },
      /keywords must be an array/,
    ],
    [(artifact) => (artifact.keywords = []), /keywords must not be empty/],
    [(artifact) => (artifact.keywords[0] = null), /must be an object/],
    [
      (artifact) =>
        artifact.keywords.push({ ...artifact.keywords[0] }),
      /Duplicate keyword id/,
    ],
    [
      (artifact) => (artifact.keywords[0].publication_count = 0),
      /publication_count is invalid/,
    ],
    [
      (artifact) => (artifact.keywords[0].coordinates.tsne.x = "bad"),
      /invalid tsne coordinates/,
    ],
    [(artifact) => (artifact.keywords[0].label = ""), /label must be/],
  ];
  for (const [mutate, pattern] of cases) {
    const artifact = makeArtifact();
    mutate(artifact);
    assert.throws(() => core.parseArtifact(artifact), pattern);
  }
});

test("points with missing or unknown keyword ids are omitted safely", () => {
  for (const keywordId of ["", "unknown-keyword"]) {
    const artifact = makeArtifact();
    artifact.points[0].keyword_id = keywordId;
    const parsed = core.parseArtifact(artifact);
    assert.equal(parsed.points.length, 7);
    assert.equal(parsed.omitted_point_count, 1);
  }
});

test("older additive artifacts remain readable without keyword metadata", () => {
  const artifact = makeArtifact();
  artifact.schema_version = 5;
  delete artifact.keywords;
  artifact.points.forEach((point) => delete point.keyword_id);

  const parsed = core.parseArtifact(artifact);

  assert.deepEqual(parsed.keywords, []);
  assert.equal(parsed.points.length, 8);
});

test("schema-six artifacts may represent an empty publication corpus", () => {
  const artifact = makeArtifact();
  artifact.points = [];
  artifact.point_count = 0;
  artifact.keywords = [];

  const parsed = core.parseArtifact(artifact);

  assert.deepEqual(parsed.keywords, []);
  assert.deepEqual(parsed.points, []);
});

test("filter matching is OR within dimensions and AND across dimensions", () => {
  const artifact = core.parseArtifact(makeArtifact());
  const robotIds = artifact.points
    .filter((point) =>
      core.pointMatches(point, { titleQueries: ["robot", "battery"] }),
    )
    .map((point) => point.work_id);
  assert.deepEqual(robotIds, ["p1", "p2", "p3"]);

  const crossDimensionIds = artifact.points
    .filter((point) =>
      core.pointMatches(point, {
        departmentIds: new Set(["d-ece"]),
        facultyIds: new Set(["f-alice", "f-bob"]),
      }),
    )
    .map((point) => point.work_id);
  assert.deepEqual(crossDimensionIds, ["p3", "p6"]);
});

test("fit calculations center finite points and reject bad dimensions", () => {
  const view = core.fitView(
    [
      { x: -1, y: -0.5 },
      { x: 1, y: 0.5 },
    ],
    1000,
    600,
  );
  assert.ok(view.scale >= 0.7 && view.scale <= 25);
  assert.equal(Math.abs(view.offsetX), 0);
  assert.equal(Math.abs(view.offsetY), 0);
  assert.throws(() => core.fitView([{ x: 0, y: 0 }], 0, 600), TypeError);
});

test("dates are rendered in UTC and invalid values stay blank", () => {
  assert.match(core.formatUtcDate("2026-07-17T00:00:00Z", "en-US"), /Jul 17, 2026/);
  assert.equal(core.formatUtcDate("not-a-date", "en-US"), "");
});
