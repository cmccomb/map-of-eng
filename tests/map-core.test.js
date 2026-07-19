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

test("configuration parsing supports absolute artifacts and dataset defaults", () => {
  const direct = core.parseConfig({
    title: " Research ",
    heading: " Map ",
    artifact_url: "https://example.test/map data.json",
  });
  assert.deepEqual(direct, {
    title: "Research",
    heading: "Map",
    default_layout_id: "",
    artifact_url: "https://example.test/map%20data.json",
    artifact_bytes: 0,
  });
  assert.equal(core.artifactUrl(direct), direct.artifact_url);

  const defaults = core.parseConfig({ title: "Research", dataset_id: "cmu/map" });
  assert.equal(defaults.dataset_revision, "main");
  assert.equal(defaults.artifact_path, "maps/publications.json");
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

  const invalidConfigurations = [
    null,
    { title: "", dataset_id: "owner/data" },
    { title: "x".repeat(201), dataset_id: "owner/data" },
    { title: "Map", dataset_id: "not-a-dataset" },
    { title: "Map", dataset_id: "owner/data", artifact_path: "/map.json" },
    { title: "Map", dataset_id: "owner/data", artifact_path: "maps\\map.json" },
    { title: "Map", dataset_id: "owner/data", artifact_path: "maps//map.json" },
  ];
  for (const invalid of invalidConfigurations) {
    assert.throws(() => core.parseConfig(invalid), core.ArtifactError);
  }
});

test("text and URL helpers normalize untrusted values", () => {
  assert.equal(core.normalizedText("  CAFE\u0301  "), "café");
  assert.equal(core.normalizedText(null), "");
  assert.equal(
    core.safeHttpUrl("https://example.test/a b"),
    "https://example.test/a%20b",
  );
  assert.equal(core.safeHttpUrl("mailto:test@example.test"), "");
  assert.equal(core.safeHttpUrl("not a URL"), "");
  assert.equal(core.safeHttpUrl(null), "");
});

test("artifact parser accepts additive fields and sanitizes publication links", () => {
  const source = makeArtifact();
  source.points[0].source_url = "javascript:alert(1)";
  source.points[0].future_field = { retained: true };
  const artifact = core.parseArtifact(source);
  assert.equal(artifact.points.length, 8);
  assert.equal(artifact.points[0].source_url, "");
  assert.deepEqual(artifact.points[0].keyword_ids, ["keyword-1", "keyword-1-1"]);
  assert.equal(artifact.keywords[0].label, "robotic design");
  assert.equal(artifact.keyword_levels[1].label, "Detailed topics");
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

test("artifact parser rejects malformed top-level structure", () => {
  const cases = [
    [null, /Artifact must be an object/],
    [{}, /unsupported schema/],
    [{ schema_version: 4 }, /Artifact points/],
    [{ schema_version: 4, points: [] }, /catalogs are missing/],
  ];
  for (const [value, message] of cases) {
    assert.throws(() => core.parseArtifact(value), message);
  }
});

test("artifact parser validates every layout invariant", () => {
  const mutations = [
    [
      (artifact) => artifact.layouts.push(...Array(7).fill(artifact.layouts[0])),
      /At most 8/,
    ],
    [
      (artifact) => {
        artifact.layouts[0] = null;
      },
      /layouts\[0\] must be an object/,
    ],
    [
      (artifact) => {
        artifact.layouts[1].layout_id = "pca";
      },
      /Duplicate layout id/,
    ],
    [
      (artifact) => {
        artifact.layouts[0].x_field = "not-valid!";
      },
      /invalid coordinate fields/,
    ],
    [
      (artifact) => {
        artifact.layouts[0].y_field = "pca_x";
      },
      /invalid coordinate fields/,
    ],
    [
      (artifact) => {
        artifact.default_layout_id = "missing";
      },
      /default layout/,
    ],
  ];
  for (const [mutate, message] of mutations) {
    const artifact = makeArtifact();
    mutate(artifact);
    assert.throws(() => core.parseArtifact(artifact), message);
  }
});

test("artifact parser validates catalog invariants", () => {
  const mutations = [
    [
      (artifact) => {
        artifact.catalogs.departments = [];
      },
      /catalog must not be empty/,
    ],
    [
      (artifact) => {
        artifact.catalogs.departments[0] = null;
      },
      /must be an object/,
    ],
    [
      (artifact) => {
        artifact.catalogs.departments[0].publication_count = -1;
      },
      /publication_count is invalid/,
    ],
    [
      (artifact) => {
        artifact.catalogs.faculty[0].display_name = "";
      },
      /must be a non-empty string/,
    ],
  ];
  for (const [mutate, message] of mutations) {
    const artifact = makeArtifact();
    mutate(artifact);
    assert.throws(() => core.parseArtifact(artifact), message);
  }
});

function parseWithInvalidPoint(mutator) {
  const artifact = makeArtifact();
  mutator(artifact.points[0], artifact);
  return core.parseArtifact(artifact);
}

test("artifact parser omits each class of invalid publication row", () => {
  const mutations = [
    (point, artifact) => {
      artifact.points[0] = null;
    },
    (point) => {
      point.work_id = "";
    },
    (point) => {
      point.pca_x = "0";
    },
    (point) => {
      point.title = "";
    },
    (point) => {
      point.department_ids = "d-me";
    },
    (point) => {
      point.department_ids = [];
    },
    (point) => {
      point.department_ids = ["unknown"];
    },
    (point) => {
      point.department_ids = ["d-me", "d-me"];
    },
    (point) => {
      point.faculty_ids = ["unknown"];
    },
  ];
  for (const mutate of mutations) {
    const parsed = parseWithInvalidPoint(mutate);
    assert.equal(parsed.omitted_point_count, 1);
    assert.equal(parsed.points.length, 7);
    assert.equal(parsed.warnings.length, 1);
  }
});

test("artifact parser records duplicate rows and metadata disagreements", () => {
  const source = makeArtifact();
  source.points.push({ ...source.points[0] });
  source.point_count = 99;
  const artifact = core.parseArtifact(source);
  assert.equal(artifact.omitted_point_count, 1);
  assert.equal(artifact.warnings.length, 2);
  assert.match(artifact.warnings[0], /does not match/);
  assert.match([...artifact.omission_reasons.keys()][0], /duplicates work_id/);
});

test("artifact parser bounds oversized inputs before parsing them", () => {
  const source = makeArtifact();
  source.points = { length: 500001 };
  assert.throws(() => core.parseArtifact(source), /at most 500000/);
});

test("publication defaults are bounded and long optional text is truncated", () => {
  const source = makeArtifact();
  Object.assign(source.points[0], {
    authors: "a".repeat(10001),
    year: 9,
    citation_count: -4,
    observation_count: 0,
    venue: null,
    doi: null,
  });
  const point = core.parseArtifact(source).points[0];
  assert.equal(point.authors.length, 10000);
  assert.equal(point.year, null);
  assert.equal(point.citation_count, 0);
  assert.equal(point.observation_count, 1);
  assert.equal(point.venue, "");
  assert.equal(point.doi, "");
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
      (artifact) => (artifact.keywords[0].coordinates = null),
      /coordinates is invalid/,
    ],
    [
      (artifact) => (artifact.keywords[0].coordinates.tsne.x = "bad"),
      /invalid tsne coordinates/,
    ],
    [(artifact) => (artifact.keywords[0].label = ""), /label must be/],
    [(artifact) => (artifact.keywords[0].level = -1), /level is invalid/],
    [
      (artifact) => (artifact.keywords[0].parent_keyword_id = "keyword-2"),
      /must not have a parent/,
    ],
    [
      (artifact) => (artifact.keywords[3].parent_keyword_id = "unknown"),
      /has an invalid parent/,
    ],
    [
      (artifact) => {
        delete artifact.keyword_levels;
      },
      /keyword_levels must be an array/,
    ],
    [(artifact) => (artifact.keyword_levels = []), /must not be empty/],
    [
      (artifact) => (artifact.keyword_levels[1].level = 3),
      /keyword_levels\[1\] is invalid/,
    ],
    [
      (artifact) => (artifact.keyword_levels[1].keyword_count = 5),
      /keyword_count is invalid/,
    ],
  ];
  for (const [mutate, pattern] of cases) {
    const artifact = makeArtifact();
    mutate(artifact);
    assert.throws(() => core.parseArtifact(artifact), pattern);
  }
});

test("points with malformed keyword hierarchies are omitted safely", () => {
  const invalidKeywordIds = [
    undefined,
    [],
    ["keyword-1"],
    ["unknown-keyword", "keyword-1-1"],
    ["keyword-1", "keyword-1"],
    ["keyword-1", "keyword-2"],
    ["keyword-2", "keyword-1-1"],
  ];
  for (const keywordIds of invalidKeywordIds) {
    const artifact = makeArtifact();
    artifact.points[0].keyword_ids = keywordIds;
    const parsed = core.parseArtifact(artifact);
    assert.equal(parsed.points.length, 7);
    assert.equal(parsed.omitted_point_count, 1);
  }
});

test("older additive artifacts remain readable without keyword metadata", () => {
  const artifact = makeArtifact();
  artifact.schema_version = 5;
  delete artifact.keywords;
  delete artifact.keyword_levels;
  artifact.points.forEach((point) => delete point.keyword_ids);

  const parsed = core.parseArtifact(artifact);

  assert.deepEqual(parsed.keywords, []);
  assert.deepEqual(parsed.keyword_levels, []);
  assert.equal(parsed.points.length, 8);
});

test("schema-five flat keywords are upgraded to an overview hierarchy", () => {
  const artifact = makeArtifact();
  artifact.schema_version = 5;
  delete artifact.keyword_levels;
  artifact.keywords.forEach((keyword) => {
    delete keyword.level;
    delete keyword.parent_keyword_id;
  });
  artifact.points.forEach((point) => {
    point.keyword_id = point.keyword_ids[0];
    delete point.keyword_ids;
  });

  const parsed = core.parseArtifact(artifact);

  assert.equal(parsed.keyword_levels.length, 1);
  assert.equal(parsed.keyword_levels[0].keyword_count, artifact.keywords.length);
  assert.ok(parsed.keywords.every((keyword) => keyword.level === 0));
  assert.deepEqual(parsed.points[0].keyword_ids, ["keyword-1"]);
});

test("schema-six artifacts may represent an empty publication corpus", () => {
  const artifact = makeArtifact();
  artifact.points = [];
  artifact.point_count = 0;
  artifact.keywords = [];
  artifact.keyword_levels.forEach((level) => (level.keyword_count = 0));

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

  const point = artifact.points[0];
  assert.equal(core.pointMatches(point), true);
  assert.equal(core.pointMatches(point, { titleQueries: ["missing"] }), false);
  assert.equal(
    core.pointMatches(point, { departmentIds: new Set(["d-ece"]) }),
    false,
  );
  assert.equal(
    core.pointMatches(point, { facultyIds: new Set(["f-bob"]) }),
    false,
  );
});

test("preferred IDs prioritize active, available choices", () => {
  const ids = ["first", "second", "third"];
  assert.equal(
    core.preferredId(ids, new Set(["second"]), new Set(["first", "second"])),
    "second",
  );
  assert.equal(core.preferredId(ids, new Set(), new Set(["third"])), "third");
  assert.equal(core.preferredId(ids, new Set(), new Set()), "");
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
  assert.equal(core.fitView([], 100, 100), null);
  assert.equal(core.fitView([{ x: Number.NaN, y: 0 }], 100, 100), null);
  assert.equal(core.baseScale(10, 10), 1);
});

test("dates are rendered in UTC and invalid values stay blank", () => {
  assert.match(core.formatUtcDate("2026-07-17T00:00:00Z", "en-US"), /Jul 17, 2026/);
  assert.equal(core.formatUtcDate("not-a-date", "en-US"), "");
  assert.equal(core.formatUtcDate(null, "en-US"), "");
});
