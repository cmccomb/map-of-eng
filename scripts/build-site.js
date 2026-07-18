"use strict";

const crypto = require("node:crypto");
const fs = require("node:fs/promises");
const path = require("node:path");
const core = require("../assets/map-core.js");

const DEFAULT_OUTPUT_DIRECTORY = "_site";
const MAX_API_BYTES = 1024 * 1024;
const MAX_ARTIFACT_BYTES = 64 * 1024 * 1024;
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
  const artifactName = `publications.${revision}.json`;
  const deployedConfig = {
    ...sourceConfig,
    dataset_revision: revision,
    artifact_url: `data/${artifactName}`,
    artifact_sha256: artifactDigest,
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
  ]);

  return {
    artifact_bytes: Buffer.byteLength(artifactDocument.text, "utf8"),
    artifact_path: deployedConfig.artifact_url,
    artifact_sha256: artifactDigest,
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
  resolveDatasetRevision,
  stampAssetUrls,
};
