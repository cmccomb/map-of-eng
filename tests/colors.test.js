"use strict";

const assert = require("node:assert/strict");

require("../assets/colors.js");

const { generatePerceptualPalette } = globalThis.ResearchMapColors;

for (const count of [0, 1, 42, 312, 512]) {
  const palette = generatePerceptualPalette(count);
  assert.equal(palette.length, count);
  assert.equal(new Set(palette).size, count);
  assert.deepEqual(palette, generatePerceptualPalette(count));
  assert.ok(palette.every((color) => /^#[0-9a-f]{6}$/.test(color)));
}

const nestedPalette = generatePerceptualPalette(313);
assert.deepEqual(
  nestedPalette.slice(0, 312),
  generatePerceptualPalette(312),
);
