"use strict";

const assert = require("node:assert/strict");

require("../assets/colors.js");

const { generatePerceptualPalette, generateSequentialPalette } =
  globalThis.ResearchMapColors;

function relativeLuminance(hex) {
  const channels = hex
    .match(/[0-9a-f]{2}/gi)
    .map((channel) => Number.parseInt(channel, 16) / 255)
    .map((value) =>
      value <= 0.04045
        ? value / 12.92
        : ((value + 0.055) / 1.055) ** 2.4,
    );
  return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2];
}

function contrastRatio(left, right) {
  const bright = Math.max(relativeLuminance(left), relativeLuminance(right));
  const dark = Math.min(relativeLuminance(left), relativeLuminance(right));
  return (bright + 0.05) / (dark + 0.05);
}

const backgrounds = { dark: "#071019", light: "#edf2f4" };

for (const theme of ["dark", "light"]) {
  for (const count of [0, 1, 42, 312, 512]) {
    const palette = generatePerceptualPalette(count, theme);
    assert.equal(palette.length, count);
    assert.equal(new Set(palette).size, count);
    assert.deepEqual(palette, generatePerceptualPalette(count, theme));
    assert.ok(palette.every((color) => /^#[0-9a-f]{6}$/.test(color)));
    assert.ok(
      palette.every((color) => contrastRatio(color, backgrounds[theme]) >= 3.1),
    );
  }

  const nestedPalette = generatePerceptualPalette(313, theme);
  assert.deepEqual(
    nestedPalette.slice(0, 312),
    generatePerceptualPalette(312, theme),
  );
}

assert.notDeepEqual(
  generatePerceptualPalette(42, "light"),
  generatePerceptualPalette(42, "dark"),
);

for (const theme of ["dark", "light"]) {
  for (const mode of ["year", "citations"]) {
    const palette = generateSequentialPalette(48, theme, mode);
    assert.equal(palette.length, 48);
    assert.equal(new Set(palette).size, 48);
    assert.deepEqual(palette, generateSequentialPalette(48, theme, mode));
    assert.ok(palette.every((color) => /^#[0-9a-f]{6}$/.test(color)));
    assert.ok(
      palette.every((color) => contrastRatio(color, backgrounds[theme]) >= 3.1),
    );
    assert.notEqual(palette[0], palette.at(-1));
  }
}

assert.notDeepEqual(
  generateSequentialPalette(48, "light", "year"),
  generateSequentialPalette(48, "dark", "year"),
);
assert.notDeepEqual(
  generateSequentialPalette(48, "dark", "year"),
  generateSequentialPalette(48, "dark", "citations"),
);
