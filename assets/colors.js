"use strict";

(() => {
  const MIN_CONTRAST = 3.2;
  const THEMES = Object.freeze({
    dark: Object.freeze({
      backgroundRgb: [7 / 255, 16 / 255, 25 / 255],
      lightnesses: [0.62, 0.67, 0.72, 0.77, 0.82, 0.87],
    }),
    light: Object.freeze({
      backgroundRgb: [237 / 255, 242 / 255, 244 / 255],
      lightnesses: [0.34, 0.39, 0.44, 0.49, 0.54, 0.59],
    }),
  });

  function linearToSrgb(value) {
    return value <= 0.0031308
      ? 12.92 * value
      : 1.055 * value ** (1 / 2.4) - 0.055;
  }

  function srgbToLinear(value) {
    return value <= 0.04045
      ? value / 12.92
      : ((value + 0.055) / 1.055) ** 2.4;
  }

  function oklabToLinearRgb([lightness, a, b]) {
    const lRoot = lightness + 0.3963377774 * a + 0.2158037573 * b;
    const mRoot = lightness - 0.1055613458 * a - 0.0638541728 * b;
    const sRoot = lightness - 0.0894841775 * a - 1.291485548 * b;
    const l = lRoot ** 3;
    const m = mRoot ** 3;
    const s = sRoot ** 3;
    return [
      4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
      -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
      -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s,
    ];
  }

  function linearRgbToOklab([red, green, blue]) {
    const l = 0.4122214708 * red + 0.5363325363 * green + 0.0514459929 * blue;
    const m = 0.2119034982 * red + 0.6806995451 * green + 0.1073969566 * blue;
    const s = 0.0883024619 * red + 0.2817188376 * green + 0.6299787005 * blue;
    const lRoot = Math.cbrt(l);
    const mRoot = Math.cbrt(m);
    const sRoot = Math.cbrt(s);
    return [
      0.2104542553 * lRoot + 0.793617785 * mRoot - 0.0040720468 * sRoot,
      1.9779984951 * lRoot - 2.428592205 * mRoot + 0.4505937099 * sRoot,
      0.0259040371 * lRoot + 0.7827717662 * mRoot - 0.808675766 * sRoot,
    ];
  }

  function relativeLuminance(rgb) {
    const [red, green, blue] = rgb.map(srgbToLinear);
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue;
  }

  function contrastRatio(left, right) {
    const bright = Math.max(relativeLuminance(left), relativeLuminance(right));
    const dark = Math.min(relativeLuminance(left), relativeLuminance(right));
    return (bright + 0.05) / (dark + 0.05);
  }

  function squaredDistance(left, right) {
    return (
      (left[0] - right[0]) ** 2 +
      (left[1] - right[1]) ** 2 +
      (left[2] - right[2]) ** 2
    );
  }

  function rgbToHex(rgb) {
    return `#${rgb
      .map((value) => Math.round(value * 255).toString(16).padStart(2, "0"))
      .join("")}`;
  }

  function buildCandidates({ backgroundRgb, lightnesses }) {
    const candidatesByHex = new Map();
    const chromas = [0.07, 0.1, 0.13, 0.16, 0.19, 0.22];

    for (const lightness of lightnesses) {
      for (const chroma of chromas) {
        for (let hue = 0; hue < 360; hue += 3) {
          const radians = (hue * Math.PI) / 180;
          const lab = [
            lightness,
            chroma * Math.cos(radians),
            chroma * Math.sin(radians),
          ];
          const linearRgb = oklabToLinearRgb(lab);
          if (linearRgb.some((value) => value < 0 || value > 1)) continue;
          const rgb = linearRgb.map(linearToSrgb);
          if (contrastRatio(rgb, backgroundRgb) < MIN_CONTRAST) continue;
          const hex = rgbToHex(rgb);
          if (!candidatesByHex.has(hex)) {
            candidatesByHex.set(hex, { hex, lab });
          }
        }
      }
    }
    return [...candidatesByHex.values()];
  }

  const themeData = new Map(
    Object.entries(THEMES).map(([theme, configuration]) => [
      theme,
      {
        candidates: buildCandidates(configuration),
        backgroundLab: linearRgbToOklab(
          configuration.backgroundRgb.map(srgbToLinear),
        ),
        paletteCache: new Map([[0, Object.freeze([])]]),
      },
    ]),
  );

  function generatePerceptualPalette(requestedCount, requestedTheme = "dark") {
    const count = Math.max(0, Math.floor(Number(requestedCount) || 0));
    const theme = themeData.has(requestedTheme) ? requestedTheme : "dark";
    const { candidates, backgroundLab, paletteCache } = themeData.get(theme);
    if (paletteCache.has(count)) return [...paletteCache.get(count)];
    if (count > candidates.length) {
      throw new RangeError(
        `Cannot generate ${count} distinct ${theme} colors from ${candidates.length} displayable candidates`,
      );
    }

    const minimumDistances = candidates.map((candidate) =>
      squaredDistance(candidate.lab, backgroundLab),
    );
    const palette = [];
    for (let selection = 0; selection < count; selection += 1) {
      let nextIndex = 0;
      for (let index = 1; index < minimumDistances.length; index += 1) {
        if (minimumDistances[index] > minimumDistances[nextIndex]) {
          nextIndex = index;
        }
      }
      const chosen = candidates[nextIndex];
      palette.push(chosen.hex);
      minimumDistances[nextIndex] = -1;
      for (let index = 0; index < candidates.length; index += 1) {
        if (minimumDistances[index] < 0) continue;
        minimumDistances[index] = Math.min(
          minimumDistances[index],
          squaredDistance(candidates[index].lab, chosen.lab),
        );
      }
    }

    paletteCache.set(count, Object.freeze([...palette]));
    return palette;
  }

  globalThis.ResearchMapColors = Object.freeze({
    generatePerceptualPalette,
  });
})();
