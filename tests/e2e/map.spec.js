"use strict";

const { test, expect } = require("@playwright/test");
const AxeBuilder = require("@axe-core/playwright").default;
const { makeArtifact, makeLargeArtifact } = require("../fixtures/artifact.js");

async function serveArtifact(page, artifact = makeArtifact()) {
  await page.route("https://huggingface.co/**", async (route) => {
    await route.fulfill({ json: artifact });
  });
}

async function openMap(page, artifact = makeArtifact()) {
  await serveArtifact(page, artifact);
  await page.goto("/");
  await expect(page.locator(".filter-panel")).toHaveAttribute(
    "aria-busy",
    "false",
  );
  await expect(page.locator("#map-status")).toContainText("publications");
  await expect(page.locator("#loading-overlay")).toBeHidden();
}

async function expectPublicationStatus(page, count) {
  await expect(page.locator("#map-status")).toContainText(count);
}

async function addTitleTerm(page, term) {
  const input = page.locator("#title-search");
  await input.fill(term);
  await input.press("Enter");
}

async function chooseComboOption(page, inputId, query) {
  const input = page.locator(inputId);
  await input.fill(query);
  await expect(input).toHaveAttribute("aria-expanded", "true");
  await input.press("Enter");
}

test("loads cleanly with useful defaults and a complete department key", async ({
  page,
}) => {
  const consoleErrors = [];
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  await openMap(page);

  await expectPublicationStatus(page, "8 publications");
  await expect(page.locator("#map-status")).toContainText(
    "newest Scholar profile refresh Jul 17, 2026",
  );
  await expect(page.locator("#clear-filters")).toBeDisabled();
  await expect(page.locator("#zoom-results")).toBeDisabled();
  await expect(
    page.getByLabel("Local neighborhoods", { exact: true }),
  ).toBeChecked();
  await expect(page.locator("#layout-note")).toContainText("t-SNE");
  await expect(page.locator("#legend-summary")).toHaveText(
    "3 departments represented",
  );
  await expect(page.locator("#map-legend .color-key-item")).toHaveCount(3);
  await expect(page.locator("#map-legend .color-key-item > span")).toHaveText([
    "Biomedical Engineering",
    "Electrical and Computer Engineering",
    "Mechanical Engineering",
  ]);
  await expect(page.locator("#research-map")).toHaveAttribute(
    "aria-label",
    /showing 8 publications/,
  );
  await expect(page.locator("#match-count, footer")).toHaveCount(0);
  const contentOrder = await page
    .locator(".filter-panel")
    .evaluate((panel) =>
      [
        "#map-title",
        "#map-status",
        ".appearance-fieldset",
        ".panel-heading",
      ].map((selector) => panel.querySelector(selector).offsetTop),
    );
  expect(contentOrder).toEqual(
    [...contentOrder].sort((left, right) => left - right),
  );
  expect(consoleErrors).toEqual([]);
});

test("reports determinate progress while publication data streams", async ({
  page,
}) => {
  await page.addInitScript((artifact) => {
    const originalFetch = window.fetch.bind(window);
    window.fetch = async (input, init) => {
      const url = input instanceof Request ? input.url : String(input);
      if (!url.startsWith("https://huggingface.co/")) {
        return originalFetch(input, init);
      }

      const bytes = new TextEncoder().encode(JSON.stringify(artifact));
      const chunkSize = Math.ceil(bytes.length / 10);
      let offset = 0;
      const body = new ReadableStream({
        async pull(controller) {
          await new Promise((resolve) => window.setTimeout(resolve, 120));
          if (offset >= bytes.length) {
            controller.close();
            return;
          }
          const nextOffset = Math.min(bytes.length, offset + chunkSize);
          controller.enqueue(bytes.slice(offset, nextOffset));
          offset = nextOffset;
        },
      });
      return new Response(body, {
        headers: {
          "content-length": String(bytes.length),
          "content-type": "application/json",
        },
      });
    };
  }, makeArtifact());

  await page.goto("/");
  const overlay = page.locator("#loading-overlay");
  const progress = page.locator("#loading-progress");
  const percent = page.locator("#loading-percent");
  await expect(overlay).toBeVisible();
  await expect
    .poll(async () => {
      const value = Number((await percent.textContent()).replace("%", ""));
      return value > 0 && value < 100;
    })
    .toBe(true);
  await expect(progress).toHaveAttribute("aria-valuenow", /[1-9][0-9]?/);
  await expect(page.locator("#loading-detail")).toContainText(" of ");
  await expect(page.locator(".filter-panel")).toHaveAttribute(
    "aria-busy",
    "false",
  );
  await expect(overlay).toBeHidden();
  await expectPublicationStatus(page, "8 publications");
});

test("uses the map as an edge-to-edge backdrop for floating controls", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await openMap(page);

  const layout = await page.evaluate(() => {
    const bounds = (selector) => {
      const rectangle = document
        .querySelector(selector)
        .getBoundingClientRect();
      return {
        top: rectangle.top,
        right: rectangle.right,
        bottom: rectangle.bottom,
        left: rectangle.left,
        width: rectangle.width,
        height: rectangle.height,
      };
    };
    const mapColumn = document.querySelector(".map-column");
    const legend = document.querySelector("#map-legend");
    return {
      canvas: bounds("#research-map"),
      filters: bounds(".filter-panel"),
      legendPanel: bounds(".legend-panel"),
      legend: bounds("#map-legend"),
      legendColumns: getComputedStyle(legend).gridTemplateColumns,
      legendOverflow: getComputedStyle(legend).overflowY,
      mapBorderWidth: getComputedStyle(mapColumn).borderTopWidth,
      bodyOverflow: getComputedStyle(document.body).overflow,
    };
  });

  expect(layout.canvas).toMatchObject({
    top: 0,
    left: 0,
    width: 1440,
    height: 900,
  });
  expect(layout.mapBorderWidth).toBe("0px");
  expect(layout.bodyOverflow).toBe("hidden");
  expect(layout.filters.left).toBeGreaterThan(layout.canvas.left);
  expect(layout.filters.bottom).toBeLessThan(layout.canvas.bottom);
  expect(layout.legendPanel.top).toBeGreaterThan(layout.canvas.top);
  expect(layout.legendPanel.right).toBeLessThan(layout.canvas.right);
  expect(layout.legendPanel.left).toBeGreaterThan(layout.filters.right);
  expect(layout.legend.left).toBeGreaterThan(layout.legendPanel.left);
  expect(layout.legend.right).toBeLessThanOrEqual(layout.legendPanel.right);
  expect(layout.legendColumns.split(" ")).toHaveLength(1);
  expect(layout.legendOverflow).toBe("auto");
  await expect(page.locator(".map-controls")).toBeVisible();
});

test("appearance control persists light, follows system, and redraws dark", async ({
  page,
}) => {
  await page.emulateMedia({ colorScheme: "dark" });
  await openMap(page);
  const root = page.locator("html");
  const canvas = page.locator("#research-map");

  await expect(root).toHaveAttribute("data-theme", "system");
  await expect(root).toHaveAttribute("data-resolved-theme", "dark");
  await expect(page.getByLabel("System", { exact: true })).toBeChecked();
  const systemDarkImage = await canvas.screenshot();

  await page.getByText("Light", { exact: true }).click();
  await expect(root).toHaveAttribute("data-theme", "light");
  await expect(root).toHaveAttribute("data-resolved-theme", "light");
  await expect
    .poll(() =>
      page.evaluate(() => localStorage.getItem("cmu-research-map-theme")),
    )
    .toBe("light");
  const lightImage = await canvas.screenshot();
  expect(lightImage.equals(systemDarkImage)).toBe(false);

  await page.reload();
  await expect(page.locator(".filter-panel")).toHaveAttribute(
    "aria-busy",
    "false",
  );
  await expect(page.getByLabel("Light", { exact: true })).toBeChecked();
  await expect(root).toHaveAttribute("data-resolved-theme", "light");

  await page.getByText("System", { exact: true }).click();
  await expect(root).toHaveAttribute("data-theme", "system");
  await expect(root).toHaveAttribute("data-resolved-theme", "dark");
  await page.emulateMedia({ colorScheme: "light" });
  await expect(root).toHaveAttribute("data-resolved-theme", "light");

  await page.getByText("Dark", { exact: true }).click();
  await expect(root).toHaveAttribute("data-theme", "dark");
  await expect(root).toHaveAttribute("data-resolved-theme", "dark");
});

test("title pills are normalized, deduplicated, ORed, and removable", async ({
  page,
}) => {
  await openMap(page);
  await addTitleTerm(page, "robot");
  await expectPublicationStatus(page, "2 of 8 publications match");
  await addTitleTerm(page, "battery");
  await expectPublicationStatus(page, "3 of 8 publications match");
  await addTitleTerm(page, "  ROBOT  ");
  await expect(page.locator("#selected-titles .filter-chip")).toHaveCount(2);

  await page
    .locator(".color-control")
    .getByText("Title", { exact: true })
    .click();
  await expect(page.locator("#map-legend")).toContainText("robot");
  await expect(page.locator("#map-legend")).toContainText("battery");
  await page.getByRole("button", { name: "Remove title term robot" }).click();
  await expectPublicationStatus(page, "1 of 8 publications match");
  await expect(page.locator("#clear-filters")).toBeEnabled();
});

test("author and department comboboxes support keyboard OR/AND filtering", async ({
  page,
}) => {
  await openMap(page);
  await chooseComboOption(page, "#author-search", "Alice");
  await expectPublicationStatus(page, "3 of 8 publications match");
  await chooseComboOption(page, "#author-search", "Bob");
  await expectPublicationStatus(page, "5 of 8 publications match");
  await chooseComboOption(page, "#department-search", "Electrical");
  await expectPublicationStatus(page, "2 of 8 publications match");

  const authorInput = page.locator("#author-search");
  await authorInput.fill("Nobody Named This");
  await expect(page.locator("#author-suggestions")).toContainText("No matches");
  await authorInput.press("Escape");
  await expect(page.locator("#author-suggestions")).toBeHidden();

  await page.getByRole("button", { name: "Remove Bob Brown" }).click();
  await expectPublicationStatus(page, "0 of 8 publications match");
  await expect(page.locator("#empty-state")).toBeVisible();
  await expect(page.locator("#zoom-results")).toBeDisabled();
});

test("department and faculty modes recolor the canvas and expose every represented color", async ({
  page,
}) => {
  await openMap(page);
  const canvas = page.locator("#research-map");
  const departmentImage = await canvas.screenshot();

  const legend = page.locator("#map-legend");
  await expect(legend.locator(".color-key-item")).toHaveCount(3);
  const departmentColors = await page
    .locator("#map-legend .color-key-item > i")
    .evaluateAll((swatches) =>
      swatches.map((swatch) => getComputedStyle(swatch).backgroundColor),
    );
  expect(new Set(departmentColors).size).toBe(3);
  await legend.getByRole("button", { name: /Biomedical Engineering/ }).click();
  await expect(page.locator("#selected-departments")).toContainText(
    "Biomedical Engineering",
  );
  await expectPublicationStatus(page, "2 of 8 publications match");
  await legend.getByRole("button", { name: /Biomedical Engineering/ }).click();
  await expectPublicationStatus(page, "8 publications");

  await page
    .locator(".color-control")
    .getByText("Faculty", { exact: true })
    .click();
  await expect(page.locator("#legend-summary")).toHaveText(
    "4 faculty members represented",
  );
  await expect(legend.locator(".color-key-item")).toHaveCount(4);
  await expect(legend.locator(".color-key-item > span")).toHaveText([
    "Alice Adams",
    "Bob Brown",
    "Carol Chen",
    "Dan Diaz",
  ]);
  const facultyImage = await canvas.screenshot();
  expect(facultyImage.equals(departmentImage)).toBe(false);

  await legend.getByRole("button", { name: /Dan Diaz/ }).click();
  await expect(page.locator("#selected-authors")).toContainText("Dan Diaz");
  await expectPublicationStatus(page, "1 of 8 publications match");
  await expect(page.locator("#map-legend")).toContainText("Dan Diaz");
});

test("year and citation modes use stable ordered scales and recolor the map", async ({
  page,
}) => {
  await openMap(page);
  const canvas = page.locator("#research-map");
  const departmentImage = await canvas.screenshot();

  await page
    .locator(".color-control")
    .getByText("Year", { exact: true })
    .click();
  await expect(page.getByLabel("Year", { exact: true })).toBeChecked();
  const yearLegend = page.locator(".continuous-legend");
  await expect(yearLegend).toContainText("Publication year");
  await expect(yearLegend).toContainText("2018");
  await expect(yearLegend).toContainText("2025");
  await expect(yearLegend).toHaveAttribute(
    "aria-label",
    "Publication year color scale from 2018 to 2025",
  );
  const yearImage = await canvas.screenshot();
  expect(yearImage.equals(departmentImage)).toBe(false);

  await chooseComboOption(page, "#author-search", "Alice");
  await expectPublicationStatus(page, "3 of 8 publications match");
  await expect(yearLegend).toContainText("2018");
  await expect(yearLegend).toContainText("2025");
  await page.locator("#clear-filters").click();

  await page
    .locator(".color-control")
    .getByText("Citations", { exact: true })
    .click();
  await expect(page.getByLabel("Citations", { exact: true })).toBeChecked();
  const citationLegend = page.locator(".continuous-legend");
  await expect(citationLegend).toContainText("Citations · log scale");
  await expect(citationLegend).toContainText("0");
  await expect(citationLegend).toContainText("49");
  await expect(citationLegend).toHaveAttribute(
    "aria-label",
    "Citation count color scale from 0 to 49, logarithmic when nonzero",
  );
  const citationImage = await canvas.screenshot();
  expect(citationImage.equals(yearImage)).toBe(false);
});

test("quantitative colors label robust ranges without hiding outliers", async ({
  page,
}) => {
  const artifact = makeLargeArtifact(200);
  artifact.points.forEach((point, index) => {
    point.year = 2000 + (index % 25);
    point.citation_count = index % 50;
  });
  artifact.points[0].year = 1800;
  artifact.points.at(-1).year = 2099;
  artifact.points.at(-1).citation_count = 9999;
  await openMap(page, artifact);

  await page
    .locator(".color-control")
    .getByText("Year", { exact: true })
    .click();
  const yearLegend = page.locator(".continuous-legend");
  await expect(yearLegend).toContainText("≤2000");
  await expect(yearLegend).toContainText("≥2024");
  await expect(yearLegend).toHaveAttribute(
    "aria-label",
    "Publication year color scale from 2000 and earlier to 2024 and later",
  );

  await page
    .locator(".color-control")
    .getByText("Citations", { exact: true })
    .click();
  const citationLegend = page.locator(".continuous-legend");
  await expect(citationLegend).toContainText("≥49");
  await expect(citationLegend).toHaveAttribute(
    "aria-label",
    "Citation count color scale from 0 to 49 and higher, logarithmic when nonzero",
  );
});

test("layout changes geometry while preserving filters, color mode, and details", async ({
  page,
}) => {
  await openMap(page);
  await chooseComboOption(page, "#author-search", "Alice");
  await page
    .locator(".color-control")
    .getByText("Faculty", { exact: true })
    .click();
  const canvas = page.locator("#research-map");
  await canvas.focus();
  await canvas.press("Enter");
  await expect(page.locator("#detail-panel")).toBeVisible();
  const firstLayout = await canvas.screenshot();

  await page.getByText("Global structure", { exact: true }).click();
  await expect(page.locator("#layout-note")).toContainText("PCA");
  await expect(page.locator("#selected-authors")).toContainText("Alice Adams");
  await expect(page.getByLabel("Faculty", { exact: true })).toBeChecked();
  await expect(page.locator("#detail-panel")).toBeVisible();
  const secondLayout = await canvas.screenshot();
  expect(secondLayout.equals(firstLayout)).toBe(false);
});

test("display, zoom, reset, clear, and canvas keyboard controls stay coherent", async ({
  page,
}) => {
  await openMap(page);
  await addTitleTerm(page, "term that does not exist");
  await expect(page.locator("#empty-state")).toBeVisible();
  await page.getByText("Show matches only", { exact: true }).click();
  await expect(page.locator("#zoom-results")).toBeDisabled();
  await page.locator("#clear-filters").click();
  await expectPublicationStatus(page, "8 publications");
  await expect(page.locator("#empty-state")).toBeHidden();
  await expect(page.locator("#clear-filters")).toBeDisabled();

  const canvas = page.locator("#research-map");
  const initial = await canvas.screenshot();
  await page.locator("#map-zoom-in").click();
  const buttonZoomed = await canvas.screenshot();
  expect(buttonZoomed.equals(initial)).toBe(false);
  await page.locator("#map-fit").click();
  await canvas.focus();
  await canvas.press("+");
  await canvas.press("ArrowRight");
  const moved = await canvas.screenshot();
  expect(moved.equals(initial)).toBe(false);
  await canvas.press("Home");
  await canvas.press("Enter");
  await expect(page.locator("#detail-panel")).toBeVisible();
  await canvas.focus();
  await canvas.press("Escape");
  await expect(page.locator("#detail-panel")).toBeHidden();
});

test("mobile starts map-first and opens one panel at a time", async ({
  page,
}) => {
  await page.setViewportSize({ width: 360, height: 740 });
  await openMap(page);

  await expect(page.locator(".mobile-toolbar")).toBeVisible();
  await expect(page.locator(".filter-panel")).toBeHidden();
  await expect(page.locator(".legend-panel")).toBeHidden();
  await expect(page.locator("#research-map")).toBeVisible();
  await expect(page.locator(".map-controls")).toBeVisible();

  const settings = page.locator("#toggle-filters");
  const colorKey = page.locator("#toggle-legend");
  await settings.click();
  await expect(settings).toHaveAttribute("aria-expanded", "true");
  await expect(page.locator(".filter-panel")).toBeVisible();
  await expect(page.locator(".legend-panel")).toBeHidden();

  await colorKey.click();
  await expect(settings).toHaveAttribute("aria-expanded", "false");
  await expect(colorKey).toHaveAttribute("aria-expanded", "true");
  await expect(page.locator(".filter-panel")).toBeHidden();
  await expect(page.locator(".legend-panel")).toBeVisible();

  await colorKey.click();
  await expect(page.locator(".legend-panel")).toBeHidden();
});

test("a transient artifact failure recovers automatically", async ({
  page,
}) => {
  let requests = 0;
  await page.route("https://huggingface.co/**", async (route) => {
    requests += 1;
    if (requests === 1)
      await route.fulfill({ status: 503, body: "Unavailable" });
    else await route.fulfill({ json: makeArtifact() });
  });
  await page.goto("/");
  await expectPublicationStatus(page, "8 publications");
  await expect(page.locator("#loading-overlay")).toBeHidden();
  expect(requests).toBe(2);
  await expect(page.locator("#retry-load")).toBeHidden();
});

test("persistent failures show a retry path and a later retry restores the map", async ({
  page,
}) => {
  let requests = 0;
  await page.route("https://huggingface.co/**", async (route) => {
    requests += 1;
    if (requests <= 2)
      await route.fulfill({ status: 503, body: "Unavailable" });
    else await route.fulfill({ json: makeArtifact() });
  });
  await page.goto("/");
  await expect(page.locator("#retry-load")).toBeVisible();
  await expect(page.locator("#loading-overlay")).toBeHidden();
  await expect(page.locator("#empty-title")).toHaveText(
    "The map could not be loaded.",
  );
  await page.locator("#retry-load").click();
  await expectPublicationStatus(page, "8 publications");
  await expect(page.locator("#retry-load")).toBeHidden();
});

test("one malformed row is omitted without taking down valid publications", async ({
  page,
}) => {
  const artifact = makeArtifact();
  artifact.points.push({
    ...artifact.points[0],
    work_id: "broken-coordinate",
    pca_x: "not-a-number",
  });
  artifact.point_count = artifact.points.length;
  await openMap(page, artifact);
  await expectPublicationStatus(page, "8 publications");
  await expect(page.locator("#map-status")).toContainText(
    "1 invalid record omitted",
  );
  await expect(page.locator("#map-status")).toHaveClass(/warning/);
});

test("fatal schema failures keep controls inert and offer retry", async ({
  page,
}) => {
  const artifact = makeArtifact();
  artifact.schema_version = 3;
  await serveArtifact(page, artifact);
  await page.goto("/");
  await expect(page.locator("#retry-load")).toBeVisible();
  await expect(page.locator(".filter-controls")).toHaveAttribute("inert", "");
  await expect(page.locator("#map-status")).toHaveText(
    "The publication map is temporarily unavailable. The dataset and provenance remain available.",
  );
  await expect(page.locator(".dataset-link")).toBeEnabled();
  await expect(page.getByLabel("System", { exact: true })).toBeEnabled();
});

test("the interface has no serious accessibility violations", async ({
  page,
}) => {
  await openMap(page);
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
    .analyze();
  expect(
    results.violations.filter((violation) =>
      ["serious", "critical"].includes(violation.impact),
    ),
  ).toEqual([]);
});

for (const viewport of [
  { width: 360, height: 740 },
  { width: 768, height: 900 },
  { width: 1440, height: 1000 },
]) {
  test(`fits a ${viewport.width}px viewport without horizontal overflow`, async ({
    page,
  }) => {
    await page.setViewportSize(viewport);
    await openMap(page);
    const measurements = await page.evaluate(() => {
      const filterElement = document.querySelector(".filter-panel");
      const legendElement = document.querySelector(".legend-panel");
      const filters = filterElement.getBoundingClientRect();
      const legend = legendElement.getBoundingClientRect();
      const panelsVisible = [filterElement, legendElement].every(
        (panel) => getComputedStyle(panel).visibility !== "hidden",
      );
      const surfacesOverlap = panelsVisible && !(
        filters.right <= legend.left ||
        legend.right <= filters.left ||
        filters.bottom <= legend.top ||
        legend.bottom <= filters.top
      );
      return {
        canvasHeight: document
          .querySelector("#research-map")
          .getBoundingClientRect().height,
        documentWidth: document.documentElement.scrollWidth,
        panelsVisible,
        surfacesOverlap,
        viewportWidth: window.innerWidth,
      };
    });
    expect(measurements.documentWidth).toBeLessThanOrEqual(
      measurements.viewportWidth,
    );
    expect(measurements.canvasHeight).toBeGreaterThan(300);
    expect(measurements.surfacesOverlap).toBe(false);
    if (viewport.width <= 680) {
      expect(measurements.panelsVisible).toBe(false);
      await expect(page.locator(".mobile-toolbar")).toBeVisible();
      await page.locator("#toggle-filters").click();
      await expect(page.locator("#clear-filters")).toBeVisible();
      await page.locator("#toggle-legend").click();
      await expect(page.locator("#map-legend")).toBeVisible();
    } else {
      expect(measurements.panelsVisible).toBe(true);
      await expect(page.locator("#map-legend")).toBeVisible();
      await expect(page.locator("#clear-filters")).toBeVisible();
    }
  });
}

test("renders and filters a production-sized 32,958-point artifact", async ({
  page,
}) => {
  test.slow();
  const started = Date.now();
  await openMap(page, makeLargeArtifact());
  const elapsed = Date.now() - started;
  await expectPublicationStatus(page, "32,958 publications");
  await addTitleTerm(page, "robot");
  await expectPublicationStatus(page, "of 32,958 publications match");
  expect(elapsed).toBeLessThan(15000);
});
