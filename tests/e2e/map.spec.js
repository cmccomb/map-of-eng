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
  await expect(page.locator("#match-count")).not.toHaveText("—");
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

  await expect(page.locator("#match-count")).toHaveText("8");
  await expect(page.locator("#map-status")).toContainText(
    "newest Scholar profile refresh Jul 17, 2026",
  );
  await expect(page.locator("#clear-filters")).toBeDisabled();
  await expect(page.locator("#zoom-results")).toBeDisabled();
  await expect(
    page.getByRole("button", { name: "View 3 department colors" }),
  ).toBeVisible();
  await expect(page.locator("#research-map")).toHaveAttribute(
    "aria-label",
    /showing 8 publications/,
  );
  expect(consoleErrors).toEqual([]);
});

test("uses the map as an edge-to-edge backdrop for floating controls", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await openMap(page);

  const layout = await page.evaluate(() => {
    const bounds = (selector) => {
      const rectangle = document.querySelector(selector).getBoundingClientRect();
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
    return {
      canvas: bounds("#research-map"),
      filters: bounds(".filter-panel"),
      toolbar: bounds(".map-toolbar"),
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
  expect(layout.toolbar.top).toBeGreaterThan(layout.canvas.top);
  expect(layout.toolbar.right).toBeLessThan(layout.canvas.right);
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
    .poll(() => page.evaluate(() => localStorage.getItem("cmu-research-map-theme")))
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
  await expect(page.locator("#match-count")).toHaveText("2 of 8");
  await addTitleTerm(page, "battery");
  await expect(page.locator("#match-count")).toHaveText("3 of 8");
  await addTitleTerm(page, "  ROBOT  ");
  await expect(page.locator("#selected-titles .filter-chip")).toHaveCount(2);

  await page.locator(".color-control").getByText("Title", { exact: true }).click();
  await expect(page.locator("#map-legend")).toContainText("robot");
  await expect(page.locator("#map-legend")).toContainText("battery");
  await page.getByRole("button", { name: "Remove title term robot" }).click();
  await expect(page.locator("#match-count")).toHaveText("1 of 8");
  await expect(page.locator("#clear-filters")).toBeEnabled();
});

test("author and department comboboxes support keyboard OR/AND filtering", async ({
  page,
}) => {
  await openMap(page);
  await chooseComboOption(page, "#author-search", "Alice");
  await expect(page.locator("#match-count")).toHaveText("3 of 8");
  await chooseComboOption(page, "#author-search", "Bob");
  await expect(page.locator("#match-count")).toHaveText("5 of 8");
  await chooseComboOption(page, "#department-search", "Electrical");
  await expect(page.locator("#match-count")).toHaveText("2 of 8");

  const authorInput = page.locator("#author-search");
  await authorInput.fill("Nobody Named This");
  await expect(page.locator("#author-suggestions")).toContainText("No matches");
  await authorInput.press("Escape");
  await expect(page.locator("#author-suggestions")).toBeHidden();

  await page.getByRole("button", { name: "Remove Bob Brown" }).click();
  await expect(page.locator("#match-count")).toHaveText("0 of 8");
  await expect(page.locator("#empty-state")).toBeVisible();
  await expect(page.locator("#zoom-results")).toBeDisabled();
});

test("department and faculty modes recolor the canvas and expose every represented color", async ({
  page,
}) => {
  await openMap(page);
  const canvas = page.locator("#research-map");
  const departmentImage = await canvas.screenshot();

  await page.getByRole("button", { name: "View 3 department colors" }).click();
  await expect(page.locator("#color-key-dialog")).toBeVisible();
  await expect(page.locator("#color-key-list button")).toHaveCount(3);
  const departmentColors = await page
    .locator("#color-key-list .faculty-legend-item > i")
    .evaluateAll((swatches) =>
      swatches.map((swatch) => getComputedStyle(swatch).backgroundColor),
    );
  expect(new Set(departmentColors).size).toBe(3);
  await page
    .locator("#color-key-list")
    .getByRole("button", { name: /Biomedical Engineering/ })
    .click();
  await expect(page.locator("#selected-departments")).toContainText(
    "Biomedical Engineering",
  );
  await expect(page.locator("#match-count")).toHaveText("2 of 8");
  await page
    .locator("#color-key-list")
    .getByRole("button", { name: /Biomedical Engineering/ })
    .click();
  await expect(page.locator("#match-count")).toHaveText("8");
  await page.locator("#close-color-key").click();

  await page
    .locator(".color-control")
    .getByText("Faculty", { exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "View 4 faculty colors" }),
  ).toBeVisible();
  const facultyImage = await canvas.screenshot();
  expect(facultyImage.equals(departmentImage)).toBe(false);

  await page.getByRole("button", { name: "View 4 faculty colors" }).click();
  await page.locator("#color-key-search").fill("Dan");
  await expect(page.locator("#color-key-list button")).toHaveCount(1);
  await page.getByRole("button", { name: /Dan Diaz/ }).click();
  await expect(page.locator("#selected-authors")).toContainText("Dan Diaz");
  await expect(page.locator("#match-count")).toHaveText("1 of 8");
  await expect(page.locator("#map-legend")).toContainText("Dan Diaz");
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

  await page.getByText("Local neighborhoods", { exact: true }).click();
  await expect(page.locator("#layout-note")).toContainText("t-SNE");
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
  await expect(page.locator("#match-count")).toHaveText("8");
  await expect(page.locator("#empty-state")).toBeHidden();
  await expect(page.locator("#clear-filters")).toBeDisabled();

  const canvas = page.locator("#research-map");
  const initial = await canvas.screenshot();
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

test("a transient artifact failure recovers automatically", async ({ page }) => {
  let requests = 0;
  await page.route("https://huggingface.co/**", async (route) => {
    requests += 1;
    if (requests === 1) await route.fulfill({ status: 503, body: "Unavailable" });
    else await route.fulfill({ json: makeArtifact() });
  });
  await page.goto("/");
  await expect(page.locator("#match-count")).toHaveText("8");
  expect(requests).toBe(2);
  await expect(page.locator("#retry-load")).toBeHidden();
});

test("persistent failures show a retry path and a later retry restores the map", async ({
  page,
}) => {
  let requests = 0;
  await page.route("https://huggingface.co/**", async (route) => {
    requests += 1;
    if (requests <= 2) await route.fulfill({ status: 503, body: "Unavailable" });
    else await route.fulfill({ json: makeArtifact() });
  });
  await page.goto("/");
  await expect(page.locator("#retry-load")).toBeVisible();
  await expect(page.locator("#empty-title")).toHaveText(
    "The map could not be loaded.",
  );
  await page.locator("#retry-load").click();
  await expect(page.locator("#match-count")).toHaveText("8");
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
  await expect(page.locator("#match-count")).toHaveText("8");
  await expect(page.locator("#map-status")).toContainText(
    "1 invalid record omitted",
  );
  await expect(page.locator("#map-status")).toHaveClass(/warning/);
});

test("fatal schema failures keep controls inert and offer retry", async ({ page }) => {
  const artifact = makeArtifact();
  artifact.schema_version = 3;
  await serveArtifact(page, artifact);
  await page.goto("/");
  await expect(page.locator("#retry-load")).toBeVisible();
  await expect(page.locator(".filter-panel")).toHaveAttribute("inert", "");
  await expect(page.locator("#match-label")).toHaveText("map unavailable");
});

test("the interface has no serious accessibility violations", async ({ page }) => {
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
    const measurements = await page.evaluate(() => ({
      canvasHeight: document.querySelector("#research-map").getBoundingClientRect().height,
      documentWidth: document.documentElement.scrollWidth,
      viewportWidth: window.innerWidth,
    }));
    expect(measurements.documentWidth).toBeLessThanOrEqual(
      measurements.viewportWidth,
    );
    expect(measurements.canvasHeight).toBeGreaterThan(300);
    await expect(page.locator(".map-toolbar")).toBeVisible();
    await expect(page.locator("#clear-filters")).toBeVisible();
  });
}

test("renders and filters a production-sized 32,958-point artifact", async ({
  page,
}) => {
  test.slow();
  const started = Date.now();
  await openMap(page, makeLargeArtifact());
  const elapsed = Date.now() - started;
  await expect(page.locator("#match-count")).toHaveText("32,958");
  await addTitleTerm(page, "robot");
  await expect(page.locator("#match-count")).toHaveText(/of 32,958/);
  expect(elapsed).toBeLessThan(15000);
});
