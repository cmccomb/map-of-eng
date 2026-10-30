"use strict";

const fs = require("node:fs");
const { test, expect } = require("@playwright/test");
const AxeBuilder = require("@axe-core/playwright").default;
const { makeArtifact, makeSchoolArtifact, makeLargeArtifact } = require("../fixtures/artifact.js");

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
    "newest Scholar profile refresh in full corpus Jul 17, 2026",
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

test("pending selected works use a nonspatial keyboard-accessible list", async ({ page }) => {
  const artifact = makeArtifact();
  artifact.schema_version = 8;
  artifact.pending_publications = Array.from({ length: 51 }, (_, index) => ({
    work_id: `pending-${index + 1}`,
    title: index === 0 ? "<img src=x onerror=alert(1)> Reviewed work" : `Reviewed work ${index + 1}`,
    year: 2026,
    faculty_ids: ["f-alice"],
    source_url: `https://example.org/publication/pending-${index + 1}`,
    release_decision: "selected",
    map_hold_reason: "pending_topic_review",
    collaboration_eligible: index % 2 === 0,
  }));
  artifact.pending_publication_count = 51;
  await openMap(page, artifact);
  const marker = page.getByRole("button", { name: /51 publications Awaiting topic review/ });
  await expect(marker).toBeVisible();
  await expectPublicationStatus(page, "8 publications");
  await marker.focus();
  await page.keyboard.press("Enter");
  const dialog = page.getByRole("dialog", { name: "Publications awaiting topic review" });
  await expect(dialog).toBeVisible();
  await expect(page.locator("#pending-publications-close")).toBeFocused();
  await expect(dialog.locator("li")).toHaveCount(50);
  await expect(dialog).toContainText("Reviewed work 2");
  await expect(dialog).toContainText("Reviewed work 3");
  await expect(dialog.locator("img")).toHaveCount(0);
  await expect(dialog).toContainText("<img src=x onerror=alert(1)> Reviewed work");
  await expect(dialog.getByRole("link", { name: /Open publication source/ }).first())
    .toHaveAttribute("rel", "noopener noreferrer");
  const accessibility = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
    .analyze();
  expect(accessibility.violations.filter((violation) =>
    ["serious", "critical"].includes(violation.impact))).toEqual([]);
  await dialog.getByRole("button", { name: "Show more" }).click();
  await expect(dialog.locator("li")).toHaveCount(51);
  await page.keyboard.press("Escape");
  await expect(dialog).toBeHidden();
  await expect(marker).toBeFocused();
});

test("pending marker stays in interface chrome on a small screen", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 720 });
  const artifact = makeArtifact();
  artifact.schema_version = 8;
  artifact.pending_publications = [{
    work_id: "pending-small",
    title: "A reviewed work",
    year: null,
    faculty_ids: ["f-alice"],
    source_url: "https://example.org/publication/pending-small",
    release_decision: "selected",
    map_hold_reason: "pending_topic_review",
    collaboration_eligible: false,
  }];
  artifact.pending_publication_count = 1;
  await openMap(page, artifact);
  const marker = page.locator("#pending-publications-open");
  const box = await marker.boundingBox();
  expect(box).not.toBeNull();
  expect(box.x).toBeGreaterThanOrEqual(0);
  expect(box.x + box.width).toBeLessThanOrEqual(390);
  await marker.click();
  await expect(page.getByRole("dialog")).toContainText("Year unknown");
});

test("malformed pending collaboration eligibility fails the whole artifact closed", async ({ page }) => {
  const artifact = makeArtifact();
  artifact.schema_version = 8;
  artifact.pending_publications = [{
    work_id: "pending-invalid-eligibility",
    title: "Reviewed work with malformed eligibility",
    year: 2026,
    faculty_ids: ["f-alice", "f-bob"],
    source_url: "https://example.org/publication/pending-invalid-eligibility",
    release_decision: "selected",
    map_hold_reason: "pending_topic_review",
    collaboration_eligible: "false",
  }];
  artifact.pending_publication_count = 1;
  await serveArtifact(page, artifact);
  await page.goto("/");
  await expect(page.locator("#retry-load")).toBeVisible();
  await expect(page.locator(".filter-controls")).toHaveAttribute("inert", "");
  await expect(page.locator("#pending-publications-open")).toBeHidden();
});

test("school selection includes joint publications and persists in the view URL", async ({ page }) => {
  await openMap(page, makeSchoolArtifact());
  await expect(page.locator("#school-field")).toBeVisible();
  await chooseComboOption(page, "#school-search", "Computer Science");
  await expectPublicationStatus(page, "2 of 10 publications match");
  await expect(page).toHaveURL(/schools=scs/);
  await chooseComboOption(page, "#department-search", "Mechanical Engineering");
  await expectPublicationStatus(page, "1 of 10 publications match");
  await page.reload();
  await expect(page.locator("#selected-schools")).toContainText("School of Computer Science");
  await expectPublicationStatus(page, "1 of 10 publications match");
});

test("college colors include joint work and the color key selects its school", async ({ page }) => {
  const artifact = makeSchoolArtifact();
  artifact.points = artifact.points.filter((point) => point.work_id !== "p10");
  artifact.point_count = artifact.points.length;
  artifact.catalogs.schools.find((school) => school.school_id === "scs").publication_count = 1;
  artifact.catalogs.departments.find((department) => department.department_id === "d-csd").publication_count = 1;
  artifact.catalogs.faculty.find((person) => person.person_id === "f-erin").publication_count = 1;
  await openMap(page, artifact);

  await page.locator(".color-control").getByText("College", { exact: true }).click();
  await expect(page.locator('input[name="color-mode"][value="school"]')).toBeChecked();
  await expect(page).toHaveURL(/color=school/);

  const legend = page.locator("#map-legend");
  const engineering = legend.getByRole("button", { name: /College of Engineering/ });
  const scs = legend.getByRole("button", { name: /School of Computer Science/ });
  await expect(legend.locator(".color-key-item")).toHaveCount(2);
  await expect(engineering.locator("small")).toHaveText("8 matches");
  await expect(scs.locator("small")).toHaveText("1 match");
  const schoolColors = await legend.locator(".color-key-item > i").evaluateAll((swatches) =>
    swatches.map((swatch) => getComputedStyle(swatch).backgroundColor),
  );
  expect(new Set(schoolColors).size).toBe(2);

  // The only SCS work is jointly affiliated with Engineering. It must already
  // use the SCS color before selecting SCS in the key.
  await addTitleTerm(page, "Robotic Design Collaboration");
  await expectPublicationStatus(page, "1 of 9 publications match");
  await expect(legend.locator(".color-key-item")).toHaveCount(1);
  await expect(scs.locator("small")).toHaveText("1 match");

  await scs.click();
  await expect(page.locator("#selected-schools")).toContainText("School of Computer Science");
  await expect(scs).toHaveAttribute("aria-pressed", "true");
  await expectPublicationStatus(page, "1 of 9 publications match");
  await expect(page).toHaveURL(/schools=scs/);

  await page.reload();
  await expect(page.locator('input[name="color-mode"][value="school"]')).toBeChecked();
  await expect(page.locator("#selected-schools")).toContainText("School of Computer Science");
  await expectPublicationStatus(page, "1 of 9 publications match");
});

test("zero-work faculty and departments remain selectable with clear coverage states", async ({ page }) => {
  const artifact = makeSchoolArtifact();
  artifact.catalogs.schools.push({
    school_id: "mcs",
    title: "Mellon College of Science",
    publication_count: 0,
  });
  artifact.catalogs.departments.push({
    department_id: "math",
    title: "Mathematical Sciences",
    school_id: "mcs",
    publication_count: 0,
  });
  artifact.catalogs.faculty.push({
    person_id: "f-nora",
    display_name: "Nora Newton",
    school_ids: ["mcs"],
    publication_count: 0,
  });
  await openMap(page, artifact);

  const departmentSearch = page.locator("#department-search");
  await departmentSearch.focus();
  await expect(page.locator("#department-suggestions button")).toHaveCount(5);
  await departmentSearch.press("Escape");

  const authorSearch = page.locator("#author-search");
  await authorSearch.fill("Nora Newton");
  await expect(page.locator("#author-suggestions")).toContainText("0 mapped works");
  await authorSearch.press("Enter");
  await expect(page.locator("#selected-authors")).toContainText("Nora Newton");
  await expectPublicationStatus(page, "0 of 10 publications match");
  await expect(page.locator("#empty-title")).toHaveText("Nora Newton is in the faculty roster.");
  await expect(page.locator("#empty-copy")).toContainText("No publications are mapped to this faculty member yet");
  await expect(page.locator("#legend-summary")).toHaveText("No matching publications");

  await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  const emptyCanvas = await page.locator("#research-map").evaluate((canvas) => canvas.toDataURL());
  await page.getByText("Show matches only", { exact: true }).click();
  await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  expect(await page.locator("#research-map").evaluate((canvas) => canvas.toDataURL())).toBe(emptyCanvas);

  await page.locator("#clear-filters").click();
  await expect(page.locator("#empty-state")).toBeHidden();
  await departmentSearch.fill("Mathematical Sciences");
  await expect(page.locator("#department-suggestions")).toContainText("0 mapped works");
  await departmentSearch.press("Enter");
  await expect(page.locator("#selected-departments")).toContainText("Mathematical Sciences");
  await expect(page.locator("#empty-title")).toHaveText("Mathematical Sciences is in the department roster.");
  await expect(page.locator("#empty-copy")).toContainText("No publications are mapped to this department yet");
});

test("Math faculty without a Scholar account keep a visible coverage note", async ({ page }) => {
  const artifact = makeSchoolArtifact();
  artifact.catalogs.schools.push({
    school_id: "mcs",
    title: "Mellon College of Science",
    publication_count: 4,
  });
  artifact.catalogs.departments.push({
    department_id: "math",
    title: "Mathematical Sciences",
    school_id: "mcs",
    publication_count: 4,
  });
  artifact.catalogs.faculty.push(
    {
      person_id: "f-shlomo",
      display_name: "Shlomo Ta'asan",
      scholar_id: "",
      memberships: [{ department_id: "math", role: "faculty" }],
      school_ids: ["mcs"],
      publication_count: 3,
    },
    {
      person_id: "f-verified-math",
      display_name: "Verified Math Faculty",
      scholar_id: "exampleAAAAJ",
      memberships: [{ department_id: "math", role: "faculty" }],
      school_ids: ["mcs"],
      publication_count: 1,
    },
  );
  artifact.catalogs.faculty.find((person) => person.person_id === "f-erin").memberships = [
    { department_id: "d-csd", role: "faculty" },
  ];
  for (const [index, personId] of [
    "f-shlomo",
    "f-shlomo",
    "f-shlomo",
    "f-verified-math",
  ].entries()) {
    artifact.points.push({
      ...artifact.points[index],
      work_id: `math-${index}`,
      faculty_ids: [personId],
      department_ids: ["math"],
      school_ids: ["mcs"],
    });
  }
  artifact.point_count = artifact.points.length;
  await openMap(page, artifact);

  const search = page.locator("#author-search");
  const note = page.locator("#faculty-coverage-note");
  await search.fill("Shlomo");
  await expect(page.locator("#author-suggestions")).toContainText("3 mapped works");
  await search.press("Enter");
  await expect(note).toHaveText(
    "Shlomo Ta'asan: Scholar account unresolved; mapped works are observed records, not a complete publication history.",
  );
  await expect(note).toBeVisible();
  await page.reload();
  await expect(note).toBeVisible();
  await expect(note).toContainText("Scholar account unresolved");
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator("#toggle-filters").click();
  await expect(note).toBeVisible();
  const mobileWidth = await page.evaluate(() => ({
    document: document.documentElement.scrollWidth,
    viewport: window.innerWidth,
  }));
  expect(mobileWidth.document).toBeLessThanOrEqual(mobileWidth.viewport);

  await page.locator("#selected-authors button").click();
  await expect(note).toBeHidden();
  await search.fill("Verified Math");
  await search.press("Enter");
  await expect(note).toBeHidden();
  await page.locator("#selected-authors button").click();
  await search.fill("Erin Ellis");
  await search.press("Enter");
  await expect(note).toBeHidden();
});

test("faculty search shows the supported primary and retains secondary department filters", async ({ page }) => {
  const artifact = makeSchoolArtifact();
  artifact.catalogs.faculty[0].memberships = [
    { department_id: "d-bme", role: "faculty" },
    { department_id: "d-me", role: "faculty" },
  ];
  artifact.catalogs.faculty[0].primary_department_id = "d-me";
  artifact.catalogs.faculty[0].primary_department_basis = "inferred_advisee_count";
  Object.assign(artifact.catalogs.faculty[0], {
    primary_academic_home_type: "department",
    primary_academic_home_id: "d-me",
    primary_academic_home_title: "Mechanical Engineering",
    primary_academic_home_school_id: "engineering",
    primary_academic_home_source_url: "https://example.org/advisees",
    primary_academic_home_verified_at: "2026-10-24",
    primary_academic_home_basis: "inferred_advisee_count",
  });
  artifact.catalogs.faculty[1].memberships = [
    { department_id: "d-ece", role: "faculty" },
    { department_id: "d-me", role: "faculty" },
  ];
  artifact.catalogs.faculty[1].primary_department_id = "";
  artifact.catalogs.faculty[1].primary_department_basis = "unresolved";
  artifact.catalogs.faculty[1].primary_academic_home_type = "unresolved";
  artifact.catalogs.faculty[1].primary_academic_home_basis = "unresolved";
  artifact.points[0].department_ids = ["d-me", "d-bme"];
  await openMap(page, artifact);

  const authorSearch = page.locator("#author-search");
  await authorSearch.fill("Alice");
  await expect(page.locator("#author-suggestions")).toContainText(
    "Primary department: Mechanical Engineering (inferred from PhD students) · Also: Biomedical Engineering",
  );
  await authorSearch.press("Enter");
  await expect(page.locator("#selected-authors")).toContainText(
    "Alice Adams · Mechanical Engineering (inferred)",
  );

  await chooseComboOption(page, "#department-search", "Biomedical Engineering");
  await expectPublicationStatus(page, "2 of 10 publications match");
  await expect(page.locator("#selected-authors")).toContainText("Alice Adams");

  await page.locator("#clear-filters").click();
  await authorSearch.fill("Bob");
  await expect(page.locator("#author-suggestions")).toContainText("Primary academic home unresolved");
  await expect(page.locator("#author-suggestions")).not.toContainText(
    "Primary department: Electrical and Computer Engineering",
  );
  await authorSearch.press("Enter");
  await expect(page.locator("#selected-authors")).toContainText("Bob Brown");
  await expect(page.locator("#selected-authors")).not.toContainText(
    "Electrical and Computer Engineering",
  );
});

test("joint faculty show a display tie-break without claiming an institutional primary", async ({ page }) => {
  const artifact = makeSchoolArtifact();
  const person = artifact.catalogs.faculty[0];
  person.memberships = [
    { department_id: "d-me", role: "faculty" },
    { department_id: "d-bme", role: "faculty" },
  ];
  Object.assign(person, {
    primary_department_id: "d-me",
    primary_academic_home_type: "department",
    primary_academic_home_id: "d-me",
    primary_academic_home_title: "Mechanical Engineering",
    primary_academic_home_school_id: "engineering",
    primary_academic_home_source_url: "https://example.org/faculty-profile",
    primary_academic_home_verified_at: "2026-10-24",
    primary_academic_home_basis: "display_tie_break_first_listed",
  });
  await openMap(page, artifact);

  const search = page.locator("#author-search");
  await search.fill("Alice");
  await expect(page.locator("#author-suggestions")).toContainText(
    "Display department: Mechanical Engineering (display tie-break: first-listed department) · Also: Biomedical Engineering",
  );
  await expect(page.locator("#author-suggestions")).not.toContainText("Primary department");
  await search.press("Enter");
  await expect(page.locator("#selected-authors")).toContainText(
    "Alice Adams · Mechanical Engineering (display tie-break: first-listed department)",
  );
});

test("faculty search labels school, institute, and campus-approved homes without roster-container titles", async ({ page }) => {
  const artifact = makeSchoolArtifact();
  artifact.catalogs.schools.push(
    { school_id: "heinz", title: "Heinz College", publication_count: 1 },
    { school_id: "tepper", title: "Tepper School of Business", publication_count: 1 },
  );
  artifact.catalogs.departments.push(
    { department_id: "heinz-faculty", school_id: "heinz", title: "Heinz College faculty (shared roster container)", unit_type: "roster_container", publication_count: 0 },
    { department_id: "tepper-faculty", school_id: "tepper", title: "Tepper School of Business faculty", unit_type: "roster_container", publication_count: 0 },
    { department_id: "heinz-ism", school_id: "heinz", title: "School of Information Systems and Management", unit_type: "constituent_school", publication_count: 0 },
    { department_id: "hcii", school_id: "scs", title: "Human-Computer Interaction Institute", unit_type: "institute", publication_count: 0 },
    { department_id: "cmu-africa", school_id: "engineering", title: "CMU-Africa", unit_type: "department", publication_count: 0 },
    { department_id: "cmu-silicon-valley", school_id: "engineering", title: "CMU Silicon Valley", unit_type: "campus", publication_count: 0 },
  );
  const [tepper, pedro, institute, africa, unresolved] = artifact.catalogs.faculty;
  const home = (person, type, id, title, schoolId) => Object.assign(person, {
    primary_department_id: type === "department" ? id : "",
    primary_academic_home_type: type,
    primary_academic_home_id: id,
    primary_academic_home_title: title,
    primary_academic_home_school_id: schoolId,
    primary_academic_home_source_url: "https://example.org/cmu-profile",
    primary_academic_home_verified_at: "2026-10-24",
    primary_academic_home_basis: "reviewed",
  });
  tepper.memberships = [{ department_id: "tepper-faculty", role: "faculty" }];
  tepper.school_ids = ["tepper"];
  home(tepper, "school", "tepper", "Tepper School of Business", "tepper");
  tepper.primary_academic_home_basis = "school_fallback";
  pedro.display_name = "Pedro Ferreira";
  pedro.memberships = [
    { department_id: "heinz-faculty", role: "faculty" },
    { department_id: "d-me", role: "faculty" },
  ];
  pedro.school_ids = ["engineering", "heinz"];
  home(pedro, "school", "heinz", "Heinz College", "heinz");
  institute.memberships = [{ department_id: "hcii", role: "faculty" }];
  institute.school_ids = ["scs"];
  home(institute, "institute", "hcii", "Human-Computer Interaction Institute", "scs");
  africa.memberships = [{ department_id: "cmu-africa", role: "faculty" }];
  home(africa, "department", "cmu-africa", "CMU-Africa", "engineering");
  unresolved.memberships = [
    { department_id: "hcii", role: "faculty" },
    { department_id: "d-csd", role: "faculty" },
  ];
  unresolved.primary_academic_home_type = "unresolved";
  unresolved.primary_academic_home_basis = "unresolved";
  const constituent = {
    person_id: "f-constituent",
    display_name: "Isabel Singh",
    publication_count: 0,
    school_ids: ["heinz"],
    memberships: [{ department_id: "heinz-ism", role: "faculty" }],
  };
  home(constituent, "constituent_school", "heinz-ism", "School of Information Systems and Management", "heinz");
  const area = {
    person_id: "f-area",
    display_name: "Ari Chen",
    publication_count: 0,
    school_ids: ["tepper"],
    memberships: [{ department_id: "tepper-faculty", role: "faculty" }],
  };
  home(area, "academic_area", "tepper:operations", "Operations Research", "tepper");
  const campus = {
    person_id: "f-campus",
    display_name: "Karimulla Shaikh",
    publication_count: 0,
    school_ids: ["engineering"],
    memberships: [{ department_id: "cmu-silicon-valley", role: "teaching" }],
  };
  home(campus, "campus", "cmu-silicon-valley", "CMU Silicon Valley", "engineering");
  artifact.catalogs.faculty.push(constituent, area, campus);
  await openMap(page, artifact);

  const search = page.locator("#author-search");
  for (const [name, expected] of [
    ["Alice", "Primary school: Tepper School of Business (no narrower unit verified)"],
    ["Pedro", "Primary school: Heinz College · Also: Mechanical Engineering"],
    ["Carol", "Primary institute: Human-Computer Interaction Institute"],
    ["Dan", "Primary department: CMU-Africa"],
    ["Erin", "Primary academic home unresolved"],
    ["Isabel", "Primary constituent school: School of Information Systems and Management"],
    ["Ari", "Primary academic area: Operations Research"],
    ["Karimulla", "Primary campus: CMU Silicon Valley"],
  ]) {
    await search.fill(name);
    await expect(page.locator("#author-suggestions")).toContainText(expected);
    await expect(page.locator("#author-suggestions")).not.toContainText("roster container");
  }
  await search.fill("Pedro");
  await search.press("Enter");
  await expect(page.locator("#selected-authors")).toContainText("Pedro Ferreira · Heinz College");
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator("#toggle-filters").click();
  await search.fill("Isabel");
  await expect(page.locator("#author-suggestions")).toContainText("Primary constituent school");
  const mobileWidth = await page.evaluate(() => ({
    document: document.documentElement.scrollWidth,
    viewport: window.innerWidth,
  }));
  expect(mobileWidth.document).toBeLessThanOrEqual(mobileWidth.viewport);
  await page.locator("#department-search").fill("Tepper");
  await expect(page.locator("#department-suggestions")).toContainText("Tepper School of Business");
  await expect(page.locator("#department-suggestions")).not.toContainText("Business faculty");
});

test("publication details disclose a separately verified record", async ({ page }) => {
  const artifact = makeArtifact();
  for (const point of artifact.points) {
    point.source_types = ["curated_verified"];
    point.source_verified_at = "2026-10-24";
    point.source_verification_urls = ["https://www.cmu.edu/math/people/faculty/tetali.html"];
  }
  await openMap(page, artifact);
  const canvas = page.locator("#research-map");
  await canvas.focus();
  await canvas.press("Home");
  await canvas.press("Enter");
  await expect(page.locator("#detail-provenance")).toContainText(
    "not collected from Google Scholar",
  );
  await expect(page.locator("#detail-meta")).toContainText("citation count unavailable");
  await expect(page.locator("#detail-keyword")).toHaveText(
    "Automated topic not reviewed for this verified record",
  );
  await expect(page.locator("#detail-verification-link")).toHaveAttribute(
    "href",
    "https://www.cmu.edu/math/people/faculty/tetali.html",
  );
  await page.locator("#open-results").click();
  await expect(page.locator("#results-list")).toContainText("citation count unavailable");
  await expect(page.locator("#results-metrics")).toContainText("without counts");
  await expect(page.locator("#results-topics")).toContainText(
    "Automated topics not reviewed for verified records",
  );
});

test("reviewed context labels contributors and links source evidence", async ({ page }) => {
  const artifact = makeArtifact();
  for (const point of artifact.points) {
    point.publication_type = "patent";
    point.context_evidence_urls = ["https://patents.google.com/patent/US6486123B1/en"];
  }
  await openMap(page, artifact);
  const canvas = page.locator("#research-map");
  await canvas.focus();
  await canvas.press("Home");
  await canvas.press("Enter");
  await expect(page.locator("#detail-meta")).toContainText("Patent ·");
  await expect(page.locator("#detail-contributors-label")).toHaveText("Inventors");
  await expect(page.locator("#detail-verification-link")).toHaveAttribute(
    "href", "https://patents.google.com/patent/US6486123B1/en",
  );

  await page.locator("#open-results").click();
  const downloadPromise = page.waitForEvent("download");
  await page.locator("#export-results").click();
  const download = await downloadPromise;
  const csv = fs.readFileSync(await download.path(), "utf8");
  expect(csv).toContain('"Publication type","Contributors"');
  expect(csv).toContain(',"Patent",');
});

test("edited book shows source-verified editors", async ({ page }) => {
  const artifact = makeArtifact();
  const editors = "C. Ho; W. A. Eaton; J. P. Collman; Q. H. Gibson; J. S. Leigh, Jr.; E. Margoliash; K. Moffat; W. R. Scheidt";
  for (const point of artifact.points) {
    point.publication_type = "edited_book";
    point.display_contributors = editors;
    point.context_evidence_urls = ["https://www.cmu.edu/bio/people/faculty/images/ho_publications.pdf"];
  }
  await openMap(page, artifact);
  const canvas = page.locator("#research-map");
  await canvas.focus();
  await canvas.press("Home");
  await canvas.press("Enter");
  await expect(page.locator("#detail-contributors-label")).toHaveText("Editors");
  await expect(page.locator("#detail-authors")).toHaveText(editors);

  await page.locator("#open-results").click();
  const downloadPromise = page.waitForEvent("download");
  await page.locator("#export-results").click();
  const csv = fs.readFileSync(await (await downloadPromise).path(), "utf8");
  expect(csv).toContain('"Publication type","Contributors"');
  expect(csv).toContain(editors);
});

test("conference presentations keep their type and author attribution", async ({ page }) => {
  const artifact = makeArtifact();
  for (const point of artifact.points) {
    point.publication_type = "conference_presentation";
  }
  await openMap(page, artifact);
  await expectPublicationStatus(page, "8 publications");
  const canvas = page.locator("#research-map");
  await canvas.focus();
  await canvas.press("Home");
  await canvas.press("Enter");
  await expect(page.locator("#detail-meta")).toContainText("Conference presentation ·");
  await expect(page.locator("#detail-contributors-label")).toHaveText("Authors");
});

test("doctoral dissertation keeps its type and author attribution", async ({ page }) => {
  const artifact = makeArtifact();
  for (const point of artifact.points) {
    point.publication_type = "doctoral_dissertation";
  }
  await openMap(page, artifact);
  await expectPublicationStatus(page, "8 publications");
  const canvas = page.locator("#research-map");
  await canvas.focus();
  await canvas.press("Home");
  await canvas.press("Enter");
  await expect(page.locator("#detail-meta")).toContainText("Doctoral dissertation ·");
  await expect(page.locator("#detail-contributors-label")).toHaveText("Authors");
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

test("topic and year filters use contextual counts and survive a permalink reload", async ({
  page,
}) => {
  await openMap(page);

  await chooseComboOption(page, "#topic-search", "robot control");
  await expectPublicationStatus(page, "2 of 8 publications match");
  await expect(page.locator("#selected-topics")).toContainText("robot control");
  await expect(page.locator("#topic-selection-note")).toContainText(
    "Detailed topic",
  );

  const authorSearch = page.locator("#author-search");
  await authorSearch.fill("Alice");
  await expect(
    page.getByRole("option", { name: "Alice Adams 1 mapped work" }),
  ).toBeVisible();
  await authorSearch.press("Escape");

  await page.getByRole("button", { name: "Last 5" }).click();
  await expectPublicationStatus(page, "1 of 8 publications match");
  await expect(page.locator("#year-min")).toHaveValue("2021");
  await expect(page.locator("#year-max")).toHaveValue("2025");
  await expect(page).toHaveURL(/t=keyword-1-1/);
  await expect(page).toHaveURL(/ymin=2021/);

  await page.reload();
  await expect(page.locator(".filter-panel")).toHaveAttribute(
    "aria-busy",
    "false",
  );
  await expectPublicationStatus(page, "1 of 8 publications match");
  await expect(page.locator("#selected-topics")).toContainText("robot control");
  await expect(page.locator("#year-min")).toHaveValue("2021");
});

test("results surface summarizes, sorts, exports, and remains accessible", async ({
  page,
}) => {
  await openMap(page);
  await addTitleTerm(page, "robot");
  await expectPublicationStatus(page, "2 of 8 publications match");

  await page.locator("#open-results").click();
  const panel = page.locator("#results-panel");
  await expect(panel).toBeVisible();
  await expect(page.locator("#results-summary")).toHaveText(
    "2 of 8 publications",
  );
  await expect(page.locator("#results-list .result-item")).toHaveCount(2);
  await expect(page.locator("#results-list .result-item").first()).toContainText(
    "Soft Robot Control",
  );
  await expect(page.locator("#results-topics li")).toHaveCount(2);

  const accessibility = await new AxeBuilder({ page })
    .include("#results-panel")
    .analyze();
  expect(accessibility.violations).toEqual([]);

  const downloadPromise = page.waitForEvent("download");
  await page.locator("#export-results").click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe(
    "cmu-engineering-publications-filtered.csv",
  );
  const downloadPath = await download.path();
  const csv = fs.readFileSync(downloadPath, "utf8");
  expect(csv).toContain('"Title","Authors","Year"');
  expect(csv).toContain("Robotic Grasp Planning");
  expect(csv).toContain("Topic path");
});

test("zoom to matches trims spatial outliers and can restore every match", async ({
  page,
}) => {
  const artifact = makeLargeArtifact(100);
  artifact.points.forEach((point, index) => {
    point.title = `Focus publication ${index}`;
    const outlier = index >= 98;
    point.pca_x = outlier ? 0.95 : ((index % 10) - 5) * 0.005;
    point.pca_y = outlier ? -0.95 : ((index % 7) - 3) * 0.005;
    point.tsne_x = point.pca_x;
    point.tsne_y = point.pca_y;
  });
  await openMap(page, artifact);
  await addTitleTerm(page, "focus");

  await page.locator("#zoom-results").click();
  await expect(page.locator("#fit-note")).toContainText(
    "central 98% · 2 outliers",
  );
  await page.locator("#show-all-results").click();
  await expect(page.locator("#fit-note")).toHaveText(
    "Showing every matching publication.",
  );
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

test("topic keywords annotate the map and dot sizing reveals age or impact", async ({
  page,
}) => {
  await openMap(page);
  const canvas = page.locator("#research-map");
  await expect(canvas).toHaveAttribute(
    "aria-label",
    /showing 8 publications across 3 topic regions and 6 detailed topics/,
  );
  await expect(canvas).toHaveAttribute("data-visible-keyword-levels", "0");
  const uniformImage = await canvas.screenshot();
  await expect(page.locator("#size-legend")).toBeHidden();

  await page.locator(".size-control").getByText("Oldest", { exact: true }).click();
  await expect(page.getByLabel("Oldest", { exact: true })).toBeChecked();
  await expect(page.locator("#size-note")).toContainText(
    "Older publications appear larger",
  );
  await expect(page.locator("#size-legend")).toContainText(
    "Larger dots are older",
  );
  const oldestImage = await canvas.screenshot();
  expect(oldestImage.equals(uniformImage)).toBe(false);

  await page.locator(".size-control").getByText("Newest", { exact: true }).click();
  const newestImage = await canvas.screenshot();
  expect(newestImage.equals(oldestImage)).toBe(false);

  await page
    .locator(".size-control")
    .getByText("Widely cited", { exact: true })
    .click();
  await expect(page.locator("#size-note")).toContainText("log scale");
  await expect(page.locator("#size-legend")).toContainText(
    "Larger dots are more cited",
  );
  const citedImage = await canvas.screenshot();
  expect(citedImage.equals(newestImage)).toBe(false);

  await page.getByRole("button", { name: "Zoom in" }).click();
  await expect(canvas).toHaveAttribute("data-visible-keyword-levels", "0,1");

  await canvas.focus();
  await canvas.press("Enter");
  await expect(page.locator("#detail-keyword")).toHaveText(
    /robotic design|energy systems|biomedical interfaces.*›/,
  );
});

test("isolated and active child topics join the overview", async ({ page }) => {
  const artifact = makeLargeArtifact(120);
  const parent = artifact.keywords.find(
    (keyword) => keyword.keyword_id === "keyword-1",
  );
  const island = artifact.keywords.find(
    (keyword) => keyword.keyword_id === "keyword-1-1",
  );
  parent.coordinates.tsne = { x: 0, y: 0 };
  island.coordinates.tsne = { x: 0, y: -0.9 };
  artifact.keywords
    .filter((keyword) => keyword.level === 1 && keyword.keyword_id !== island.keyword_id)
    .forEach((keyword, index) => {
      const keywordParent = artifact.keywords.find(
        (candidate) => candidate.keyword_id === keyword.parent_keyword_id,
      );
      keyword.coordinates.tsne = {
        x: keywordParent.coordinates.tsne.x + 0.02 * ((index % 2) + 1),
        y: keywordParent.coordinates.tsne.y + 0.02 * ((index % 3) + 1),
      };
    });
  artifact.points.forEach((point, index) => {
    const isIsland = index < 60;
    point.keyword_ids = ["keyword-1", isIsland ? "keyword-1-1" : "keyword-1-2"];
    point.title = `${isIsland ? "Island" : "Mainland"} publication ${index}`;
    point.tsne_x = (index % 10) * 0.002;
    point.tsne_y = (isIsland ? -0.9 : 0) + (index % 7) * 0.002;
  });
  await openMap(page, artifact);

  const canvas = page.locator("#research-map");
  await expect(canvas).toHaveAttribute(
    "data-overview-keywords",
    /keyword-1-1/,
  );
  await chooseComboOption(page, "#topic-search", "robot control");
  await expectPublicationStatus(page, "60 of 120 publications match");
  await expect(page.locator("#topic-selection-note")).toContainText(
    "Overview topic",
  );
  await expect(page.locator("#topic-selection-note")).not.toContainText(
    "promoted",
  );
  await expect(canvas).toHaveAttribute(
    "data-overview-keywords",
    /keyword-1-1/,
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
  const results = page.locator("#toggle-results");
  await settings.click();
  await expect(settings).toHaveAttribute("aria-expanded", "true");
  await expect(page.locator(".filter-panel")).toBeVisible();
  await expect(page.locator(".legend-panel")).toBeHidden();

  await colorKey.click();
  await expect(settings).toHaveAttribute("aria-expanded", "false");
  await expect(colorKey).toHaveAttribute("aria-expanded", "true");
  await expect(page.locator(".filter-panel")).toBeHidden();
  await expect(page.locator(".legend-panel")).toBeVisible();

  await results.click();
  await expect(colorKey).toHaveAttribute("aria-expanded", "false");
  await expect(results).toHaveAttribute("aria-expanded", "true");
  await expect(page.locator(".legend-panel")).toBeHidden();
  await expect(page.locator("#results-panel")).toBeVisible();

  await results.click();
  await expect(page.locator("#results-panel")).toBeHidden();
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
