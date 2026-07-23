"use strict";

const { defineConfig, devices } = require("@playwright/test");

const crossBrowserSmoke =
  /loads cleanly with useful defaults|mobile starts map-first/;
const productionLoadBudget =
  /renders and filters a production-sized 32,958-point artifact/;

module.exports = defineConfig({
  testDir: "./tests/e2e",
  outputDir: "./output/playwright/test-results",
  fullyParallel: true,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  workers: 2,
  reporter: process.env.CI
    ? [["line"], ["html", { outputFolder: "output/playwright/report", open: "never" }]]
    : "line",
  use: {
    baseURL: "http://127.0.0.1:4173",
    trace: "on-first-retry",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
    {
      name: "firefox-smoke",
      grep: crossBrowserSmoke,
      use: { ...devices["Desktop Firefox"] },
    },
    {
      name: "webkit-smoke",
      grep: crossBrowserSmoke,
      use: { ...devices["Desktop Safari"] },
    },
    {
      name: "low-end-mobile-budget",
      grep: productionLoadBudget,
      metadata: {
        load_budget_ms: 15000,
        profile:
          "Galaxy S9+ landscape, Chromium low-end mode, 512 MB JavaScript heap",
      },
      use: {
        ...devices["Galaxy S9+"],
        viewport: { width: 720, height: 360 },
        launchOptions: {
          args: [
            "--enable-low-end-device-mode",
            "--renderer-process-limit=2",
            "--js-flags=--max-old-space-size=512",
          ],
        },
      },
    },
  ],
  webServer: {
    command: "node tests/test-server.js",
    url: "http://127.0.0.1:4173",
    reuseExistingServer: !process.env.CI,
    timeout: 10000,
  },
  expect: { timeout: 5000 },
  timeout: 30000,
});
