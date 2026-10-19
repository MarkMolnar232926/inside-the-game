// node design/render.cjs [names...]  -> design/renders/<name>.png (headless Chromium, file:// URLs)
const { chromium } = require("playwright");
const fs = require("fs"), path = require("path");
(async () => {
  const dir = __dirname;
  const names = process.argv.slice(2).length ? process.argv.slice(2)
    : fs.readdirSync(dir).filter((f) => /^\d\d_.*\.html$/.test(f)).map((f) => f.replace(".html", ""));
  const browser = await chromium.launch({ executablePath: process.env.CHROME || "/opt/pw-browsers/chromium-1194/chrome-linux/chrome" });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  page.on("pageerror", (e) => console.error("pageerror", e.message));
  for (const n of names) {
    await page.goto("file://" + path.join(dir, n + ".html"));
    await page.evaluate(() => document.fonts.ready);
    await page.waitForTimeout(300);
    await page.screenshot({ path: path.join(dir, "renders", n + ".png") });
    console.log("rendered", n);
  }
  await browser.close();
})();
