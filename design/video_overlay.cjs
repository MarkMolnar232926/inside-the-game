// node design/video_overlay.cjs -> design/renders/video_overlay/<frame>.png (transparent), bug.png (score bug only)
const { chromium } = require("playwright");
const fs = require("fs"), path = require("path");
(async () => {
  const dir = __dirname, out = path.join(dir, "renders", "video_overlay");
  fs.rmSync(out, { recursive: true, force: true }); fs.mkdirSync(out, { recursive: true });
  const browser = await chromium.launch({ executablePath: process.env.CHROME || "/opt/pw-browsers/chromium-1194/chrome-linux/chrome" });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  page.on("pageerror", (e) => console.error("pageerror", e.message));
  await page.goto("file://" + path.join(dir, "14_video_overlay.html"));
  await page.evaluate(() => document.fonts.ready);
  const [f0, f1] = await page.evaluate(() => [VSCENE.f0, VSCENE.f1]);
  await page.evaluate((f) => { applyFrame(f); seek(-5); }, f0);     // everything hidden except the score bug
  await page.screenshot({ path: path.join(out, "bug.png"), omitBackground: true });
  for (let f = f0; f <= f1; f++) {
    await page.evaluate((f) => seekFrame(f), f);
    await page.screenshot({ path: path.join(out, String(f).padStart(4, "0") + ".png"), omitBackground: true });
  }
  await browser.close();
  console.log("overlay frames", f0, "..", f1);
})();
