// node design/anim.cjs [name] [fps] -> design/renders/<name>.mp4 (frame-stepped via window.seek(t), H.264)
const { chromium } = require("playwright");
const fs = require("fs"), path = require("path"), { execFileSync } = require("child_process");
(async () => {
  const name = process.argv[2] || "13_animated", fps = +(process.argv[3] || 30);
  const dir = __dirname, tmp = path.join(dir, "renders", name + "_frames");
  fs.rmSync(tmp, { recursive: true, force: true }); fs.mkdirSync(tmp, { recursive: true });
  const browser = await chromium.launch({ executablePath: process.env.CHROME || "/opt/pw-browsers/chromium-1194/chrome-linux/chrome" });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  page.on("pageerror", (e) => console.error("pageerror", e.message));
  await page.goto("file://" + path.join(dir, name + ".html") + "#t");
  await page.evaluate(() => document.fonts.ready);
  const end = await page.evaluate(() => window.T.end);
  const n = Math.round(end * fps);
  for (let f = 0; f <= n; f++) {
    await page.evaluate((t) => window.seek(t), f / fps);
    await page.screenshot({ path: path.join(tmp, String(f).padStart(4, "0") + ".jpg"), type: "jpeg", quality: 94 });
  }
  await browser.close();
  const out = path.join(dir, "renders", name + ".mp4");
  execFileSync("ffmpeg", ["-y", "-loglevel", "error", "-framerate", String(fps), "-i", path.join(tmp, "%04d.jpg"),
    "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]);
  console.log("wrote", out, n + 1, "frames");
})();
