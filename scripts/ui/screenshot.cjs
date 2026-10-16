// Headless UI capture: NODE_PATH=$(npm root -g) node scripts/ui/screenshot.cjs <url> <out.png> [seekSeconds] [clickSelector...]
const { chromium } = require("playwright");
(async () => {
const [url, out, seek = "11", ...clicks] = process.argv.slice(2);
const browser = await chromium.launch({ executablePath: "/opt/pw-browsers/chromium-1194/chrome-linux/chrome" });
const page = await browser.newPage({ viewport: { width: 1680, height: 1050 }, deviceScaleFactor: 1 });
const logs = [];
page.on("console", (m) => logs.push(`${m.type()}: ${m.text()}`));
page.on("pageerror", (e) => logs.push(`pageerror: ${e.message}`));
await page.goto(url, { waitUntil: "networkidle" });
await page.waitForTimeout(1500);
await page.evaluate((s) => { const v = document.querySelector("#video"); if (v) { v.currentTime = +s; } }, seek);
await page.waitForTimeout(1500);
for (const c of clicks) { await page.click(c); await page.waitForTimeout(1200); }
await page.screenshot({ path: out });
console.log(logs.join("\n"));
await browser.close();

})();
