// End-to-end check of the human-in-the-loop flow (server must be running):
// producer console receives a proposal -> operator approves (A) -> on-air output plays it.
// node scripts/ui/producer_e2e.cjs [base=http://localhost:8000] [outDir=outputs/producer]
const { chromium } = require("playwright");
const path = require("path");
(async () => {
  const base = process.argv[2] || "http://localhost:8000", out = process.argv[3] || "outputs/producer";
  const browser = await chromium.launch({ executablePath: process.env.CHROME || "/opt/pw-browsers/chromium-1194/chrome-linux/chrome" });
  const ctx = await browser.newContext({ viewport: { width: 1920, height: 1080 } });
  const prod = await ctx.newPage(), air = await ctx.newPage();
  for (const p of [prod, air]) p.on("pageerror", (e) => console.error("pageerror", e.message));
  await prod.goto(base + "/producer");
  await air.goto(base + "/onair?bg=1");
  const t0 = Date.now();
  await prod.waitForSelector("#pending .prop", { timeout: 120000 });
  await prod.waitForTimeout(900);
  const id = await prod.$eval("#pending .prop", (n) => n.dataset.id);
  console.log("proposal", id, "after", ((Date.now() - t0) / 1000).toFixed(1), "s");
  await prod.screenshot({ path: path.join(out, "producer_console.png") });
  // operator edit with a number that is not in the evidence: the Verifier must flag it
  await prod.keyboard.press("e");
  const ta = await prod.$("#pending .prop .edit");
  const orig = await ta.inputValue();
  await ta.fill(orig.replace(/\d+/, (n) => String(+n + 4)));
  await prod.waitForSelector("#pending .prop .checkmsg.bad", { timeout: 5000 });
  console.log("edit check:", await prod.$eval("#pending .prop .checkmsg", (n) => n.textContent));
  await prod.screenshot({ path: path.join(out, "producer_edit_check.png") });
  await ta.fill(orig + " A big reason for the pressure up front.");
  await prod.waitForSelector("#pending .prop .checkmsg.ok", { timeout: 5000 });
  await prod.click("body", { position: { x: 5, y: 1070 } });
  await prod.keyboard.press("a");
  await air.waitForFunction((id) => window.__onair === id, id, { timeout: 10000 });
  await air.waitForTimeout(2600);
  await air.screenshot({ path: path.join(out, "onair_output.png") });
  await prod.waitForTimeout(400);
  await prod.screenshot({ path: path.join(out, "producer_after_approve.png") });
  console.log("on air:", await air.evaluate(() => window.__onair), "· ticker rows:", await prod.$$eval("#ticker li", (l) => l.length));
  await browser.close();
})();
