// On-air output: score bug always on; an approved insight plays the column build-in / hold / build-out.
// Approvals that arrive while a card is on air wait their turn (never two cards at once).
(() => {
  const qs = new URLSearchParams(location.search);
  if (qs.get("bg")) document.body.classList.add("preview");
  const stage = document.getElementById("stage"), status = document.getElementById("status");
  const fit = () => { const k = Math.min(innerWidth / 1920, innerHeight / 1080); stage.style.transform = `scale(${k})`; };
  addEventListener("resize", fit); fit();
  let bug = null, clock = {t: 0, score: [0, 0], running: false, speed: 1, at: performance.now()};
  const queue = [];
  let current = null;

  function onMessage(m) {
    if (m.type === "clock" || m.type === "play") {
      clock = {t: m.t, score: m.score, running: m.running, speed: m.speed, at: performance.now()};
      if (!bug && m.teams) bug = PLCard.bug(stage, m.teams);
    }
    if (m.type === "play") queue.push(m.proposal);
  }

  function frame(now) {
    // the match clock keeps ticking between messages (feed speed x real time)
    const t = clock.running ? clock.t + (now - clock.at) / 1000 * clock.speed : clock.t;
    if (bug) bug.update(t, clock.score);
    if (!current && queue.length) {
      const p = queue.shift();
      current = {p, col: PLCard.column(stage, p), t0: now};
      window.__onair = p.id;  // for automated checks
    }
    if (current) {
      const tt = (now - current.t0) / 1000;
      current.col.seek(tt);
      if (tt > PLCard.T.end) { current.col.remove(); current = null; window.__onair = null; }
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  function connect() {
    const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/onair`);
    ws.onopen = () => { status.textContent = "on air: connected"; };
    ws.onmessage = (ev) => onMessage(JSON.parse(ev.data));
    ws.onclose = () => { status.textContent = "reconnecting..."; setTimeout(connect, 1000); };
  }
  connect();
})();
