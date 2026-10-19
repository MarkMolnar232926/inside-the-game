// Producer console: proposals from the analyst agent pop up; nothing reaches air without a click.
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
  let teams = [], players = {}, clock = {t: 0, score: [0, 0], running: false, speed: 10, at: performance.now()};
  const cards = new Map();  // id -> {p, node}
  const TYPE = {kick_off: "Kick-off", pass: "Pass", carry: "Carry", shot: "Shot", goal: "GOAL", save: "Save", tackle: "Tackle",
    interception: "Interception", duel: "Duel", pressure: "Pressure", ball_recovery: "Recovery", clearance: "Clearance",
    foul: "Foul", free_kick: "Free kick", corner: "Corner", throw_in: "Throw-in", goal_kick: "Goal kick",
    out_of_play: "Out of play", possession_change: "Possession", substitution: "Substitution",
    period_start: "Period start", period_end: "Period end"};

  // ------------------------------------------------------------------ clock + header
  const now = () => clock.running ? clock.t + (performance.now() - clock.at) / 1000 * clock.speed : clock.t;
  function setClock(m) {
    clock = {t: m.t, score: m.score, running: m.running, speed: m.speed, at: performance.now()};
    $("#score").textContent = `${m.score[0]} – ${m.score[1]}`;
    $("#live").classList.toggle("paused", !m.running);
    $("#live").lastChild.textContent = m.finished ? "FULL TIME" : m.running ? "EVENT FEED" : "PAUSED";
    $("#pause").textContent = m.running ? "Pause" : "Resume";
    $("#speed").value = String(m.speed);
  }
  function tick() {
    $("#clock").textContent = PLCard.clockText(now());
    for (const {p, node} of cards.values()) {
      const k = Math.max(0, Math.min(1, (p.expires_t - now()) / (p.expires_t - p.t)));
      node.querySelector(".timer span").style.width = (k * 100) + "%";
    }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);

  // ------------------------------------------------------------------ ticker
  function addEvent(e, prepend = true) {
    const li = document.createElement("li");
    const team = teams.find((t) => t.id === e.team);
    li.className = e.kind;
    li.innerHTML = `<span class="m">${e.minute}'</span><span class="c" style="background:${team ? team.primary : "#555"}"></span>
      <span>${esc(TYPE[e.kind] || e.kind)} <span class="p">${esc(players[e.player] || "")}${e.outcome === "fail" ? " ✕" : ""}</span></span>`;
    const ol = $("#ticker");
    prepend ? ol.prepend(li) : ol.append(li);
    while (ol.children.length > 60) ol.lastChild.remove();
  }

  // ------------------------------------------------------------------ proposals
  function pitch(svg, evs) {
    svg.innerHTML = `<rect class="ln" x="0" y="0" width="105" height="68"/><line class="ln" x1="52.5" y1="0" x2="52.5" y2="68"/>
      <circle class="ln" cx="52.5" cy="34" r="9.15"/><rect class="ln" x="0" y="13.84" width="16.5" height="40.32"/>
      <rect class="ln" x="88.5" y="13.84" width="16.5" height="40.32"/>` +
      evs.filter((e) => e.x != null).map((e) => `<circle cx="${e.x}" cy="${68 - e.y}" r="2.4" fill="#ff4f9a" stroke="#fff" stroke-width=".6"/>`).join("");
  }
  function chime() {
    try {
      const a = chime.ctx || (chime.ctx = new AudioContext()), o = a.createOscillator(), g = a.createGain();
      o.frequency.setValueAtTime(880, a.currentTime); o.frequency.setValueAtTime(1320, a.currentTime + .09);
      g.gain.setValueAtTime(.0001, a.currentTime); g.gain.exponentialRampToValueAtTime(.12, a.currentTime + .02);
      g.gain.exponentialRampToValueAtTime(.0001, a.currentTime + .35);
      o.connect(g).connect(a.destination); o.start(); o.stop(a.currentTime + .4);
    } catch { /* audio is optional */ }
  }
  function addProposal(p, fresh = true) {
    if (cards.has(p.id)) return;
    const node = $("#tpl-proposal").content.firstElementChild.cloneNode(true);
    node.dataset.id = p.id;
    $(".pid", node).textContent = p.id;
    $(".minute", node).textContent = `${p.minute}' · ${p.detector.replaceAll("_", " ")}`;
    $(".score i", node).style.width = (p.score * 100) + "%";
    $(".verified", node).textContent = p.verified ? "✓ VERIFIED" : "✕ NOT VERIFIED";
    $(".verified", node).className = "verified " + (p.verified ? "ok" : "bad");
    $(".verified", node).title = p.checks.join("\n");
    $(".title", node).textContent = `${p.player.first} ${p.player.last} · ${p.value_text} ${p.label.toLowerCase()} ${p.sublabel}`;
    $(".reasons", node).innerHTML = p.reasons.map((r) => `<li>${esc(r)}</li>`).join("");
    $(".why", node).textContent = p.why;
    $(".evs", node).innerHTML = p.evidence.map((e) => `<li><b>${e.minute}'</b> ${esc(TYPE[e.type] || e.type)}</li>`).join("");
    pitch($(".mini", node), p.evidence);
    const col = PLCard.column($(".preview .pl-stage", node), p);
    col.seek(4);  // settled state = what goes to air
    $(".approve", node).onclick = () => decide(p.id, true);
    $(".reject", node).onclick = () => decide(p.id, false);
    $(".editbtn", node).onclick = () => edit(p.id);
    cards.set(p.id, {p, node, col});
    $("#empty").hidden = true;
    const list = $("#pending");
    // strongest first, like the queue on the server
    const after = [...list.querySelectorAll(".prop")].find((n) => cards.get(n.dataset.id).p.score < p.score);
    after ? list.insertBefore(node, after) : list.append(node);
    $("#pendingCount").textContent = cards.size;
    if (fresh) chime();
  }
  function removeProposal(id) {
    const c = cards.get(id);
    if (!c) return;
    cards.delete(id);
    c.node.classList.add("leaving");
    setTimeout(() => c.node.remove(), 300);
    $("#pendingCount").textContent = cards.size;
    $("#empty").hidden = cards.size > 0;
  }
  function addHistory(p, status) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="st ${status}">${status}</span><span>${esc(p.player.last)} · ${esc(p.value_text)} ${esc(p.label.toLowerCase())}
      <small>${p.minute}' · ${esc(p.id)}${p.edited ? " · edited" : ""}</small></span>`;
    $("#history").prepend(li);
  }

  async function decide(id, approve) {
    const c = cards.get(id);
    if (!c) return;
    const ta = $(".edit", c.node), body = !ta.hidden && ta.value !== c.p.why ? {why: ta.value} : null;
    const r = await fetch(`/api/proposals/${id}/${approve ? "approve" : "reject"}`, {method: "POST",
      headers: {"content-type": "application/json"}, body: body ? JSON.stringify(body) : "null"});
    if (!r.ok) {
      const msg = $(".checkmsg", c.node);
      msg.hidden = false; msg.className = "checkmsg bad";
      msg.textContent = "Not approved: " + ((await r.json()).detail || r.statusText);
    }
  }
  let checkT = 0;
  function edit(id) {
    const c = cards.get(id);
    if (!c) return;
    const ta = $(".edit", c.node), why = $(".why", c.node), msg = $(".checkmsg", c.node);
    const open = ta.hidden;
    ta.hidden = !open; why.hidden = open;
    if (!open) { msg.hidden = true; return; }
    ta.value = c.p.why; ta.focus();
    ta.oninput = () => {
      clearTimeout(checkT);
      checkT = setTimeout(async () => {
        const r = await (await fetch(`/api/proposals/${id}/check`, {method: "POST", headers: {"content-type": "application/json"},
          body: JSON.stringify({why: ta.value})})).json();
        msg.hidden = false;
        msg.className = "checkmsg " + (r.verified ? "ok" : "bad");
        const badge = $(".verified", c.node);
        badge.textContent = r.verified ? "✓ VERIFIED (EDITED)" : "✕ EDIT NOT VERIFIED";
        badge.className = "verified " + (r.verified ? "ok" : "bad");
        msg.textContent = r.verified ? "✓ Every number and name is in the evidence" : "✕ " + r.failures.join("; ");
      }, 250);
    };
  }

  // ------------------------------------------------------------------ live channel
  function onMessage(m) {
    if (m.type === "snapshot") {
      teams = m.teams; players = m.players || {};
      $("#home").textContent = teams[0].short; $("#away").textContent = teams[1].short;
      for (const id of [...cards.keys()]) removeProposal(id);
      $("#ticker").innerHTML = ""; $("#history").innerHTML = "";
      m.ticker.slice().reverse().forEach((e) => addEvent(e, false));
      m.pending.forEach((p) => addProposal(p, false));
      m.history.forEach((p) => addHistory(p, p.status));
    }
    if (m.type === "clock" || m.type === "snapshot") setClock(m);
    if (m.type === "event") addEvent(m);
    if (m.type === "proposal") addProposal(m.proposal);
    if (m.type === "status" && cards.has(m.id)) {
      addHistory(cards.get(m.id).p, m.status);
      removeProposal(m.id);
    }
  }
  function connect() {
    const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/producer`);
    ws.onmessage = (ev) => onMessage(JSON.parse(ev.data));
    ws.onclose = () => setTimeout(connect, 1000);
  }
  connect();

  // ------------------------------------------------------------------ controls
  const post = (u) => fetch(u, {method: "POST"});
  $("#pause").onclick = () => post(`/api/session/${clock.running ? "pause" : "resume"}`);
  $("#speed").onchange = (e) => post(`/api/session/speed?x=${e.target.value}`);
  $("#restart").onclick = () => post(`/api/session/start?speed=${$("#speed").value}&start=${$("#startAt").value}`);
  addEventListener("keydown", (e) => {
    if (e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT") return;
    const top = $("#pending .prop:not(.leaving)");
    if (e.key === " ") { e.preventDefault(); $("#pause").click(); }
    if (!top) return;
    if (e.key === "a" || e.key === "A") decide(top.dataset.id, true);
    if (e.key === "r" || e.key === "R") decide(top.dataset.id, false);
    if (e.key === "e" || e.key === "E") { e.preventDefault(); edit(top.dataset.id); }
  });
  // program monitor: scale the 1920x1080 on-air page into its box
  const mon = $(".monitor"), frame = $("#program");
  const fitMon = () => { frame.style.transform = `scale(${mon.clientWidth / 1920})`; };
  addEventListener("resize", fitMon); fitMon();
})();
