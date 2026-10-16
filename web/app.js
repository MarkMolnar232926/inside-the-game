// PitchLens web app: broadcast viewer, live feed, recap. Plain JS, no build step.
/* global OverlayRenderer, applyH */
const $ = (s) => document.querySelector(s);
const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html !== undefined) e.innerHTML = html; return e; };
const fmtClock = (s) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const LANGS = ["en", "hu", "nl", "fr", "es", "de", "pt"];
const getJSON = async (u) => { const r = await fetch(u); if (!r.ok) throw new Error(`${u}: ${r.status}`); return r.json(); };

const S = { profile: "casual", lang: "en", timelines: {}, camera: null, tracks: null, manifest: null, traces: [], labels: {}, profiles: [] };
const qs = new URLSearchParams(location.search);
if (qs.get("lang")) S.lang = qs.get("lang");
if (qs.get("profile")) S.profile = qs.get("profile");

async function init() {
  const [tokens, meta, profiles, health] = await Promise.all([getJSON("/api/tokens"), getJSON("/api/meta"), getJSON("/api/profiles"), getJSON("/api/health")]);
  S.tokens = tokens; S.meta = meta; S.profiles = profiles;
  $("#mode-pill").textContent = `mode: ${health.mode}`;
  const locs = await Promise.all(LANGS.map((l) => getJSON(`/api/locales/${l}`)));
  locs.forEach((L) => { S.labels[L.lang] = L.labels; });
  S.renderer = new OverlayRenderer($("#overlay"), tokens, meta);
  S.liveRenderer = new OverlayRenderer($("#live-canvas"), tokens, meta);
  for (const r of [S.renderer, S.liveRenderer]) for (const l of LANGS) r.setLabels(l, S.labels[l]);
  buildChips("#profiles", profiles.map((p) => [p.id, p.name]), () => S.profile, (v) => { S.profile = v; loadProfile(); });
  buildChips("#live-profiles", profiles.map((p) => [p.id, p.name]), () => S.profile, (v) => { S.profile = v; renderChips(); });
  for (const sel of ["#langs", "#live-langs", "#recap-langs"]) buildChips(sel, LANGS.map((l) => [l, l.toUpperCase()]), () => S.lang, (v) => { S.lang = v; renderChips(); renderRecap(); refreshNow(true); });
  document.querySelectorAll(".tab").forEach((b) => b.onclick = () => showView(b.dataset.view));
  $("#drawer-close").onclick = () => $("#drawer").classList.add("hidden");
  try {
    S.manifest = await getJSON("/api/demo");
    [S.camera, S.tracks, S.traces, S.metrics] = await Promise.all([getJSON("/api/camera"), getJSON("/api/tracks"), getJSON("/api/trace?limit=5000"), getJSON("/api/metrics")]);
    S.useWebm = !$("#video").canPlayType('video/mp4; codecs="avc1.640028"') && S.manifest.video_webm;
    $("#video").src = S.useWebm ? S.manifest.video_webm : S.manifest.video; await loadProfile();
  } catch (e) { console.warn(e); $("#no-video").classList.remove("hidden"); }
  setupTransport(); renderMetrics(); setupLive(); renderRecap();
  if (qs.get("view")) showView(qs.get("view"));
  requestAnimationFrame(loop);
}

function buildChips(sel, items, get, set) {
  const box = $(sel); box.innerHTML = ""; box._get = get;
  for (const [v, label] of items) { const c = el("button", "chip", label); c.dataset.v = v; c.onclick = () => { set(v); renderChips(); }; box.appendChild(c); }
  renderChips();
}
function renderChips() {
  document.querySelectorAll(".chips").forEach((box) => { if (!box._get) return; box.querySelectorAll(".chip").forEach((c) => c.classList.toggle("active", c.dataset.v === box._get())); });
  const p = S.profiles.find((x) => x.id === S.profile); if (p) $("#profile-desc").textContent = p.description;
}
function showView(v) {
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b.dataset.view === v));
  document.querySelectorAll(".view").forEach((s) => s.classList.toggle("active", s.id === `view-${v}`));
}

// ------------------------------------------------------------ broadcast
async function loadProfile() {
  if (!S.timelines[S.profile]) S.timelines[S.profile] = await getJSON(`/api/timeline/${S.profile}`);
  const tl = S.timelines[S.profile];
  const marks = $("#scrub-marks"); marks.innerHTML = "";
  for (const it of tl.items.filter((i) => i.type === "stat_card")) {
    const m = el("div", "scrub-mark"); m.style.left = `${(it.t_start / tl.media.duration_s) * 100}%`; m.title = (it.text.en || {}).headline || ""; marks.appendChild(m);
  }
  const burned = S.manifest && S.manifest.burned && S.manifest.burned[S.profile];
  $("#burned").disabled = !burned; if (!burned) $("#burned").checked = false; applyBurned();
  renderUpNext(); refreshNow(true);
}
function applyBurned() {
  const v = $("#video"), on = $("#burned").checked && S.manifest && S.manifest.burned && S.manifest.burned[S.profile];
  const base = S.useWebm ? S.manifest.video_webm : S.manifest && S.manifest.video;
  const src = on ? S.manifest.burned[S.profile] : base;
  if (src && !v.src.endsWith(src)) { const t = v.currentTime, playing = !v.paused; v.src = src; v.currentTime = t; if (playing) v.play(); }
  $("#overlay").style.visibility = on ? "hidden" : "visible";
}
function setupTransport() {
  const v = $("#video");
  $("#play").onclick = () => (v.paused ? v.play() : v.pause());
  v.onplay = () => ($("#play").textContent = "❚❚"); v.onpause = () => ($("#play").textContent = "▶");
  document.addEventListener("keydown", (e) => { if (e.code === "Space" && e.target === document.body) { e.preventDefault(); $("#play").click(); } });
  $("#scrub").onclick = (e) => { const r = e.currentTarget.getBoundingClientRect(); v.currentTime = ((e.clientX - r.left) / r.width) * (v.duration || 30); };
  $("#burned").onchange = applyBurned;
}
let lastNowKey = "";
function loop() {
  const v = $("#video"), tl = S.timelines[S.profile];
  if (tl && S.camera) {
    const t = v.currentTime || 0, f = Math.min(Math.round(t * S.camera.fps), S.camera.homographies.length - 1);
    const active = S.renderer.draw(tl, t, S.camera.homographies[Math.max(f, 0)], S.tracks, S.lang);
    $("#scrub-fill").style.width = `${(t / (v.duration || tl.media.duration_s)) * 100}%`;
    $("#clock").textContent = fmtClock(t + tl.media.match_t_at_media_zero);
    const key = active.filter((i) => i.insight_id).map((i) => i.insight_id).join("|");
    if (key !== lastNowKey) { lastNowKey = key; refreshNow(); }
    renderTrace(t + tl.media.match_t_at_media_zero);
  }
  requestAnimationFrame(loop);
}
function insightGroups(tl, t) {
  const groups = {};
  for (const it of tl.items) if (it.insight_id && it.t_start <= t && t < it.t_end) (groups[it.insight_id] = groups[it.insight_id] || []).push(it);
  return groups;
}
function refreshNow(force) {
  const tl = S.timelines[S.profile]; if (!tl) return;
  const t = $("#video").currentTime || 0, box = $("#now"); box.innerHTML = "";
  const groups = insightGroups(tl, t);
  if (!Object.keys(groups).length) box.appendChild(el("div", "desc", "Nothing on screen. The Director is waiting for a moment worth showing."));
  for (const [iid, items] of Object.entries(groups)) {
    const card = items.find((i) => i.type === "stat_card"), lower = items.find((i) => i.type === "narrative_lower_third");
    const ttl = card ? `${card.value} · ${(card.text[S.lang] || card.text.en || {}).headline || ""}` : (items[0].type);
    const why = lower ? (lower.text[S.lang] || lower.text.en || {}).body : "";
    const g = items[0].grounding || {};
    const div = el("div", "now-item");
    div.appendChild(el("div", "ttl", ttl)); if (why) div.appendChild(el("div", "why", why));
    const row = el("div", "row");
    row.appendChild(el("span", g.verified ? "badge" : "badge bad", g.verified ? "✓ grounded" : "✗ unverified"));
    row.appendChild(el("span", "badge gen", g.generator || "template"));
    row.appendChild(el("span", "badge gen", items.map((i) => i.type.replace(/_/g, " ")).join(" · ")));
    const b = el("button", "why-btn", "Why am I seeing this?"); b.onclick = () => openDrawer(iid, items); row.appendChild(b);
    div.appendChild(row); box.appendChild(div);
  }
}
function renderUpNext() {
  const tl = S.timelines[S.profile], box = $("#upnext"); box.innerHTML = "";
  for (const it of tl.items.filter((i) => i.type === "stat_card")) {
    const u = el("div", "u", `<b>${fmtClock(it.t_start + tl.media.match_t_at_media_zero)}</b> ${it.value} · ${(it.text[S.lang] || it.text.en || {}).headline || ""}`);
    u.onclick = () => { $("#video").currentTime = it.t_start + 0.6; }; box.appendChild(u);
  }
  if (!box.children.length) box.appendChild(el("div", "desc", "No insight cards for this profile in this clip."));
}
let traceShown = -1;
function renderTrace(tMatch) {
  const vis = S.traces.filter((t) => t.t_match <= tMatch && t.t_match >= tMatch - 90 && !t.action.startsWith("tool:"));
  const style = (S.profiles.find((p) => p.id === S.profile) || {}).style;
  const relevant = vis.filter((t) => (!["personalizer", "director"].includes(t.agent) || t.summary.startsWith(S.profile + ":"))
    && (t.agent !== "explainer" || t.summary.startsWith(`[${S.lang}/${style}]`))).slice(-40);
  const key = `${S.profile}|${S.lang}|${relevant.length}`; if (key === traceShown) return; traceShown = key;
  const ol = $("#trace"); ol.innerHTML = "";
  for (const t of relevant.reverse()) {
    const li = el("li", "", `<span class="mono">${fmtClock(t.t_match)}</span><span class="agent ${t.agent}">${t.agent.replace("_", " ")}</span><span><b>${t.action}</b> ${escapeHtml(t.summary)}</span>`);
    ol.appendChild(li);
  }
}
function renderMetrics() {
  const m = S.metrics || {}, box = $("#metrics"); box.innerHTML = "";
  const rows = [["events", m.events], ["insights proposed", m.insights_proposed], ["insights shown", m.insights_shown], ["expired / held", m.insights_expired],
    ["rejected by verifier", (m.template_rejected || 0) + (m.llm_rejected || 0)], ["decision p99", m.latency_ms_p99 !== undefined ? `${m.latency_ms_p99} ms` : "–"]];
  for (const [k, v] of rows) box.appendChild(el("div", "metric", `<div class="v">${v ?? "–"}</div><div class="k">${k}</div>`));
}
const escapeHtml = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

async function openDrawer(iid, items) {
  const [ev, tr] = await Promise.all([getJSON(`/api/evidence/${encodeURIComponent(iid)}`), getJSON(`/api/trace?insight_id=${encodeURIComponent(iid)}`)]);
  const ins = ev.insight, f = ins.evidence.facts, markers = items.find((i) => i.type === "pitch_markers");
  const pts = markers ? markers.points : ins.evidence.points;
  const body = $("#drawer-body"); body.innerHTML = "";
  body.appendChild(el("p", "desc", `<b>${ins.detector}</b> found <b>${ins.value_text}</b> (${ins.metric}) at ${fmtClock(ins.t)}. Every number on screen comes from the facts below; the Verifier checks this before anything renders.`));
  body.appendChild(el("h4", "", "Evidence on the pitch"));
  body.appendChild(pitchSVG(pts));
  body.appendChild(el("h4", "", `Events (${ev.events.length})`));
  const tb = el("table"); tb.innerHTML = "<tr><th>time</th><th>event</th><th>player</th><th>x, y (m)</th></tr>" + ev.events.slice(-12).map((e) =>
    `<tr><td class="mono">${fmtClock(e.t)}</td><td>${e.type}${e.outcome !== "na" ? " · " + e.outcome : ""}</td><td>${e.player_id || ""}</td><td class="mono">${e.x?.toFixed(1)}, ${e.y?.toFixed(1)}</td></tr>`).join("");
  body.appendChild(tb);
  body.appendChild(el("h4", "", "Grounded facts"));
  const facts = el("div", "facts"); for (const [k, v] of Object.entries(f)) if (!Array.isArray(v)) facts.appendChild(el("span", "fact", `${k}: <b>${escapeHtml(v)}</b>`)); body.appendChild(facts);
  const g = items[0].grounding || {};
  body.appendChild(el("h4", "", "Verifier"));
  body.appendChild(el("div", "facts", [...(g.checks || []).map((c) => `<span class="badge">✓ ${c}</span>`), ...(g.failures || []).map((c) => `<span class="badge bad">✗ ${c}</span>`)].join(" ")));
  body.appendChild(el("h4", "", "Agent chain"));
  const ol = el("ol", "trace"); ol.style.maxHeight = "none";
  const style = (S.profiles.find((p) => p.id === S.profile) || {}).style;
  for (const t of tr.filter((t) => (!["personalizer", "director"].includes(t.agent) || t.summary.startsWith(S.profile + ":"))
    && (t.agent !== "explainer" || t.summary.startsWith(`[${S.lang}/${style}]`)))) ol.appendChild(el("li", "", `<span class="mono">${fmtClock(t.t_match)}</span><span class="agent ${t.agent}">${t.agent.replace("_", " ")}</span><span><b>${t.action}</b> ${escapeHtml(t.summary)}</span>`));
  body.appendChild(ol);
  $("#drawer").classList.remove("hidden");
}
function pitchSVG(points) {
  const s = 5, NS = "http://www.w3.org/2000/svg", svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `-4 -4 ${105 * s + 8} ${68 * s + 8}`); svg.setAttribute("class", "pitchmap");
  const P = (x, y) => [x * s, (68 - y) * s];
  const path = (pts, attrs) => { const p = document.createElementNS(NS, "polyline"); p.setAttribute("points", pts.map(([x, y]) => P(x, y).join(",")).join(" "));
    for (const [k, v] of Object.entries(attrs)) p.setAttribute(k, v); svg.appendChild(p); };
  const L = 105, cy = 34;
  path([[0, 0], [L, 0], [L, 68], [0, 68], [0, 0]], { fill: "none", stroke: "#d9f2df", "stroke-width": 1.5 }); path([[L / 2, 0], [L / 2, 68]], { fill: "none", stroke: "#d9f2df", "stroke-width": 1.5 });
  for (const [x0, sg] of [[0, 1], [L, -1]]) path([[x0, cy - 20.16], [x0 + sg * 16.5, cy - 20.16], [x0 + sg * 16.5, cy + 20.16], [x0, cy + 20.16]], { fill: "none", stroke: "#d9f2df", "stroke-width": 1.5 });
  const c = document.createElementNS(NS, "circle"); const [ccx, ccy] = P(L / 2, cy); c.setAttribute("cx", ccx); c.setAttribute("cy", ccy); c.setAttribute("r", 9.15 * s);
  c.setAttribute("fill", "none"); c.setAttribute("stroke", "#d9f2df"); c.setAttribute("stroke-width", 1.5); svg.appendChild(c);
  points.forEach((p, i) => { const [x, y] = P(p.x, p.y); const g = document.createElementNS(NS, "g");
    g.innerHTML = `<circle cx="${x}" cy="${y}" r="9" fill="#c21f3f" stroke="#fff" stroke-width="2"/><text x="${x}" y="${y + 4}" font-size="11" font-weight="900" text-anchor="middle" fill="#fff">${i + 1}</text>`;
    svg.appendChild(g); });
  return svg;
}

// ----------------------------------------------------------------- live
function setupLive() {
  const cv = $("#live-canvas"), s = 17, ox = (1920 - 105 * s) / 2, oy = (1240 - 68 * s) / 2;
  S.liveH = [s, 0, ox, 0, -s, oy + 68 * s, 0, 0, 1];
  S.live = { items: [], players: {}, events: [], t: 0 };
  drawLivePitch();
  $("#live-start").onclick = () => {
    if (S.ws) { S.ws.close(); S.ws = null; $("#live-start").textContent = "Start live feed"; return; }
    S.live = { items: [], players: {}, events: [], t: 0 };
    $("#live-events").innerHTML = ""; $("#live-trace").innerHTML = ""; $("#live-overlays").innerHTML = "";
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws/live?profile=${S.profile}&speed=${$("#live-speed").value}&start=${$("#live-min").value * 60}`);
    S.ws = ws; $("#live-start").textContent = "Stop";
    ws.onmessage = (m) => onLive(JSON.parse(m.data));
    ws.onclose = () => { S.ws = null; $("#live-start").textContent = "Start live feed"; };
  };
  const tick = () => { drawLive(); requestAnimationFrame(tick); }; tick();
  void cv;
}
function onLive(msg) {
  const L = S.live; L.t = msg.t; L.wall = performance.now();
  if (msg.players) for (const [id, x, y] of msg.players) L.players[id] = [x, y];
  if (msg.event) {
    const e = msg.event; if (e.x !== null && !["pressure", "possession_change"].includes(e.type)) L.events.push({ ...e, t: msg.t });
    if (!["pressure"].includes(e.type)) { const li = el("li", "", `<span class="mono">${fmtClock(msg.t)}</span><span class="agent">${e.type}</span><span>${e.player_id || e.team_id || ""} ${e.outcome !== "na" ? "· " + e.outcome : ""} <b>${e.score.join("-")}</b></span>`);
      $("#live-events").prepend(li); while ($("#live-events").children.length > 60) $("#live-events").lastChild.remove(); }
  }
  if (msg.overlays) {
    L.items.push(...msg.overlays); $("#live-latency").textContent = `event → overlay ${msg.latency_ms} ms`;
    const card = msg.overlays.find((i) => i.type === "stat_card") || msg.overlays[0], lower = msg.overlays.find((i) => i.type === "narrative_lower_third");
    const d = el("div", "now-item", `<div class="ttl">${fmtClock(msg.t)} · ${card.value || ""} ${((card.text || {})[S.lang] || (card.text || {}).en || {}).headline || card.type}</div>` +
      (lower ? `<div class="why">${(lower.text[S.lang] || lower.text.en).body}</div>` : "") + `<div class="row">${msg.overlays.map((i) => `<span class="badge gen">${i.type}</span>`).join("")}</div>`);
    $("#live-overlays").prepend(d);
  }
  if (msg.traces) for (const t of msg.traces) { const li = el("li", "flash", `<span class="mono">${fmtClock(t.t_match)}</span><span class="agent ${t.agent}">${t.agent.replace("_", " ")}</span><span><b>${t.action}</b> ${escapeHtml(t.summary)}</span>`);
    $("#live-trace").prepend(li); setTimeout(() => li.classList.remove("flash"), 900); while ($("#live-trace").children.length > 80) $("#live-trace").lastChild.remove(); }
  if (msg.metrics) { const box = $("#live-metrics"); box.innerHTML = "";
    for (const [k, v] of [["events / s", msg.metrics.events_per_s], ["proposed", msg.metrics.insights_proposed], ["shown", msg.metrics.insights_shown], ["expired", msg.metrics.insights_expired]])
      box.appendChild(el("div", "metric", `<div class="v">${v ?? 0}</div><div class="k">${k}</div>`)); }
  $("#live-clock").textContent = fmtClock(msg.t);
}
function drawLivePitch() {
  const ctx = $("#live-canvas").getContext("2d"), H = S.liveH;
  ctx.fillStyle = "#0e4a26"; ctx.fillRect(0, 0, 1920, 1240);
  for (let k = 0; k < 18; k++) { if (k % 2) continue; const [x0] = applyH(H, k * 105 / 18, 0), [x1] = applyH(H, (k + 1) * 105 / 18, 0); ctx.fillStyle = "#11542b"; ctx.fillRect(x0, 0, x1 - x0, 1240); }
  ctx.strokeStyle = "#e6f5ea"; ctx.lineWidth = 3;
  const line = (pts) => { ctx.beginPath(); pts.forEach(([x, y], i) => { const [u, v] = applyH(H, x, y); i ? ctx.lineTo(u, v) : ctx.moveTo(u, v); }); ctx.stroke(); };
  line([[0, 0], [105, 0], [105, 68], [0, 68], [0, 0]]); line([[52.5, 0], [52.5, 68]]);
  for (const [x0, s] of [[0, 1], [105, -1]]) { line([[x0, 13.84], [x0 + s * 16.5, 13.84], [x0 + s * 16.5, 54.16], [x0, 54.16]]); line([[x0, 24.84], [x0 + s * 5.5, 24.84], [x0 + s * 5.5, 43.16], [x0, 43.16]]); }
  line(Array.from({ length: 65 }, (_, k) => [52.5 + 9.15 * Math.cos(k / 64 * 2 * Math.PI), 34 + 9.15 * Math.sin(k / 64 * 2 * Math.PI)]));
}
function drawLive() {
  const L = S.live; if (!L) return; drawLivePitch();
  const ctx = $("#live-canvas").getContext("2d"), H = S.liveH;
  const tNow = L.t + (S.ws && L.wall ? ((performance.now() - L.wall) / 1000) * +$("#live-speed").value * 0 : 0);
  for (const e of L.events.filter((e) => tNow - e.t < 12)) {
    const [u, v] = applyH(H, e.x, e.y), a = 1 - (tNow - e.t) / 12; ctx.globalAlpha = a;
    ctx.fillStyle = e.team_id === S.meta.home.id ? S.meta.home.kit.shirt : S.meta.away.primary; ctx.beginPath(); ctx.arc(u, v, 7, 0, 2 * Math.PI); ctx.fill();
  }
  ctx.globalAlpha = 1;
  const tracks = { fps: 30, tracks: {} };
  for (const [id, [x, y]] of Object.entries(L.players)) {
    const [u, v] = applyH(H, x, y); const team = id.startsWith(S.meta.home.id) ? S.meta.home : S.meta.away;
    ctx.fillStyle = team.kit.shirt; ctx.strokeStyle = "#0008"; ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(u, v, 13, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    ctx.fillStyle = team.kit.number; ctx.font = "900 13px Inter"; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(id.split("-")[1].replace(/^0/, ""), u, v + 1);
    tracks.tracks[id] = { frames: [Math.round(tNow * 30)], boxes: [[u - 13, v - 16, u + 13, v + 13]], pitch_xy: [[x, y]], speed_kmh: [0] };
  }
  const tl = { media: { fps: 30 }, items: L.items.filter((i) => i.t_end > tNow - 1) };
  S.liveRenderer.draw(tl, tNow, H, tracks, S.lang, { clear: false, filter: (i) => i.type !== "minimap" });
}

// ---------------------------------------------------------------- recap
async function renderRecap() {
  if (!S.recap) { try { S.recap = await getJSON("/api/recap"); } catch (e) { return; } }
  const r = S.recap, L = S.lang, m = S.meta; if (!r || !r.headline) return;
  const potm = Object.values(m.home.players).concat(m.away.players).find((p) => p.id === r.player_of_the_match.player_id);
  const lab = S.labels[L] || S.labels.en;
  $("#recap-card").innerHTML = `
    <div class="comp">${m.competition} · ${lab.full_time} · ${m.venue}</div>
    <h1>${r.headline[L]}</h1>
    <div class="score"><img src="/assets/generated/crest_${m.home.id}.png"/><span class="n">${r.final_score[0]} – ${r.final_score[1]}</span><img src="/assets/generated/crest_${m.away.id}.png"/></div>
    <div class="sum">${r.summary[L]}</div>
    <h4 style="margin:22px 0 4px;color:#c4cdfc;text-transform:uppercase;font-size:12px;letter-spacing:1px">${lab.key_moments}</h4>
    <ul class="moments">${r.moments.map((mo) => `<li><span class="m">${mo.minute}'</span><span>${mo.text[L]}</span></li>`).join("")}</ul>
    <div class="potm"><img src="/assets/generated/portrait_${r.player_of_the_match.player_id}.png"/>
      <div><div class="t">${lab.player_of_the_match}</div><div class="n">${potm ? potm.name : ""}</div><div class="sum">${r.potm_text[L]}</div></div></div>
    <div style="margin-top:14px;font-size:12px;color:#9aa3d6">${r.verified ? "✓ every number verified against match data" : "✗ verification failed"} · synthetic match ${r.match_id}</div>`;
  $("#recap-png").href = `/media/recap_${L}.png`;
  $("#recap-share").onclick = async () => { const u = `${location.origin}/?view=recap&lang=${L}`; try { await navigator.clipboard.writeText(u); } catch (e) { /* ignore */ } $("#recap-share").textContent = "Link copied ✓"; setTimeout(() => ($("#recap-share").textContent = "Copy share link"), 1500); };
}

init();
