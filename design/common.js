// Shared helpers for the static design comps. Every pitch-anchored shape comes from
// out/scene.js (projected through the frame's homography), so it sits on the real grass.
(() => {
const S = window.SCENE;
const NS = "http://www.w3.org/2000/svg";

function svgEl(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}

function stage() {
  const root = document.getElementById("stage");
  const img = document.createElement("img");
  img.src = "out/frame.jpg";
  img.className = "bg";
  root.appendChild(img);
  const svg = svgEl("svg", { viewBox: "0 0 1920 1080", class: "pitch", width: 1920, height: 1080 });
  root.appendChild(svg);
  return { root, svg, defs: svgEl("defs", {}, svg) };
}

function html(parent, markup) {
  const t = document.createElement("template");
  t.innerHTML = markup.trim();
  const n = t.content.firstElementChild;
  parent.appendChild(n);
  return n;
}

// pitch (x, y) -> minimap px, far touchline (y = 68) at the top, attacking to the right
function mini(x, y, w, h, pad = 0) {
  return [pad + (x / 105) * (w - 2 * pad), pad + ((68 - y) / 68) * (h - 2 * pad)];
}

function minimapSVG(w, h, o = {}) {
  const line = o.line || "rgba(255,255,255,.55)", pad = o.pad ?? 8;
  const P = (x, y) => mini(x, y, w, h, pad).map((v) => v.toFixed(1)).join(",");
  const rect = (x0, y0, x1, y1) => `M${P(x0, y0)} L${P(x1, y0)} L${P(x1, y1)} L${P(x0, y1)} Z`;
  const cx = mini(52.5, 34, w, h, pad);
  const r = (9.15 / 105) * (w - 2 * pad);
  let s = `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" xmlns="${NS}">`;
  if (o.zone) s += `<path d="${rect(70, 0, 105, 68)}" fill="${o.zone}"/>`;
  if (o.frustum) s += `<path d="M${S.frustum.map(([x, y]) => P(Math.max(-3, Math.min(108, x)), Math.max(-3, Math.min(71, y)))).join(" L")} Z" fill="${o.frustum}" stroke="${o.frustumStroke || "none"}" stroke-width="1"/>`;
  s += `<g fill="none" stroke="${line}" stroke-width="1.2">
    <path d="${rect(0, 0, 105, 68)}"/><path d="M${P(52.5, 0)} L${P(52.5, 68)}"/>
    <circle cx="${cx[0]}" cy="${cx[1]}" r="${r}"/>
    <path d="${rect(0, 13.84, 16.5, 54.16)}"/><path d="${rect(88.5, 13.84, 105, 54.16)}"/>
    <path d="${rect(0, 24.84, 5.5, 43.16)}"/><path d="${rect(99.5, 24.84, 105, 43.16)}"/></g>`;
  if (o.dots !== false) for (const d of S.dots) {
    const [x, y] = mini(d.x, d.y, w, h, pad);
    const hero = Math.abs(d.x - S.player.pitch[0]) < 0.05 && Math.abs(d.y - S.player.pitch[1]) < 0.05;
    s += `<circle cx="${x}" cy="${y}" r="${hero ? 5 : 3.4}" fill="${d.team === 1 ? o.home || "#e0233f" : o.away || "#f2f2f2"}" ${hero ? `stroke="${o.heroStroke || "#fff"}" stroke-width="2"` : ""}/>`;
  }
  if (o.markers) for (const m of S.markers) {
    const [x, y] = mini(m.x, m.y, w, h, pad);
    s += o.markers(x, y, m);
  }
  return s + "</svg>";
}

window.DESIGN = { S, svgEl, stage, html, mini, minimapSVG };
})();
