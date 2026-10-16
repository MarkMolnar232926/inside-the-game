// PitchLens canvas overlay renderer.
// A second, independent consumer of the overlay-timeline JSON contract (the Python compositor is the first).
// Draws pitch-anchored graphics through the per-frame homography so they stay glued to the grass.
/* exported OverlayRenderer, applyH */

function applyH(H, x, y) {
  const w = H[6] * x + H[7] * y + H[8];
  return [(H[0] * x + H[1] * y + H[2]) / w, (H[3] * x + H[4] * y + H[5]) / w];
}
function invertH(H) {
  const [a, b, c, d, e, f, g, h, i] = H;
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  return [A / det, -(b * i - c * h) / det, (b * f - c * e) / det,
          B / det, (a * i - c * g) / det, -(a * f - c * d) / det,
          C / det, -(a * h - b * g) / det, (a * e - b * d) / det];
}
const clamp = (u) => Math.min(Math.max(u, 0), 1);
const easeOut = (u) => 1 - Math.pow(1 - clamp(u), 3);
const easeIn = (u) => Math.pow(clamp(u), 3);
const easeBack = (u) => { u = clamp(u) - 1; const s = 1.9; return 1 + u * u * ((s + 1) * u + s); };

const PITCH_LINES = (() => {
  const L = 105, W = 68, cy = 34, lines = [];
  const arc = (cx, cy0, r, a0, a1, n) => Array.from({ length: n }, (_, k) => {
    const a = a0 + (a1 - a0) * k / (n - 1); return [cx + r * Math.cos(a), cy0 + r * Math.sin(a)];
  });
  lines.push([[0, 0], [L, 0], [L, W], [0, W], [0, 0]], [[L / 2, 0], [L / 2, W]], arc(L / 2, cy, 9.15, 0, 2 * Math.PI, 64));
  for (const [x0, s] of [[0, 1], [L, -1]]) {
    lines.push([[x0, cy - 20.16], [x0 + s * 16.5, cy - 20.16], [x0 + s * 16.5, cy + 20.16], [x0, cy + 20.16]]);
    lines.push([[x0, cy - 9.16], [x0 + s * 5.5, cy - 9.16], [x0 + s * 5.5, cy + 9.16], [x0, cy + 9.16]]);
    const half = Math.acos(5.5 / 9.15), base = s === 1 ? 0 : Math.PI;
    lines.push(arc(x0 + s * 11, cy, 9.15, base - half, base + half, 24));
  }
  return lines;
})();

class OverlayRenderer {
  constructor(canvas, tokens, meta) {
    this.canvas = canvas; this.ctx = canvas.getContext("2d");
    this.tok = tokens; this.c = tokens.colors; this.lay = tokens.layout; this.mo = tokens.motion;
    this.meta = meta; this.images = {}; this.labels = {};
    this.players = {}; for (const t of [meta.home, meta.away]) for (const p of t.players) this.players[p.id] = { ...p, team: t };
  }
  setLabels(lang, labels) { this.labels[lang] = labels; }
  label(key, lang) { return (this.labels[lang] && this.labels[lang][key]) || (this.labels.en && this.labels.en[key]) || key; }
  img(src) {
    if (!this.images[src]) { const im = new Image(); im.src = src; this.images[src] = im; }
    const im = this.images[src]; return im.complete && im.naturalWidth ? im : null;
  }
  text(item, lang) { const t = item.text || {}; return t[lang] || t.en || Object.values(t)[0] || null; }

  // --------------------------------------------------------------- frame
  draw(timeline, t, H, tracks, lang, opts = {}) {
    const ctx = this.ctx, W = this.canvas.width, Hh = this.canvas.height;
    this.s = Hh / 1080; if (opts.clear !== false) ctx.clearRect(0, 0, W, Hh);
    this.frame = Math.round(t * (timeline.media.fps || 30)); this.H = H; this.tracks = tracks; this.lang = lang; this.t = t;
    const order = { pitch_markers: 0, pass_line: 1, player_highlight: 2, speed_tag: 3, score_bug: 4, minimap: 5,
      stat_card: 6, narrative_lower_third: 7, recap_card: 8 };
    const active = timeline.items.filter((i) => i.t_start <= t && t < i.t_end && (!opts.filter || opts.filter(i)))
      .sort((a, b) => (order[a.type] ?? 9) - (order[b.type] ?? 9));
    for (const it of active) { const fn = this["draw_" + it.type]; if (fn) fn.call(this, it); }
    return active;
  }
  px(v) { return v * this.s; }
  env(it, tin, tout) {
    return [easeOut((this.t - it.t_start) / tin), easeIn((this.t - (it.t_end - tout)) / tout)];
  }
  font(weight, size) { return `${weight} ${Math.round(size)}px Inter, system-ui, sans-serif`; }
  groundPath(pts) {
    const ctx = this.ctx; ctx.beginPath();
    pts.forEach(([x, y], i) => { const [u, v] = applyH(this.H, x, y); i ? ctx.lineTo(u, v) : ctx.moveTo(u, v); });
  }
  ribbon(pts, width) { // pitch polyline -> polygon of real width (metres)
    const L = [], R = [];
    for (let i = 0; i < pts.length; i++) {
      const a = pts[Math.max(i - 1, 0)], b = pts[Math.min(i + 1, pts.length - 1)];
      let dx = b[0] - a[0], dy = b[1] - a[1]; const n = Math.hypot(dx, dy) || 1; dx /= n; dy /= n;
      L.push([pts[i][0] - dy * width / 2, pts[i][1] + dx * width / 2]); R.push([pts[i][0] + dy * width / 2, pts[i][1] - dx * width / 2]);
    }
    return L.concat(R.reverse());
  }
  circle(cx, cy, r, n = 40, a0 = 0, a1 = 2 * Math.PI) {
    return Array.from({ length: n }, (_, k) => { const a = a0 + (a1 - a0) * k / (n - 1); return [cx + r * Math.cos(a), cy + r * Math.sin(a)]; });
  }
  star(cx, cy, r, inner = 0.45) {
    return Array.from({ length: 10 }, (_, k) => { const rr = k % 2 ? r * inner : r, a = -Math.PI / 2 + k * Math.PI / 5;
      return [cx + rr * Math.cos(a), cy + rr * Math.sin(a)]; });
  }
  roundRect(x, y, w, h, r) { const c = this.ctx; c.beginPath(); c.roundRect ? c.roundRect(x, y, w, h, r) : c.rect(x, y, w, h); }

  // ----------------------------------------------------------- score bug
  draw_score_bug(it) {
    const ctx = this.ctx; let home = 0, away = 0;
    for (const [mt, h, a] of it.score_track) if (this.t >= mt) { home = h; away = a; }
    const clock = this.t + it.clock_offset, mm = Math.floor(clock / 60), ss = Math.floor(clock % 60);
    const h = this.px(this.lay.bug_h), x0 = this.canvas.width * this.lay.safe_x, y = this.canvas.height * this.lay.safe_y;
    ctx.font = this.font(800, this.px(this.lay.bug_font_px));
    const tw = Math.max(ctx.measureText(it.home_short).width, ctx.measureText(it.away_short).width) + h * 0.64;
    const ws = h * 0.95, wc = ctx.measureText("00:00").width + h * 0.64, stripe = h * 0.12;
    const W = stripe + 2 * tw + 2 * ws + wc;
    ctx.save(); ctx.globalAlpha = 0.94; ctx.fillStyle = this.c.bug_bg; this.roundRect(x0, y, W, h, h * 0.14); ctx.fill();
    ctx.globalAlpha = 1; ctx.fillStyle = it.home_color; ctx.fillRect(x0, y, stripe, h);
    let x = x0 + stripe; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillStyle = "#fff";
    ctx.fillText(it.home_short, x + tw / 2, y + h / 2); x += tw;
    ctx.fillStyle = it.home_color; ctx.fillRect(x, y, ws, h); ctx.fillStyle = "#fff"; ctx.fillText(home, x + ws / 2, y + h / 2); x += ws;
    ctx.fillStyle = it.away_color; ctx.fillRect(x, y, ws, h); ctx.fillStyle = "#fff"; ctx.fillText(away, x + ws / 2, y + h / 2); x += ws;
    ctx.fillText(it.away_short, x + tw / 2, y + h / 2); x += tw;
    ctx.fillStyle = this.c.bug_clock_bg; this.roundRect(x, y, wc, h, h * 0.14); ctx.fill(); ctx.fillRect(x, y, h * 0.2, h);
    ctx.fillStyle = "#fff"; ctx.font = this.font(600, this.px(this.lay.bug_font_px - 2));
    ctx.fillText(`${String(mm).padStart(2, "0")}:${String(ss).padStart(2, "0")}`, x + wc / 2, y + h / 2);
    ctx.restore();
  }

  // ----------------------------------------------------------- stat card
  draw_stat_card(it) {
    const ctx = this.ctx, L = this.lay, tb = this.text(it, this.lang) || {};
    const Wp = this.px(L.panel_w), margin = this.canvas.width * L.safe_x;
    const xf = it.placement === "right" ? this.canvas.width - margin - Wp : margin;
    const blocks = [["header", this.px(L.panel_header_h)]];
    if (it.player_id && it.density !== "minimal") blocks.push(["portrait", this.px(L.panel_portrait_h) * (it.density === "dense" ? 0.72 : 1)]);
    if (it.player_name || it.team_name) blocks.push(["plate", this.px(L.panel_plate_h)]);
    if (it.density === "dense" && it.extra_metrics && it.extra_metrics.length) blocks.push(["chips", this.px(64) * Math.ceil(it.extra_metrics.length / 2)]);
    let y = this.px(L.panel_top);
    blocks.forEach(([kind, h], k) => {
      const [fin, fout] = [easeOut((this.t - it.t_start - k * this.mo.stagger_s) / this.mo.slide_in_s),
        easeIn((this.t - (it.t_end - (blocks.length - 1 - k) * this.mo.stagger_s * 0.6 - this.mo.slide_out_s)) / this.mo.slide_out_s)];
      const off = (1 - fin) * (Wp + margin) + fout * (Wp + margin);
      const x = it.placement === "right" ? xf + off : xf - off;
      ctx.save(); ctx.globalAlpha = Math.min(1, fin * 1.4) * (1 - fout * 0.6);
      this["card_" + kind](it, tb, x, y, Wp, h); ctx.restore();
      y += h + (k === 0 ? this.px(L.panel_gap) : this.px(L.panel_gap) / 2);
    });
  }
  grad(x, y, w, h, a, b) {
    const g = this.ctx.createLinearGradient(x, y, x + w * 0.35, y + h); g.addColorStop(0, a); g.addColorStop(1, b); return g;
  }
  card_header(it, tb, x, y, w, h) {
    const ctx = this.ctx; ctx.fillStyle = this.grad(x, y, w, h, this.c.panel_header_from, this.c.panel_header_to); ctx.fillRect(x, y, w, h);
    const pad = w * 0.06; let size = this.px(this.lay.big_number_px);
    ctx.font = this.font(900, size); while (ctx.measureText(it.value).width > w * 0.42 && size > 30) { size *= 0.9; ctx.font = this.font(900, size); }
    const vw = ctx.measureText(it.value).width; ctx.fillStyle = "#fff"; ctx.textBaseline = "middle"; ctx.textAlign = "left";
    ctx.fillText(it.value, x + pad, y + h / 2 + size * 0.04);
    let hs = this.px(this.lay.headline_px); const maxw = w - pad * 3 - vw; let lines;
    do { ctx.font = this.font(800, hs); lines = wrapText(ctx, (tb.headline || "").toUpperCase(), maxw); hs *= 0.92; } while (lines.length > 3 && hs > 12);
    const lh = hs / 0.92 * 1.08, y0 = y + h / 2 - lh * (lines.length - 1) / 2;
    lines.forEach((l, i) => ctx.fillText(l, x + pad * 2 + vw, y0 + i * lh));
  }
  card_portrait(it, tb, x, y, w, h) {
    const ctx = this.ctx; ctx.fillStyle = this.grad(x, y, w, h, this.c.portrait_from, this.c.portrait_to); ctx.fillRect(x, y, w, h);
    const im = this.img(`/assets/generated/portrait_${it.player_id}.png`);
    if (im) { const s = Math.min(w * 0.95 / im.width, h * 0.96 / im.height); ctx.drawImage(im, x + (w - im.width * s) / 2, y + h - im.height * s, im.width * s, im.height * s); }
  }
  card_plate(it, tb, x, y, w, h) {
    const ctx = this.ctx, pad = w * 0.06; ctx.fillStyle = this.c.plate; ctx.fillRect(x, y, w, h);
    ctx.fillStyle = this.c.plate_edge; ctx.fillRect(x, y, w, h * 0.04);
    let tw = w - 2 * pad;
    if (it.team_id) {
      const cb = h * 0.78, cx = x + w - pad - cb; ctx.fillStyle = "#fff"; this.roundRect(cx, y + (h - cb) / 2, cb, cb, cb * 0.08); ctx.fill();
      const cr = this.img(`/assets/generated/crest_${it.team_id}.png`); if (cr) ctx.drawImage(cr, cx + cb * 0.08, y + (h - cb) / 2 + cb * 0.08, cb * 0.84, cb * 0.84);
      tw = cx - x - pad * 1.6;
    }
    ctx.textBaseline = "alphabetic"; ctx.textAlign = "left"; ctx.fillStyle = "#fff";
    if (it.player_name) {
      const [first, ...rest] = it.player_name.split(" "), num = it.player_number ? `#${it.player_number} ` : "";
      let fs = this.px(this.lay.plate_name_px);
      const measure = () => { ctx.font = this.font(900, fs); let a = ctx.measureText(num).width + ctx.measureText(rest.join(" ").toUpperCase()).width; ctx.font = this.font(500, fs); return a + ctx.measureText(first.toUpperCase() + " ").width; };
      while (measure() > tw && fs > 10) fs *= 0.92;
      let xx = x + pad; const yy = y + h * 0.47;
      ctx.font = this.font(900, fs); ctx.fillText(num, xx, yy); xx += ctx.measureText(num).width;
      ctx.font = this.font(500, fs); ctx.fillText(first.toUpperCase() + " ", xx, yy); xx += ctx.measureText(first.toUpperCase() + " ").width;
      ctx.font = this.font(900, fs); ctx.fillText(rest.join(" ").toUpperCase(), xx, yy);
    }
    if (it.team_name) { ctx.font = this.font(800, this.px(this.lay.plate_team_px)); ctx.fillStyle = this.c.text_dim; ctx.fillText(it.team_name.toUpperCase(), x + pad, y + h * 0.82); }
  }
  card_chips(it, tb, x, y, w, h) {
    const ctx = this.ctx; ctx.fillStyle = this.c.chip_bg; ctx.globalAlpha *= 0.94; ctx.fillRect(x, y, w, h);
    const cw = w / 2, chh = this.px(64);
    it.extra_metrics.forEach((ch, i) => {
      const cx = x + (i % 2) * cw, cy = y + Math.floor(i / 2) * chh;
      ctx.fillStyle = this.c.plate_edge; ctx.fillRect(cx, cy, cw, 1);
      ctx.fillStyle = this.c.text_dim; ctx.font = this.font(600, this.px(17)); ctx.textBaseline = "top";
      ctx.fillText(this.label(ch.label, this.lang).toUpperCase(), cx + this.px(16), cy + this.px(8));
      ctx.fillStyle = "#fff"; ctx.font = this.font(900, this.px(28)); ctx.fillText(ch.value + (ch.unit || ""), cx + this.px(16), cy + this.px(28));
    });
  }

  // ------------------------------------------------------- pitch markers
  draw_pitch_markers(it) {
    const ctx = this.ctx, r = this.lay.marker_radius_m, col = this.c[it.color] || it.color || this.c.marker;
    const fadeOut = easeIn((this.t - (it.t_end - this.mo.fade_s)) / this.mo.fade_s);
    const shown = [];
    it.points.forEach((p, k) => {
      const ta = it.t_start + (p.appear_at || 0); if (this.t < ta) return;
      const u = (this.t - ta) / this.mo.marker_pop_s, pop = u < 1 ? easeBack(u) : 1;
      const pulse = 1 + 0.07 * Math.sin(2 * Math.PI * this.mo.marker_pulse_hz * (this.t - ta)) * Math.min(u, 1);
      shown.push({ k, p, ta, scale: Math.max(pop, 0) * pulse * (1 - 0.3 * fadeOut), alpha: Math.min(1, u * 2) * (1 - fadeOut) });
    });
    if (it.connect && shown.length > 1) {
      ctx.save(); ctx.globalAlpha = 0.55 * Math.min(...shown.map((s) => s.alpha)); ctx.fillStyle = "#fff";
      for (let i = 1; i < shown.length; i++) {
        const a = shown[i - 1].p, b = shown[i].p, n = Math.max(2, Math.floor(Math.hypot(b.x - a.x, b.y - a.y) / 0.8));
        for (let j = 0; j < n; j += 2) {
          const u0 = j / n, u1 = Math.min((j + 1) / n, 1);
          this.groundPath(this.ribbon([[a.x + (b.x - a.x) * u0, a.y + (b.y - a.y) * u0], [a.x + (b.x - a.x) * u1, a.y + (b.y - a.y) * u1]], 0.16)); ctx.fill();
        }
      }
      ctx.restore();
    }
    for (const s of shown) {
      if (s.scale < 0.01) continue;
      const { p } = s; ctx.save();
      const u = (this.t - s.ta) / 0.9;
      if (u < 1) { ctx.globalAlpha = s.alpha * (1 - u) * 0.9; ctx.fillStyle = this.c.marker_glow;
        this.groundPath(this.ribbon(this.circle(p.x, p.y, r * (1 + 1.6 * u)), 0.12 + 0.2 * (1 - u))); ctx.fill(); }
      const shape = it.shape === "star" ? this.star(p.x, p.y, r * s.scale) : it.shape === "ring"
        ? this.ribbon(this.circle(p.x, p.y, r * s.scale), 0.25) : this.circle(p.x, p.y, r * 0.6 * s.scale, 28);
      ctx.globalAlpha = s.alpha * 0.25; ctx.fillStyle = "#000"; ctx.translate(0, 2 * this.s); this.groundPath(shape); ctx.closePath(); ctx.fill(); ctx.translate(0, -2 * this.s);
      ctx.globalAlpha = s.alpha; ctx.fillStyle = col; this.groundPath(shape); ctx.closePath(); ctx.fill();
      ctx.strokeStyle = this.c.marker_edge; ctx.lineWidth = Math.max(1, this.px(1.6)); ctx.stroke();
      if (it.numbered) {
        const [cx, cy] = applyH(this.H, p.x, p.y), rr = this.px(15), lift = this.px(34) * Math.min(s.scale, 1);
        ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(cx, cy - lift, rr, 0, 2 * Math.PI); ctx.fill();
        ctx.fillStyle = this.c.plate; ctx.beginPath(); ctx.arc(cx, cy - lift, rr * 0.82, 0, 2 * Math.PI); ctx.fill();
        ctx.fillStyle = "#fff"; ctx.font = this.font(900, rr * 1.05); ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText(String(s.k + 1), cx, cy - lift + rr * 0.05);
      }
      ctx.restore();
    }
  }

  // ---------------------------------------------------- player highlight
  playerAt(pid) {
    const tr = this.tracks && this.tracks.tracks && this.tracks.tracks[pid]; if (!tr || !tr.frames.length) return null;
    let lo = 0, hi = tr.frames.length - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; if (tr.frames[m] < this.frame) lo = m + 1; else hi = m; }
    let k = lo; if (k > 0 && Math.abs(tr.frames[k - 1] - this.frame) < Math.abs(tr.frames[k] - this.frame)) k--;
    if (Math.abs(tr.frames[k] - this.frame) > 6) return null;
    return { box: tr.boxes[k], xy: tr.pitch_xy[k], speed: tr.speed_kmh[k] };
  }
  draw_player_highlight(it) {
    const got = this.playerAt(it.player_id); if (!got) return;
    const ctx = this.ctx, [fin, fout] = this.env(it, 0.35, 0.35), alpha = fin * (1 - fout), col = this.c[it.color] || this.c.accent;
    ctx.save(); ctx.globalAlpha = alpha;
    if (got.xy) {
      const r = this.lay.ring_radius_m * (0.6 + 0.4 * easeBack(fin)), spin = this.t * 1.4;
      ctx.fillStyle = col; this.groundPath(this.ribbon(this.circle(got.xy[0], got.xy[1], r, 40, spin, spin + 1.6 * Math.PI), 0.2)); ctx.fill();
    }
    const [x1, y1, x2] = got.box, cx = (x1 + x2) / 2, top = y1 - this.px(14) - this.px(4) * Math.sin(this.t * 5);
    ctx.fillStyle = col; ctx.beginPath(); ctx.moveTo(cx - this.px(9), top - this.px(14)); ctx.lineTo(cx + this.px(9), top - this.px(14)); ctx.lineTo(cx, top); ctx.fill();
    let y = top - this.px(22);
    if (it.show_name && it.display_name) { y = this.tag(cx, y, it.display_name, this.c.plate, "#fff", it.player_number ? `#${it.player_number}` : null) - this.px(6); }
    for (const ch of it.chips || []) {
      if (this.t < it.t_start + ch.appear_at) continue;
      const u = easeBack((this.t - it.t_start - ch.appear_at) / this.mo.chip_pop_s);
      ctx.globalAlpha = alpha * Math.min(u, 1);
      y = this.tag(cx, y, `${ch.value} ${ch.unit || ""}`.trim(), this.c.speed, this.c.speed_text, this.label(ch.label, this.lang).toUpperCase()) - this.px(6);
    }
    ctx.restore();
  }
  tag(cx, bottom, text, bg, fg, prefix) {
    const ctx = this.ctx, fs = this.px(this.lay.chip_font_px), pad = fs * 0.45, h = fs * 1.5;
    ctx.font = this.font(900, fs); const pw = prefix ? ctx.measureText(prefix).width + pad : 0;
    ctx.font = this.font(800, fs); const w = ctx.measureText(text).width + pw + 2 * pad, x = cx - w / 2, y = bottom - h;
    ctx.fillStyle = bg; this.roundRect(x, y, w, h, h * 0.22); ctx.fill();
    ctx.textBaseline = "middle"; ctx.textAlign = "left";
    if (prefix) { ctx.fillStyle = this.c.accent; this.roundRect(x, y, pw + pad, h, h * 0.22); ctx.fill(); ctx.fillStyle = "#fff"; ctx.font = this.font(900, fs); ctx.fillText(prefix, x + pad * 0.6, y + h / 2); }
    ctx.fillStyle = fg; ctx.font = this.font(800, fs); ctx.fillText(text, x + pw + pad * 1.2, y + h / 2);
    return y;
  }
  draw_speed_tag(it) {
    const [fin, fout] = this.env(it, 0.25, 0.3); let anchor = null;
    if (it.player_id) { const g = this.playerAt(it.player_id); if (g) anchor = [(g.box[0] + g.box[2]) / 2, g.box[1] - this.px(70)]; }
    if (!anchor && it.style && it.style.x) { const p = applyH(this.H, +it.style.x, +it.style.y); anchor = [p[0], p[1] - this.px(40)]; }
    if (!anchor) return;
    const tb = this.text(it, this.lang), label = (tb && tb.label) || (it.subject === "ball" ? "BALL" : "SPEED");
    this.ctx.save(); this.ctx.globalAlpha = fin * (1 - fout);
    this.tag(anchor[0], anchor[1], `${Math.round(it.speed_kmh)} km/h${it.estimated ? " (est.)" : ""}`, this.c.speed, this.c.speed_text, label.toUpperCase());
    this.ctx.restore();
  }
  draw_pass_line(it) {
    const ctx = this.ctx, dur = it.t_end - it.t_start, grow = easeOut((this.t - it.t_start) / Math.min(0.9, dur / 3));
    const alpha = 1 - easeIn((this.t - (it.t_end - 0.4)) / 0.4), a = [it.start.x, it.start.y], b = [it.end.x, it.end.y];
    const e = [a[0] + (b[0] - a[0]) * grow, a[1] + (b[1] - a[1]) * grow];
    const col = { perfect: this.c.pass_good, accurate: this.c.pass_good, loose: this.c.pass_loose, failed: this.c.pass_failed }[it.accuracy];
    const pts = Array.from({ length: 30 }, (_, k) => [a[0] + (e[0] - a[0]) * k / 29, a[1] + (e[1] - a[1]) * k / 29]);
    ctx.save(); ctx.globalAlpha = alpha; ctx.fillStyle = col; this.groundPath(this.ribbon(pts, 0.35)); ctx.fill();
    const n = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1, dx = (b[0] - a[0]) / n, dy = (b[1] - a[1]) / n;
    this.groundPath([[e[0] + dx * 1.6, e[1] + dy * 1.6], [e[0] - dy, e[1] + dx], [e[0] + dy, e[1] - dx]]); ctx.fill();
    ctx.fillStyle = "#fff"; this.groundPath(this.circle(a[0], a[1], 0.7, 24)); ctx.fill();
    if (grow > 0.98) { const [mx, my] = applyH(this.H, (a[0] + b[0]) / 2, (a[1] + b[1]) / 2); const tb = this.text(it, this.lang);
      this.tag(mx, my - this.px(18), (tb && tb.label) || `${Math.round(it.distance_m)} m`, this.c.plate, "#fff", (it.difficulty * 10).toFixed(1)); }
    ctx.restore();
  }
  draw_narrative_lower_third(it) {
    const tb = this.text(it, this.lang); if (!tb) return;
    const ctx = this.ctx, [fin, fout] = this.env(it, 0.45, 0.35), x = this.canvas.width * this.lay.safe_x;
    const fl = this.px(this.lay.lower_label_px), fb = this.px(this.lay.lower_font_px), pad = fb * 0.6, maxw = this.canvas.width * 0.6;
    ctx.font = this.font(600, fb); const lines = wrapText(ctx, tb.body || tb.headline || "", maxw - 2 * pad).slice(0, 2);
    const w = Math.max(...lines.map((l) => ctx.measureText(l).width)) + 2 * pad, labH = fl * 1.7, bodyH = fb * 1.32 * lines.length + pad * 1.1;
    const y = this.canvas.height * (1 - this.lay.safe_y) - labH - bodyH + fout * this.px(30);
    const tone = it.tone !== "neutral" ? (this.c[it.tone] || this.c.lower_label_bg) : this.c.lower_label_bg;
    ctx.save(); ctx.globalAlpha = 1 - fout; ctx.beginPath(); ctx.rect(x, y - 2, w * fin, labH + bodyH + 4); ctx.clip();
    ctx.font = this.font(900, fl); const lw = ctx.measureText((tb.label || "").toUpperCase()).width + 2 * pad;
    ctx.fillStyle = tone; ctx.fillRect(x, y, lw, labH); ctx.fillStyle = "#fff"; ctx.textBaseline = "middle"; ctx.textAlign = "left";
    ctx.fillText((tb.label || "").toUpperCase(), x + pad, y + labH / 2);
    ctx.font = this.font(800, fl * 0.9); ctx.fillStyle = this.c.text_dim; ctx.fillText(this.lang.toUpperCase(), x + lw + pad * 0.6, y + labH / 2);
    ctx.globalAlpha = (1 - fout) * 0.93; ctx.fillStyle = this.c.lower_bg; ctx.fillRect(x, y + labH, w, bodyH);
    ctx.globalAlpha = 1 - fout; ctx.fillStyle = tone; ctx.fillRect(x, y + labH, this.px(4), bodyH);
    ctx.fillStyle = "#fff"; ctx.font = this.font(600, fb); ctx.textBaseline = "top";
    lines.forEach((l, i) => ctx.fillText(l, x + pad, y + labH + pad * 0.45 + i * fb * 1.32));
    ctx.restore();
  }
  draw_minimap(it) {
    const ctx = this.ctx, [fin, fout] = this.env(it, 0.4, 0.4), mw = this.px(this.lay.minimap_w), mh = this.px(this.lay.minimap_h);
    const x0 = this.canvas.width * (1 - this.lay.safe_x) - mw, y0 = this.canvas.height * (1 - this.lay.safe_y) - mh, pad = 6 * this.s;
    const m2p = (x, y) => [x0 + pad + x / 105 * (mw - 2 * pad), y0 + mh - pad - y / 68 * (mh - 2 * pad)];
    ctx.save(); ctx.globalAlpha = fin * (1 - fout) * 0.85; ctx.fillStyle = this.c.minimap_bg; ctx.fillRect(x0, y0, mw, mh);
    ctx.globalAlpha = fin * (1 - fout); ctx.strokeStyle = this.c.minimap_line; ctx.lineWidth = 1;
    for (const pl of PITCH_LINES) { ctx.beginPath(); pl.forEach(([x, y], i) => { const [u, v] = m2p(x, y); i ? ctx.lineTo(u, v) : ctx.moveTo(u, v); }); ctx.stroke(); }
    if (this.H) { const Hi = invertH(this.H), W = this.canvas.width, Hc = this.canvas.height;
      ctx.fillStyle = "rgba(255,255,255,0.14)"; ctx.beginPath();
      [[0, Hc * 0.32], [W, Hc * 0.32], [W, Hc], [0, Hc]].forEach(([u, v], i) => { let [x, y] = applyH(Hi, u, v);
        x = Math.min(Math.max(x, -3), 108); y = Math.min(Math.max(y, -3), 71); const [a, b] = m2p(x, y); i ? ctx.lineTo(a, b) : ctx.moveTo(a, b); });
      ctx.fill(); }
    if (this.tracks) for (const pid of Object.keys(this.tracks.tracks)) {
      const g = this.playerAt(pid); if (!g || !g.xy) continue; const p = this.players[pid]; const [u, v] = m2p(g.xy[0], g.xy[1]);
      ctx.fillStyle = p ? p.team.kit.shirt : "#fff"; ctx.beginPath(); ctx.arc(u, v, pid === it.focus_player_id ? 4.5 * this.s : 3 * this.s, 0, 2 * Math.PI); ctx.fill();
      if (pid === it.focus_player_id) { ctx.strokeStyle = this.c.accent; ctx.lineWidth = 2; ctx.stroke(); }
    }
    for (const p of it.points) { if (this.t < it.t_start + (p.appear_at || 0)) continue; const [u, v] = m2p(p.x, p.y);
      ctx.fillStyle = this.c.marker; ctx.beginPath(); this.star(u, v, 6 * this.s).forEach(([a, b], i) => i ? ctx.lineTo(a, b) : ctx.moveTo(a, b)); ctx.fill(); }
    ctx.restore();
  }
}

function wrapText(ctx, text, maxw) {
  const words = (text || "").split(/\s+/).filter(Boolean), lines = []; let cur = "";
  for (const w of words) { const t = cur ? cur + " " + w : w; if (ctx.measureText(t).width <= maxw || !cur) cur = t; else { lines.push(cur); cur = w; } }
  if (cur) lines.push(cur); return lines;
}
