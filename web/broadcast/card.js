// PitchLens broadcast graphics: score bug + insight column (stat / kit / name), top-flight theme.
// Shared by the on-air output (web/onair.html) and the producer console preview, so what the producer
// approves is pixel-for-pixel what goes to air. Animation is a pure function of time: card.seek(t).
(() => {
  const NS = "http://www.w3.org/2000/svg";
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
  const el = (html) => { const t = document.createElement("template"); t.innerHTML = html.trim(); return t.content.firstElementChild; };
  let uid = 0;

  function crest(a, b, mono) {
    const id = "cg" + (++uid);
    return `<svg viewBox="0 0 40 46"><defs><linearGradient id="${id}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#fff" stop-opacity=".4"/><stop offset=".5" stop-color="#fff" stop-opacity="0"/></linearGradient></defs>
      <path d="M20 1 L38 6 V22 C38 34 30 41 20 45 C10 41 2 34 2 22 V6 Z" fill="${a}" stroke="#fff" stroke-width="2"/>
      <path d="M2 18 L20 28 L38 18 V24 L20 34 L2 24 Z" fill="${b}"/>
      <path d="M20 1 L38 6 V22 C38 34 30 41 20 45 C10 41 2 34 2 22 V6 Z" fill="url(#${id})"/>
      <text x="20" y="19.5" text-anchor="middle" font-family="Big Shoulders Display" font-weight="900" font-size="12" fill="#fff">${esc(mono)}</text></svg>`;
  }
  const mono = (short) => (short || "").slice(0, 2).toUpperCase();
  const lens = `<svg viewBox="0 0 18 18"><circle cx="9" cy="9" r="7" fill="none" stroke="currentColor" stroke-width="2.2"/><circle cx="9" cy="9" r="2.6" fill="currentColor"/></svg>`;

  function shade(hex, k) {  // k < 0 darkens, k > 0 lightens
    const n = parseInt(hex.slice(1), 16), c = [n >> 16, (n >> 8) & 255, n & 255].map((v) => Math.round(k < 0 ? v * (1 + k) : v + (255 - v) * k));
    return "#" + c.map((v) => v.toString(16).padStart(2, "0")).join("");
  }

  // lit "product shot" of the player's shirt (back view), in the club's colours
  function kit(number, last, shirt, trim) {
    const i = ++uid, lo = shade(shirt, -.32), hi = shade(shirt, .12);
    const n = last.length, fs = n > 7 ? Math.max(24, 40 * 7 / n) : 40, ls = n > 7 ? 3 : 6;  // long names fit the arc
    return `<svg class="kit" viewBox="0 0 400 420"><defs>
      <linearGradient id="kf${i}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${hi}"/><stop offset=".55" stop-color="${shirt}"/><stop offset="1" stop-color="${lo}"/></linearGradient>
      <linearGradient id="ksh${i}" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#000" stop-opacity=".38"/><stop offset=".2" stop-color="#000" stop-opacity="0"/><stop offset=".55" stop-color="#fff" stop-opacity=".08"/><stop offset=".8" stop-color="#000" stop-opacity="0"/><stop offset="1" stop-color="#000" stop-opacity=".42"/></linearGradient>
      <linearGradient id="khi${i}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#fff" stop-opacity=".28"/><stop offset=".35" stop-color="#fff" stop-opacity="0"/></linearGradient>
      <pattern id="mesh${i}" width="5" height="5" patternUnits="userSpaceOnUse"><circle cx="2.5" cy="2.5" r=".9" fill="#000" fill-opacity=".14"/></pattern>
      <filter id="fold${i}"><feGaussianBlur stdDeviation="7"/></filter><filter id="rim${i}"><feGaussianBlur stdDeviation="2"/></filter>
      <clipPath id="kc${i}"><path id="kb${i}" d="M128,40 C160,58 240,58 272,40 L352,78 L398,196 L344,220 L318,166 L324,396 Q200,418 76,396 L82,166 L56,220 L2,196 L48,78 Z"/></clipPath>
      <path id="arc${i}" d="M110,128 Q200,106 290,128"/></defs>
      <use href="#kb${i}" fill="url(#kf${i})"/>
      <g clip-path="url(#kc${i})">
        <rect width="400" height="420" fill="url(#mesh${i})"/>
        <path d="M82,166 L98,410 L70,410 Z M318,166 L302,410 L330,410 Z" fill="${trim}"/>
        <path d="M330,96 L394,184 L378,192 L318,110 Z M70,96 L6,184 L22,192 L82,110 Z" fill="${trim}"/>
        <path d="M2,196 L56,220 L61,206 L8,182 Z M398,196 L344,220 L339,206 L392,182 Z" fill="#fff"/>
        <path d="M6,188 L58,212 M394,188 L342,212" stroke="${trim}" stroke-width="3"/>
        <g filter="url(#fold${i})" opacity=".9">
          <path d="M120,200 C150,260 140,330 150,400" stroke="#000" stroke-opacity=".22" stroke-width="16" fill="none"/>
          <path d="M268,190 C252,260 262,330 248,402" stroke="#000" stroke-opacity=".2" stroke-width="14" fill="none"/>
          <path d="M200,150 C210,220 196,300 205,400" stroke="#fff" stroke-opacity=".12" stroke-width="22" fill="none"/>
          <path d="M100,330 C160,350 240,350 300,326" stroke="#000" stroke-opacity=".14" stroke-width="12" fill="none"/></g>
        <path d="M48,78 L82,166 M352,78 L318,166" stroke="#000" stroke-opacity=".25" stroke-width="2" fill="none"/>
        <rect width="400" height="420" fill="url(#ksh${i})"/><rect width="400" height="420" fill="url(#khi${i})"/></g>
      <use href="#kb${i}" fill="none" stroke="#fff" stroke-opacity=".35" stroke-width="3" filter="url(#rim${i})"/>
      <path d="M128,40 C160,58 240,58 272,40 L262,28 C232,45 168,45 138,28 Z" fill="${trim}"/>
      <path d="M138,28 C168,45 232,45 262,28" fill="none" stroke="#fff" stroke-width="3"/>
      <text font-family="Big Shoulders Display" font-weight="800" font-size="${fs}" letter-spacing="${ls}" fill="#fff" text-anchor="middle"><textPath href="#arc${i}" startOffset="50%">${esc(last.toUpperCase())}</textPath></text>
      <text x="200" y="318" text-anchor="middle" font-family="Big Shoulders Display" font-weight="900" font-size="220" fill="#fff" stroke="${trim}" stroke-width="7" paint-order="stroke" letter-spacing="-4">${esc(number)}</text></svg>`;
  }

  const clockText = (t) => { const m = Math.floor(t / 60), s = Math.floor(t % 60); return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`; };

  // ---------------------------------------------------------------- score bug (always on)
  function bug(root, teams) {
    const [h, a] = teams;
    const b = el(`<div class="bug drop"><div class="seg">${crest(h.primary, h.secondary, mono(h.short))}<span class="club">${esc(h.short)}</span></div>
      <div class="sc"><b class="sh">0</b><b class="sa">0</b></div>
      <div class="seg"><span class="club">${esc(a.short)}</span>${crest(a.primary, a.secondary, mono(a.short))}</div>
      <div class="clk"><small class="per">1ST HALF</small><span class="time">00:00</span></div><div class="line"></div></div>`);
    root.appendChild(b);
    return {
      update(t, score) {
        b.querySelector(".sh").textContent = score[0]; b.querySelector(".sa").textContent = score[1];
        b.querySelector(".time").textContent = clockText(t);
        b.querySelector(".per").textContent = t >= 2700 ? "2ND HALF" : "1ST HALF";
      },
    };
  }

  // ---------------------------------------------------------------- insight column
  const clamp = (x) => Math.max(0, Math.min(1, x));
  const ease = {out: (x) => 1 - Math.pow(1 - x, 3), inOut: (x) => x < .5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2, in: (x) => x * x * x};
  const ph = (t, a, d) => clamp((t - a) / d);
  const T = {stat: 0, roll: .35, kit: .75, name: 1.15, out: 8.2, end: 9.3};

  function fmtValue(p, k) {  // roll the number up to its value with the same formatting as value_text
    const unit = (p.value_text.match(/[^\d.,\s-]+$/) || [""])[0];
    const dec = (p.value_text.split(".")[1] || "").replace(/\D/g, "").length;
    const v = p.value * k;
    return (dec ? v.toFixed(dec) : String(Math.round(v))) + unit;
  }

  function column(root, p) {
    const pl = p.player, big = p.value_text.length > 3;
    const c = el(`<div class="col"><div class="spine"></div>
      <div class="card stat cut grain drop"><div class="hair"></div>
        <div class="eyebrow">${lens}Match insight</div><div class="min">${p.minute}'</div>
        <div class="hero${big ? " long" : ""}"><span class="ghost"></span><span class="fill"></span></div>
        <div class="txt"><b>${esc(p.label)}</b><span>${esc(p.sublabel)}</span></div></div>
      <div class="card kitcard cut grain drop"><div class="num">${String(pl.number).padStart(2, "0")}</div><div class="stripes"></div><div class="dots"></div><div class="floor"></div>
        ${kit(pl.number, pl.last, pl.primary, pl.secondary)}</div>
      <div class="card name cut grain drop"><div class="no">${String(pl.number).padStart(2, "0")}</div>
        <div class="who"><small>${esc(pl.first)}</small><b${pl.last.length > 8 ? ` style="font-size:${Math.round(52 * 8 / pl.last.length)}px"` : ""}>${esc(pl.last)}</b><span>${esc(pl.team_name)}</span></div>
        <div class="crest cut">${crest(pl.primary, pl.secondary, mono(pl.team_short))}</div></div></div>`);
    root.appendChild(c);
    const st = c.querySelector(".stat"), kc = c.querySelector(".kitcard"), nm = c.querySelector(".name");
    const spine = c.querySelector(".spine"), hero = c.querySelector(".hero"), heroS = [...c.querySelectorAll(".hero span")];
    const jersey = kc.querySelector("svg.kit"), wm = kc.querySelector(".num"), stripes = kc.querySelector(".stripes");
    const card = (e, a, q, dx = 70) => {
      const v = ease.out(a) * (1 - ease.in(q));
      e.style.opacity = v; e.style.transform = `translateX(${(1 - ease.out(a)) * dx + ease.in(q) * dx}px)`;
    };
    function seek(t) {
      const out = (k) => ph(t, T.out + k, .45);
      card(st, ph(t, T.stat, .55), out(.16));
      spine.style.transformOrigin = "top";
      spine.style.transform = `scaleY(${ease.inOut(ph(t, T.stat, T.name + .5 - T.stat)) * (1 - ease.in(out(0)))})`;
      card(kc, ph(t, T.kit, .6), out(.08));
      jersey.style.transform = `translateX(-50%) translateY(${(1 - ease.out(ph(t, T.kit + .1, .7))) * 70}px)`;
      wm.style.transform = `translateX(calc(-50% + ${(1 - ease.out(ph(t, T.kit, 1.1))) * -60}px))`;
      stripes.style.transform = `skewY(-14deg) translateX(${(1 - ease.out(ph(t, T.kit + .05, .8))) * 120}px)`;
      card(nm, ph(t, T.name, .5), out(0));
      const r = ph(t, T.roll, .8), txt = fmtValue(p, ease.out(r));
      for (const e of heroS) e.textContent = txt;
      const pop = ph(t, T.roll + .8, .35);
      hero.style.transformOrigin = "40% 60%";
      hero.style.transform = `scale(${1 + .18 * (pop > 0 && pop < 1 ? 1 - ease.out(pop) : 0)})`;
    }
    seek(0);
    return {el: c, seek, remove: () => c.remove()};
  }

  window.PLCard = {T, bug, column, crest, kit, clockText};
})();
