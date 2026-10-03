// The markup of every part of the tool, as strings built from the prepared data and a state.
// It touches no DOM: app.js places what these functions return.

import * as core from "./core.js";

const { fixed, ratio } = core;

// The words beside the numbers, each left whole on its line.
const COPY = {
  example:
    "Leicester City won the 2015/16 Premier League. This is what Football Scout suggests for them; choose any of the 80 clubs to see theirs.",
  gem: "How far his value lies above that of players of his price, age, league and position. Its test passed narrowly: a correlation of 0.066 (0.007 to 0.125) with the next year's change in market value.",
  momentum:
    "His price over his 2015 valuation. Momentum predicted the next year's change better than the gem score did (0.196), so it has its own column and is never combined with the gem score.",
  fit: "How far his style leans this club's way. A search preference only: its test did not pass.",
  value:
    "What his actions on the ball were worth per 90 minutes, against his league's level. The range is a 90% interval.",
  price: "His Transfermarkt valuation in summer 2016, shown as a band.",
  percentile: (pct, group) => `Value higher than that of ${pct}% of ${group}.`,
  noise:
    "As close to him as a typical player is to himself between two halves of the season, so the order among these means nothing.",
  style: "Defensive depth and shot creation are the least reliable of the eight.",
  shape: (club) =>
    `How ${club}'s minutes split across positions, rounded to ten places. It is not a formation.`,
  plans: [
    "This squad method's test against unshrunk values passed at none of its budgets.",
    "Values are measured against each player's league level, so a move between leagues assumes his output carries over.",
    "The range is a 90% interval of the noise in each player's estimate, nothing more.",
  ],
  short: (club, places) =>
    `${club}'s own players can't fill its shape: ${places} short. Its plans list signings only, with no gain.`,
  band: "The gem score compares a player with others at his price, so the list starts with stars as well as bargains. Set a highest band to search within a budget.",
  hindsight: "Hindsight. A buyer in summer 2016 did not have this.",
  headline:
    "Of the 117 players with the highest gem scores, 17 were valued at one and a half times their price or more a year later, against 152 of all 1,166.",
  carried: "No valuation in summer 2017; his latest before September 2017 is used.",
  unchanged: "No valuation after his price; counted as unchanged.",
  noMomentum: "No 2015 valuation.",
  later: "His summer 2017 valuation over his price.",
};

const NAMES = {
  gems: "Hidden gems",
  replace: "Replace a player",
  plans: "Upgrade plans",
  later: "One year later",
};
// the terms of the notes, by the key of their words in COPY
const TERMS = {
  gem: "Gem score",
  momentum: "Momentum",
  fit: "Fit",
  value: "Value",
  price: "Price",
  noise: "Within the noise",
  plans: "Plans",
  later: "One year later",
};
const MARK = { carried: "†", price: "‡" };
const FOOTNOTES = { carried: `† ${COPY.carried}`, price: `‡ ${COPY.unchanged}` };
const NOISE = `<span class="flag">Within the noise</span>`;
const SIGNING = `<span class="tag">Signing</span>`;

const esc = (text) => String(text).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
const whole = (n) => n.toLocaleString("en-GB");
const cap = (text) => text[0].toUpperCase() + text.slice(1);
const lower = (text) => text[0].toLowerCase() + text.slice(1);
// a position group in lower case: "centre-backs", or "centre-back" for one
const noun = (data, key, n = 2) => esc(data.group.get(key).toLowerCase()) + (n === 1 ? "" : "s");
const link = (data, state, text, extra = "") =>
  `<a data-go href="${esc(core.href(state, data))}"${extra}>${text}</a>`;

// One radial scale for every fingerprint: the centre is −LIMIT, the ring 0, the edge LIMIT, and
// a score beyond the edge is drawn at it.
const LIMIT = 3;

function ring(scores, radius, cx, cy) {
  return scores.map((score, i) => {
    const r = (radius * (Math.max(-LIMIT, Math.min(LIMIT, score)) + LIMIT)) / (2 * LIMIT);
    const angle = (2 * Math.PI * i) / scores.length;
    return [(cx + r * Math.sin(angle)).toFixed(1), (cy - r * Math.cos(angle)).toFixed(1)];
  });
}

const points = (xy) => xy.map((point) => point.join(",")).join(" ");

// A fingerprint in words, for the small ones: where the style scores are highest and lowest.
function leaning(data, { style, missing = [] }) {
  const scored = data.meta.dimensions
    .map((d, i) => ({ label: d.label.toLowerCase(), score: style[i] }))
    .filter((_, i) => !missing.includes(i))
    .sort((a, b) => b.score - a.score);
  if (!scored.length) return "Style: no scores";
  return `Style: highest on ${scored[0].label}, lowest on ${scored.at(-1).label}`;
}

// kind is "club", drawn filled, or "player", drawn outlined
function glyph(data, subject, kind) {
  const label = esc(leaning(data, subject));
  return (
    `<svg class="glyph" viewBox="0 0 36 36" width="36" height="36" role="img" aria-label="${label}">` +
    `<circle class="edge" cx="18" cy="18" r="17"/><circle class="zero" cx="18" cy="18" r="8.5"/>` +
    `<polygon class="${kind}" points="${points(ring(subject.style, 17, 18, 18))}"/></svg>`
  );
}

// The large fingerprint: the club's shape filled and, in player detail, the player's outlined
// over it. The less reliable dimensions have a dashed axis and a starred label.
function chart(data, club, player) {
  const [cx, cy, r] = [220, 140, 104];
  const dims = data.meta.dimensions;
  const edge = dims.map(() => LIMIT);
  const spokes = ring(edge, r, cx, cy).map(([x, y], i) => {
    const less = dims[i].less_reliable ? " less" : "";
    return `<line class="spoke${less}" x1="${cx}" y1="${cy}" x2="${x}" y2="${y}"/>`;
  });
  const labels = ring(edge, r + 12, cx, cy).map(([x, y], i) => {
    const angle = (2 * Math.PI * i) / dims.length;
    const side = Math.sin(angle);
    const anchor = side > 0.1 ? "start" : side < -0.1 ? "end" : "middle";
    // the baseline drops from on the point at the top to a line below it at the bottom
    const base = (Number(y) + 5 * (1 - Math.cos(angle))).toFixed(1);
    const text = esc(dims[i].label) + (dims[i].less_reliable ? " *" : "");
    return `<text x="${x}" y="${base}" text-anchor="${anchor}">${text}</text>`;
  });
  const outline = player ? ring(player.style, r, cx, cy) : [];
  const dots = outline.map(([x, y]) => `<circle class="dot" cx="${x}" cy="${y}" r="3"/>`);
  const label = player
    ? `Style of ${player.name}, outlined, over that of ${club.name}, filled.`
    : `Style of ${club.name}.`;
  return (
    `<svg class="fingerprint" viewBox="0 0 440 280" role="img" ` +
    `aria-label="${esc(label)} The table lists the scores.">` +
    `<polygon class="club" points="${points(ring(club.style, r, cx, cy))}"/>` +
    `<circle class="edge" cx="${cx}" cy="${cy}" r="${r}"/>${spokes.join("")}` +
    `<circle class="zero" cx="${cx}" cy="${cy}" r="${r / 2}"/>` +
    (player ? `<polygon class="player" points="${points(outline)}"/>${dots.join("")}` : "") +
    `${labels.join("")}</svg>`
  );
}

// The eight scores as text. Its headings carry the swatches that tell the two shapes apart.
function scores(data, club, player) {
  const columns = player ? [[player, "player"], [club, "club"]] : [[club, "club"]];
  const head = columns.map(
    ([subject, kind]) =>
      `<th scope="col" class="num"><svg class="swatch" viewBox="0 0 18 12" width="18" height="12" ` +
      `aria-hidden="true"><rect class="${kind}" x="1.5" y="1.5" width="15" height="9"/></svg> ` +
      `${esc(subject.name)}</th>`,
  );
  const rows = data.meta.dimensions.map((d, i) => {
    const cells = columns.map(([subject]) => {
      const score = subject.missing?.includes(i) ? "No score" : fixed(subject.style[i]);
      return `<td class="num">${score}</td>`;
    });
    const star = d.less_reliable ? " *" : "";
    return `<tr><th scope="row">${esc(d.label)}${star}</th>${cells.join("")}</tr>`;
  });
  return (
    `<table class="scores"><caption class="sr-only">Style scores</caption>` +
    `<thead><tr><th scope="col">Style</th>${head.join("")}</tr></thead>` +
    `<tbody>${rows.join("")}</tbody></table>`
  );
}

const LEGEND =
  `<p class="legend">The ring is 0, the average. The centre is −${LIMIT} and the edge ${LIMIT}; ` +
  `a score beyond them is drawn at the edge.</p><p class="legend">* ${COPY.style}</p>`;

function shape(data, club) {
  const places = data.meta.groups.map((g) => {
    const n = club.shape[g.key];
    return `${n} ${noun(data, g.key, n)}`;
  });
  return (
    `<p class="shape"><strong>Shape</strong> ${places.join(", ")}</p>` +
    `<p class="note">${COPY.shape(esc(club.name))}</p>`
  );
}

export function clubList(data, state, text) {
  const found = core.searchClubs(data, text);
  const leagues = data.meta.leagues.map((league) => {
    const items = found
      .filter((c) => c.league === league.key)
      .map((c) => {
        const next = { ...state, club: c.key, replace: null, player: null };
        const current = c.key === state.club ? ' aria-current="true"' : "";
        return `<li>${glyph(data, c, "club")}${link(data, next, esc(c.name), current)}</li>`;
      });
    if (!items.length) return "";
    return `<section><h3>${esc(league.label)}</h3><ul>${items.join("")}</ul></section>`;
  });
  return leagues.join("") || `<p class="empty">No club by that name.</p>`;
}

// The top of the tool: the chosen club, the picker, its fingerprint and its shape.
export function club(data, state) {
  const c = data.club.get(state.club);
  const example = state.club === core.DEFAULT_CLUB ? `<p class="example">${COPY.example}</p>` : "";
  return (
    `<div class="club-head"><div><h2 id="club-name" tabindex="-1">${esc(c.name)}</h2>` +
    `<p class="league">${esc(data.league.get(c.league))}, 2015/16</p></div>` +
    `<button type="button" id="club-toggle" aria-expanded="false" aria-controls="club-panel">` +
    `Choose a club</button></div>` +
    `<div id="club-panel" class="club-panel" hidden>` +
    `<div class="field"><label for="club-filter">Find a club</label>` +
    `<input id="club-filter" type="text" autocomplete="off" spellcheck="false"></div>` +
    `<div id="club-list" class="club-list"></div></div>${example}` +
    `<div class="style"><figure>${chart(data, c)}</figure><div>${scores(data, c)}</div>` +
    `<div class="style-notes">${LEGEND}${shape(data, c)}</div></div>`
  );
}

export function tabs(data, state) {
  return core.VIEWS.map((view) => {
    const current = view === state.view ? ' aria-current="page"' : "";
    return link(data, { ...state, view, player: null }, NAMES[view], ` id="tab-${view}"${current}`);
  }).join("");
}

const td = (text) => `<td>${text}</td>`;
const num = (text) => `<td class="num">${text}</td>`;
const tr = (cells) => `<tr>${cells}</tr>`;
const position = (data, p) => esc(data.group.get(p.group));
const age = (p) => p.age ?? "Not known";
const band = (data, p) => (p.band === null ? "No price" : esc(data.meta.bands[p.band]));
const range = (low, high) => `<span class="sub">${fixed(low)} to ${fixed(high)}</span>`;
const value = (p) => num(fixed(p.value) + range(p.low, p.high));
const momentum = (p) =>
  p.momentum === null ? `<span class="sub">${COPY.noMomentum}</span>` : ratio(p.momentum);
const later = (p) => ratio(p.later) + (MARK[p.how] ? `<span class="mark">${MARK[p.how]}</span>` : "");

// A player's name as the link that opens his detail, beside his small fingerprint.
function who(data, state, p, club = true) {
  const sub = club ? `<span class="sub">${esc(data.club.get(p.club).name)}</span>` : "";
  const name = link(data, { ...state, player: p.id }, esc(p.name));
  return `<th scope="row"><div class="who">${glyph(data, p, "player")}<div>${name}${sub}</div></div></th>`;
}

// the cells a signing's row starts with in an upgrade plan
const candidate = (data, state, p) => who(data, state, p) + td(position(data, p)) + num(age(p));

// The rows of one position group under its heading.
function group(data, g, columns, rows) {
  const heading = cap(noun(data, g.key, g.players.length));
  return (
    `<tbody><tr><th class="group" colspan="${columns}" scope="rowgroup">${heading}</th></tr>` +
    `${rows.join("")}</tbody>`
  );
}

// A table in a box that scrolls sideways when it must. In head, "#" ends the heading of a
// column of numbers and "~" starts one that only a screen reader is given.
function table(caption, head, bodies) {
  const cells = head.split("|").map((heading) => {
    const text = heading.replace(/[#~]/g, "");
    const shown = heading.startsWith("~") ? `<span class="sr-only">${text}</span>` : text;
    return `<th scope="col"${heading.endsWith("#") ? ' class="num"' : ""}>${shown}</th>`;
  });
  return (
    `<div class="table-scroll" role="region" tabindex="0" aria-label="${esc(caption)}"><table>` +
    `<caption class="sr-only">${esc(caption)}</caption>` +
    `<thead><tr>${cells.join("")}</tr></thead>${bodies}</table></div>`
  );
}

// The lists shown a page at a time: how many rows a page holds, the headings, what is listed
// and the cells of a row. The name stays in view while the other columns scroll, so the column
// a list is ordered by comes straight after it.
const LISTS = {
  gems: {
    size: 25,
    head: "Player|Gem score#|Price|Age#|Value#|Momentum#|Fit#|Position",
    items: (data, state, file) => core.gems(data, file, state),
    cells: (data, state, { player: p, fit }) =>
      who(data, state, p) +
      num(fixed(p.gem)) +
      td(band(data, p)) +
      num(age(p)) +
      value(p) +
      num(momentum(p)) +
      num(fixed(fit)) +
      td(position(data, p)),
  },
  later: {
    size: 25,
    head: "Player|Gem score#|One year later#|Price|Age#|Position",
    items: (data, state, file) => core.gems(data, file, state),
    cells: (data, state, { player: p }) =>
      who(data, state, p) +
      num(fixed(p.gem)) +
      num(later(p)) +
      td(band(data, p)) +
      num(age(p)) +
      td(position(data, p)),
  },
  matches: {
    size: 10,
    head: "Player|Distance#|~Marker|Value#|Price|Age#|Position",
    items: (data, state, file) => core.matches(data, file, state.replace),
    cells: (data, state, { player: p, distance, noise }) =>
      who(data, state, p) +
      num(fixed(distance)) +
      td(noise ? NOISE : "") +
      value(p) +
      td(band(data, p)) +
      num(age(p)) +
      td(position(data, p)),
  },
};

// The rows of a long list from one position on, with how many are then shown and how many exist.
export function page(kind, data, state, file, from = 0) {
  const list = LISTS[kind];
  const all = list.items(data, state, file);
  const shown = Math.min(from + list.size, all.length);
  const rows = all.slice(from, shown).map((item) => tr(list.cells(data, state, item)));
  return { rows: rows.join(""), shown, total: all.length };
}

export function tally(shown, total) {
  if (!total) return "No player matches these filters.";
  const players = `${whole(total)} ${total === 1 ? "player" : "players"}`;
  return shown < total ? `Showing ${shown} of ${players}` : players;
}

function paged(kind, caption, data, state, file) {
  const { rows, shown, total } = page(kind, data, state, file);
  const more = `data-more="${kind}" data-from="${shown}"${shown < total ? "" : " hidden"}`;
  return (
    `<div class="paged"><p class="tally" role="status">${tally(shown, total)}</p>` +
    (total ? table(caption, LISTS[kind].head, `<tbody>${rows}</tbody>`) : "") +
    `<button type="button" ${more}>Show ${LISTS[kind].size} more</button></div>`
  );
}

// A view's four parts. The stylesheet sets the notes beside the list where there is room.
function frame(view, lede, controls, terms, results) {
  const notes = terms.map((key) => {
    const texts = [COPY[key]].flat().map((text) => `<dd>${text}</dd>`);
    return `<div><dt>${TERMS[key]}</dt>${texts.join("")}</div>`;
  });
  return (
    `<div class="view view-${view}">` +
    `<div class="view-head"><h3 id="view-title">${NAMES[view]}</h3>${lede}</div>` +
    `<div class="view-controls">${controls}</div>` +
    `<aside class="notes" aria-label="Notes on the numbers"><dl>${notes.join("")}</dl></aside>` +
    `<div class="view-results">${results}</div></div>`
  );
}

function select(id, label, key, options, chosen, extra = "") {
  const items = options.map(([v, text]) => {
    const selected = String(v) === String(chosen) ? " selected" : "";
    return `<option value="${v}"${selected}>${esc(text)}</option>`;
  });
  return (
    `<div class="field"><label for="${id}">${label}</label>` +
    `<select id="${id}" data-key="${key}"${extra}>${items.join("")}</select></div>`
  );
}

function check(id, key, label, on) {
  const input = `<input type="checkbox" id="${id}" data-key="${key}"${on ? " checked" : ""}>`;
  return `<label class="check">${input} ${label}</label>`;
}

function radios(name, key, legend, options, chosen) {
  const items = options.map(([v, text]) => {
    const input =
      `<input type="radio" name="${name}" id="${name}-${v}" data-key="${key}" value="${v}"` +
      `${v === chosen ? " checked" : ""}>`;
    return `<label class="check">${input} ${text}</label>`;
  });
  return `<fieldset><legend>${legend}</legend>${items.join("")}</fieldset>`;
}

function gemsView(data, state, file) {
  const club = data.club.get(state.club);
  const [youngest, oldest] = data.ages;
  const ages = Array.from({ length: oldest - youngest + 1 }, (_, i) => [youngest + i, youngest + i]);
  const groups = data.meta.groups.map((g) => [g.key, cap(noun(data, g.key))]);
  groups.unshift(["", "All positions"]);
  // the last band has no upper edge, so as a highest band it would be the same as none
  const bands = data.meta.bands.slice(0, -1).map((label, i) => [i, label]);
  bands.unshift(["", "No highest band"]);
  const to = `<span class="sr-only">Age </span>to`;
  const hinted = ' aria-describedby="band-hint"';
  const controls =
    `<div class="filters" role="group" aria-label="Filters">` +
    select("filter-position", "Position", "group", groups, state.group ?? "") +
    select("filter-age-from", "Age from", "age", ages, state.age[0], ' data-end="0"') +
    select("filter-age-to", to, "age", ages, state.age[1], ' data-end="1"') +
    select("filter-band", "Highest price band", "band", bands, state.band ?? "", hinted) +
    check("filter-fit", "fit", "Fit above zero", state.fit) +
    `</div><p class="hint" id="band-hint">${COPY.band}</p>`;
  return frame(
    "gems",
    `<p>The priced players who never played for ${esc(club.name)}, highest gem score first.</p>`,
    controls,
    ["gem", "momentum", "fit", "value", "price"],
    paged("gems", `Hidden gems for ${club.name}`, data, state, file),
  );
}

function replaceView(data, state, file) {
  const club = data.club.get(state.club);
  const chosen = data.player.get(state.replace);
  const own = core.byGroup(data, core.ranked(data, club.players)).map((g) => {
    const rows = g.players.map((p) => {
      const current = p === chosen ? ' aria-current="true"' : "";
      const name = link(data, { ...state, replace: p.id }, esc(p.name), ` id="own-${p.id}"${current}`);
      return (
        `<tr${p === chosen ? ' class="chosen"' : ""}><th scope="row">` +
        `<div class="who">${glyph(data, p, "player")}${name}</div></th>` +
        `${num(age(p))}${num(fixed(p.value))}</tr>`
      );
    });
    return group(data, g, 3, rows);
  });
  const results = chosen
    ? `<h4 id="matches-title" tabindex="-1">Closest to ` +
      `${link(data, { ...state, player: chosen.id }, esc(chosen.name))}</h4>` +
      paged("matches", `Players closest to ${chosen.name}`, data, state, file)
    : `<p id="matches-title" class="empty" tabindex="-1">Choose a player to list his matches here.</p>`;
  return frame(
    "replace",
    `<p>Choose one of ${esc(club.name)}'s players to see who plays most like him, nearest first.</p>`,
    table(`${club.name}'s players`, "Player|Age#|Value#", own.join("")),
    ["noise", "value", "price"],
    results,
  );
}

// What the club is short of, when its own players cannot fill its shape.
function shortNote(data, club, file) {
  const parts = Object.entries(file.short).map(([key, n]) => `${n} ${noun(data, key, n)}`);
  const places = parts.length > 1 ? `${parts.slice(0, -1).join(", ")} and ${parts.at(-1)}` : parts[0];
  return `<p class="note">${COPY.short(esc(club.name), places)}</p>`;
}

function plan(data, state, file) {
  const club = data.club.get(state.club);
  const found = core.findPlan(file, state.budget, state.cap, state.planFit);
  if (found?.status !== "ok") {
    return `<p class="empty">No plan at these settings: ${esc(found?.reason ?? "none in the file")}.</p>`;
  }
  const ten = core.planTen(data, file, found);
  const signings = core.ranked(data, found.in).map((p) =>
    tr(candidate(data, state, p) + value(p) + td(band(data, p)) + num(fixed(core.fitOf(file, p.id)))),
  );
  const gain = ten
    ? `<div><dt>Gain over ${esc(club.name)}'s own best ten</dt>` +
      `<dd class="num">${fixed(found.gain)}${range(found.low, found.high)}</dd></div>`
    : "";
  const head =
    `<dl class="summary"><div><dt>Cost</dt><dd>Within budget</dd></div>` +
    `<div><dt>Total value of the ten</dt><dd class="num">${fixed(found.total)}</dd></div>${gain}</dl>` +
    `<h4>Signings</h4>` +
    table(
      `Signings of ${club.name}'s plan`,
      "Player|Position|Age#|Value#|Price|Fit#",
      `<tbody>${signings.join("")}</tbody>`,
    );
  // without a baseline the rest of the ten is not published
  if (!ten) return head;
  const replaced = core.ranked(data, found.out).map((p) =>
    tr(who(data, state, p, false) + td(position(data, p)) + num(age(p)) + value(p)),
  );
  const groups = core.byGroup(data, ten).map((g) => {
    const rows = g.players.map((p) => {
      const signed = found.in.includes(p.id);
      return tr(
        who(data, state, p, signed) + num(age(p)) + num(fixed(p.value)) + td(signed ? SIGNING : ""),
      );
    });
    return group(data, g, 4, rows);
  });
  return (
    `${head}<h4>Players they replace</h4>` +
    table(
      `${club.name}'s players the signings replace`,
      "Player|Position|Age#|Value#",
      `<tbody>${replaced.join("")}</tbody>`,
    ) +
    `<h4>The resulting ten</h4>` +
    table(`${club.name}'s ten under the plan`, "Player|Age#|Value#|~Marker", groups.join(""))
  );
}

function plansView(data, state, file) {
  const club = data.club.get(state.club);
  const budgets = data.meta.budgets_m.map((b) => [b, `€${b}m`]);
  const caps = data.meta.caps.map((k) => [k, k]);
  const controls =
    `<div class="filters" role="group" aria-label="Plan settings">` +
    radios("budget", "budget", "Budget", budgets, state.budget) +
    radios("signings", "cap", "Signings, up to", caps, state.cap) +
    check("plan-fit", "planFit", "Only signings whose fit is above zero", state.planFit) +
    `</div>${shape(data, club)}`;
  return frame(
    "plans",
    `<p>The signings that would raise ${esc(club.name)}'s side most within a budget.</p>`,
    controls,
    ["plans", "value", "price", "fit"],
    (file.baseline ? "" : shortNote(data, club, file)) + plan(data, state, file),
  );
}

// The gems filters in words, for the view that lists the gems again without their controls.
function filtersInWords(data, state) {
  const highest = state.band === null ? null : esc(lower(data.meta.bands[state.band]));
  return [
    state.group ? noun(data, state.group) : "all positions",
    `ages ${state.age[0]} to ${state.age[1]}`,
    highest ? `highest price band ${highest}` : "no highest price band",
    ...(state.fit ? ["fit above zero"] : []),
  ].join(", ");
}

function laterView(data, state, file) {
  const club = data.club.get(state.club);
  const found = core.findPlan(file, state.budget, state.cap, state.planFit);
  const signings = (found?.status === "ok" ? core.ranked(data, found.in) : []).map((p) =>
    tr(
      who(data, state, p) +
        num(later(p)) +
        td(band(data, p)) +
        value(p) +
        num(age(p)) +
        td(position(data, p)),
    ),
  );
  const settings =
    `€${state.budget}m, up to ${state.cap} ${state.cap === 1 ? "signing" : "signings"}` +
    (state.planFit ? ", only signings whose fit is above zero" : "");
  const results =
    `<h4>The hidden gems</h4><p class="settings">As filtered in ` +
    `${link(data, { ...state, view: "gems" }, NAMES.gems)}: ${filtersInWords(data, state)}.</p>` +
    paged("later", `Hidden gems for ${club.name}, one year later`, data, state, file) +
    `<h4>The plan's signings</h4><p class="settings">As set in ` +
    `${link(data, { ...state, view: "plans" }, NAMES.plans)}: ${settings}.</p>` +
    (signings.length
      ? table(
          `Signings of ${club.name}'s plan, one year later`,
          "Player|One year later#|Price|Value#|Age#|Position",
          `<tbody>${signings.join("")}</tbody>`,
        )
      : `<p class="empty">No plan at these settings.</p>`) +
    `<ul class="footnotes"><li>${FOOTNOTES.carried}</li><li>${FOOTNOTES.price}</li></ul>`;
  return frame(
    "later",
    `<p class="hindsight">${COPY.hindsight}</p>`,
    `<p class="headline">${COPY.headline}</p>`,
    ["later", "gem", "value", "price"],
    results,
  );
}

const VIEW = { gems: gemsView, replace: replaceView, plans: plansView, later: laterView };

export const view = (data, state, file) => VIEW[state.view](data, state, file);

export const loading = (club) =>
  `<p class="waiting">Loading ${esc(club.name)}'s lists. They will appear here.</p>`;

// Player detail. file is the chosen club's, for his fit to it, and home his own club's, for his
// matches; either may still be on its way.
export function detail(data, state, p, file, home) {
  const club = data.club.get(state.club);
  const full = p.full.replaceAll("''", "'");
  const fit = file ? core.fitOf(file, p.id) : null;
  const facts = [
    ["Club", esc(data.club.get(p.club).name)],
    ["Position", position(data, p)],
    ["Age", age(p)],
    ["Minutes", whole(p.minutes)],
  ].map(([term, text]) => `<div><dt>${term}</dt><dd>${text}</dd></div>`);
  // a term, its number, and the notes under it
  const numbers = [
    [
      TERMS.value,
      fixed(p.value) + range(p.low, p.high),
      COPY.value,
      COPY.percentile(p.pct, noun(data, p.group)),
    ],
    [TERMS.price, band(data, p), COPY.price],
  ];
  if (p.band !== null) {
    const shown = p.momentum === null ? COPY.noMomentum : ratio(p.momentum);
    numbers.push([TERMS.gem, fixed(p.gem), COPY.gem], [TERMS.momentum, shown, COPY.momentum]);
  }
  if (fit !== null) numbers.push([`Fit to ${esc(club.name)}`, fixed(fit), COPY.fit]);
  if (p.later !== null) {
    const hindsight = `<span class="flag">${COPY.hindsight}</span>`;
    const footnote = FOOTNOTES[p.how] ? [FOOTNOTES[p.how]] : [];
    numbers.push([TERMS.later, later(p), hindsight, COPY.later, ...footnote]);
  }
  const rows = numbers.map(([term, number, ...texts]) => {
    const notes = texts.map((text) => `<p class="note">${text}</p>`);
    return `<div><dt>${term}</dt><dd><span class="num">${number}</span>${notes.join("")}</dd></div>`;
  });
  const near = (home ? core.matches(data, home, p.id).slice(0, 5) : []).map(
    ({ player: m, distance, noise }) =>
      tr(who(data, state, m) + num(fixed(distance)) + td(noise ? NOISE : "")),
  );
  const matches = home
    ? table(`Players closest to ${p.name}`, "Player|Distance#|~Marker", `<tbody>${near.join("")}</tbody>`) +
      `<p class="note">${NOISE} ${COPY.noise}</p>`
    : `<p class="waiting">Loading his matches. They will appear here.</p>`;
  return (
    `<div class="detail"><div class="detail-head">` +
    `<div><h2 id="detail-name" tabindex="-1">${esc(p.name)}</h2>` +
    (full === p.name ? "" : `<p class="full">${esc(full)}</p>`) +
    `</div><button type="button" id="detail-close" data-close>Close</button></div>` +
    `<dl class="facts">${facts.join("")}</dl><dl class="numbers">${rows.join("")}</dl>` +
    `<h3>Style</h3><div class="style"><figure>${chart(data, club, p)}</figure>` +
    `<div>${scores(data, club, p)}</div><div class="style-notes">${LEGEND}</div></div>` +
    `<h3>Nearest matches</h3>${matches}</div>`
  );
}

export function suggestions(data, found) {
  const items = found.map(
    (p) =>
      `<li role="option" id="suggestion-${p.id}" data-player="${p.id}" aria-selected="false">` +
      `${esc(p.name)}<span class="sub">${esc(data.club.get(p.club).name)}</span></li>`,
  );
  return items.join("");
}
