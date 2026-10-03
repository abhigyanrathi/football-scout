// Data shaping, filtering, ordering, name search and the address state.
// Nothing here touches the DOM or any browser interface, so it runs under Node's test runner.

export const DEFAULT_CLUB = "premier_league-22"; // Leicester City
export const DEFAULT_BUDGET = 20;
export const DEFAULT_CAP = 2;
export const VIEWS = ["gems", "replace", "plans", "later"];

// letters that Unicode decomposition leaves whole
const FOLD = { ð: "d", đ: "d", ł: "l", þ: "th" };

// Lower case, without accents, and with any run of apostrophes, straight or curly, as one.
export function normalise(text) {
  return text
    .normalize("NFD")
    .replace(/\p{M}/gu, "")
    .toLowerCase()
    .replace(/[ðđłþ]/g, (c) => FOLD[c])
    .replace(/['’]+/g, "'")
    .trim();
}

const priced = (p) => p.band !== null;
const byValue = (a, b) => b.value - a.value || a.id - b.id;

// The three files read at start, with lookups by key and the players in name order for search.
export function prepare(meta, clubs, players) {
  const ages = players.filter(priced).map((p) => p.age);
  return {
    meta,
    clubs,
    players,
    club: new Map(clubs.map((c) => [c.key, c])),
    player: new Map(players.map((p) => [p.id, p])),
    group: new Map(meta.groups.map((g) => [g.key, g.label])),
    league: new Map(meta.leagues.map((l) => [l.key, l.label])),
    ages: [Math.min(...ages), Math.max(...ages)],
    names: players
      .map((p) => ({ player: p, name: normalise(p.name), full: normalise(p.full) }))
      .sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : a.player.id - b.player.id)),
  };
}

// Players whose display or full name holds the query, alphabetical by display name.
export function searchPlayers(data, query, limit = 10) {
  const q = normalise(query);
  const found = [];
  if (!q) return found;
  for (const { player, name, full } of data.names) {
    if (name.includes(q) || full.includes(q)) found.push(player);
    if (found.length === limit) break;
  }
  return found;
}

export function searchClubs(data, query) {
  const q = normalise(query);
  return data.clubs.filter((c) => normalise(c.name).includes(q));
}

// The club's candidate pool restricted to priced players and to the filters, highest gem score
// first. Nothing reverses this order.
export function gems(data, file, { group = null, age = data.ages, band = null, fit = false } = {}) {
  return file.fit
    .map(([id, score]) => ({ player: data.player.get(id), fit: score }))
    .filter(
      ({ player: p, fit: score }) =>
        priced(p) &&
        (group === null || p.group === group) &&
        p.age >= age[0] &&
        p.age <= age[1] &&
        (band === null || p.band <= band) &&
        (!fit || score > data.meta.fit_condition),
    )
    .sort((a, b) => a.player.rank - b.player.rank);
}

// A player's fit to the club of this file, or null if he is not in its pool.
export function fitOf(file, id) {
  return file.fit.find(([candidate]) => candidate === id)?.[1] ?? null;
}

// The players of these ids, highest value first.
export function ranked(data, ids) {
  return ids.map((id) => data.player.get(id)).sort(byValue);
}

// Players by position group in the groups' order, highest value first within each.
export function byGroup(data, players) {
  return data.meta.groups
    .map((g) => ({ ...g, players: players.filter((p) => p.group === g.key).sort(byValue) }))
    .filter((g) => g.players.length);
}

// An own player's matches in the file's order, which is by distance.
export function matches(data, file, id) {
  return (file.similar[id] ?? []).map(([candidate, distance, closeness, noise]) => ({
    player: data.player.get(candidate),
    distance,
    closeness,
    noise: noise === 1,
  }));
}

export function findPlan(file, budget, cap, fit) {
  return file.plans.find((p) => p.budget === budget && p.cap === cap && p.fit === fit) ?? null;
}

// The plan's ten: the baseline less the players replaced, plus the signings. Null when the club
// has no baseline or the plan could not be filled.
export function planTen(data, file, plan) {
  if (!file.baseline || plan?.status !== "ok") return null;
  const out = new Set(plan.out);
  const ids = [...file.baseline.players.filter((id) => !out.has(id)), ...plan.in];
  return ids.map((id) => data.player.get(id));
}

export function defaultState(data) {
  return {
    club: DEFAULT_CLUB,
    view: VIEWS[0],
    group: null,
    age: [...data.ages],
    band: null,
    fit: false,
    replace: null,
    budget: DEFAULT_BUDGET,
    cap: DEFAULT_CAP,
    planFit: false,
    player: null,
  };
}

// The first value of each key of a query string. A malformed escape drops its pair.
function query(search) {
  const pairs = new Map();
  for (const part of search.replace(/^\?/, "").split("&")) {
    const at = part.indexOf("=");
    if (at < 1) continue;
    try {
      const key = decodeURIComponent(part.slice(0, at));
      if (!pairs.has(key)) pairs.set(key, decodeURIComponent(part.slice(at + 1)));
    } catch {
      // the pair is dropped and its default stands
    }
  }
  return pairs;
}

const whole = (text) => (/^\d{1,9}$/.test(text ?? "") ? Number(text) : null);

// The state an address asks for. Unknown keys are ignored and a value that is not valid leaves
// its default. No key orders a list.
export function parseState(search, data) {
  const q = query(search);
  const state = defaultState(data);
  if (data.club.has(q.get("club"))) state.club = q.get("club");
  if (VIEWS.includes(q.get("view"))) state.view = q.get("view");
  if (data.group.has(q.get("position"))) state.group = q.get("position");
  const age = /^(\d{1,3})-(\d{1,3})$/.exec(q.get("age") ?? "")?.slice(1).map(Number);
  if (age && data.ages[0] <= age[0] && age[0] <= age[1] && age[1] <= data.ages[1]) state.age = age;
  // the last band has no upper edge, so as a highest band it is the same as none
  const band = whole(q.get("band"));
  if (band !== null && band < data.meta.bands.length - 1) state.band = band;
  state.fit = q.get("fit") === "1";
  const replace = whole(q.get("replace"));
  if (data.club.get(state.club).players.includes(replace)) state.replace = replace;
  const budget = whole(q.get("budget"));
  if (data.meta.budgets_m.includes(budget)) state.budget = budget;
  const cap = whole(q.get("signings"));
  if (data.meta.caps.includes(cap)) state.cap = cap;
  state.planFit = q.get("planfit") === "1";
  const player = whole(q.get("player"));
  if (data.player.has(player)) state.player = player;
  return state;
}

// The query string of a state, without the question mark: only what differs from the default,
// so the default state gives an empty string.
export function serialiseState(state, data) {
  const usual = defaultState(data);
  return [
    ["club", state.club, usual.club],
    ["view", state.view, usual.view],
    ["position", state.group, usual.group],
    ["age", state.age.join("-"), usual.age.join("-")],
    ["band", state.band, usual.band],
    ["fit", state.fit ? 1 : null, null],
    ["replace", state.replace, usual.replace],
    ["budget", state.budget, usual.budget],
    ["signings", state.cap, usual.cap],
    ["planfit", state.planFit ? 1 : null, null],
    ["player", state.player, usual.player],
  ]
    .filter(([, value, base]) => value !== base)
    .map(([key, value]) => `${key}=${encodeURIComponent(value)}`)
    .join("&");
}

// The relative address of a state; the default state is the bare folder.
export function href(state, data) {
  const q = serialiseState(state, data);
  return q ? `?${q}` : "./";
}

// x to a fixed number of decimals with a true minus sign. Halves round away from zero on the
// decimal digits as written, which toFixed does not do.
export function fixed(x, places = 3) {
  const digits = String(Math.abs(x));
  const scaled = digits.includes("e") ? Math.abs(x) * 10 ** places : Number(`${digits}e${places}`);
  const rounded = Math.round(scaled) / 10 ** places;
  return (x < 0 && rounded > 0 ? "−" : "") + rounded.toFixed(places);
}

export const ratio = (x) => `${fixed(x, 2)}×`;
