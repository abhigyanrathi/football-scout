import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import * as core from "../../web/js/core.js";

const read = (path) =>
  JSON.parse(readFileSync(new URL(`../../web/data/${path}`, import.meta.url), "utf8"));
const meta = read("meta.json");
const { clubs } = read("clubs.json");
const { players } = read("players.json");
const data = core.prepare(meta, clubs, players);
const file = (key) => read(`clubs/${key}.json`);

const LEICESTER = core.DEFAULT_CLUB;
const GENOA = "serie_a-233";
const leicester = file(LEICESTER);
const own = data.club.get(LEICESTER).players;
const names = (found) => found.map((p) => p.name);
const inRankOrder = (rows) => rows.every((r, i) => i === 0 || rows[i - 1].player.rank < r.player.rank);
const byValue = (list) => list.every((p, i) => i === 0 || list[i - 1].value >= p.value);

test("the module imports nothing and names no browser interface", () => {
  const source = readFileSync(new URL("../../web/js/core.js", import.meta.url), "utf8");
  const browser = /\b(window|document|fetch|location|history|localStorage|sessionStorage|navigator)\b/;
  assert.equal(browser.exec(source), null);
  assert.ok(!/^import\s/m.test(source));
});

test("the defaults are Leicester City and a budget and a number of signings the files hold", () => {
  assert.equal(data.club.get(core.DEFAULT_CLUB).name, "Leicester City");
  assert.ok(meta.budgets_m.includes(core.DEFAULT_BUDGET));
  assert.ok(meta.caps.includes(core.DEFAULT_CAP));
  assert.deepEqual(data.ages, [18, 39]);
});

test("names are compared without case, accents or doubled apostrophes", () => {
  assert.equal(core.normalise("  Ángel Di MARÍA "), "angel di maria");
  assert.equal(core.normalise("N''Golo Kanté"), "n'golo kante");
  assert.equal(core.normalise("N’Golo"), "n'golo");
  assert.equal(core.normalise("Filip Đorđević"), "filip dordevic");
  assert.equal(core.normalise("Gylfi Sigurðsson"), "gylfi sigurdsson");
  assert.equal(core.normalise("Paweł Wszołek"), "pawel wszolek");
});

test("search finds a player by either name, however the query is written", () => {
  for (const q of ["kanté", "KANTE", "n'golo", "N''Golo", "n’golo kante"]) {
    assert.deepEqual(names(core.searchPlayers(data, q)), ["N'Golo Kanté"], q);
  }
  // ten full names hold a doubled apostrophe, and no display name does
  assert.equal(players.filter((p) => p.full.includes("''")).length, 10);
  assert.ok(!players.some((p) => p.name.includes("''")));
  // only his full name holds this query
  const nzonzi = players.find((p) => p.name === "Steven Nzonzi");
  assert.ok(!core.normalise(nzonzi.name).includes("kemboanza"));
  for (const q of ["kemboanza", "n'kemboanza", "N''Kemboanza Mike"]) {
    assert.deepEqual(core.searchPlayers(data, q), [nzonzi], q);
  }
  // the display name has no apostrophe and the full name has two
  const dalessandro = players.find((p) => p.full === "Marco D''Alessandro");
  assert.deepEqual(core.searchPlayers(data, "dalessandro"), [dalessandro]);
  assert.deepEqual(core.searchPlayers(data, "d'alessandro"), [dalessandro]);
  for (const q of ["", "   ", "no such player"]) assert.deepEqual(core.searchPlayers(data, q), [], q);
});

test("suggestions are alphabetical by display name and at most ten", () => {
  const found = core.searchPlayers(data, "an");
  const keys = found.map((p) => core.normalise(p.name));
  const every = players
    .filter((p) => [p.name, p.full].some((n) => core.normalise(n).includes("an")))
    .map((p) => core.normalise(p.name))
    .sort();
  assert.ok(every.length > 10);
  assert.deepEqual(keys, every.slice(0, 10));
  assert.equal(core.searchPlayers(data, "an", 3).length, 3);
  // an accented initial sorts with its plain letter
  const order = data.names.map((n) => n.player.name);
  assert.ok(order.indexOf("Ángel Di María") < order.findIndex((name) => name.startsWith("B")));
});

test("club search uses the same matching and keeps the file's order", () => {
  const found = (q) => core.searchClubs(data, q).map((c) => c.name);
  assert.deepEqual(found("atletico"), ["Atlético Madrid"]);
  assert.deepEqual(found("SAINT-ETIENNE"), ["Saint-Étienne"]);
  assert.deepEqual(found("malaga"), ["Málaga"]);
  assert.deepEqual(found("no such club"), []);
  assert.deepEqual(core.searchClubs(data, ""), clubs);
  const real = core.searchClubs(data, "real");
  assert.ok(real.length > 1);
  assert.deepEqual(real, clubs.filter((c) => real.includes(c)));
});

test("hidden gems are the priced players of the club's pool, highest gem score first", () => {
  const rows = core.gems(data, leicester);
  const pool = new Map(leicester.fit);
  const priced = leicester.fit.filter(([id]) => data.player.get(id).band !== null);
  assert.equal(rows.length, priced.length);
  assert.ok(rows.length < leicester.fit.length);
  for (const { player, fit } of rows) {
    assert.equal(fit, pool.get(player.id));
    assert.notEqual(player.band, null);
    assert.ok(!own.includes(player.id));
  }
  assert.ok(inRankOrder(rows));
  assert.equal(rows[0].player.rank, 1);
  assert.deepEqual(core.gems(data, leicester, core.defaultState(data)), rows);
});

test("each gems filter narrows the list, alone and with the others", () => {
  const all = core.gems(data, leicester);
  const above = (r) => r.fit > meta.fit_condition;
  const cases = [
    [{ group: "FWD" }, (r) => r.player.group === "FWD"],
    [{ age: [20, 23] }, (r) => r.player.age >= 20 && r.player.age <= 23],
    [{ band: 2 }, (r) => r.player.band <= 2],
    [{ fit: true }, above],
    [
      { group: "MID", age: [18, 21], band: 3, fit: true },
      (r) => r.player.group === "MID" && r.player.age <= 21 && r.player.band <= 3 && above(r),
    ],
  ];
  for (const [filters, kept] of cases) {
    const rows = core.gems(data, leicester, filters);
    assert.deepEqual(rows, all.filter(kept), JSON.stringify(filters));
    assert.ok(rows.length > 0 && rows.length < all.length, JSON.stringify(filters));
    assert.ok(inRankOrder(rows));
  }
  // the highest band and both ends of the age range are inclusive
  assert.ok(core.gems(data, leicester, { band: 2 }).some((r) => r.player.band === 2));
  const ages = core.gems(data, leicester, { age: [20, 23] }).map((r) => r.player.age);
  assert.ok(ages.includes(20) && ages.includes(23));
  // the fit filter reads its condition from the files
  assert.equal(meta.fit_condition, 0);
  assert.ok(all.some((r) => r.fit <= 0) && core.gems(data, leicester, { fit: true }).every(above));
});

test("matches keep the file's order and its noise flags", () => {
  let flagged = 0;
  let plain = 0;
  for (const [id, rows] of Object.entries(leicester.similar)) {
    const found = core.matches(data, leicester, Number(id));
    assert.deepEqual(found.map((m) => [m.player.id, m.distance]), rows.map((r) => [r[0], r[1]]));
    assert.deepEqual(found.map((m) => m.noise), rows.map((r) => r[3] === 1));
    assert.ok(found.every((m, i) => i === 0 || found[i - 1].distance <= m.distance));
    flagged += found.filter((m) => m.noise).length;
    plain += found.filter((m) => !m.noise).length;
  }
  assert.ok(flagged > 0 && plain > 0);
  assert.deepEqual(Object.keys(leicester.similar).map(Number), own);
  // a player of another club has no matches in this file
  assert.deepEqual(core.matches(data, leicester, leicester.fit[0][0]), []);
});

test("players are listed highest value first, and by position in the groups' order", () => {
  const ranked = core.ranked(data, own);
  assert.equal(ranked.length, own.length);
  assert.ok(byValue(ranked));
  const groups = core.byGroup(data, ranked);
  assert.deepEqual(groups.map((g) => g.key), meta.groups.map((g) => g.key));
  for (const g of groups) {
    assert.ok(g.players.every((p) => p.group === g.key) && byValue(g.players));
  }
  assert.equal(groups.flatMap((g) => g.players).length, own.length);
});

test("a player's fit to a club is read from its pool", () => {
  const [id, score] = leicester.fit[5];
  assert.equal(core.fitOf(leicester, id), score);
  assert.equal(core.fitOf(leicester, own[0]), null);
});

test("a plan is found by its budget, signings and condition", () => {
  for (const key of [LEICESTER, GENOA]) {
    const f = file(key);
    const found = new Set();
    for (const budget of meta.budgets_m) {
      for (const cap of meta.caps) {
        for (const fit of [false, true]) {
          const plan = core.findPlan(f, budget, cap, fit);
          assert.deepEqual([plan.budget, plan.cap, plan.fit], [budget, cap, fit]);
          found.add(plan);
        }
      }
    }
    assert.equal(found.size, 30);
    assert.equal(f.plans.length, 30);
    assert.equal(core.findPlan(f, 7, 1, false), null);
  }
});

test("a plan's ten is the baseline less the players replaced plus the signings, in the club's shape", () => {
  let plans = 0;
  for (const club of clubs) {
    const f = file(club.key);
    if (!f.baseline) continue;
    for (const plan of f.plans) {
      const ten = core.planTen(data, f, plan);
      const ids = ten.map((p) => p.id);
      assert.equal(new Set(ids).size, 10);
      assert.ok(plan.in.every((id) => ids.includes(id)));
      assert.ok(!plan.out.some((id) => ids.includes(id)));
      const kept = f.baseline.players.filter((id) => !plan.out.includes(id));
      assert.deepEqual(ids.filter((id) => !plan.in.includes(id)), kept);
      const places = meta.groups.map((g) => [g.key, ten.filter((p) => p.group === g.key).length]);
      assert.deepEqual(Object.fromEntries(places), club.shape);
      plans += 1;
    }
  }
  assert.equal(plans, 79 * 30);
});

test("Genoa, with no baseline, has no ten and no gain", () => {
  const f = file(GENOA);
  assert.equal(data.club.get(GENOA).name, "Genoa");
  assert.equal(f.baseline, null);
  assert.deepEqual(f.short, { FB: 1 });
  for (const plan of f.plans) {
    assert.equal(core.planTen(data, f, plan), null);
    assert.deepEqual([plan.gain, plan.low, plan.high, plan.out], [null, null, null, []]);
    assert.ok(plan.in.length > 0 && plan.total !== null);
  }
  // a plan that could not be filled has no ten either
  const failed = { ...leicester.plans[0], status: "infeasible" };
  assert.equal(core.planTen(data, leicester, failed), null);
  assert.equal(core.planTen(data, leicester, null), null);
});

test("the default state has no query string", () => {
  const state = core.defaultState(data);
  assert.deepEqual(state, {
    club: LEICESTER,
    view: "gems",
    group: null,
    age: [18, 39],
    band: null,
    fit: false,
    replace: null,
    budget: 20,
    cap: 2,
    planFit: false,
    player: null,
  });
  assert.equal(core.serialiseState(state, data), "");
  assert.equal(core.href(state, data), "./");
  assert.deepEqual(core.parseState("", data), state);
  assert.deepEqual(core.parseState("?", data), state);
});

test("an address restores the state it was written from", () => {
  const usual = core.defaultState(data);
  const states = [];
  for (const club of [LEICESTER, GENOA, clubs[0].key, clubs.at(-1).key]) {
    const theirs = data.club.get(club).players;
    for (const view of core.VIEWS) {
      states.push(
        { ...usual, club, view },
        { ...usual, club, view, group: "WIDE", age: [19, 24], band: 3, fit: true },
        { ...usual, club, view, replace: theirs[0], budget: 80, cap: 3, planFit: true },
        { ...usual, club, view, age: [18, 30], band: 0, replace: theirs.at(-1), budget: 5, cap: 1 },
        { ...usual, club, view, group: "CB", age: [39, 39], player: theirs[1] },
        { ...usual, club, view, fit: true, planFit: true, player: players[0].id },
      );
    }
  }
  for (const state of states) {
    const address = core.serialiseState(state, data);
    assert.deepEqual(core.parseState(address, data), state, address);
    assert.deepEqual(core.parseState(`?${address}`, data), state, address);
    assert.equal(core.href(state, data), address ? `?${address}` : "./");
  }
  assert.equal(new Set(states.map((s) => core.serialiseState(s, data))).size, states.length);
  assert.equal(
    core.serialiseState({ ...usual, club: GENOA, view: "plans", budget: 40, player: 2936 }, data),
    "club=serie_a-233&view=plans&budget=40&player=2936",
  );
});

test("unknown keys are ignored and values that are not valid fall back to defaults", () => {
  const usual = core.defaultState(data);
  const addresses = [
    "?utm_source=x&foo=bar&=1&&x",
    "?club=nowhere-1&view=worst&position=GK&age=30-20&band=6&fit=yes",
    "?age=17-25",
    "?age=18-40",
    "?age=18",
    "?age=abc",
    "?band=-1",
    "?band=1.5",
    "?band=7",
    "?replace=1",
    "?replace=abc",
    "?budget=7&signings=4&planfit=true",
    "?player=1&player=2936",
    "?player=%E0%A4%A",
    "?club=%",
  ];
  for (const address of addresses) {
    assert.deepEqual(core.parseState(address, data), usual, address);
  }
  // one value that is not valid leaves the others standing
  assert.deepEqual(core.parseState("?view=plans&budget=7&signings=3&x=1", data), {
    ...usual,
    view: "plans",
    cap: 3,
  });
  // a replaced player must be one of the chosen club's own
  const theirs = data.club.get(GENOA).players[0];
  assert.equal(core.parseState(`?replace=${theirs}`, data).replace, null);
  assert.equal(core.parseState(`?club=${GENOA}&replace=${theirs}`, data).replace, theirs);
});

test("no address reverses a list", () => {
  const usual = core.defaultState(data);
  const address = "?sort=later&order=asc&dir=desc&reverse=1&by=fit&direction=ascending&desc=0";
  const state = core.parseState(address, data);
  assert.deepEqual(state, usual);
  assert.deepEqual(Object.keys(state), [
    "club",
    "view",
    "group",
    "age",
    "band",
    "fit",
    "replace",
    "budget",
    "cap",
    "planFit",
    "player",
  ]);
  const rows = core.gems(data, leicester);
  assert.deepEqual(core.gems(data, leicester, state), rows);
  assert.deepEqual(core.gems(data, leicester, { sort: "later", order: "asc", reverse: true }), rows);
  for (const view of core.VIEWS) {
    assert.equal(core.serialiseState(core.parseState(`${address}&view=${view}`, data), data), {
      gems: "",
      replace: "view=replace",
      plans: "view=plans",
      later: "view=later",
    }[view]);
  }
});

test("numbers show a true minus sign and round halves on their decimal digits", () => {
  assert.equal(core.fixed(0.1234), "0.123");
  assert.equal(core.fixed(-0.1758), "−0.176");
  assert.equal(core.fixed(0.0245), "0.025");
  assert.equal(core.fixed(-0.0245), "−0.025");
  assert.equal(core.fixed(1.0005), "1.001");
  assert.equal(core.fixed(-0.0004), "0.000");
  assert.equal(core.fixed(0), "0.000");
  assert.equal(core.fixed(2), "2.000");
  assert.equal(core.ratio(1.25), "1.25×");
  assert.equal(core.ratio(27.585), "27.59×");
  assert.equal(core.ratio(1), "1.00×");
  assert.ok(!core.fixed(-1.5).includes("-"));
});
