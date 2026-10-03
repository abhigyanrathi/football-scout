// The DOM side: loads the data, places the markup of views.js and keeps the view in the address.

import * as core from "./core.js";
import * as views from "./views.js";

const $ = (id) => document.getElementById(id);
const clubBox = $("club");
const tabs = $("views");
const view = $("view");
const dialog = $("detail");
const search = $("player-search");
const options = $("player-suggestions");

const requests = new Map();
const files = new Map(); // the club files that have arrived, by club key
let data;
let state;
let drawn = null; // the state last drawn
let opener = null; // what had the focus when player detail opened
let closes = 0; // closes of the dialog asked for here whose events are still to come
let active = -1; // the suggestion the arrow keys are on

function getJSON(path) {
  if (!requests.has(path)) {
    const request = fetch(`data/${path}`).then((response) => {
      if (!response.ok) throw new Error(`${path}: ${response.status}`);
      return response.json();
    });
    // a failed request is forgotten, so the next need for the file tries again
    request.catch(() => requests.delete(path));
    requests.set(path, request);
  }
  return requests.get(path);
}

const clubFile = (key) => getJSON(`clubs/${key}.json`).then((file) => files.set(key, file));

// Put the message for a file that would not load in place of whatever was waiting for it.
function fail(error) {
  console.error(error);
  for (const waiting of document.querySelectorAll(".waiting:not([hidden])")) {
    const message = $("load-error").cloneNode(true);
    message.removeAttribute("id");
    message.hidden = false;
    waiting.replaceWith(message);
  }
}

const SHOWN = ["club", "view", "group", "age", "band", "fit", "replace", "budget", "cap", "planFit"];
// whether two states differ in what the view shows; an opened player alone does not count
const moved = (a, b) => !a || SHOWN.some((key) => String(a[key]) !== String(b[key]));

// Replace an element's content and keep the focus on the part of it that had it.
function fill(element, html) {
  const id = element.contains(document.activeElement) ? document.activeElement.id : "";
  element.innerHTML = html;
  if (id) $(id)?.focus();
}

function drawView() {
  const file = files.get(state.club);
  if (file) return fill(view, views.view(data, state, file));
  const asked = state;
  view.innerHTML = views.loading(data.club.get(state.club));
  clubFile(state.club).then(() => moved(asked, state) || drawView(), fail);
}

function drawDetail() {
  if (state.player === null) {
    if (dialog.open) {
      closes += 1;
      dialog.close();
    }
    if (drawn && drawn.player !== null) {
      if (opener?.isConnected) opener.focus();
      opener = null;
    }
    return;
  }
  const player = data.player.get(state.player);
  const keys = [state.club, player.club];
  fill(dialog, views.detail(data, state, player, ...keys.map((key) => files.get(key))));
  if (!dialog.open) dialog.showModal();
  if (drawn?.player !== state.player) {
    dialog.scrollTop = 0;
    $("detail-name").focus();
  }
  if (!keys.every((key) => files.has(key))) {
    const asked = state;
    Promise.all(keys.map(clubFile)).then(() => asked === state && drawDetail(), fail);
  }
}

function draw() {
  const club = data.club.get(state.club);
  document.title = state.club === core.DEFAULT_CLUB ? "Football Scout" : `${club.name}, Football Scout`;
  if (drawn?.club !== state.club) clubBox.innerHTML = views.club(data, state);
  if (moved(drawn, state)) {
    fill(tabs, views.tabs(data, state));
    drawView();
  }
  drawDetail();
  drawn = state;
}

// Move to a state. A club, a view or an opened player adds an entry to the history, so back and
// forward move between them; anything else replaces the current entry.
function go(next) {
  const entry = ["club", "view", "player"].some((key) => next[key] !== state[key]);
  history[entry ? "pushState" : "replaceState"](null, "", core.href(next, data));
  state = next;
  draw();
}

// Every link inside the tool carries the whole state it leads to in its address.
function follow(link) {
  const before = state;
  const next = core.parseState(link.search, data);
  if (next.player !== null && before.player === null) opener = link;
  go(next);
  if (next.club !== before.club) $("club-name").focus();
  else if (next.replace !== before.replace) $("matches-title")?.focus();
}

function showMore(button) {
  const box = button.closest(".paged");
  const body = box.querySelector("tbody");
  const last = body.lastElementChild;
  const from = Number(button.dataset.from);
  const { rows, shown, total } = views.page(button.dataset.more, data, state, files.get(state.club), from);
  body.insertAdjacentHTML("beforeend", rows);
  box.querySelector(".tally").textContent = views.tally(shown, total);
  button.dataset.from = shown;
  button.hidden = shown >= total;
  // the focus moves to the first new row, so the keyboard carries on from there
  last.nextElementSibling?.querySelector("a")?.focus();
}

function suggest(open = true) {
  const found = open ? core.searchPlayers(data, search.value) : [];
  options.innerHTML = views.suggestions(data, found);
  options.hidden = !found.length;
  search.setAttribute("aria-expanded", found.length > 0);
  search.removeAttribute("aria-activedescendant");
  active = -1;
  const none = open && search.value.trim() && !found.length;
  $("search-status").textContent = none ? "No player by that name." : "";
}

function choose(option) {
  if (!option) return;
  suggest(false);
  opener = search;
  go({ ...state, player: Number(option.dataset.player) });
}

function listen() {
  addEventListener("popstate", () => {
    state = core.parseState(location.search, data);
    draw();
  });

  document.addEventListener("click", (event) => {
    if (event.button || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    const target = event.target;
    if (target.closest("[data-close]")) return go({ ...state, player: null });
    const more = target.closest("[data-more]");
    if (more) return showMore(more);
    // a click on a row, other than on a control or to select text, opens what its name opens
    const inert = target.closest("a, button, input, select, label") || String(getSelection());
    const link = target.closest("a[data-go]") ?? (inert ? null : target.closest("tr")?.querySelector("a[data-go]"));
    if (!link) return;
    event.preventDefault();
    follow(link);
  });

  document.addEventListener("change", ({ target }) => {
    const key = target.dataset.key;
    if (!key) return;
    const next = { ...state };
    if (key === "age") {
      const end = Number(target.dataset.end);
      next.age = [...state.age];
      next.age[end] = Number(target.value);
      // the other end follows, so the range stays in order
      if (next.age[0] > next.age[1]) next.age[1 - end] = next.age[end];
    } else if (target.type === "checkbox") next[key] = target.checked;
    else if (target.value === "") next[key] = null;
    else next[key] = key === "group" ? target.value : Number(target.value);
    go(next);
  });

  // The dialog reports a close a task later. One asked for here is only counted off; any other
  // is the reader's, with Esc, and takes the player out of the address.
  dialog.addEventListener("close", () => {
    if (closes) closes -= 1;
    else if (state.player !== null) go({ ...state, player: null });
  });

  clubBox.addEventListener("click", ({ target }) => {
    const toggle = target.closest("#club-toggle");
    if (!toggle) return;
    const panel = $("club-panel");
    panel.hidden = !panel.hidden;
    toggle.setAttribute("aria-expanded", !panel.hidden);
    // drawn on opening, so its links carry the view and filters as they are now
    if (!panel.hidden) $("club-list").innerHTML = views.clubList(data, state, $("club-filter").value);
  });
  clubBox.addEventListener("input", ({ target }) => {
    if (target.id === "club-filter") $("club-list").innerHTML = views.clubList(data, state, target.value);
  });
  clubBox.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && event.target.id === "club-filter") {
      $("club-list").querySelector("a")?.click();
    } else if (event.key === "Escape" && !$("club-panel").hidden) {
      $("club-toggle").click();
      $("club-toggle").focus();
    }
  });

  search.addEventListener("input", () => suggest());
  search.addEventListener("blur", () => suggest(false));
  search.addEventListener("keydown", (event) => {
    if (event.key === "Escape") return suggest(false);
    if (event.key === "Enter") return choose(options.children[Math.max(active, 0)]);
    if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
    event.preventDefault();
    if (options.hidden) suggest();
    const items = options.children;
    if (!items.length) return;
    items[active]?.setAttribute("aria-selected", "false");
    if (event.key === "ArrowDown") active = (active + 1) % items.length;
    else active = (active < 1 ? items.length : active) - 1;
    items[active].setAttribute("aria-selected", "true");
    items[active].scrollIntoView({ block: "nearest" });
    search.setAttribute("aria-activedescendant", items[active].id);
  });
  // a press on the list must not take the focus from the input, or the list would close first
  options.addEventListener("mousedown", (event) => event.preventDefault());
  options.addEventListener("click", ({ target }) => choose(target.closest("[role=option]")));
}

async function start() {
  try {
    const [meta, clubs, players] = await Promise.all(["meta.json", "clubs.json", "players.json"].map(getJSON));
    data = core.prepare(meta, clubs.clubs, players.players);
  } catch (error) {
    return fail(error);
  }
  state = core.parseState(location.search, data);
  $("status").remove();
  clubBox.hidden = tabs.hidden = search.disabled = false;
  listen();
  draw();
}

start();
