// Hiring-RH · Talent Orbit page. Talks to the FastAPI backend in api/main.py.
import { TalentOrbit } from "./talent-orbit.js";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (m) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[m]));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const PRESETS = {
  custom: { label: "Blank form", title: "", skills: "", minexp: 0, criteria: "", desc: "" },
  backend: {
    label: "Backend Python Developer", title: "Backend Python Developer",
    skills: "Python, FastAPI, PostgreSQL, Docker, REST API", minexp: 3,
    criteria: "Comfortable working in an agile team",
    desc: "We build the APIs behind our recruiting platform. You will design REST services in Python (FastAPI), model data in PostgreSQL, ship with Docker and review code with the team.",
  },
  data: {
    label: "Data Analyst", title: "Data Analyst", skills: "SQL, Power BI, Python, Pandas, Statistics", minexp: 2,
    criteria: "Can present insights to non-technical managers",
    desc: "You will query our sales data with SQL, build Power BI dashboards and run analyses in Python to support business decisions.",
  },
  frontend: {
    label: "Frontend Developer (React)", title: "Frontend Developer (React)",
    skills: "JavaScript, TypeScript, React, CSS, HTML", minexp: 2, criteria: "Attention to accessibility",
    desc: "You will own our customer-facing web app in React and TypeScript, working closely with designers.",
  },
};
const STATUS = { pending: "Needs review", shortlisted: "Shortlisted", rejected: "Not selected" };
const SUBS = [["skills_score", "Skills"], ["experience_score", "Experience"], ["domain_score", "Field"]];

const state = { candidates: [], task: null, decisions: {}, selected: null, revealed: new Set(), sig: "" };

const orbit = new TalentOrbit($("orbit"), {
  minScore: 0,
  quarantineOrbit: "score", // flagged resumes keep their score; the injected text was removed before scoring
  strongScore: 0.75,
  thresholds: [
    { score: 0.75, label: "fit 75", highlight: true },
    { score: 0.5, label: "fit 50" },
    { score: 0.25, label: "fit 25" },
  ],
});
orbit.on("select", (c) => select(c.id));

// ------------------------------------------------------------------ API

async function api(path, options = {}) {
  const res = await fetch(path, options);
  let body = null;
  try { body = await res.json(); } catch { /* empty */ }
  if (!res.ok) {
    const detail = body?.detail;
    throw new Error(Array.isArray(detail) ? detail.map((d) => d.msg).join(", ") : detail || res.statusText);
  }
  return body;
}

async function poll(taskId, onUpdate) {
  for (;;) {
    const t = await api(`/api/tasks/${taskId}`);
    onUpdate(t);
    if (t.status === "done") return t;
    await sleep(1000);
  }
}

function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.style.display = "block";
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (t.style.display = "none"), 2600);
}

function setProgress(el, done, total, text) {
  el.hidden = false;
  el.querySelector("i").style.width = `${total ? (100 * done) / total : 0}%`;
  el.querySelector("span").textContent = text;
}

function showLog(el, messages) {
  el.innerHTML = "";
  for (const m of messages) {
    const li = document.createElement("li");
    li.className = m.level;
    li.textContent = m.text;
    el.append(li);
  }
}

// ------------------------------------------------------------------ candidates

async function loadCandidates() {
  state.candidates = await api("/api/candidates");
  render();
}

async function upload(fileList) {
  const files = [...fileList];
  if (!files.length) return;
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  try {
    const task = await api("/api/candidates/upload", { method: "POST", body: form });
    await poll(task.id, (t) => {
      setProgress($("upProgress"), t.done, t.total, `Reading resumes with the LLM: ${t.done} / ${t.total}`);
      showLog($("upLog"), t.messages);
    });
    await loadCandidates();
  } catch (e) {
    toast(`Upload failed: ${e.message}`);
  }
}

$("files").onchange = (e) => { upload(e.target.files); e.target.value = ""; };
const drop = $("drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => upload(e.dataTransfer.files));

$("demoBtn").onclick = async () => {
  try {
    const r = await api("/api/candidates/demo", { method: "POST" });
    showLog($("upLog"), r.messages.map((text) => ({ level: text.includes("removed") ? "warn" : "ok", text })));
    await loadCandidates();
  } catch (e) { toast(e.message); }
};

$("clearBtn").onclick = async () => {
  if (!confirm("Remove all candidates and decisions?")) return;
  await api("/api/candidates", { method: "DELETE" });
  Object.assign(state, { task: null, decisions: {}, selected: null, sig: "" });
  $("upLog").innerHTML = ""; $("matchLog").innerHTML = "";
  $("upProgress").hidden = true; $("matchProgress").hidden = true;
  await loadCandidates();
};

// ------------------------------------------------------------------ job + scoring

for (const [key, p] of Object.entries(PRESETS)) {
  const o = document.createElement("option");
  o.value = key; o.textContent = p.label;
  $("preset").append(o);
}
$("preset").onchange = () => {
  const p = PRESETS[$("preset").value];
  $("title").value = p.title; $("skills").value = p.skills; $("minexp").value = p.minexp;
  $("criteria").value = p.criteria; $("desc").value = p.desc;
};
$("preset").value = "backend";
$("preset").onchange();

$("scoreBtn").onclick = async () => {
  const body = {
    job_title: $("title").value.trim(),
    required_skills: $("skills").value.split(",").map((s) => s.trim()).filter(Boolean),
    min_experience: Number($("minexp").value) || 0,
    num_candidates: Math.max(1, Number($("topn").value) || 5),
    additional_criteria: $("criteria").value.trim(),
    job_description: $("desc").value.trim() || null,
  };
  if (body.job_title.length < 2) { toast("Give the job a title first."); $("title").focus(); return; }
  $("scoreBtn").disabled = true;
  try {
    const task = await api("/api/match", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    state.task = task;
    state.decisions = await api(`/api/decisions?job_key=${encodeURIComponent(task.job_key)}`);
    await poll(task.id, (t) => {
      state.task = t;
      setProgress($("matchProgress"), t.done, t.total,
        t.status === "done" ? `Scored ${t.results.length} of ${t.total}` : `Scoring with the LLM: ${t.done} / ${t.total}`);
      showLog($("matchLog"), t.messages);
      render();
    });
  } catch (e) {
    toast(e.message);
  } finally {
    $("scoreBtn").disabled = false;
  }
};

// ------------------------------------------------------------------ rendering

const resultOf = (id) => state.task?.results.find((r) => r.id === id);
const isBlind = () => $("blind").checked;
const labelOf = (c) => (isBlind() && !state.revealed.has(c.id) ? c.id : `${c.id} · ${c.name}`);
const statusOf = (id) => state.decisions[id] || "pending";

function render() {
  const scored = state.task?.results || [];
  $("count").textContent = state.candidates.length;
  $("sTotal").textContent = state.candidates.length;
  $("sScored").textContent = scored.length;
  $("sStrong").textContent = scored.filter((r) => r.fit_score >= 75).length;
  $("sFlag").textContent = state.candidates.filter((c) => c.injection.flagged).length;
  const job = state.task?.job;
  $("jobTitle").textContent = job ? job.job_title : "Talent Orbit";
  $("headNote").textContent = job
    ? "Distance from the job = 100 − fit score. Each profile orbits on its own plane. Grey moons are not scored yet."
    : "Load candidates, describe the job, then score. Grey moons are waiting to be scored.";

  const planets = state.candidates.map((c) => {
    const r = resultOf(c.id);
    return {
      id: c.id, label: isBlind() ? c.id : c.name, profile: c.profile,
      score: r ? r.score : 0, pending: !r, quarantined: c.injection.flagged, status: statusOf(c.id),
    };
  });
  const sig = JSON.stringify([job?.job_title, planets]);
  if (sig !== state.sig) {
    state.sig = sig;
    orbit.setData({ job: { title: job ? job.job_title : "Your job offer" }, candidates: planets });
    const colors = orbit.getProfileColors();
    $("legend").innerHTML = Object.entries(colors).map(([p, c]) => `<span><i style="background:${c}"></i>${esc(p)}</span>`).join("")
      + `<span><i style="background:#FF6B4F;border-radius:2px;transform:rotate(45deg)"></i>Flagged</span>`
      + `<span><i style="background:#5B6673"></i>Not scored</span>`;
  }
  renderRows(scored);
  renderDetail();
}

function bar10(v) {
  return `<span class="bar10"><span class="t"><i style="width:${v * 10}%"></i></span>${v}</span>`;
}

function renderRows(scored) {
  const topN = state.task?.job?.num_candidates || 0;
  if (!scored.length) {
    $("rows").innerHTML = `<tr><td colspan="8" class="muted">${state.task ? "Scoring… the first results appear in a few seconds." : "No scores yet."}</td></tr>`;
    return;
  }
  $("rows").innerHTML = scored.map((r, i) => {
    const flagged = r.injection.flagged;
    const st = statusOf(r.id);
    return `<tr class="r${r.id === state.selected ? " sel" : ""}${i === topN - 1 ? " cut" : ""}" data-id="${esc(r.id)}" tabindex="0">
      <td>${r.rank}</td>
      <td><b>${esc(labelOf(r))}</b>${i < topN ? '<span class="pill top">suggested</span>' : ""}${flagged ? '<span class="pill flag">flagged</span>' : ""}</td>
      <td>${esc(r.profile)}</td><td><b>${r.fit_score}</b></td>
      ${SUBS.map(([k]) => `<td>${bar10(r.justification[k])}</td>`).join("")}
      <td><span class="status ${st}">${STATUS[st]}</span></td></tr>`;
  }).join("");
  document.querySelectorAll("tr.r").forEach((tr) => {
    tr.onclick = () => select(tr.dataset.id);
    tr.onkeydown = (e) => { if (e.key === "Enter") select(tr.dataset.id); };
  });
}

function select(id) {
  state.selected = id;
  orbit.select(id);
  renderRows(state.task?.results || []);
  renderDetail();
}

function renderDetail() {
  const d = $("detail");
  const c = state.candidates.find((x) => x.id === state.selected);
  if (!c) { d.innerHTML = `<p class="muted">Click a planet or a row to see the LLM’s justification.</p>`; return; }
  const r = resultOf(c.id);
  const hidden = isBlind() && !state.revealed.has(c.id);
  const inj = c.injection;
  let html = `
    <h2>${esc(hidden ? c.id : c.name)}</h2>
    <div class="muted">${esc(c.profile)} profile · ${esc(c.source === "demo" ? "demo data" : c.filename)}</div>
    ${hidden ? `<button class="reveal" id="reveal">Reveal identity (after your assessment)</button>` : ""}
    ${inj.flagged ? `<div class="alert"><b>Instruction aimed at the AI found in this resume.</b> It was removed before the LLM read the resume, so it could not influence the score. Review this candidate yourself.${inj.excerpts.map((e) => `<code>“${esc(e)}”</code>`).join("")}${inj.hidden_chars ? "<div>Invisible characters were also found.</div>" : ""}</div>` : ""}`;

  if (!r) {
    html += `<p class="muted" style="margin-top:16px">Not scored for this job yet. Fill in the job offer and press “Score candidates”.</p>`;
  } else {
    const j = r.justification;
    const st = statusOf(c.id);
    html += `
      <div class="big">${r.fit_score}<span class="muted" style="font-size:16px"> / 100</span></div>
      <div class="muted">fit for ${esc(state.task.job.job_title)} · rank ${r.rank} of ${state.task.results.length}</div>
      <div class="subs">${SUBS.map(([k, label]) => `<div><span>${label}</span><span class="t"><i style="width:${j[k] * 10}%"></i></span><span>${j[k]}/10</span></div>`).join("")}</div>
      <dl>
        <dt>Skills</dt><dd>${esc(j.skill_match)}</dd>
        <dt>Experience</dt><dd>${esc(j.exp_match)}</dd>
        <dt>Education</dt><dd>${esc(j.edu_match)}</dd>
        <dt>Role fit</dt><dd>${esc(j.role_match)}</dd>
        <dt>Team player</dt><dd>${esc(j.team_player)}</dd>
        <dt>SWOT</dt><dd>${esc(j.swot_analysis)}</dd>
        <dt>Final justification</dt><dd>${esc(j.final_justification)}</dd>
      </dl>
      <h4 style="margin-top:16px">Profile (as extracted)</h4>
      <div class="facts">
        <div>Current title: ${esc(r.current_title)}</div>
        <ul>${r.experience.map((e) => `<li>${esc(e.title)} at ${esc(e.company)} (${esc(e.year)})</li>`).join("") || "<li>No experience listed</li>"}</ul>
        <div class="chips">${r.skills.map((s) => `<span class="chip">${esc(s)}</span>`).join("")}</div>
        <ul>${r.education.map((e) => `<li>${esc(e.degree)}, ${esc(e.university)}${e.year ? ` (${esc(e.year)})` : ""}</li>`).join("")}</ul>
        ${r.certifications.length ? `<div>Certifications: ${r.certifications.map((x) => esc(x.name)).join(", ")}</div>` : ""}
      </div>
      <div class="actions">
        <button class="btn primary" id="aShort">${st === "shortlisted" ? "Shortlisted" : "Shortlist"}</button>
        <button class="btn" id="aRej">Not selected</button>
        ${st !== "pending" ? `<button class="btn ghost" id="aReset">Reset</button>` : ""}
      </div>
      <p class="note">The agent only suggests. Your decision is recorded, with an audit entry, only when you click.</p>`;
  }
  d.innerHTML = html;
  $("reveal")?.addEventListener("click", () => { state.revealed.add(c.id); state.sig = ""; render(); });
  $("aShort")?.addEventListener("click", () => decide(c, "shortlisted"));
  $("aRej")?.addEventListener("click", () => decide(c, "rejected"));
  $("aReset")?.addEventListener("click", () => decide(c, "pending"));
}

async function decide(c, status) {
  if (c.injection.flagged && status === "shortlisted" &&
      !confirm("This resume tried to instruct the AI. Shortlist it anyway after your own review?")) return;
  try {
    await api("/api/decisions", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_key: state.task.job_key, candidate_id: c.id, status }),
    });
    state.decisions[c.id] = status;
    toast(status === "pending" ? "Decision cleared" : `${c.id}: ${STATUS[status].toLowerCase()}`);
    render();
  } catch (e) { toast(e.message); }
}

$("blind").onchange = () => { state.sig = ""; render(); };
$("pauseBtn").onclick = () => { $("pauseBtn").textContent = orbit.togglePaused() ? "Play orbits" : "Pause orbits"; };

// ------------------------------------------------------------------ start

(async () => {
  try {
    const h = await api("/api/health");
    const demo = h.llm.provider === "demo";
    $("llm").className = "llm" + (demo ? " demo" : "");
    $("llm").textContent = demo
      ? "Demo mode: rule-based scores, no LLM. Set LLM_PROVIDER in .env for real scoring."
      : `LLM: ${h.llm.provider} · ${h.llm.model} · storage: ${h.store}`;
    await loadCandidates();
  } catch (e) {
    $("llm").textContent = `Cannot reach the API (${e.message}). Start it with: uvicorn api.main:app`;
  }
})();
