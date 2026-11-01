/**
 * TalentOrbit — 3D "solar system" view for job ↔ candidate matching.
 *
 * The job offer is the sun at the center. Each candidate is a planet whose
 * orbit radius is derived from its match score:  closer = better match.
 *
 *   radius = clamp((1 - score) / (1 - minScore) * maxRadius, minRadius, maxRadius)
 *
 * Candidates with the same `profile` share a tilted orbital plane, so similar
 * profiles cluster together in 3D.
 *
 * Usage:
 *   import { TalentOrbit } from "./talent-orbit.js";
 *   const orbit = new TalentOrbit(document.getElementById("orbit"));
 *   orbit.setData({ job: { title: "Backend Engineer" }, candidates: [...] });
 *   orbit.on("select", (candidate) => console.log(candidate));
 *
 * Candidate shape:
 *   {
 *     id: "C-001",            // required, unique
 *     score: 0.82,            // required, 0..1
 *     profile: "Backend",     // optional, groups candidates on the same plane
 *     status: "pending",      // "pending" | "shortlisted" | "rejected"
 *     quarantined: false,     // true → red crystal (see `quarantineOrbit`)
 *     warning: false,         // true → small amber moon
 *     pending: false,         // true → not scored yet: small grey moon on the outer edge
 *     label: "C-001"          // optional display label
 *   }
 *
 * Requires three.js (tested with 0.160.0).
 */
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

export const DEFAULT_PALETTE = [
  "#4FC3BE", "#9D8CF0", "#EE8DB4", "#8CCB5E", "#C9B79C",
  "#6FA8FF", "#FF9F6B", "#E6D86A",
];

const DEFAULTS = {
  maxRadius: 7.5,          // world radius for the lowest score shown
  minRadius: 1.3,          // closest any planet can get to the sun
  minScore: 0.2,           // scores at or below this sit on the outermost orbit
  thresholds: [
    { score: 0.9, label: "90%" },
    { score: 0.75, label: "75% shortlist shell", highlight: true },
    { score: 0.5, label: "50%" },
  ],
  palette: DEFAULT_PALETTE,
  background: 0x04060b,
  sunColor: "#FFB547",
  accent: "#4FC3BE",       // highlighted threshold + strong orbits
  danger: "#FF6B4F",       // quarantined candidates
  warn: "#FFC04A",         // warning moons
  ringColor: [255, 214, 140],
  autoRotate: true,
  orbitSpeed: 0.55,        // Kepler factor: speed = orbitSpeed * r^-1.5
  fog: 0.035,
  stars: 3000,
  showLabels: true,
  strongScore: 0.75,       // orbits above this get the accent color
  // Where flagged candidates orbit: "outer" (pushed out) or "score" (at their real score,
  // for pipelines where a flag must never change the ranking)
  quarantineOrbit: "outer",
  tooltip: (c) =>
    c.pending ? `<b>${esc(c.label || c.id)}</b> · waiting to be scored` : `<b>${esc(c.label || c.id)}</b> · ${Math.round(c.score * 100)}% match${c.quarantined ? " · flagged" : ""}${c.profile ? " · " + esc(c.profile) : ""}`,
};

function esc(s) {
  return String(s).replace(/[&<>"']/g, (m) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[m]));
}

export class TalentOrbit {
  constructor(container, options = {}) {
    if (!container) throw new Error("TalentOrbit: container element is required");
    this.container = container;
    this.opts = { ...DEFAULTS, ...options };
    this.listeners = { select: [], hover: [] };
    this.job = null;
    this.candidates = [];
    this.selectedId = null;
    this.paused = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches || false;
    this._texCache = new Map();
    this._planets = [];
    this._labels = [];
    this._profileIndex = new Map();
    this._init();
  }

  // ---------- Public API ----------

  /** Replace the job and candidates. Keeps orbital positions of candidates that remain. */
  setData({ job, candidates }) {
    this.job = job || { title: "Job offer" };
    this.candidates = (candidates || []).map((c) => ({
      status: "pending", quarantined: false, warning: false, ...c,
      score: Math.max(0, Math.min(1, Number(c.score) || 0)),
    }));
    this._profileIndex = new Map();
    for (const c of this.candidates) {
      const p = c.profile || "Other";
      if (!this._profileIndex.has(p)) this._profileIndex.set(p, this._profileIndex.size);
    }
    this._build();
  }

  /** Highlight a candidate (or pass null to clear). Does not fire "select". */
  select(id) {
    this.selectedId = id;
    this._build();
  }

  /** Update one candidate's status ("pending" | "shortlisted" | "rejected"). */
  setStatus(id, status) {
    const c = this.candidates.find((x) => x.id === id);
    if (c) { c.status = status; this._build(); }
  }

  setPaused(paused) { this.paused = !!paused; }
  togglePaused() { this.paused = !this.paused; return this.paused; }

  /** Colors used per profile, for building a legend outside the canvas. */
  getProfileColors() {
    const out = {};
    for (const [p, i] of this._profileIndex) out[p] = this.opts.palette[i % this.opts.palette.length];
    return out;
  }

  /** Orbit radius for a score, useful if you want to explain the scale. */
  radiusFor(score) {
    const { maxRadius, minRadius, minScore } = this.opts;
    return Math.min(maxRadius, Math.max(minRadius, ((1 - score) / (1 - minScore)) * maxRadius));
  }

  on(event, fn) {
    (this.listeners[event] ||= []).push(fn);
    return () => { this.listeners[event] = this.listeners[event].filter((f) => f !== fn); };
  }

  destroy() {
    cancelAnimationFrame(this._raf);
    this._ro?.disconnect();
    this.controls.dispose();
    this._clear(this.world);
    this._clear(this.nodes);
    this._texCache.forEach((t) => t.dispose());
    this.renderer.dispose();
    this.renderer.domElement.remove();
    this._overlay.remove();
    this._tip.remove();
  }

  // ---------- Setup ----------

  _init() {
    const o = this.opts;
    const el = this.container;
    if (getComputedStyle(el).position === "static") el.style.position = "relative";
    el.style.overflow = "hidden";

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    renderer.setClearColor(o.background, 1);
    if ("outputColorSpace" in renderer) renderer.outputColorSpace = THREE.SRGBColorSpace;
    el.prepend(renderer.domElement);
    renderer.domElement.style.display = "block";

    const scene = new THREE.Scene();
    if (o.fog) scene.fog = new THREE.FogExp2(o.background, o.fog);
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 300);
    camera.position.set(4, 6, 15);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.07;
    controls.minDistance = 3;
    controls.maxDistance = 40;
    controls.autoRotate = o.autoRotate && !this.paused;
    controls.autoRotateSpeed = 0.25;
    controls.addEventListener("start", () => { controls.autoRotate = false; });

    scene.add(this._makeStars(o.stars));
    scene.add(new THREE.AmbientLight(0xffffff, 0.35));
    scene.add(new THREE.PointLight(0xffe2b0, 3, 0, 0)); // decay 0: works with any three.js lighting mode

    this.world = new THREE.Group();
    this.nodes = new THREE.Group();
    scene.add(this.world, this.nodes);
    Object.assign(this, { renderer, scene, camera, controls });

    // HTML overlay for labels + tooltip
    this._overlay = document.createElement("div");
    Object.assign(this._overlay.style, { position: "absolute", inset: "0", pointerEvents: "none", font: "11px system-ui, sans-serif" });
    el.append(this._overlay);
    this._tip = document.createElement("div");
    Object.assign(this._tip.style, {
      position: "absolute", display: "none", pointerEvents: "none", whiteSpace: "nowrap",
      background: "rgba(14,18,26,.92)", color: "#E8EEF3", border: "1px solid rgba(255,255,255,.15)",
      borderRadius: "6px", padding: "5px 9px", font: "12.5px system-ui, sans-serif",
    });
    el.append(this._tip);

    const resize = () => {
      const w = el.clientWidth, h = el.clientHeight || 600;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    this._ro = new ResizeObserver(resize);
    this._ro.observe(el);
    resize();

    // Picking
    const ray = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    const canvas = renderer.domElement;
    const setMouse = (e) => {
      const b = canvas.getBoundingClientRect();
      mouse.set(((e.clientX - b.left) / b.width) * 2 - 1, -((e.clientY - b.top) / b.height) * 2 + 1);
      return b;
    };
    const pick = () => {
      ray.setFromCamera(mouse, camera);
      const h = ray.intersectObjects(this.nodes.children, true).find((i) => i.object.userData.candidate);
      return h ? h.object.userData.candidate : null;
    };
    canvas.addEventListener("pointermove", (e) => {
      const b = setMouse(e);
      const c = pick();
      canvas.style.cursor = c ? "pointer" : "grab";
      if (c) {
        this._tip.innerHTML = o.tooltip(c);
        this._tip.style.display = "block";
        this._tip.style.left = e.clientX - b.left + 14 + "px";
        this._tip.style.top = e.clientY - b.top - 10 + "px";
      } else this._tip.style.display = "none";
      this._emit("hover", c);
    });
    canvas.addEventListener("pointerleave", () => { this._tip.style.display = "none"; });
    let downAt = null;
    canvas.addEventListener("pointerdown", (e) => { downAt = [e.clientX, e.clientY]; });
    canvas.addEventListener("pointerup", (e) => {
      if (!downAt) return;
      const moved = Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]);
      downAt = null;
      if (moved > 5) return; // it was a drag, not a click
      setMouse(e);
      const c = pick();
      if (c) {
        controls.autoRotate = false;
        this.selectedId = c.id;
        this._build();
        this._emit("select", c);
      }
    });

    // Render loop
    const v = new THREE.Vector3();
    let last = 0;
    const loop = (t) => {
      this._raf = requestAnimationFrame(loop);
      const dt = Math.min(0.05, (t - last) / 1000 || 0);
      last = t;
      for (const p of this._planets) {
        if (!this.paused) { p.ang += p.speed * dt; p.mesh.rotation.y += dt * 0.6; }
        p.rad += (p.target - p.rad) * Math.min(1, dt * 1.6); // glide to a new score
        p.g.position.set(Math.cos(p.ang) * p.rad, 0, Math.sin(p.ang) * p.rad);
        p.line.scale.setScalar(p.rad);
      }
      if (!this.paused) {
        if (this._sun) {
          this._sun.rotation.y += dt * 0.2;
          this._corona.scale.setScalar(4.2 * (1 + Math.sin(t * 0.0018) * 0.05));
          this._corona2.scale.setScalar(8 + Math.sin(t * 0.0011) * 0.6);
        }
      }
      if (this._halo) this._halo.material.opacity = 0.55 + Math.sin(t * 0.005) * 0.25;
      controls.update();
      renderer.render(scene, camera);
      const w = canvas.clientWidth, h = canvas.clientHeight;
      for (const [d, get] of this._labels) {
        v.copy(get()).project(camera);
        d.style.display = v.z < 1 ? "block" : "none";
        d.style.left = ((v.x + 1) / 2) * w + "px";
        d.style.top = ((1 - v.y) / 2) * h + "px";
      }
    };
    this._raf = requestAnimationFrame(loop);
  }

  // ---------- Scene building ----------

  _build() {
    const o = this.opts;
    const prev = new Map(this._planets.map((p) => [p.id, p]));
    this._clear(this.world);
    this._clear(this.nodes);
    this._overlay.innerHTML = "";
    this._labels = [];
    this._planets = [];
    this._halo = null;
    if (!this.job) return;

    // Sun = job offer
    this._sun = new THREE.Mesh(new THREE.SphereGeometry(0.7, 64, 64), new THREE.MeshBasicMaterial({ map: this._bandTex(o.sunColor, 3) }));
    this._corona = this._glowSprite("#FFD27A", 4.2, 1);
    this._corona2 = this._glowSprite("#FF8A3D", 8, 0.35);
    this.world.add(this._corona2, this._corona, this._sun);
    this._label(this.job.title || "Job offer", new THREE.Vector3(0, -1.25, 0), "#FFE3B3", true);

    // Threshold shells
    for (const th of o.thresholds) {
      const r = this.radiusFor(th.score);
      const color = new THREE.Color(th.highlight ? o.accent : "#FFFFFF");
      this.world.add(new THREE.Mesh(new THREE.SphereGeometry(r, 48, 32),
        new THREE.MeshBasicMaterial({ color, transparent: true, opacity: th.highlight ? 0.07 : 0.025, depthWrite: false, side: THREE.BackSide })));
      this.world.add(new THREE.Mesh(new THREE.SphereGeometry(r, 24, 16),
        new THREE.MeshBasicMaterial({ color, wireframe: true, transparent: true, opacity: th.highlight ? 0.1 : 0.04, depthWrite: false })));
      this._label(th.label, new THREE.Vector3(0, r + 0.2, 0), th.highlight ? o.accent : "#9AA7B4");
    }

    // Group by profile so each profile gets its own orbital plane
    const groups = new Map();
    for (const c of this.candidates) {
      const p = c.profile || "Other";
      if (!groups.has(p)) groups.set(p, []);
      groups.get(p).push(c);
    }
    const nProfiles = Math.max(1, groups.size);
    const span = (2 * Math.PI) / nProfiles;
    const best = [...this.candidates].filter((c) => !c.quarantined && !c.pending).sort((a, b) => b.score - a.score)[0];

    for (const [profile, list] of groups) {
      const i = this._profileIndex.get(profile) ?? 0;
      const baseColor = o.palette[i % o.palette.length];
      const tilt = (i - (nProfiles - 1) / 2) * (1.6 / Math.max(1, nProfiles - 1 || 1));
      list.forEach((c, k) => {
        const frac = (k + 0.5) / list.length;
        const outer = c.quarantined && o.quarantineOrbit === "outer";
        const s = outer || c.pending ? 0 : c.score;
        const rad = c.pending ? o.maxRadius + 0.4 : outer ? o.maxRadius + 0.9 : this.radiusFor(s);
        const isSel = c.id === this.selectedId;
        const dim = c.status === "rejected";
        const color = c.quarantined ? o.danger : c.pending ? "#5B6673" : baseColor;

        const pivot = new THREE.Object3D();
        pivot.rotation.set(tilt + ((k % 4) - 1.5) * 0.08, i * 1.2 + ((k % 3) - 1) * 0.15, ((k % 5) - 2) * 0.06);
        const strong = !c.quarantined && !c.pending && s >= o.strongScore;
        const line = this._circle(1, isSel ? "#FFFFFF" : c.quarantined ? o.danger : strong ? o.accent : "#FFFFFF",
          isSel ? 0.85 : c.quarantined ? 0.25 : c.pending ? 0.04 : strong ? 0.3 : 0.08);
        pivot.add(line);

        const g = new THREE.Group();
        const size = c.quarantined ? 0.26 : c.pending ? 0.12 : 0.16 + s * 0.22;
        const mesh = c.quarantined
          ? new THREE.Mesh(new THREE.OctahedronGeometry(size),
              new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.5, roughness: 0.4, metalness: 0.3 }))
          : new THREE.Mesh(new THREE.SphereGeometry(size, 32, 32),
              new THREE.MeshStandardMaterial({ map: this._bandTex(color, k), roughness: 0.85, metalness: 0.05, transparent: dim, opacity: dim ? 0.18 : 1 }));
        mesh.rotation.z = 0.3;
        mesh.userData.candidate = c;
        g.add(mesh);

        if (c.status === "shortlisted") {
          const ring = new THREE.Mesh(new THREE.RingGeometry(size * 1.45, size * 2.3, 64),
            new THREE.MeshBasicMaterial({ map: this._ringTex(), side: THREE.DoubleSide, transparent: true, depthWrite: false }));
          ring.rotation.x = -Math.PI / 2.4;
          g.add(ring);
        }
        if (c.warning && !c.quarantined) {
          const moon = new THREE.Mesh(new THREE.SphereGeometry(0.06, 12, 12), new THREE.MeshBasicMaterial({ color: o.warn }));
          moon.position.set(size + 0.15, 0.1, 0);
          g.add(moon);
        }
        if (c.quarantined) g.add(this._glowSprite(o.danger, size * 5, 0.7));

        const above = () => g.getWorldPosition(new THREE.Vector3()).add(new THREE.Vector3(0, size + 0.4, 0));
        if (isSel) {
          this._halo = this._glowSprite("#FFFFFF", size * 6, 1);
          g.add(this._halo);
          this._label(`${c.label || c.id} · ${c.pending ? "not scored yet" : Math.round(c.score * 100) + "%"}${c.quarantined ? " · flagged" : ""}`, above, "#FFFFFF", true);
        } else if (c === best && o.showLabels) {
          this._label(`Best match ${c.label || c.id}`, above, "#FFE3B3", true);
        }

        const old = prev.get(c.id);
        const ang = old ? old.ang : i * span + span * (0.1 + 0.8 * frac);
        const cur = old ? old.rad : rad;
        g.position.set(Math.cos(ang) * cur, 0, Math.sin(ang) * cur);
        line.scale.setScalar(cur);
        pivot.add(g);
        this.nodes.add(pivot);
        this._planets.push({ id: c.id, g, mesh, line, rad: cur, target: rad, ang, speed: o.orbitSpeed * Math.pow(rad, -1.5) });
      });
    }
  }

  // ---------- Helpers ----------

  _emit(evt, payload) { (this.listeners[evt] || []).forEach((fn) => fn(payload)); }

  _label(text, posOrGetter, color, bold) {
    if (!this.opts.showLabels) return;
    const d = document.createElement("div");
    d.textContent = text;
    Object.assign(d.style, {
      position: "absolute", transform: "translate(-50%,-50%)", whiteSpace: "nowrap",
      color: color || "#9AA7B4", textShadow: "0 1px 3px #000", fontWeight: bold ? "600" : "400",
      fontSize: bold ? "12.5px" : "11px",
    });
    this._overlay.append(d);
    this._labels.push([d, typeof posOrGetter === "function" ? posOrGetter : () => posOrGetter]);
  }

  _clear(group) {
    while (group.children.length) {
      const obj = group.children.pop();
      obj.traverse((x) => { x.geometry?.dispose(); x.material?.dispose(); });
    }
  }

  _tex(canvas) {
    const t = new THREE.CanvasTexture(canvas);
    if ("colorSpace" in t) t.colorSpace = THREE.SRGBColorSpace;
    return t;
  }

  _glowSprite(color, scale, opacity) {
    const key = "glow" + color;
    if (!this._texCache.has(key)) {
      const c = document.createElement("canvas");
      c.width = c.height = 128;
      const x = c.getContext("2d");
      const g = x.createRadialGradient(64, 64, 0, 64, 64, 64);
      g.addColorStop(0, color); g.addColorStop(0.25, color + "AA"); g.addColorStop(0.6, color + "22"); g.addColorStop(1, color + "00");
      x.fillStyle = g; x.fillRect(0, 0, 128, 128);
      this._texCache.set(key, this._tex(c));
    }
    const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: this._texCache.get(key), blending: THREE.AdditiveBlending, depthWrite: false, opacity }));
    s.scale.setScalar(scale);
    return s;
  }

  _bandTex(color, variant) {
    const key = "band" + color + variant;
    if (!this._texCache.has(key)) {
      const c = document.createElement("canvas");
      c.width = 256; c.height = 128;
      const x = c.getContext("2d");
      x.fillStyle = color; x.fillRect(0, 0, 256, 128);
      let seed = variant * 97 + 13;
      const r = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
      for (let i = 0; i < 22; i++) {
        const y = r() * 128, h = 2 + r() * 12;
        x.fillStyle = r() > 0.5 ? `rgba(255,255,255,${0.05 + r() * 0.18})` : `rgba(0,0,0,${0.06 + r() * 0.22})`;
        x.fillRect(0, y, 256, h);
      }
      for (let i = 0; i < 3; i++) {
        x.fillStyle = `rgba(255,255,255,${0.12 + r() * 0.1})`;
        x.beginPath(); x.ellipse(r() * 256, 20 + r() * 88, 8 + r() * 14, 3 + r() * 5, 0, 0, Math.PI * 2); x.fill();
      }
      this._texCache.set(key, this._tex(c));
    }
    return this._texCache.get(key);
  }

  _ringTex() {
    if (!this._texCache.has("ring")) {
      const [R, G, B] = this.opts.ringColor;
      const c = document.createElement("canvas");
      c.width = 256; c.height = 8;
      const x = c.getContext("2d");
      for (let i = 0; i < 256; i++) {
        const a = ((Math.sin(i * 0.35) + 1) / 2) * 0.5 + 0.25;
        x.fillStyle = `rgba(${R},${G},${B},${a * (i < 20 || i > 236 ? 0.3 : 1)})`;
        x.fillRect(i, 0, 1, 8);
      }
      this._texCache.set("ring", this._tex(c));
    }
    return this._texCache.get("ring");
  }

  _circle(r, color, opacity) {
    const pts = [];
    for (let i = 0; i <= 160; i++) {
      const a = (i / 160) * Math.PI * 2;
      pts.push(new THREE.Vector3(Math.cos(a) * r, 0, Math.sin(a) * r));
    }
    return new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
      new THREE.LineBasicMaterial({ color, transparent: true, opacity, depthWrite: false }));
  }

  _makeStars(n) {
    const pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const u = Math.random() * 2 - 1, th = Math.random() * Math.PI * 2, rr = 55 + Math.random() * 45, q = Math.sqrt(1 - u * u);
      pos.set([rr * q * Math.cos(th), rr * u, rr * q * Math.sin(th)], i * 3);
      const t = Math.random(), b = 0.6 + Math.random() * 0.4;
      col.set(t < 0.1 ? [b, 0.85 * b, 0.7 * b] : t < 0.2 ? [0.75 * b, 0.85 * b, b] : [b, b, b], i * 3);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.BufferAttribute(col, 3));
    return new THREE.Points(g, new THREE.PointsMaterial({ size: 0.22, vertexColors: true, transparent: true, opacity: 0.9, depthWrite: false, fog: false }));
  }
}

export default TalentOrbit;
