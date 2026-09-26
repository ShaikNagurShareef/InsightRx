// RetiLink: progressive enhancement only. Every form works without JS.
(() => {
  const MAX_SIDE = 2048, MAX_BYTES = 3.5 * 1024 * 1024;

  // ---- uploads: drag & drop, previews, client-side downscale of oversized photos
  async function shrink(file) {
    if (!/^image\/(jpeg|png)$/.test(file.type)) return file;
    const bmp = await createImageBitmap(file).catch(() => null);
    if (!bmp) return file;
    const scale = Math.min(1, MAX_SIDE / Math.max(bmp.width, bmp.height));
    if (scale === 1 && file.size <= MAX_BYTES) return file;
    const c = document.createElement("canvas");
    c.width = Math.round(bmp.width * scale); c.height = Math.round(bmp.height * scale);
    c.getContext("2d").drawImage(bmp, 0, 0, c.width, c.height);
    const blob = await new Promise(r => c.toBlob(r, "image/jpeg", 0.92));
    return new File([blob], file.name.replace(/\.\w+$/, "") + ".jpg", { type: "image/jpeg" });
  }
  document.querySelectorAll("[data-drop]").forEach(zone => {
    const input = zone.querySelector("input[type=file]");
    const list = zone.querySelector(".previews");
    const note = zone.querySelector("[data-drop-note]");
    const render = files => {
      if (!list) return;
      list.replaceChildren();
      [...files].forEach(f => {
        const fig = document.createElement("figure");
        const img = document.createElement("img");
        img.alt = ""; img.src = URL.createObjectURL(f);
        const cap = document.createElement("figcaption");
        cap.textContent = (f.size / 1024 / 1024).toFixed(1) + " MB";
        fig.append(img, cap); list.append(fig);
      });
    };
    const accept = async files => {
      const out = new DataTransfer(); let resized = 0;
      for (const f of files) { const s = await shrink(f); if (s !== f) resized++; out.items.add(s); }
      input.files = out.files; render(out.files);
      if (note) note.textContent = resized ? `${resized} large photo${resized > 1 ? "s" : ""} resized to 2048 px for upload.` : "";
    };
    input.addEventListener("change", () => accept(input.files));
    ["dragenter", "dragover"].forEach(e => zone.addEventListener(e, ev => { ev.preventDefault(); zone.classList.add("over"); }));
    ["dragleave", "drop"].forEach(e => zone.addEventListener(e, () => zone.classList.remove("over")));
    zone.addEventListener("drop", ev => { ev.preventDefault(); if (ev.dataTransfer.files.length) accept(ev.dataTransfer.files); });
  });

  // ---- small behaviours kept out of inline handlers (strict CSP)
  document.querySelectorAll("[data-autosubmit]").forEach(i => i.addEventListener("change", () => i.form.requestSubmit()));
  document.querySelectorAll("[data-back]").forEach(a => a.addEventListener("click", e => {
    if (history.length > 1) { e.preventDefault(); history.back(); }
  }));

  // ---- busy state on long actions (analysis, sending)
  document.querySelectorAll("form[data-busy]").forEach(f => f.addEventListener("submit", () => {
    const b = f.querySelector("button[type=submit], button:not([type])");
    if (!b) return;
    b.classList.add("busy"); b.setAttribute("aria-busy", "true");
    const label = b.querySelector("[data-label]");
    if (label && f.dataset.busy) label.textContent = f.dataset.busy;
    setTimeout(() => { b.disabled = true; }, 0);
  }));

  // ---- fundus viewer: thumbnails, attention overlay, opacity
  document.querySelectorAll("[data-eye]").forEach(col => {
    const frame = col.querySelector(".fundus");
    const main = frame && frame.querySelector("img.base");
    const over = frame && frame.querySelector("img.overlay");
    const toggle = col.querySelector("[data-attn-toggle]");
    const range = col.querySelector("[data-attn-range]");
    const select = id => {
      col.querySelectorAll("[data-img]").forEach(b => b.setAttribute("aria-pressed", b.dataset.img === id));
      col.querySelectorAll("[data-meta]").forEach(m => { m.hidden = m.dataset.meta !== id; });
      const btn = col.querySelector(`[data-img="${id}"]`);
      if (!btn || !main) return;
      main.src = btn.dataset.src; main.alt = btn.dataset.alt || "";
      if (over) { over.dataset.src = btn.dataset.attn || ""; over.removeAttribute("src"); frame.classList.remove("show-attn"); }
      if (toggle) { toggle.setAttribute("aria-pressed", "false"); toggle.hidden = !btn.dataset.attn; }
      const corner = frame.querySelector(".corner");
      if (corner) {
        corner.replaceChildren();
        const q = btn.dataset.q;
        if (q) {
          const t = document.createElement("span");
          t.className = "tag " + (q === "assessable" ? "clear" : q === "uncertain" ? "caution" : "signal");
          t.textContent = "Quality: " + q;
          corner.append(t);
        }
      }
    };
    col.querySelectorAll("[data-img]").forEach(b => b.addEventListener("click", () => select(b.dataset.img)));
    if (toggle && over) toggle.addEventListener("click", () => {
      const on = toggle.getAttribute("aria-pressed") !== "true";
      toggle.setAttribute("aria-pressed", on);
      if (on && !over.getAttribute("src") && over.dataset.src) {
        toggle.classList.add("busy");
        over.onload = () => toggle.classList.remove("busy");
        over.onerror = () => { toggle.classList.remove("busy"); toggle.hidden = true; };
        over.src = over.dataset.src;
      }
      frame.classList.toggle("show-attn", on);
      if (range) range.disabled = !on;
    });
    if (range) range.addEventListener("input", () => frame.style.setProperty("--attn", range.value / 100));
  });

  // ---- oculomics: linked highlighting between retina features, organ rows and the body map
  const light = (organs, on, feature) => {
    organs.forEach(o => document.querySelectorAll(`.bodymap [data-organ="${o}"], .organs [data-organ="${o}"]`)
      .forEach(el => el.classList.toggle("hl", on)));
    if (feature) document.querySelectorAll(".retina").forEach(r => {
      r.classList.toggle("focus", on);
      r.querySelectorAll(`[data-f="${feature}"]`).forEach(f => f.classList.toggle("hl", on));
    });
  };
  document.querySelectorAll(".features [data-links]").forEach(li => {
    const organs = li.dataset.links.split(" "), f = li.dataset.f;
    ["mouseenter", "focus"].forEach(e => li.addEventListener(e, () => light(organs, true, f)));
    ["mouseleave", "blur"].forEach(e => li.addEventListener(e, () => light(organs, false, f)));
  });
  document.querySelectorAll(".organs [data-organ]").forEach(li => {
    const o = [li.dataset.organ];
    ["mouseenter", "focus"].forEach(e => li.addEventListener(e, () => light(o, true)));
    ["mouseleave", "blur"].forEach(e => li.addEventListener(e, () => light(o, false)));
  });
})();
