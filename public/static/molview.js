// Protein viewer for target pages (3Dmol.js, self-hosted). Reads everything from data attributes (strict CSP).
//   data-structure: URL of a PDB file    data-mode: "model" (AlphaFold, colour by pLDDT) | "complex"
//   data-partners:  space-separated chain IDs to colour as the drug (antibody / peptide) in a complex
(() => {
  const el = document.querySelector("[data-structure]");
  if (!el) return;
  const fallback = el.querySelector("[data-fallback]");
  const fail = msg => { if (fallback) { fallback.hidden = false; fallback.textContent = msg; } };
  const gl = (() => { try { return !!document.createElement("canvas").getContext("webgl"); } catch (e) { return false; } })();
  if (!gl || !window.$3Dmol) { fail("3D view needs WebGL, which is not available in this browser."); return; }

  // AlphaFold confidence bands (same colours as the AlphaFold DB)
  const plddt = b => b >= 90 ? "#0053d6" : b >= 70 ? "#65cbf3" : b >= 50 ? "#ffdb13" : "#ff7d45";
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  fetch(el.dataset.structure).then(r => { if (!r.ok) throw new Error(r.status); return r.text(); }).then(pdb => {
    const v = $3Dmol.createViewer(el.querySelector(".mol-canvas"), { backgroundColor: "white", antialias: true });
    v.addModel(pdb, "pdb");
    if (el.dataset.mode === "model") {
      v.setStyle({}, { cartoon: { colorfunc: a => plddt(a.b) } });
    } else {
      const partners = (el.dataset.partners || "").split(" ").filter(Boolean);
      v.setStyle({}, { cartoon: { color: "#9fb3c8" } });
      if (partners.length) v.setStyle({ chain: partners }, { cartoon: { color: "#e0782b" } });
      v.setStyle({ hetflag: true }, { stick: { colorscheme: "orangeCarbon", radius: 0.28 } });
      v.setStyle({ resn: ["HOH", "SO4", "CL", "NA", "GOL", "EDO", "PEG"] }, {});
      v.setStyle({ resn: ["NAP", "PLM", "NAG"] }, { stick: { color: "#8a9aa9", radius: 0.15 } });
      const lig = { hetflag: true, not: { resn: ["HOH", "SO4", "CL", "NA", "GOL", "EDO", "PEG", "NAP", "PLM", "NAG", "ZN"] } };
      v.zoomTo(partners.length ? {} : lig);
    }
    if (el.dataset.mode === "model") v.zoomTo();
    v.render();
    if (!reduced) v.spin("y", 0.4);
    const spin = document.querySelector("[data-spin]");
    if (spin) {
      spin.setAttribute("aria-pressed", String(!reduced));
      spin.addEventListener("click", () => {
        const on = spin.getAttribute("aria-pressed") !== "true";
        spin.setAttribute("aria-pressed", String(on));
        v.spin(on ? "y" : false, 0.4);
      });
    }
    const reset = document.querySelector("[data-reset]");
    if (reset) reset.addEventListener("click", () => { v.zoomTo(); v.render(); });
    new ResizeObserver(() => { v.resize(); v.render(); }).observe(el);
  }).catch(() => fail("The structure could not be loaded."));
})();
