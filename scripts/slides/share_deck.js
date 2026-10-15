// A deck to share after a session: the presented slides as pictures, with map pictures
// inserted after a given slide (one per slide, with a caption), as .pptx.
// Needs pptxgenjs (npm install pptxgenjs@3.12.0). Spec (JSON):
//   { "title", "slides_dir" (slide-N.png), "n_slides", "insert_after", "out",
//     "maps_title", "maps_intro", "maps": [[png, name, sub, note], ...] }
// Run: node scripts/slides/share_deck.js spec.json
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");
const spec = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "Alexander Bakker, Moraine Water";
pres.title = spec.title;
pres.theme = { headFontFace: "Calibri", bodyFontFace: "Calibri" };
const TEAL = "2AB4A8", INK = "333332", MUTED = "7D8688";
const pad = (n) => (spec.n_slides >= 10 ? String(n).padStart(2, "0") : String(n));

const slide = (i) => {
  const s = pres.addSlide();
  s.addImage({ path: path.join(spec.slides_dir, `slide-${pad(i)}.png`), x: 0, y: 0, w: 10, h: 5.625, altText: `Dia ${i}` });
};
for (let i = 1; i <= spec.insert_after; i++) slide(i);

const intro = pres.addSlide();
intro.background = { color: TEAL };
intro.addText(spec.maps_title, { x: 0.6, y: 1.5, w: 8.8, h: 1.0, fontSize: 36, bold: true, color: "FFFFFF", isTextBox: true, margin: 0 });
intro.addText(spec.maps_intro, { x: 0.6, y: 2.7, w: 8.8, h: 2.0, fontSize: 16, color: "FFFFFF", isTextBox: true, margin: 0, valign: "top" });
for (const [file, name, sub, note] of spec.maps) {
  const s = pres.addSlide();
  s.addImage({ path: file, x: 0, y: 0, w: 7.875, h: 5.625, altText: `Kaartbeeld ${name}` });
  s.addText(name, { x: 8.1, y: 0.5, w: 1.7, h: 1.4, fontSize: 18, bold: true, color: TEAL, isTextBox: true, margin: 0, valign: "top" });
  s.addText(sub, { x: 8.1, y: 2.0, w: 1.7, h: 0.9, fontSize: 12, color: MUTED, isTextBox: true, margin: 0, valign: "top" });
  if (note) s.addText(note, { x: 8.1, y: 3.0, w: 1.7, h: 2.2, fontSize: 12, color: INK, isTextBox: true, margin: 0, valign: "top" });
}
for (let i = spec.insert_after + 1; i <= spec.n_slides; i++) slide(i);
pres.writeFile({ fileName: spec.out }).then((f) => console.log(f));
