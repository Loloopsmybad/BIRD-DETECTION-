# PROGRESS.md — shared work log (append-only, newest first)

Convention: one dated block per work session. `Agent:` = which OpenCode
session wrote it. Keep entries as terse facts with paths/numbers.

---

## 2026-11-16 — kickoff extras (agent: gentle-wizard, build)

- **Obsidian notes written** (user's standing request):
  `Project Logs/Bird Detector From Scratch.md` (live diary with production
  numbers + kickoff entry) + concept notes `Object Detection.md`,
  `Training From Scratch vs Fine-Tuning.md`, `Knowledge Distillation.md`,
  `Quantization.md`; index `Artificial Intelligence.md` updated with all
  links. Style follows the vault conventions from session `cosmic-lagoon`.
- `bird.ipynb` validated (JSON parses, 11 cells).
- Waiting on: user to Run-all `training/new_model_fast/bird.ipynb` on
  Colab T4 → results land in `MyDrive/BIRD/scratch_v1/`. After that:
  distillation data + custom model design phase.

---

## 2026-11-16 — new_model_fast project kickoff (agent: gentle-wizard, build)

- **Board stopped & verified clean** (user request, pre-model work):
  `birdcpp` + `birdboard` + `camkit` all stopped (`systemctl is-active`:
  inactive/inactive/failed), no bird/frame_pumper processes, no 8888/8889/
  8890 listeners, no docker containers. Note: `birdboard` restarted itself
  once right after first SSH (~06:49 UTC) — if seen again, check who starts
  it (it is `disabled` at boot; suspect manual start scripts).
- **Committed the folder reorg** as renames: commit `571f710`
  "reorganize training: consolidate V8 fine-tuning era into training/V8
  fine tuning". Also folded in the pre-existing `fastyolo/` deletion
  (user's cleanup — it was the YOLO-FastestV2 PC-side checker folder).
- **Session archaeology done** (via `tools/read_session.py` on the local
  OpenCode DB):
  - `curious-mountain` (Nov 4-10) → board/deployment facts summarized and
    baked into `AGENTS.md` (benchmarks table, ORT dlopen, benchmark
    protocol, gotchas, INT8 = standing next step, fv2-retrain conclusion).
  - `cosmic-lagoon` (Nov 15) → vault conventions: explain first, then
    full-rigor notes, tables/LaTeX/`[[links]]`, keep an index note.
- **Colab bridge status**: user's "bird" Colab kernel is browser-only (no
  local jupyter/tunnel on PC). Decision: notebook
  `training/new_model_fast/bird.ipynb` is self-contained (Run all → results
  to `MyDrive/BIRD/scratch_v1/`), plus an OPTIONAL ssh/cloudflared tunnel
  cell for interactive GPU work later.
- **Created** `tools/read_session.py`, `AGENTS.md`, `PROGRESS.md`,
  `training/new_model_fast/bird.ipynb` (from-scratch baseline,
  yolov8n.yaml + pretrained=False, 250 epochs), `training/new_model_fast/README.md`.
- **Next**: user Run-all's `bird.ipynb` on Colab T4 (~2.5 h) → then custom
  PyTorch model design in `training/new_model_fast/custom_model/`.
