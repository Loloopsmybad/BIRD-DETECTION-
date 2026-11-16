# PROGRESS.md — shared work log (append-only, newest first)

Convention: one dated block per work session. `Agent:` = which OpenCode
session wrote it. Keep entries as terse facts with paths/numbers.

---

## 2026-11-16 — BirdNet v1 built + smoke-validated; sandbox staged (agent: gentle-wizard, build)

- Custom-model skeleton committed: `training/new_model_fast/custom_model/`
  (`birdnet.py` ReLU modules + register(), `birdnet_v1.yaml` channel plan,
  `train_birdnet.py` VM runner, `README.md`). Commits `7b2d14f`, `03c995c`,
  `7953add`, `1f96a4b`.
- **Build verified on VM (ultralytics 8.4.215): BirdNet v1 = 194 layers,
  1.23 M params, 5.3 GFLOPs @640 (~2.6 @448); forward @448 decodes to
  `(1, 5, 4116)` = exact birdcpp contract.** Head rebuilt slim (32/32
  hidden vs v8n's 16/64).
- Integration fixes for ultralytics 8.4.215: `BConv.default_act` class attr
  (parse_model reads it as process-wide state), yaml `concat`→`Concat`,
  `BirdDetect(nc, reg_max, end2end, ch)` signature, `amp=False` (trainer's
  `check_amp()` builds a stock yolo26n.yaml that the monkey-patch corrupts).
- **Smoke run (2 ep @224) green end-to-end:** train → val → fuse (BN folded,
  1,229,235 params) → ONNX export `(1, 5, 1029)` @224 → Drive copy. BirdNet
  v1 launch-ready for when the T4 frees.
- Board sandbox staged: `/home/root/birdcpp-scratch/` (binary+frame_pumper+
  camera_url.txt copied from production) + `birdcpp-scratch.service`
  (:8890, `--img 448`, manual start). Production :8888 untouched, camera is
  viewer-gated so sandbox trials don't fight production.
- File transfer PC↔VM: dpaste up (curl `--data-urlencode content@file`),
  verified; VM→PC via cloudflared quick tunnel
  `https://general-concerning-lives-diameter.trycloudflare.com` → VM :8000.
- Baseline progress: ep 157/250, val mAP50 0.688 / mAP50-95 0.448, ~14 s/ep.

## 2026-11-16 — baseline training LAUNCHED on Colab T4 (agent: gentle-wizard, build)

- **Route used: official `googlecolab/colab-mcp` MCP server** (user
  suggested it — correct call). Installed `uv` into repo venv
  (`venv/Scripts/uvx.exe`), added `colab-mcp` to global
  `~\.config\opencode\opencode.jsonc` (needs `--python 3.13`:
  colab-mcp 1.0.1 requires >=3.13; uv defaults to 3.15 which fails
  building pyyaml). `open_colab_browser_connection()` → user connected
  their Colab browser tab → MCP unlocked live notebook tools
  (get_cells/update_cell/run_code_cell/add_code_cell).
- Hand-cracking the VS Code↔Colab secret store was abandoned mid-way
  (DPAPI v10 blobs = Chromium os_crypt AES-GCM; Local State key exists
  at `%APPDATA%\Code\Local State`) — MCP route made it moot.
- Colab runtime switched to **T4** (user) — verified: Tesla T4 15360 MiB,
  torch 2.14.1+cu130, CUDA ok, python 3.13.16.
- Drive mounted in notebook (`/content/drive`), dataset confirmed at
  `MyDrive/BIRD/data_detection` (760 train / 190 val, labels all present,
  1895 train boxes).
- `train_baseline.py` uploaded to VM `/content/train_baseline.py`
  (via MCP cell, base64 round-trip), py_compile OK.
- ultralytics 8.4.215 installed on VM.
- **Training launched**: PID 3776, `nohup`-style background
  (`/content/train_baseline.py --drive-save`), log `/content/train.log`,
  250 epochs @ 640, batch 16, patience 30, seed 0, from scratch
  (`yolov8n.yaml` + `pretrained=False`), results →
  `/content/scratch_runs/yolov8n_scratch_v1` + copy to
  `MyDrive/BIRD/scratch_v1`. Est. wall 2-3.5 h on T4.
- Notebook cells (current): 0=env check, 1=drive mount, 2=writer(script
  via base64), 3=decode+compile, 4=pip install+sanity, 5=launch, 6=log tail.

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
