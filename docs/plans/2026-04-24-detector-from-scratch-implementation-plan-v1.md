# Camera Agent Detector From-Scratch Implementation Plan (v1)

> **For Agent:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a complete pipeline to train person + face detector from scratch, from data preparation -> train -> evaluate -> checkpoint -> export, so it can be retrained and handed off for stable operation.

**Architecture:** Keep the current inference stack in `camera_agent/multi_task` (backbone + heads + unified detector), add an independent training stack with clear modules. Training stack uses FCOS-style target assignment compatible with current outputs (`*_cls`, `*_reg`, `*_ctr`).

**Tech Stack:** Python, PyTorch, TorchVision, timm, OpenCV, NumPy, PyYAML, tqdm, pycocotools (or equivalent evaluator), TensorBoard.

---

## 0) Mandatory Context for New Agents (No Prior Project History)

### 0.1 Current Status
- The `camera_agent/multi_task` branch already has inference for 2 tasks: person detection + face detection.
- Hand detection and face landmarks/keypoints have been removed.
- Optional teacher recognition using ArcFace in inference (`unified_detector.py`) but NOT in scope for training detector this time.
- Currently no complete training pipeline for the multi_task branch.

### 0.2 This Time's Scope
- In scope:
1. Train detector person+face from scratch.
2. Data pipeline + validation + experiment reproducibility.
3. Evaluate AP/AR and basic benchmark.
4. Export best checkpoint for reuse in inference.
- Out of scope:
1. Train/fine-tune ArcFace teacher recognizer.
2. Multi-camera identity fusion/handoff logic.
3. Device control logic in control_agent.

### 0.3 Data Contract Must Be Strictly Followed
- Class id detector:
1. `0 = person`
2. `1 = face`
- Class `roi` if exists in annotation source, DO NOT include in detector targets.
- Label format for training detector:
1. Each line: `class_id x_center y_center width height` (YOLO normalized).
2. Box within [0, 1].

### 0.4 Mandatory Rules When Running Commands
- Always activate conda env first:
  - `conda activate mas`
- Run commands from repo root: `MAS-Clr/`.

---

## 1) Technical Approaches and Selection Reasons

### Approach A (Recommended, Selected)
- 1 unified detector (person + face) sharing backbone/FPN, 2 separate heads.
- Advantages: lightweight, easy to deploy, utilizes shared features, fits current code.
- Disadvantages: need to balance loss and sampling well to avoid task imbalance.

### Approach B
- 2 separate detectors (person model, face model).
- Advantages: easy to optimize each task independently.
- Disadvantages: doubles inference cost, increases deployment complexity.

### Conclusion
- Choose Approach A as it fits real-time goals and current repo architecture.

---

## 2) Final Output Goals (Deliverables)

1. Have independent training modules for `camera_agent/multi_task`.
2. Have CLI-runnable train/eval/export scripts.
3. Have unit + smoke tests for critical components.
4. Have benchmark + reports on metrics per person/face.
5. Have handoff docs so new agents can join mid-project and continue.

---

## 3) Target File Structure

```text
camera_agent/
  multi_task/
    training/
      __init__.py
      configs/
        detector_base.yaml
        detector_smoke.yaml
      data/
        __init__.py
        dataset_spec.py
        index_builder.py
        yolo_parser.py
        transforms.py
        collate.py
        sanity_check.py
      targets/
        __init__.py
        fcos_assigner.py
      engine/
        __init__.py
        trainer.py
        evaluator.py
        metrics.py
        checkpoint.py
        ema.py
      utils/
        __init__.py
        seed.py
        logging_utils.py
        distributed.py
      cli/
        train_detector.py
        eval_detector.py
        export_best.py
    tests/
      test_dataset_parser.py
      test_transforms.py
      test_fcos_assigner.py
      test_loss_integration.py
      test_trainer_smoke.py
      test_evaluator_smoke.py
```

Note: file names can be adjusted slightly if needed, but must maintain the module meaning as above.

---

## 4) Overall Execution Order (Dependency Order)

1. Task 1-2: Finalize data contract + index + sanity.
2. Task 3-4: DataLoader + transforms + collate.
3. Task 5: FCOS target assignment.
4. Task 6: Loss integration.
5. Task 7-9: Trainer + evaluator + checkpoint/EMA.
6. Task 10: CLI train/eval/export.
7. Task 11-12: Test suite + benchmark + doc handoff.

Do not reverse Task 5 after Task 7 (trainer depends on target assignment).

---

## 5) Detailed Task Plan (For Handing to Small Agents)

### Task 1: Finalize Data Contract and Base Configs

**Objective:** Create a single source of truth for classes, image size, stride levels, training hyperparams.

**Files:**
- Create: `camera_agent/multi_task/training/configs/detector_base.yaml`
- Create: `camera_agent/multi_task/training/configs/detector_smoke.yaml`
- Create: `camera_agent/multi_task/training/data/dataset_spec.py`

**Steps:**
1. Declare detector classes only including `person`, `face`.
2. Declare default train image size (e.g. 640), strides `[8, 16, 32]`.
3. Declare optimizer, scheduler, batch size, num_workers, AMP, grad clip.
4. Declare from-scratch flag clearly: `pretrained_backbone: false`.
5. Write `dataset_spec.py` with dataclass to parse config + validate class map.
6. Add check for roi class if exists in labels to be skipped with logging warning.

**Module/Code Requirements:**
- Config has clear schema, with default values.
- If class map is wrong or missing class, fail-fast with clear message.

**DoD:**
- Have 2 runnable config files (base + smoke).
- Parse config successfully and print summary.

**Verify Command:**
- `conda activate mas && python -m camera_agent.multi_task.training.data.dataset_spec --config camera_agent/multi_task/training/configs/detector_smoke.yaml`

---

### Task 2: Create Dataset Index + Sanity Check Data

**Objective:** Convert annotation/raw data into unified index for train/eval.

**Files:**
- Create: `camera_agent/multi_task/training/data/index_builder.py`
- Create: `camera_agent/multi_task/training/data/sanity_check.py`

**Expected Input:**
- Root dataset (images + labels).
- List of splits train/val/test (if not exist, script creates split).

**Expected Output:**
- `manifests/train.jsonl`, `manifests/val.jsonl`, `manifests/test.jsonl`
- Each jsonl line: image_path, label_path, camera_id, frame_ts, split.

**Steps:**
1. Scan all paired image + label by filename.
2. Check file existence, empty labels, valid class_id.
3. Check bbox normalized within [0, 1].
4. Write jsonl for each split.
5. Generate stats report: num images, num bbox/class, ratio images without face.
6. Fail if error ratio exceeds threshold (e.g. >1%).

**Module/Code Requirements:**
- Script has strict and permissive modes.
- Clear error codes for each data error.

**DoD:**
- Run index script without crash on official dataset.
- Have stats report saved as markdown/json.

**Verify Command:**
- `conda activate mas && python -m camera_agent.multi_task.training.data.index_builder --config camera_agent/multi_task/training/configs/detector_base.yaml`
- `conda activate mas && python -m camera_agent.multi_task.training.data.sanity_check --manifest data/manifests/train.jsonl`

---

### Task 3: Implement YOLO Parser + Dataset Class

**Objective:** Create Dataset class returning normalized samples for trainer.

**Files:**
- Create: `camera_agent/multi_task/training/data/yolo_parser.py`
- Modify/Create: `camera_agent/multi_task/training/data/__init__.py`
- Create: Dataset part in `camera_agent/multi_task/training/data/dataset_spec.py` or separate `detector_dataset.py`
- Test: `camera_agent/multi_task/tests/test_dataset_parser.py`

**Steps:**
1. Parse 1 YOLO label file -> list bbox xyxy pixel + class.
2. Skip classes outside detector scope (roi), with counter.
3. Validate bbox minimum size (e.g. >= 2 px after scale) to avoid NaN.
4. Return target dict: boxes, labels, image_id, camera_id.
5. Read image with cv2/PIL, convert RGB float tensor.
6. Write unit tests for valid labels, invalid labels, empty labels.

**Module/Code Requirements:**
- Robust check for empty files, malformed lines.
- Do not silent fail.

**DoD:**
- Unit test parser pass.
- Dataset can iterate 100 samples continuously without crash.

**Verify Command:**
- `conda activate mas && pytest camera_agent/multi_task/tests/test_dataset_parser.py -v`

---

### Task 4: Augmentations + Collate Function

**Objective:** Have stable image/box transformation pipeline for train and val.

**Files:**
- Create: `camera_agent/multi_task/training/data/transforms.py`
- Create: `camera_agent/multi_task/training/data/collate.py`
- Test: `camera_agent/multi_task/tests/test_transforms.py`

**Steps:**
1. Train transforms: resize, random flip, color jitter, light random affine.
2. Val transforms: deterministic resize + normalize.
3. Ensure bbox updated correctly after each transform.
4. Clamp bbox to image boundary.
5. Remove bbox lost after transform.
6. Collate function batches variable-size targets.

**Module/Code Requirements:**
- Train/val transforms separated, configurable via config.
- Deterministic when seed set.

**DoD:**
- Test checks tensor shape correct.
- Test checks bbox after transform still valid.

**Verify Command:**
- `conda activate mas && pytest camera_agent/multi_task/tests/test_transforms.py -v`

---

### Task 5: FCOS Target Assignment (Core)

**Objective:** Generate cls/reg/ctr targets for person and face matching current head outputs.

**Files:**
- Create: `camera_agent/multi_task/training/targets/fcos_assigner.py`
- Test: `camera_agent/multi_task/tests/test_fcos_assigner.py`

**Steps:**
1. Implement grid points creation per strides `[8,16,32]`.
2. Assign GT to each level per size range (configurable).
3. Calculate ltrb regression targets for each positive point.
4. Calculate centerness target from ltrb.
5. Resolve conflicts multiple GT at 1 point (choose smaller box or higher IoU, clear rule).
6. Return target tensors for 2 tasks person/face.

**Module/Code Requirements:**
- Maximize vectorized, avoid large Python loops.
- Check NaN/Inf in output targets.

**DoD:**
- Unit test pass for toy case with expected target.
- Positive sample ratio not abnormal (with log).

**Verify Command:**
- `conda activate mas && pytest camera_agent/multi_task/tests/test_fcos_assigner.py -v`

---

### Task 6: Loss Integration into Training Graph

**Objective:** Connect model outputs + targets -> scalar losses for backward.

**Files:**
- Modify: `camera_agent/multi_task/losses.py` (if need to add helper)
- Create: `camera_agent/multi_task/training/engine/metrics.py` (loss meters part)
- Test: `camera_agent/multi_task/tests/test_loss_integration.py`

**Steps:**
1. Define loss compute per task: cls focal, reg giou, ctr bce.
2. Combine with `MultiTaskLoss` uncertainty weighting or fixed weights (config).
3. Ignore invalid/empty target samples properly.
4. Return detailed loss dict for logging.
5. Unit test loss finite on simulated batch.

**Module/Code Requirements:**
- No NaN when batch has no positive for 1 task.
- Assert shape before loss calculation.

**DoD:**
- Backward() runs on 1 smoke batch.
- Test loss integration pass.

**Verify Command:**
- `conda activate mas && pytest camera_agent/multi_task/tests/test_loss_integration.py -v`

---

### Task 7: Trainer Loop (Complete 1 Epoch Train)

**Objective:** Have trainer running train loop with AMP, optimizer, scheduler.

**Files:**
- Create: `camera_agent/multi_task/training/engine/trainer.py`
- Create: `camera_agent/multi_task/training/utils/seed.py`
- Create: `camera_agent/multi_task/training/utils/logging_utils.py`
- Test: `camera_agent/multi_task/tests/test_trainer_smoke.py`

**Steps:**
1. Initialize model with `pretrained=False`.
2. Create train/val dataloaders.
3. Forward -> target assign -> loss -> backward.
4. Apply AMP autocast + GradScaler.
5. Gradient clipping per config.
6. Optimizer step + scheduler step.
7. Log loss per step/epoch (console + tensorboard).
8. Save train history json.

**Module/Code Requirements:**
- Resume-safe: can continue from any epoch.
- Reproducible when seed set.

**DoD:**
- Smoke train 1 epoch completes.
- No crash when batch has image without face.

**Verify Command:**
- `conda activate mas && python -m camera_agent.multi_task.training.cli.train_detector --config camera_agent/multi_task/training/configs/detector_smoke.yaml`

---

### Task 8: Evaluator AP/AR per Task

**Objective:** Have module to evaluate detector quality on val/test.

**Files:**
- Create: `camera_agent/multi_task/training/engine/evaluator.py`
- Modify/Create: `camera_agent/multi_task/training/engine/metrics.py`
- Test: `camera_agent/multi_task/tests/test_evaluator_smoke.py`

**Steps:**
1. Normalize prediction format to COCO-like or internal evaluator format.
2. Calculate AP50, AP50-95 for person and face separately.
3. Calculate additional recall @score threshold.
4. Save results json + markdown summary.
5. Return main metric for best-model selection.

**Module/Code Requirements:**
- Metrics calculated separately per class, not combined.
- Assert no mismatch image_id to avoid wrong calculation.

**DoD:**
- Eval runs on any checkpoint.
- Have metric output file for experiment comparison.

**Verify Command:**
- `conda activate mas && python -m camera_agent.multi_task.training.cli.eval_detector --config camera_agent/multi_task/training/configs/detector_smoke.yaml --checkpoint outputs/smoke/latest.pth`

---

### Task 9: Checkpoint, EMA, Best Model Selection

**Objective:** Manage full checkpoints and have best model on val.

**Files:**
- Create: `camera_agent/multi_task/training/engine/checkpoint.py`
- Create: `camera_agent/multi_task/training/engine/ema.py`
- Modify: `camera_agent/multi_task/training/engine/trainer.py`

**Steps:**
1. Save `latest.pth` each epoch (model, optimizer, scheduler, scaler, epoch).
2. Save `best.pth` per priority metric (`face_ap50_95` + `person_ap50_95` weighted).
3. Add EMA weights update each step.
4. Allow eval with raw weights and EMA weights.
5. Support resume from latest/best.

**Module/Code Requirements:**
- Resume must restore full state (optimizer/scheduler/scaler).
- Checkpoint compatibility versioned.

**DoD:**
- Stop train midway, resume successful.
- Best checkpoint updated with correct logic.

**Verify Command:**
- `conda activate mas && python -m camera_agent.multi_task.training.cli.train_detector --config camera_agent/multi_task/training/configs/detector_smoke.yaml --resume outputs/smoke/latest.pth`

---

### Task 10: CLI Wrappers Train/Eval/Export

**Objective:** Standardize 1 entrypoint for each operation.

**Files:**
- Create: `camera_agent/multi_task/training/cli/train_detector.py`
- Create: `camera_agent/multi_task/training/cli/eval_detector.py`
- Create: `camera_agent/multi_task/training/cli/export_best.py`

**Steps:**
1. Parse args: config, resume, output_dir, device.
2. Call appropriate engine modules.
3. Print clear summary before and after run.
4. Export best checkpoint to inference usable path.

**Module/Code Requirements:**
- CLI has full `--help`.
- Exit code != 0 on config/data error.

**DoD:**
- 3 CLI scripts run independently.
- Ops team can use without manual code import.

**Verify Command:**
- `conda activate mas && python -m camera_agent.multi_task.training.cli.train_detector --help`
- `conda activate mas && python -m camera_agent.multi_task.training.cli.eval_detector --help`
- `conda activate mas && python -m camera_agent.multi_task.training.cli.export_best --help`

---

### Task 11: Mandatory Test Strategy

**Objective:** Have tests to prevent regression on critical modules.

**Test Files:**
- `camera_agent/multi_task/tests/test_dataset_parser.py`
- `camera_agent/multi_task/tests/test_transforms.py`
- `camera_agent/multi_task/tests/test_fcos_assigner.py`
- `camera_agent/multi_task/tests/test_loss_integration.py`
- `camera_agent/multi_task/tests/test_trainer_smoke.py`
- `camera_agent/multi_task/tests/test_evaluator_smoke.py`

**Steps:**
1. Write fixture mini dataset (5-10 images).
2. Unit tests for parser/transform/assigner.
3. Smoke train 1 epoch with detector_smoke.yaml.
4. Smoke eval from just trained checkpoint.
5. Set reasonable timeout for CI tests.

**Module/Code Requirements:**
- Tests do not depend on data not in repo.
- Test output deterministic in smoke mode.

**DoD:**
- `pytest camera_agent/multi_task/tests -v` pass all.

**Verify Command:**
- `conda activate mas && pytest camera_agent/multi_task/tests -v`

---

### Task 12: Benchmark, Acceptance Gate, Handoff Docs

**Objective:** Finalize pass/fail criteria to decide integration into main pipeline.

**Files:**
- Create: `docs/plans/2026-04-24-detector-from-scratch-benchmark-template.md`
- Update: `docs/camera_agent_technical_report.md`
- Update: `docs/plans/2026-04-21-camera-agent-multicam-multitask-srs-v1.md` (reference implementation plan)

**Steps:**
1. Run train base run (at least 1 full run).
2. Run eval on val and test.
3. Record metrics, speed, failure cases into benchmark template.
4. Finalize acceptance gate.
5. Update handoff guide docs.

**Acceptance Gate to Pass Task:**
1. Train run stable, no NaN.
2. Have best + latest checkpoints.
3. AP50 person >= agreed minimum threshold.
4. AP50 face >= agreed minimum threshold.
5. Smoke inference loads best checkpoint via `UnifiedDetector`.

**Verify Command:**
- `conda activate mas && python -m camera_agent.multi_task.demo_unified --source path/to/sample.mp4 --checkpoint outputs/base/best.pth`

---

## 6) Coding Standards for Each Task

1. Public functions must have short, clear input/output docstring.
2. Have type hints for core functions.
3. Do not hardcode absolute paths.
4. Error messages must include context (`image_path`, `label_path`).
5. Each engine module must have unified logging.

---

## 7) Handoff Protocol for Small Agents

Each agent when receiving task must send back report in format:

1. `Task ID` working on.
2. `Files touched` (create/modify/test).
3. `Commands run`.
4. `Verify result` (pass/fail + reason).
5. `Blockers` (if any).
6. `Next suggested task`.

If fail test/command:
1. Stop, log error clearly.
2. Do not fix outside task scope.
3. Hand back to coordinator with small fix suggestion.

---

## 8) Risk Register and Mitigation

1. Data mismatch class map (has roi class):
- Mitigate: parser skips roi + warning + sanity report.

2. High label noise (wrong face boxes):
- Mitigate: sanity check + visual audit random 200 images.

3. Task imbalance person/face:
- Mitigate: class-aware sampling + uncertainty loss weighting.

4. Unstable train from scratch:
- Mitigate: LR warmup, grad clip, EMA, AMP fallback off if needed.

5. Inference regression:
- Mitigate: smoke demo with new checkpoint before merge.

---

## 9) Commit Plan for Easy Review

1. Commit 1: configs + dataset_spec + docs data contract.
2. Commit 2: index_builder + sanity_check + tests.
3. Commit 3: parser + transforms + collate + tests.
4. Commit 4: fcos_assigner + tests.
5. Commit 5: loss integration + tests.
6. Commit 6: trainer + checkpoint + ema + smoke tests.
7. Commit 7: evaluator + metrics + smoke eval test.
8. Commit 8: CLI wrappers + docs update + benchmark template.

Each commit must run at least related test set before push.

---

## 10) Definition of Ready (Before Coding)

Task allowed to start when all conditions met:
1. Have related config file.
2. Clear input/output contract of task.
3. Clear test file to write.
4. Clear verify command and expected result.

## 11) Definition of Done (When Whole Plan Ends)

1. All tasks 1 -> 12 completed.
2. Full test suite pass.
3. Have at least 1 full training run + val/test report.
4. Have `best.pth` checkpoint demo loaded successfully.
5. Technical report and SRS reference new training pipeline.

---

## 12) Quick Run Checklist for Coordinator

1. Data readiness:
- Have manifests train/val/test.
- Sanity report not exceed error threshold.

2. Training readiness:
- Config smoke pass.
- Trainer smoke 1 epoch pass.

3. Quality readiness:
- Evaluator returns full metrics.
- AP person/face meet acceptable threshold.

4. Deployment readiness:
- Demo loads best checkpoint successfully.
- Logs and docs full for handoff.

---

## 13) Sample Full Pipeline Command (After Implementation Done)

```bash
conda activate mas && \
python -m camera_agent.multi_task.training.data.index_builder --config camera_agent/multi_task/training/configs/detector_base.yaml && \
python -m camera_agent.multi_task.training.data.sanity_check --manifest data/manifests/train.jsonl && \
python -m camera_agent.multi_task.training.cli.train_detector --config camera_agent/multi_task/training/configs/detector_base.yaml && \
python -m camera_agent.multi_task.training.cli.eval_detector --config camera_agent/multi_task/training/configs/detector_base.yaml --checkpoint outputs/base/best.pth && \
python -m camera_agent.multi_task.training.cli.export_best --checkpoint outputs/base/best.pth --out camera_agent/multi_task/checkpoints/detector_best.pth
```

If this command chain passes, project reaches operational baseline for detector from-scratch.