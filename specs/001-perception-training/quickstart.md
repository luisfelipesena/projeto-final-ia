# Quickstart — Phase 2 Perception Model Training

Step-by-step guide for collecting datasets, training models, and exporting artifacts defined in the Phase 2 spec.

## 1. Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. LIDAR Dataset Collection
```bash
# Launch Webots with IA_20252.wbt, then run:
python scripts/collect_lidar_data.py \
  --output data/lidar/raw \
  --sessions 20 \
  --scans-per-session 60 \
  --seed 42

# Optional manual annotation/cleanup
python scripts/annotate_lidar.py \
  --input data/lidar/raw \
  --output data/lidar/annotated
```
- Ensure at least 1,000 scans after validation (`python scripts/validate_lidar_dataset.py`).
- Commit updated dataset manifest (`data/lidar/dataset_manifest.json`).

## 3. Camera Dataset Collection
```bash
python scripts/collect_camera_data.py \
  --output data/camera/raw \
  --sessions 15 \
  --frames-per-session 40 \
  --seed 1337

python scripts/annotate_camera.py \
  --input data/camera/raw \
  --output data/camera/annotated \
  --label-tool labelme
```
- Aim for ≥500 annotated frames with balanced cube colors and distances.

## 4. LIDAR Model Training & Export
```bash
python scripts/train_lidar_model.py \
  --data data/lidar/annotated \
  --config configs/lidar_default.yaml \
  --log-dir logs/perception/lidar_run_001 \
  --export models/lidar_net.pt
```
- Verify metrics via generated report (`logs/perception/lidar_run_001/metrics.json`).
- Run latency benchmark: `python scripts/profile_lidar_model.py models/lidar_net.pt`.
- Save metadata: `models/lidar_net_metadata.json`.

## 5. Camera Model Training & Export
```bash
python scripts/train_camera_model.py \
  --data data/camera/annotated \
  --config configs/camera_default.yaml \
  --log-dir logs/perception/camera_run_001 \
  --export models/camera_net.pt
```
- Validate accuracy/FPS using `python scripts/profile_camera_model.py`.
- Save metadata: `models/camera_net_metadata.json`.

## 6. Documentation & Compliance
1. Record architecture/training choices in `DECISIONS.md` (citing references).
2. Update `TODO.md` Phase 2 checklist.
3. Store plots/notebooks under `notebooks/lidar_training.ipynb` and `notebooks/camera_training.ipynb`.
4. Attach confusion matrices, FPS tables, and dataset stats to `docs/perception/`.

## 7. Verification Checklist
- [ ] LIDAR dataset ≥1,000 validated samples.
- [ ] Camera dataset ≥500 validated frames.
- [ ] LIDAR model: accuracy ≥90%, latency ≤100 ms.
- [ ] Camera model: per-color precision/recall ≥95%, FPS ≥10.
- [ ] Metadata + logs committed.
- [ ] DECISIONS.md updated with citations.

