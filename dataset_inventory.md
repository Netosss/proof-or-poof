# Complete Dataset Inventory for Phase 11 Fine-Tuning

## Available Datasets (9 total)

### 1. **Kaggle Benchmark** (973 images)
- Path: `tests/data/kaggle_benchmark/`
- AI: `AiArtData/AiArtData/` (~538 images)
- Real: `RealArt/RealArt/` (~435 images)
- Ground Truth: Folder-based
- **Status:** Known false positives on RealArt (33% FP rate in sample)

### 2. **HF 200 Benchmark** (200 images)
- Path: `tests/data/hf_200_benchmark/`
- AI: `AI/` (100 images, prefix: `ai_hf_`)
- Real: `Real/` (100 images, prefix: `real_hf_`)
- Ground Truth: Filename prefix
- **Status:** 100% accuracy in live GPU test

### 3. **Benchmark 50** (100 images)
- Path: `tests/data/benchmark_50/`
- AI: `ai/` (50 images)
- Real: `original/` (50 images)
- Ground Truth: Folder-based
- **Status:** Original mixed dataset (Phase 1-2)

### 4. **Benchmark HF 50** (48 images)
- Path: `tests/data/benchmark_hf_50/`
- AI: `ai/` (24 images)
- Real: `original/` (24 images)
- Ground Truth: Folder-based
- **Status:** Phase 3 verification dataset

### 5. **GenImage Benchmark** (200 images)
- Path: `tests/data/genimage_benchmark/`
- Ground Truth: Unknown (needs investigation)
- **Status:** Unexplored dataset

### 6. **User AI Data** (9 images)
- Path: `tests/data/ai/images/`
- Ground Truth: AI (user-provided)
- **Status:** User's personal AI samples

### 7. **User Original Data** (6 images)
- Path: `tests/data/original/images/`
- Ground Truth: Original (user-provided)
- **Status:** User's personal original samples

### 8. **Screen Records** (4 videos)
- Path: `tests/data/screen_records/`
- AI: `ai/` (2 videos)
- Real: `original/` (2 videos)
- **Status:** Video dataset (separate handling)

### 9. **Screenshots** (2 images)
- Path: `tests/data/screenshots/`
- AI: `ai/` (1 image)
- Real: `original/` (1 image)
- **Status:** Screenshot dataset

---

## Total Image Count: ~1,340 images
## Priority for Fine-Tuning:
1. **Kaggle** (largest, has known issues)
2. **HF 200** (verified accurate baseline)
3. **GenImage** (unexplored, potential goldmine)
4. **Benchmark 50** (original test set)
5. **User Data** (real-world validation)
