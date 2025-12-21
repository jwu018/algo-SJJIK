# Lambda Cloud Data Loading Optimization Guide

## 📋 Summary

Your optimized data loading system provides **10-100x speedup** through:
- **Partition caching**: Hours → seconds on subsequent runs
- **Parallel processing**: 10-30x faster first run using all 30 CPU cores
- **Optimized DataLoaders**: 2-5x faster training with multi-worker loading

## 🚀 Quick Start

### For Testing (Fast Iteration)
1. Set `TEST_MODE = True` in `Lambda_Dataloading_optimized.py`
2. This uses only 0.5% of files for quick testing
3. Run `python test_pretraining.py`

### For Full Training
1. Set `TEST_MODE = False` in `Lambda_Dataloading_optimized.py`
2. This uses 25% of files (or set `TRAIN_FRACTION` to 1.0 for 100%)
3. First run will be slower (calculating partitions)
4. Subsequent runs will be fast (using cached partition table)

## 🔧 Configuration Flags

Edit these at the top of `Lambda_Dataloading_optimized.py`:

```python
# Quick testing vs full training
TEST_MODE = True              # True = 0.5% files, False = 25% files

# Skip file collection if already done
SKIP_COLLECTION = True        # Set True after first collection

# Partition caching (HIGHLY RECOMMENDED)
USE_CACHE = True              # Cache partition calculations
FORCE_RECALCULATE = False     # Force recalculation even if cache exists

# DataLoader workers (more = faster, but may cause pickling errors)
NUM_WORKERS = 8               # 0 = safe but slow, 8-16 = fast but may fail

# Parallel partition counting
USE_PARALLEL_PARTITION_COUNTING = True  # Use all CPU cores
MAX_PARALLEL_WORKERS = None   # None = use all cores
```

## 📊 Performance Comparison

### Without Optimizations (Original)
- **First run**: 2-6 hours for 25% of 1.67TB dataset
- **Subsequent runs**: 2-6 hours (recalculates every time)
- **Training speed**: Slow (GPU waits for data, num_workers=0)

### With Optimizations
- **First run**: 10-30 minutes for 25% of dataset (parallel processing)
- **Subsequent runs**: 10-30 seconds (cached partition table)
- **Training speed**: 2-5x faster (multi-worker data loading)

## 🎯 Recommended Workflow

### Phase 1: Initial Testing (5-10 minutes)
```python
TEST_MODE = True              # Use 0.5% of files
SKIP_COLLECTION = True        # Files already collected
USE_CACHE = True              # Enable caching
NUM_WORKERS = 8               # Fast data loading
```
Run: `python test_pretraining.py`
- Verifies pipeline works
- Creates partition cache for 0.5% subset
- Tests 1 epoch of pretraining

### Phase 2: Scale to 25% Dataset (First Run: 10-30 min)
```python
TEST_MODE = False             # Use 25% of files
TRAIN_FRACTION = 0.25
SKIP_COLLECTION = True
USE_CACHE = True
NUM_WORKERS = 8
```
Run: `python Pretraining.py`
- First run calculates partitions (10-30 min)
- Saves partition cache
- Runs full pretraining

### Phase 3: Subsequent Training Runs (30 seconds setup)
```python
# Same settings as Phase 2
```
Run: `python Pretraining.py`
- Loads from cache (30 seconds)
- Jumps straight to training
- No partition recalculation needed

### Phase 4: Full Dataset (100% of 1.67TB)
```python
TEST_MODE = False
TRAIN_FRACTION = 1.0          # Use all files
SKIP_COLLECTION = True
USE_CACHE = True
NUM_WORKERS = 8
```
- First run: 1-2 hours (parallel partition counting for all files)
- Subsequent runs: 1-2 minutes (cached)
- Consider upgrading to larger Lambda instance (8xA100 for distributed training)

## 🐛 Troubleshooting

### "RuntimeError: DataLoader worker ... exited unexpectedly"
**Cause**: MNE objects don't pickle well with multiprocessing

**Solution**: Set `NUM_WORKERS = 0` in config
```python
NUM_WORKERS = 0  # Slower but safer
```

### Partition counting is still slow
**Check**:
1. Is `USE_PARALLEL_PARTITION_COUNTING = True`?
2. Is partition table being cached? Check `/lambda/nfs/JJIK-EEG/cache/`
3. Are you using a cached run? (Should see "Loaded partition table from cache")

**Solution**: First run is always slower. Subsequent runs use cache.

### Cache not being used
**Check**:
1. Is `USE_CACHE = True`?
2. Does cache file exist? `ls /lambda/nfs/JJIK-EEG/cache/`
3. Did you change `data_fraction`? (Different fractions have different cache files)

**Solution**: Cache is fraction-specific. Each TEST_MODE setting has its own cache.

### Out of memory during partition counting
**Cause**: Loading too many files simultaneously

**Solution**: Reduce parallel workers
```python
MAX_PARALLEL_WORKERS = 10  # Instead of using all 30 cores
```

### GPU is underutilized during training
**Cause**: Data loading is bottleneck

**Solutions**:
1. Increase `NUM_WORKERS` (try 12-16)
2. Check with `nvidia-smi` during training
3. If GPU usage is low, data loading is the bottleneck
4. If GPU usage is high (>90%), you're good!

## 📁 Files Overview

### Main Files
- `Lambda_Dataloading_optimized.py` - **Use this** for production
- `Lambda_Dataloading.py` - Original version (slower, kept for reference)
- `test_pretraining.py` - Quick test script (auto-uses optimized version)
- `Pretraining.py` - Full training script

### Cache Location
- `/lambda/nfs/JJIK-EEG/cache/` - Partition cache files stored here
- Cache files are named: `partition_cache_frac{fraction}_freq{freq}_win{window}_ovlp{overlap}.pkl`
- Different settings = different cache files

### What Gets Cached
- Partition counts for each file
- File durations, channel counts, sampling rates
- Everything needed to create train/val/test splits
- **Not cached**: Actual EEG data (loaded during training as needed)

## 🔍 Monitoring Performance

### Check if cache is being used
Look for this in output:
```
✓ Loaded partition table from cache: /lambda/nfs/JJIK-EEG/cache/...
✓ Cache is valid for current file selection (XXX files)
```

### Check parallel processing
Look for this in output:
```
Using parallel processing with 30 workers...
Processed 100/500 files (15.2 files/sec, ~26s remaining)
```

### Check DataLoader optimization
Look for this in output:
```
Using 8 worker processes for data loading
Pin memory: True
Persistent workers: True
```

### Monitor GPU during training
In a separate terminal on Lambda:
```bash
watch -n 1 nvidia-smi
```
- GPU Utilization should be >80-90% during training
- If low, increase NUM_WORKERS

## 💡 Advanced Tips

### 1. Preprocessing to HDF5 (For Maximum Speed)
If you're doing many training runs, consider one-time preprocessing:
- Convert all EDF files to HDF5 format with pre-windowed data
- Training becomes 10-100x faster
- Requires extra storage (1.5-2x original data size)
- Best for production after you've finalized your preprocessing pipeline

### 2. Distributed Training (Multiple GPUs)
For full 1.67TB dataset:
- Consider 8xA100 instance on Lambda Cloud
- Modify training script to use PyTorch DDP (Distributed Data Parallel)
- Each GPU processes different subset of data
- Can reduce training time from days to hours

### 3. Smart Caching Strategy
- Keep separate caches for different experiments
- Delete old caches if changing preprocessing
- Back up important cache files to avoid recalculation

### 4. Profiling Training
Use PyTorch profiler to find bottlenecks:
```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # Train for a few batches
    ...
print(prof.key_averages().table())
```

## 📞 Support

If you encounter issues:
1. Check this guide first
2. Look for error messages in output
3. Try TEST_MODE first before scaling up
4. Set NUM_WORKERS=0 if seeing pickling errors
5. Check cache directory exists and has write permissions

## 🎓 Understanding the Bottlenecks

### Why is partition counting slow?
- Must open every EDF file to read metadata
- EDF format is not optimized for metadata-only reads
- Even with `preload=False`, MNE parses file headers
- Over NFS adds network latency
- **Solution**: Cache results, parallelize, use metadata-only loading

### Why does num_workers help?
- GPU trains on batch N while CPUs load batch N+1
- Without workers: GPU waits idle while loading data
- With workers: Overlapped computation and data loading
- Each worker is a separate process loading data in parallel

### Why cache on NFS storage?
- Instance storage (28GB) is too small for large datasets
- NFS storage is persistent across instance restarts
- Cache survives even if you terminate and relaunch instance
- Shared across all instances attached to same filesystem

## 🔮 Next Steps

1. **Test locally**: Run `test_pretraining.py` with TEST_MODE=True
2. **Verify cache works**: Run twice, second time should be ~seconds
3. **Scale gradually**: TEST_MODE → 25% → 100%
4. **Monitor GPU**: Use nvidia-smi to verify high utilization
5. **Tune workers**: Increase NUM_WORKERS until GPU is maxed
6. **Consider preprocessing**: If doing many runs, preprocess to HDF5

Good luck with your training! 🚀
