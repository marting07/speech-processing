# Proximity Index Recognition (DTW Re-ranking)

## Objective

Accelerate recognition by narrowing candidates with index structures, then compute final decision using DTW over candidates.

Implementation reference: `speech_proximity.py`.

## Core Idea

1. Convert each variable-length feature matrix into a fixed pooled vector:
   - `v = [mean(features), std(features)]`
2. Insert pooled vectors into index.
3. At query time:
   - index returns candidate item IDs
   - compute exact DTW on candidate sequences
   - choose smallest DTW distance.

## Indexes Implemented

1. BK-tree (`bktree`)
- Tree keyed by bucketized distance.
- Uses DTW during insertion/search.

2. FQT (`fqt`)
- Recursive pivot tree with quantile distance bins on pooled vectors.

3. Fixed-Height FQT (`fixed_height_fqt`)
- Same as FQT with forced maximum depth.

4. Permutation Index (`permutation`)
- Pivot-based rank signatures; nearest by rank-difference score.

5. LSH (`lsh`)
- Random hyperplane hashing on pooled vectors.
- Retrieves union of matching buckets across hash tables.
- Optional DTW alignment-constellation visualization for best match.

## Persistence Layer

- Manager abstraction: `ProximityIndexManager`
- Save/load compressed index files (`.pidx.gz`) with:
  - index state
  - stored items
  - feature type and index type metadata.

## Reimplementation Steps

1. Build dictionary for one feature type.
2. Choose index type.
3. Insert items one-by-one or full ingest.
4. Save index snapshot.
5. Query with held-out utterances.
6. Measure:
   - candidate count
   - DTW final accuracy
   - query latency.

## Experimental Comparison Template

For each index type, report:
- Top-1 accuracy after DTW re-ranking
- Mean candidate pool size
- Mean query time
- Memory footprint (index stats panel)
