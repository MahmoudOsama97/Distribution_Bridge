# Domain Distribution Bridge (DDB)

**Synthesizing Continuous Style Manifolds for Robust Generalization**

Mahmoud Soliman, Ahmed Elgazwy, Ahmed Radwan, Omar Abdelaziz, Ahmad Abdel-Qader, Islam Osman, Mohamed S. Shehata
The University of British Columbia, Kelowna, BC, Canada

Domain generalization (DG) methods typically treat source domains as discrete
samples, failing to cover the continuous style manifold between them. Distribution
Bridge (DDB) represents each domain-class pair as a full empirical distribution in
a pre-trained vision-language embedding space and performs optimal transport
displacement interpolation, producing intermediate distributions whose statistics vary
continuously along the Wasserstein geodesic between source domains. `K` uniformly
spaced interpolated distributions reduce the covering radius along a source-pair
geodesic by a factor of `K+1`, motivating a structural decomposition of the target
risk. The interpolated embeddings condition a diffusion model (unCLIP) to generate
images capturing intermediate domain statistics, which are then used to densify
training support for any downstream DG algorithm.

This repository contains the full pipeline: CLIP embedding extraction, Sinkhorn-OT
interpolation and unCLIP-conditioned synthetic domain generation, integration with
[DomainBed](https://github.com/facebookresearch/DomainBed) training, and the
verification/analysis scripts used to check the paper's theoretical claims empirically
(generation fidelity, extrapolation distance, covering-radius reduction).

## Repository layout

```
distribution_bridge/
├── generation/          <-- the DDB generation pipeline (see below)
│   ├── unclip_pipeline.py       # unCLIP img2img-embeds pipeline (raw CLIP embedding -> image)
│   ├── extract_embeddings.py    # Stage 1: CLIP embedding extraction + caching
│   ├── ot_interpolate.py        # Algorithm 1: Sinkhorn-OT + barycentric projection
│   ├── generate_ot.py           # Stage 2a: OT-interpolated synthetic domain generation
│   └── generate_diffsrc.py      # Stage 2b: no-interpolation control (endpoint conditioning)
├── data_prep/            # builds condition-specific / K-sweep DomainBed data roots
├── analysis/              # empirical verification of the paper's theoretical claims
│   ├── common.py                # shared exact-OT / D_max utilities
│   ├── eps_gen.py                # generation fidelity (Assumption 3.4)
│   ├── eps_extrap.py             # extrapolation distance to the source 1-skeleton (Def. 3.4)
│   ├── table7_reconcile.py       # Table 7 covering-radius statistics, sanity-gated
│   └── eps_extrap_normalized.py  # eps_extrap under the ℓ2-normalized embedding convention
└── smoke_test/            # environment + pipeline sanity checks

domainbed/                # DomainBed (Facebook Research) training framework, extended with
                           # DDB's two-tiered domain batching and SynDomain_* auto-discovery
environment/               # environment setup script
slurm_examples/             # example SLURM job templates for each pipeline stage
```

## The generation pipeline

This is the core contribution of the paper, and the part of this repository most
worth reading if you want to reproduce or build on DDB. For each unordered pair of
source domains `(i, j)` and each class `c`:

1. **Stage 1 - embedding extraction** (`distribution_bridge/generation/extract_embeddings.py`):
   every image is encoded with an OpenCLIP ViT-H/14 image encoder into
   `R^1024`, cached per `(domain, class)` as an `.npz` of the full embedding matrix
   plus its empirical mean/covariance.

2. **Stage 2a - OT interpolation + generation** (`distribution_bridge/generation/ot_interpolate.py`,
   `generate_ot.py`): for each interpolation weight `t in {1/6, ..., 5/6}`, Algorithm 1
   computes a Sinkhorn optimal-transport plan between the two domains' class-conditional
   empirical distributions, barycentrically projects, and linearly interpolates at `t`
   to produce a set of interpolant embeddings. A subset of these condition the unCLIP
   decoder (`unclip_pipeline.py`) to synthesize images whose CLIP embeddings track the
   interpolated distribution. The resulting images are written to
   `SynDomain_OT_<domA>_<domB>_t<k>of6/<class>/*.jpg` - a naming convention DomainBed's
   `datasets.py` auto-discovers and auto-prunes per leave-one-domain-out configuration
   (any synthetic pair touching the held-out test domain is excluded automatically).

3. **Stage 2b - Diff-Src control** (`generate_diffsrc.py`): an ablation condition with
   identical generation budget and decoder settings, but conditioning vectors are
   sampled directly from the two domains' *raw* embeddings (no OT interpolation) -
   used to isolate the OT interpolation's specific contribution from the effect of
   simply adding more synthetic training data.

Both generation scripts write per-image conditioning-vector metadata alongside the
JPEGs, consumed by the fidelity-verification scripts in `distribution_bridge/analysis/`.

See `slurm_examples/01_extract_embeddings.sh` through `03_generate_diffsrc.sh` for
runnable examples.

## Training

`domainbed/scripts/train.py` implements DDB's two-tiered domain batching: every
training step draws one mini-batch from each original source domain plus
`k_syn_domains_per_step` mini-batches from a random subset of the available
synthetic domains, so synthetic data augments rather than replaces the original
training distribution. Point `--data_dir` at a directory containing the original
domain folders alongside any `SynDomain_*` folders produced by Stage 2; see
`slurm_examples/04_train_domainbed.sh`. `distribution_bridge/data_prep/` contains
helpers for building condition-specific data roots (e.g. isolating only the OT or
only the Diff-Src synthetic domains, or subsampling to a fixed `K` for a covering-
radius sweep).

## Analysis / verification

`distribution_bridge/analysis/` re-encodes generated images and measures embedding-space
quantities directly from cached CLIP embeddings (no additional generation or
training required):

* **`eps_gen.py`** - generation fidelity: how closely a generated image's CLIP
  re-encoding matches its conditioning embedding (Assumption 3.4).
* **`eps_extrap.py`** - extrapolation distance: the worst-case Wasserstein-2 distance
  from a held-out target domain's class-conditional distribution to the nearest point
  on the source 1-skeleton (Definition 3.4), i.e. how far a target lies outside the
  region DDB's covering-radius guarantee actually covers.
* **`table7_reconcile.py`** - reproduces the paper's Table 7 covering-radius statistics
  from cached embeddings via exact optimal transport, with a pass/fail sanity gate
  against the published values before proceeding to further analysis.
* **`eps_extrap_normalized.py`** - `eps_extrap.py`'s measurement under the ℓ2-normalized
  embedding convention, with argmin-t and nearest-source-pair tracking for the
  supplementary per-class breakdown.

## Pre-generated synthetic data

Running the full generation pipeline from scratch takes several GPU-hours. The
PACS synthetic domains used in the paper's experiments (10,500 OT-interpolated +
10,500 Diff-Src control images, plus per-image conditioning-embedding metadata) are
available as a [Release asset](https://github.com/MahmoudOsama97/Distribution_Bridge/releases/tag/pacs-synthetic-v1)
rather than committed to the repository. Extract it into `data/pacs/` alongside the
4 original PACS domains (downloaded separately via `domainbed/scripts/download.py`)
to reproduce the paper's training setup without re-running Stage 2.

## Setup

```sh
export DDB_VENV=/path/to/ddb_env
export DDB_HF_HOME=/path/to/hf_cache
export DDB_TORCH_HOME=/path/to/torch_cache
./environment/setup_env.sh
```

This creates a virtual environment with the pinned dependencies for both the
DomainBed training framework and the generation/analysis pipeline (see
`domainbed/requirements.txt` and `environment/setup_env.sh`), and pre-downloads the
unCLIP, OpenCLIP, and ImageNet-pretrained ResNet-50 weights into an offline-reusable
cache - useful on clusters where compute nodes have no internet access.

Download the DomainBed benchmark datasets:

```sh
python -m domainbed.scripts.download --data_dir=./data
```

## Smoke test

Before running the full pipeline, `distribution_bridge/smoke_test/smoke_test.py` checks
that the environment and unCLIP pipeline load correctly and that round-trip
embedding fidelity (encode -> generate -> re-encode -> cosine similarity) is
sane on a small sample.

---

## DomainBed

This repository builds on [DomainBed](https://github.com/facebookresearch/DomainBed),
a PyTorch suite of benchmark datasets and algorithms for domain generalization
([Gulrajani and Lopez-Paz, 2020](https://arxiv.org/abs/2007.01434)). All 30+ of
DomainBed's algorithms and datasets remain available and unmodified except for the
two-tiered synthetic-domain batching extension described above. See
[`domainbed/algorithms.py`](domainbed/algorithms.py) and
[`domainbed/datasets.py`](domainbed/datasets.py) for the full list, and
[`domainbed/hparams_registry.py`](domainbed/hparams_registry.py) for hyperparameter
grids.

Launch a standard DomainBed sweep (see the DomainBed docs for `command_launchers.py`):

```sh
python -m domainbed.scripts.sweep launch \
       --data_dir=/my/datasets/path \
       --output_dir=/my/sweep/output/path \
       --command_launcher MyLauncher
```

## License

Released under the MIT license, included [here](LICENSE).

## Citation

If you use this code, please cite the paper (citation to be added on publication).
