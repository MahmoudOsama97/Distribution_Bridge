"""
Build two isolated data roots for E1 (source-of-gain disentanglement), so the
Diff-Src and OT conditions can't leak into each other during training:

  data_diffsrc/pacs/  -> 4 original domains (symlink) + only SynDomain_DiffSrc_* folders
  data_ot/pacs/       -> 4 original domains (symlink) + only SynDomain_OT_* folders

(No "Base" data root needed - the plan reuses the paper's existing Table 9 "Base"
column numbers instead of rerunning the baseline.)

Usage:
  python prepare_condition_dataroots.py --base_data_root .../data \
      --diffsrc_root .../data_diffsrc --ot_root .../data_ot
"""
import argparse
import os

DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]


def build_root(base_pacs_dir, out_root, prefix):
    out_pacs = os.path.join(out_root, "pacs")
    os.makedirs(out_pacs, exist_ok=True)

    for d in DOMAINS:
        link_path = os.path.join(out_pacs, d)
        target = os.path.join(base_pacs_dir, d)
        if not os.path.exists(link_path):
            os.symlink(target, link_path)

    n_syn = 0
    for folder in sorted(os.listdir(base_pacs_dir)):
        if folder.startswith(prefix):
            link_path = os.path.join(out_pacs, folder)
            target = os.path.join(base_pacs_dir, folder)
            if not os.path.exists(link_path):
                os.symlink(target, link_path)
            n_syn += 1
    print(f"{out_root}: linked 4 original domains + {n_syn} '{prefix}*' synthetic domains")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_data_root", required=True, help="dir containing pacs/ with all SynDomain_* folders")
    ap.add_argument("--diffsrc_root", required=True)
    ap.add_argument("--ot_root", required=True)
    args = ap.parse_args()

    # os.symlink interprets a relative target relative to the LINK's own directory,
    # not the CWD - always resolve to absolute paths before creating links.
    args.base_data_root = os.path.abspath(args.base_data_root)
    args.diffsrc_root = os.path.abspath(args.diffsrc_root)
    args.ot_root = os.path.abspath(args.ot_root)

    base_pacs_dir = os.path.join(args.base_data_root, "pacs")
    build_root(base_pacs_dir, args.diffsrc_root, "SynDomain_DiffSrc_")
    build_root(base_pacs_dir, args.ot_root, "SynDomain_OT_")


if __name__ == "__main__":
    main()
