# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader


def two_tiered_domain_sampling_iterator(loaders_dict, core_domain_indices, syn_domain_indices, k_syn_domains):
    """
    A generator that, at each step, yields a batch from ALL core domains
    and a batch from a random subset of k_syn_domains synthetic domains.
    """
    # Create persistent iterators for each loader
    iterators = {idx: iter(loader) for idx, loader in loaders_dict.items()}
    
    # Ensure k is not larger than the available synthetic domains
    k_syn_domains = min(k_syn_domains, len(syn_domain_indices))
    
    while True:
        # 1. Get a batch from every single core domain
        core_batches = [next(iterators[i]) for i in core_domain_indices]
        
        # 2. Randomly sample k synthetic domain indices
        sampled_syn_indices = random.sample(syn_domain_indices, k_syn_domains)
        
        # 3. Get a batch from each of the sampled synthetic domains
        syn_batches = [next(iterators[i]) for i in sampled_syn_indices]
        
        # 4. Yield the combined list of batches for this step
        yield core_batches + syn_batches
# --- INSERT THIS NEW HELPER FUNCTION AT THE TOP OF train.py ---

def domain_sampling_iterator(loaders_dict, domain_indices, k_domains):
    """
    A generator that, at each step, randomly samples k domains and yields
    a batch from each of them.
    Args:
        loaders_dict (dict): A dictionary mapping domain_index -> dataloader.
        domain_indices (list): A list of the keys in loaders_dict to sample from.
        k_domains (int): The number of domains to sample at each step.
    """
    # Create persistent iterators for each loader
    iterators = {idx: iter(loader) for idx, loader in loaders_dict.items()}
    while True:
        # At each step, randomly sample k domain indices without replacement
        sampled_indices = random.sample(domain_indices, k_domains)
        # Yield a list containing one batch from each of the k sampled domains
        yield [next(iterators[i]) for i in sampled_indices]

# --- THE REST OF THE FILE FOLLOWS ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        # Check if the hparams argument is a path to a JSON file
        if args.hparams.endswith('.json'):
            print(f"--- Loading hparams from file: {args.hparams} ---")
            with open(args.hparams, 'r') as f:
                hparams_from_file = json.load(f)
            hparams.update(hparams_from_file)
        else:
            # Otherwise, treat it as a JSON string (original behavior)
            print("--- Loading hparams from command-line string ---")
            hparams.update(json.loads(args.hparams))

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams)
    else:
        raise NotImplementedError

    # In synthetic-domain mode, the dataset's internal environment ordering is
    # alphabetically sorted over ALL loaded folders (real + "SynDomain_*"/
    # "zSynDomain_*"), which shifts each real domain's position away from its
    # index in FULL_ENV_NAMES - "SynDomain_*" sorts before lowercase real domain
    # names. args.test_envs is a CLI index into the *original* FULL_ENV_NAMES
    # order, so it must be remapped to the dataset's actual position for every
    # downstream use (train/uda split, core-vs-synthetic separation, eval loader
    # setup, domain counts) - otherwise the held-out test domain silently leaks
    # into training and an unrelated synthetic domain gets wrongly excluded.
    # No-op in standard mode, where sorted(real domain names) == FULL_ENV_NAMES.
    _requested_test_domain_names = [dataset.original_domain_names[i] for i in args.test_envs]
    args.test_envs = [dataset.loaded_environments.index(name) for name in _requested_test_domain_names]
    print(f"Remapped test_envs to the dataset's actual domain ordering: "
          f"{args.test_envs} ({_requested_test_domain_names})")

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):
        uda = []

        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))

        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                int(len(in_)*args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    # 1. Get the names of all loaded environments from the dataset object.
    # This is crucial for distinguishing core vs. synthetic domains.
    all_env_names = dataset.loaded_environments    
    # 2. Separate all available training indices into 'core' and 'synthetic' tiers.
    core_train_indices = []
    syn_train_indices = []
    for i, _ in enumerate(in_splits):
        if i not in args.test_envs: # Ensure it's a training domain
            # Check the name of the domain at this index
            if "SynDomain" in all_env_names[i] or "zSynDomain" in all_env_names[i]:
                syn_train_indices.append(i)
            else:
                core_train_indices.append(i)
    
    print(f"Separated training domains into {len(core_train_indices)} CORE domains and {len(syn_train_indices)} SYNTHETIC domains.")
    print(f"Core domains (indices): {core_train_indices}")
    print(f"Synthetic domains (indices): {syn_train_indices}")

    # 3. Create the dictionary of data loaders, covering all training domains.
    all_train_indices = core_train_indices + syn_train_indices
    train_loaders_dict = {
        i: InfiniteDataLoader(
            dataset=in_splits[i][0], # env
            weights=in_splits[i][1], # env_weights
            batch_size=hparams['batch_size'],
            num_workers=dataset.N_WORKERS
        )
        for i in all_train_indices
    }
    
    # 4. Get the number of synthetic domains to sample per step (k_syn) from hparams.
    k_syn_per_step = hparams.get('k_syn_domains_per_step', 2) # Default to sampling 2 synthetic domains
    print(f"Will use ALL {len(core_train_indices)} core domains + a random sample of {k_syn_per_step} synthetic domains per step.")

    # 5. Instantiate our new two-tiered iterator.
    train_minibatches_iterator = two_tiered_domain_sampling_iterator(
        train_loaders_dict,
        core_train_indices,
        syn_train_indices,
        k_syn_per_step
    )

    # 6. Calculate the total number of domains the algorithm will see per step.
    num_domains_per_step = len(core_train_indices) + k_syn_per_step

    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(uda_splits)]

# --- REPLACE WITH THIS NEW BLOCK ---

    # ==============================================================================
    # === MODIFIED EVALUATION SETUP ================================================
    # ==============================================================================
    
    print("--- Setting up evaluation loaders for ORIGINAL domains only ---")

    # 1. Identify the indices of the original domains within the full loaded dataset.
    original_domain_indices = [
        i for i, name in enumerate(dataset.loaded_environments)
        if name in dataset.original_domain_names
    ]
    print(f"Found {len(original_domain_indices)} original domains to evaluate at indices: {original_domain_indices}")

    # 2. Build the evaluation lists by iterating ONLY over the original domain indices.
    eval_loaders = []
    eval_weights = []
    eval_loader_names = []
    
    # We use a new counter `j` to create clean log names like env0, env1, ... env5
    for j, i in enumerate(original_domain_indices):
        # Add the 'in' split for the original domain
        eval_loaders.append(FastDataLoader(
            dataset=in_splits[i][0],
            batch_size=64,
            num_workers=dataset.N_WORKERS
        ))
        eval_weights.append(in_splits[i][1])
        # The name uses the new, clean index `j`
        eval_loader_names.append(f'env{j}_in')

        # Add the 'out' split for the original domain
        eval_loaders.append(FastDataLoader(
            dataset=out_splits[i][0],
            batch_size=64,
            num_workers=dataset.N_WORKERS
        ))
        eval_weights.append(out_splits[i][1])
        eval_loader_names.append(f'env{j}_out')

    # Note: This simple version ignores UDA splits for clarity.
    # The main evaluation loop will now only see the 6 original domains.
    # ==============================================================================
    # === END OF MODIFIED EVALUATION SETUP =========================================
    # ==============================================================================

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes, num_domains_per_step, hparams)
    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

    # train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env)/hparams['batch_size'] for env,_ in in_splits])

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))


    last_results_keys = None
    for step in range(start_step, n_steps):
        step_start_time = time.time()
        minibatches_device = [(x.to(device), y.to(device))
            for x,y in next(train_minibatches_iterator)]
        if args.task == "domain_adaptation":
            uda_device = [x.to(device)
                for x,_ in next(uda_minibatches_iterator)]
        else:
            uda_device = None
        step_vals = algorithm.update(minibatches_device, uda_device)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            for name, loader, weights in evals:
                acc = misc.accuracy(algorithm, loader, weights, device)
                results[name+'_acc'] = acc

            results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                colwidth=12)

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })
            # print(f"***********************{results}**")
            # print(f"***********************{json.dumps(results, sort_keys=True)}")
            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl')

    save_checkpoint('model.pkl')

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
