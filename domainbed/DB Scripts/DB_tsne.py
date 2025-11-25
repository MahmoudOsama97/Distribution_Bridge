import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import os
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from transformers import CLIPVisionModel, CLIPImageProcessor
from torchvision import transforms  # <<< ADD THIS LINE
import argparse
# ==============================================================================
# === ARGUMENT PARSER ==========================================================
# ==============================================================================
parser = argparse.ArgumentParser(description="Generate t-SNE visualizations for Distribution Bridge.")
parser.add_argument('--class_name', type=str, default="all",
                    help="The specific class to visualize. Use 'all' to generate a multi-plot for every class in the dataset.")
# You can add more arguments here later if needed (e.g., --output_dir)

args = parser.parse_args()
# ==============================================================================
# === 1. SETUP AND CONFIGURATION ===============================================
# ==============================================================================

# --- Model and Device ---
CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- Path Parameters ---
# IMPORTANT: Update these paths to point to your data directories.
# The original domains root should contain subfolders like 'photo', 'art_painting', etc.
ORIGINAL_DOMAINS_ROOT = "/home/mosama97/UBCO/research/PEFTFusion/data/PACS"
# The synthetic domains root should contain subfolders like 'zSynDomain_Pair_photo_x_art_painting', etc.
SYNTHETIC_DOMAINS_ROOT = "/home/mosama97/UBCO/research/MoA/data/generated_Domains_slerp_two_Interpolated_PACS"

# --- REPLACE WITH THIS ---
# --- Dataset and Class Parameters ---
# The class to visualize is now controlled by the command-line argument
CLASS_TO_VISUALIZE = args.class_name
# We'll now sample per-class, per-domain
SAMPLES_PER_DOMAIN_PER_CLASS = 50 # Reduced for the multi-plot to keep it manageable

# --- t-SNE Parameters ---
TSNE_PERPLEXITY = 30
TSNE_ITERATIONS = 1000
TSNE_RANDOM_STATE = 42  # For reproducibility

# --- Plotting Parameters ---
# Use a distinct, professional color for each original domain. This will be populated automatically.
original_domain_colors = {} 
# Use a single, neutral color for all synthetic domains
SYNTHETIC_DOMAIN_COLOR = '#a9a9a9' # Dark Gray

POINT_SIZE = 25
POINT_ALPHA = 0.8

# ==============================================================================
# === 2. HELPER FUNCTIONS ======================================================
# ==============================================================================

@torch.no_grad()
def extract_clip_features(clip_model, clip_processor, loader, device):
    """Extracts CLIP features for all images in a data loader."""
    clip_model.eval()
    
    all_features = []
    all_class_labels = []
    
    for images, labels in loader:
        # The loader provides PIL images, which the processor handles
        inputs = clip_processor(images=images, return_tensors="pt").to(device)
        
        # Get the final pooled output feature
        features = clip_model(**inputs).pooler_output
        
        all_features.append(features.cpu().numpy())
        all_class_labels.append(labels.cpu().numpy())
        
    if not all_features:
        return np.array([]), np.array([])
        
    return np.vstack(all_features), np.hstack(all_class_labels)


# --- REPLACE WITH THIS ---
def load_datasets_from_root(root_path):
    """Loads all subdirectories in a root path as separate ImageFolder datasets."""
    
    # Define a transform to convert images to PyTorch tensors
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # Resize to CLIP's expected input size
        transforms.ToTensor()
    ])
    
    datasets_dict = {}
    if not os.path.isdir(root_path):
        print(f"Warning: Directory not found: {root_path}")
        return datasets_dict
    
    domain_names = sorted([d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))])
    for domain_name in domain_names:
        domain_path = os.path.join(root_path, domain_name)
        try:
            # Pass the transform to the ImageFolder constructor
            dataset = ImageFolder(domain_path, transform=transform)
            
            if len(dataset) > 0 and hasattr(dataset, 'class_to_idx'):
                datasets_dict[domain_name] = dataset
                print(f"  - Found domain '{domain_name}' with {len(dataset)} images.")
            else:
                print(f"  - Skipping '{domain_name}': No images found or not a valid ImageFolder.")
        except Exception as e:
            print(f"  - Error loading '{domain_name}': {e}. Skipping.")
    return datasets_dict
# ==============================================================================
# === 3. MAIN SCRIPT LOGIC =====================================================
# ==============================================================================

if __name__ == "__main__":
    # --- Load CLIP Model (Done once) ---
    print(f"\nLoading CLIP model: {CLIP_MODEL_ID}")
    clip_model = CLIPVisionModel.from_pretrained(CLIP_MODEL_ID).to(DEVICE)
    clip_processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID)

    # --- Load All Datasets (Done once) ---
    print(f"\nLoading ORIGINAL domains from: {ORIGINAL_DOMAINS_ROOT}")
    original_datasets_dict = load_datasets_from_root(ORIGINAL_DOMAINS_ROOT)
    print(f"\nLoading SYNTHETIC domains from: {SYNTHETIC_DOMAINS_ROOT}")
    synthetic_datasets_dict = load_datasets_from_root(SYNTHETIC_DOMAINS_ROOT)

    if not original_datasets_dict:
        raise ValueError("No original domain datasets were loaded. Check path.")
    # ==================================================================
    # === ADD THIS BLOCK TO FIX THE ERROR ==============================
    # ==================================================================
    # --- Dynamically create color palette for the original domains ---
    original_domain_names = sorted(original_datasets_dict.keys())
    palette = sns.color_palette('deep', n_colors=len(original_domain_names))
    original_domain_colors = {name: color for name, color in zip(original_domain_names, palette)}
    print(f"\nCreated color map for original domains: {original_domain_colors}")
    # ==================================================================
    # === END OF NEW BLOCK =============================================
    # ==================================================================

    # --- Extract All Features from All Domains and All Classes (Done once) ---
    print("\nExtracting features from all available domains and classes...")
    all_features_list, all_class_labels_list, all_domain_labels_list = [], [], []
    
    first_domain_name = list(original_datasets_dict.keys())[0]
    class_to_idx = original_datasets_dict[first_domain_name].class_to_idx
    
    all_datasets_dict = {**original_datasets_dict, **synthetic_datasets_dict}

    for domain_name, dataset in all_datasets_dict.items():
        print(f"  - Extracting from '{domain_name}'...")
        # Use a smaller subset per domain to speed up the global extraction
        subset_indices = np.random.choice(len(dataset), size=min(SAMPLES_PER_DOMAIN_PER_CLASS * len(class_to_idx), len(dataset)), replace=False)
        subset = Subset(dataset, subset_indices)
        loader = DataLoader(subset, batch_size=64, num_workers=4, shuffle=False)
        
        features, class_labels = extract_clip_features(clip_model, clip_processor, loader, DEVICE)
        
        if features.shape[0] > 0:
            all_features_list.append(features)
            all_class_labels_list.append(class_labels)
            all_domain_labels_list.extend([domain_name] * len(features))

    all_features_agg = np.vstack(all_features_list)
    all_class_labels_agg = np.hstack(all_class_labels_list)
    all_domain_labels_agg = np.array(all_domain_labels_list)

    # --- Determine which classes to loop through ---
    if CLASS_TO_VISUALIZE.lower() == 'all':
        class_names_to_plot = sorted(class_to_idx.keys())
    else:
        if CLASS_TO_VISUALIZE not in class_to_idx:
            raise ValueError(f"Class '{CLASS_TO_VISUALIZE}' not found. Available: {list(class_to_idx.keys())}")
        class_names_to_plot = [CLASS_TO_VISUALIZE]
    
    num_classes = len(class_names_to_plot)
    
    # --- Create the Main Figure Grid ---
    # Figure size is now dynamic based on the number of classes
    # --- Create the Main Figure Grid (Matrix Layout) ---
    # We create a 4x4 grid to fit 7 classes (14 plots).
    # The figure size is now wider than it is tall.
    num_rows = 4
    num_cols = 4 # 2 plots per class * 2 classes per row
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 18), squeeze=False)    
    print(f"\nPreparing to generate plots for: {class_names_to_plot}")

    # --- Loop Through Each Class and Generate its Row of Plots ---
    for i, class_name in enumerate(class_names_to_plot):
        print(f"--- Processing class: {class_name} ---")
        target_class_index = class_to_idx[class_name]
        row_idx = i // 2  # Integer division gives the row (0, 0, 1, 1, 2, 2, 3)
        col_idx_start = (i % 2) * 2 # Modulo gives the starting column (0, 2, 0, 2, 0, 2, 0)
        # 1. Filter the pre-extracted data for the current class
        
        # Define the left and right axes for the current class
        ax1 = axes[row_idx, col_idx_start]     # Left plot (Originals)
        ax2 = axes[row_idx, col_idx_start + 1] # Right plot (Augmented)

        class_mask = (all_class_labels_agg == target_class_index)
        features_for_tsne = all_features_agg[class_mask]
        domain_labels_for_tsne = all_domain_labels_agg[class_mask]
        
        if len(features_for_tsne) < TSNE_PERPLEXITY:
            print(f"  - Skipping class '{class_name}': Not enough data points ({len(features_for_tsne)}) for t-SNE perplexity ({TSNE_PERPLEXITY}).")
            axes[i, 0].text(0.5, 0.5, 'Not Enough Data', ha='center', va='center')
            axes[i, 1].text(0.5, 0.5, 'Not Enough Data', ha='center', va='center')
            axes[i, 0].set_title(f"(Originals: '{class_name.title()}')")
            axes[i, 1].set_title(f"(Augmented: '{class_name.title()}')")
            continue

        # 2. Run t-SNE for the current class
        print(f"  - Running t-SNE on {len(features_for_tsne)} points for class '{class_name}'...")
        tsne = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, max_iter=TSNE_ITERATIONS,
                    random_state=TSNE_RANDOM_STATE, init='pca', learning_rate='auto')
        tsne_results = tsne.fit_transform(features_for_tsne)

        # 3. Separate data for plotting
        original_domain_names = list(original_datasets_dict.keys())
        is_original_mask = np.isin(domain_labels_for_tsne, original_domain_names)
        
        tsne_originals = tsne_results[is_original_mask]
        labels_originals = domain_labels_for_tsne[is_original_mask]
        
        tsne_synthetics = tsne_results[~is_original_mask]

     # 4. Plot on the designated axes
        # Plot 1: Originals Only (on the calculated ax1)
        sns.scatterplot(x=tsne_originals[:, 0], y=tsne_originals[:, 1], hue=labels_originals,
                        palette=original_domain_colors, s=POINT_SIZE, alpha=POINT_ALPHA, linewidth=0, ax=ax1)
        ax1.set_title(f"Domains Without Distribution Bridge Augmentation: '{class_name.title()}'")
        ax1.get_legend().remove() # Remove individual legends to de-clutter

        # Plot 2: Originals + Synthetics (on the calculated ax2)
        if tsne_synthetics.shape[0] > 0:
            sns.scatterplot(x=tsne_synthetics[:, 0], y=tsne_synthetics[:, 1], color=SYNTHETIC_DOMAIN_COLOR,
                            label='Synthetic Bridge', marker='X', s=POINT_SIZE, alpha=0.6, linewidth=0, ax=ax2)
        sns.scatterplot(x=tsne_originals[:, 0], y=tsne_originals[:, 1], hue=labels_originals,
                        palette=original_domain_colors, s=POINT_SIZE + 10, alpha=0.9, linewidth=0, ax=ax2)
        ax2.set_title(f"Domains With Distribution Bridge Augmentation: '{class_name.title()}'")
        ax2.get_legend().remove() # Remove individual legends

    # --- Finalize and Save the Single Figure ---
    for ax in axes.flat:
        ax.set_xlabel(""); ax.set_ylabel(""); ax.set_xticklabels([]); ax.set_yticklabels([])
    for i in range(len(class_names_to_plot), (num_rows * num_cols) // 2):
        row_idx = i // 2
        col_idx_start = (i % 2) * 2
        axes[row_idx, col_idx_start].set_visible(False)
        axes[row_idx, col_idx_start + 1].set_visible(False)
    
    # Create a single, shared legend for the entire figure
    handles, labels = axes[0,0].get_legend_handles_labels() # Get handles from a populated plot
    # Add the synthetic handle/label
    handles.append(plt.Line2D([0], [0], marker='X', color='w', label='Synthetic Bridge', markerfacecolor=SYNTHETIC_DOMAIN_COLOR, markersize=10))
    labels.append('Synthetic Bridge')
    fig.legend(handles, labels, loc='lower right', title='Domain Type', bbox_to_anchor=(0.98, 0.05))
    fig.suptitle(f"t-SNE Visualization of Frozen Feature Space (PACS)", fontsize=22, y=1.0)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to make room for suptitle

    output_filename = f"ALL_tsne_pacs_{CLASS_TO_VISUALIZE.lower()}.png"
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    print(f"\nSaved combined figure to {output_filename}")
