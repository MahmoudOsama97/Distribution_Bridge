import os
import torch
import random
from PIL import Image
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import torch.nn.functional as F

# --- 1. CONFIGURATION ---
# IMPORTANT: Update this path to the root of your PACS dataset
SOURCE_DOMAIN_ROOT = "/home/mosama97/UBCO/research/PACS" 

# --- Model & Device Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
CLIP_COMPONENTS_REPO_ID = "openai/clip-vit-large-patch14" # Using the standard CLIP model for this experiment

# --- Experiment Configuration ---
CLASS_TO_TEST = 'dog'
DOMAINS_TO_TEST = ['photo', 'art_painting']
TRAIN_SPLIT_RATIO = 0.8  # Use 80% of data to create prototypes, 20% for testing
MAX_PROTOTYPE_IMAGES = 100 # Limit images per domain for prototype calculation to keep it fast and stable

print(f"Using device: {DEVICE}, dtype: {TORCH_DTYPE}")
print("-" * 50)

# --- 2. HELPER FUNCTIONS (Adapted from your script for consistency) ---

def get_image_paths_from_dir(directory_path):
    """Scans a directory for image files."""
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    image_files = []
    if not os.path.isdir(directory_path):
        return []
    for f_name in os.listdir(directory_path):
        if f_name.lower().endswith(supported_formats):
            image_files.append(os.path.join(directory_path, f_name))
    return image_files

def discover_class_domain_data(root_dir):
    """Discovers and maps all class/domain image paths from the root directory."""
    class_domain_data = {}
    if not os.path.isdir(root_dir):
        print(f"Error: Source domain root directory '{root_dir}' not found.")
        return class_domain_data
    print(f"Scanning for domains and classes in: {root_dir}")
    for domain_name in os.listdir(root_dir):
        domain_path = os.path.join(root_dir, domain_name)
        if not os.path.isdir(domain_path) or domain_name.startswith('.'):
            continue
        for class_name in os.listdir(domain_path):
            class_path = os.path.join(domain_path, class_name)
            if not os.path.isdir(class_path) or class_name.startswith('.'):
                continue
            image_files = get_image_paths_from_dir(class_path)
            if image_files:
                if class_name not in class_domain_data:
                    class_domain_data[class_name] = {}
                class_domain_data[class_name][domain_name] = image_files
    return class_domain_data

@torch.no_grad()
def get_image_embedding(image_path, image_encoder_model, feature_extractor_model, device, dtype):
    """Generates a CLIP image embedding for a single image."""
    try:
        image = Image.open(image_path).convert("RGB")
        # Note: CLIP uses `image_processor` which is the new name for `feature_extractor`
        processed_inputs = feature_extractor_model(images=image, return_tensors="pt")
        pixel_values = processed_inputs.pixel_values.to(device, dtype=dtype)
        embeds = image_encoder_model(pixel_values).image_embeds
        return embeds
    except Exception as e:
        print(f"Error generating embedding for {image_path}: {e}")
        return None

@torch.no_grad()
def calculate_prototypical_embeddings(class_data_map, image_encoder_model, fe_model, dev, dtype, max_images):
    """Calculates the mean embedding (prototype) for each class-domain pair."""
    print("\n--- Calculating Prototypical Embeddings from Training Split ---")
    prototypical_embeds = {}
    for class_name, domains_map in tqdm(class_data_map.items(), desc="Processing Classes"):
        prototypical_embeds[class_name] = {}
        for domain_name, image_paths in tqdm(domains_map.items(), desc=f"  Domains for {class_name}", leave=False):
            embeddings_list = []
            # Use max_images to limit the number of images used for the prototype
            selected_image_paths = image_paths[:max_images] if len(image_paths) > max_images else image_paths
            if not selected_image_paths:
                continue
            for img_path in selected_image_paths:
                embed = get_image_embedding(img_path, image_encoder_model, fe_model, dev, dtype)
                if embed is not None:
                    embeddings_list.append(embed)
            if embeddings_list:
                stacked_embeddings = torch.cat(embeddings_list, dim=0)
                mean_embedding = torch.mean(stacked_embeddings, dim=0, keepdim=True)
                prototypical_embeds[class_name][domain_name] = mean_embedding
    print("--- Prototypical Embeddings Calculation Complete ---")
    return prototypical_embeds


# --- 3. MAIN EXPERIMENT SCRIPT ---
if __name__ == "__main__":
    # --- Load CLIP Models ---
    print(f"Loading CLIP model from: {CLIP_COMPONENTS_REPO_ID}")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        CLIP_COMPONENTS_REPO_ID, torch_dtype=TORCH_DTYPE
    ).to(DEVICE)
    feature_extractor = CLIPImageProcessor.from_pretrained(CLIP_COMPONENTS_REPO_ID)

    # --- Discover and Filter Data ---
    all_data = discover_class_domain_data(SOURCE_DOMAIN_ROOT)
    
    if CLASS_TO_TEST not in all_data:
        raise ValueError(f"Class '{CLASS_TO_TEST}' not found in the dataset.")
    
    class_data = all_data[CLASS_TO_TEST]

    # --- Split Data into Train (for prototypes) and Test (for evaluation) ---
    print(f"\n--- Splitting data with {TRAIN_SPLIT_RATIO*100:.0f}/{100-TRAIN_SPLIT_RATIO*100:.0f} Train/Test ratio ---")
    train_data_split = {CLASS_TO_TEST: {}}
    test_data_split = {} # Simplified structure for testing

    for domain in DOMAINS_TO_TEST:
        if domain not in class_data:
            print(f"Warning: Domain '{domain}' not found for class '{CLASS_TO_TEST}'. Skipping.")
            continue
        
        image_paths = class_data[domain]
        random.shuffle(image_paths) # Shuffle for a random split
        
        split_idx = int(len(image_paths) * TRAIN_SPLIT_RATIO)
        
        train_paths = image_paths[:split_idx]
        test_paths = image_paths[split_idx:]
        
        train_data_split[CLASS_TO_TEST][domain] = train_paths
        test_data_split[domain] = test_paths
        
        print(f"Domain '{domain}': {len(train_paths)} training samples, {len(test_paths)} test samples.")

    # --- Calculate Prototypes using ONLY the Training Data ---
    prototypes = calculate_prototypical_embeddings(
        train_data_split, image_encoder, feature_extractor, DEVICE, TORCH_DTYPE, max_images=MAX_PROTOTYPE_IMAGES
    )
    
    # Extract the specific prototypes we need for the classifier
    try:
        p_domain1 = prototypes[CLASS_TO_TEST][DOMAINS_TO_TEST[0]]
        p_domain2 = prototypes[CLASS_TO_TEST][DOMAINS_TO_TEST[1]]
    except KeyError:
        raise RuntimeError("Failed to compute one or both prototypes. Check if data exists for the specified class/domains.")

    # --- Evaluate the Nearest-Prototype Classifier on the Test Set ---
    print("\n--- Evaluating Nearest-Prototype Classifier on Held-Out Test Set ---")
    total_tested = 0
    correctly_classified = 0

    for true_domain, test_image_paths in tqdm(test_data_split.items(), desc="Evaluating Domains"):
        for image_path in tqdm(test_image_paths, desc=f"  Images in {true_domain}", leave=False):
            z_test = get_image_embedding(image_path, image_encoder, feature_extractor, DEVICE, TORCH_DTYPE)
            
            if z_test is None:
                continue

            # Calculate Cosine Similarities
            sim_domain1 = F.cosine_similarity(z_test, p_domain1)
            sim_domain2 = F.cosine_similarity(z_test, p_domain2)
            
            # Classify based on highest similarity
            if sim_domain1 > sim_domain2:
                predicted_domain = DOMAINS_TO_TEST[0]
            else:
                predicted_domain = DOMAINS_TO_TEST[1]
            
            # Check if prediction is correct
            if predicted_domain == true_domain:
                correctly_classified += 1
            
            total_tested += 1

    # --- Report Final Accuracy ---
    if total_tested > 0:
        final_accuracy = (correctly_classified / total_tested) * 100
        print("\n" + "="*50)
        print("          Quantitative Assessment Results")
        print("="*50)
        print(f"Total Images Tested: {total_tested}")
        print(f"Correctly Classified: {correctly_classified}")
        print(f"\nFinal Accuracy: {final_accuracy:.2f}%")
        print("="*50)
        print("You can now replace the placeholder in your paper with this value.")
    else:
        print("No images were tested. Could not calculate accuracy.")