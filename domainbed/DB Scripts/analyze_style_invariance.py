import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================

# Path to your PACS dataset root folder
DATA_ROOT = "/home/mosama97/UBCO/research/PACS" 

# Paths to your trained model checkpoints
# IMPORTANT: Map the name you want in the table to the file path
CHECKPOINTS = {
    "ERM (Baseline)": "/home/mosama97/UBCO/research/DDB_Checkpoints/output/ERM__PACS/0185f9c3600aa2edb3146127e481d652/best_model.pkl",
    "Mixup":          "/home/mosama97/UBCO/research/DDB_Checkpoints/output/Mixup__PACS/14e073272e2f08e833c77301da857ed7/best_model.pkl", # Optional
    "MixStyle":      "/home/mosama97/UBCO/research/DDB_Checkpoints/output/Mixstyle2_PACS/0f4ef5a8ce29fa11d9c294382ae1dbdf/best_model.pkl",
    "Distribution Bridge":   "/home/mosama97/UBCO/research/DDB_Checkpoints/output/DDB__PACS/6c1babfd03d0fc1f6ce81eb11513ecbd/best_model.pkl"
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64

# ==========================================
# 2. UTILITY FUNCTIONS
# ==========================================

class ResNet50FeatureExtractor(nn.Module):
    """
    Wraps ResNet50 to return the 2048-dim feature vector 
    instead of class logits.
    """
    def __init__(self, checkpoint_path=None):
        super(ResNet50FeatureExtractor, self).__init__()
        # Load standard ResNet50 structure
        backbone = models.resnet50(pretrained=False)
        
        # Replace the final classification layer (fc) with Identity
        # This ensures we get the features (2048 dim)
        backbone.fc = nn.Identity()
        self.backbone = backbone

        if checkpoint_path:
            self.load_weights(checkpoint_path)

    def load_weights(self, path):
        # Load checkpoint
        state_dict = torch.load(path, map_location=DEVICE)
        
        # Handle DomainBed specific saving (sometimes nested in 'network' or 'model')
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif 'network' in state_dict:
            state_dict = state_dict['network']
            
        # Clean up keys if necessary (e.g., remove 'module.' prefix)
        new_state_dict = {}
        for k, v in state_dict.items():
            # We only care about the backbone, not the classifier head weights
            if 'fc' in k or 'classifier' in k: 
                continue
            name = k.replace('module.', '').replace('network.', '')
            new_state_dict[name] = v
            
        # Load into backbone (strict=False to ignore missing fc layer weights)
        try:
            self.backbone.load_state_dict(new_state_dict, strict=False)
            print(f"Successfully loaded weights from {path}")
        except Exception as e:
            print(f"Error loading weights: {e}")

    def forward(self, x):
        return self.backbone(x)

def get_pacs_loader(root_dir):
    """
    Loads all domains in PACS as a single dataset, 
    but keeps track of domain labels.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # We iterate domains manually to assign domain labels
    domains = ['photo', 'art_painting', 'cartoon', 'sketch']
    all_data = []
    
    for domain_idx, domain_name in enumerate(domains):
        domain_path = os.path.join(root_dir, domain_name)
        if not os.path.exists(domain_path):
            continue
            
        ds = datasets.ImageFolder(domain_path, transform=transform)
        for img, class_idx in ds:
            # Store: (Image, Class_Label, Domain_Label)
            all_data.append((img, class_idx, domain_idx))
            
    return all_data

def extract_features(model, dataset):
    """
    Passes data through model to get features.
    Returns matrices for features, class labels, and domain labels.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    features_list = []
    class_labels_list = []
    domain_labels_list = []
    
    with torch.no_grad():
        for imgs, class_lbls, domain_lbls in tqdm(loader, desc="Extracting Features"):
            imgs = imgs.to(DEVICE)
            feats = model(imgs)
            features_list.append(feats.cpu().numpy())
            class_labels_list.append(class_lbls.numpy())
            domain_labels_list.append(domain_lbls.numpy())
            
    return (np.concatenate(features_list), 
            np.concatenate(class_labels_list), 
            np.concatenate(domain_labels_list))

# ==========================================
# 3. METRIC CALCULATIONS
# ==========================================

def compute_style_predictability(features, domain_labels):
    """
    Metric 1: Can a linear classifier predict the domain from the features?
    Goal: LOW accuracy (indistinguishable domains).
    """
    # Split into train/test for the probe
    X_train, X_test, y_train, y_test = train_test_split(
        features, domain_labels, test_size=0.2, random_state=42, stratify=domain_labels
    )
    
    # Using multinomial logistic regression
    clf = LogisticRegression(max_iter=500, solver='lbfgs', multi_class='multinomial')
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return accuracy_score(y_test, preds)

def feature_covariance_similarity(X, Y):
    """
    Computes the Cosine Similarity between the feature covariance matrices.
    This replaces CKA for unpaired data.
    
    Args:
        X: Feature matrix for Domain A (N_a x D)
        Y: Feature matrix for Domain B (N_b x D)
    """
    # 1. Center the features (Subtract Mean)
    X_centered = X - np.mean(X, axis=0)
    Y_centered = Y - np.mean(Y, axis=0)
    
    # 2. Compute Covariance Matrices (D x D)
    # We use dot product. Dividing by N-1 is technically correct for covariance,
    # but since we normalize by norm later, the scalar cancels out. 
    # We keep it simple here.
    cov_X = np.dot(X_centered.T, X_centered)
    cov_Y = np.dot(Y_centered.T, Y_centered)
    
    # 3. Compute Cosine Similarity of the flattened covariance matrices
    # Sim(A, B) = <A, B> / (||A|| * ||B||)
    numerator = np.sum(cov_X * cov_Y)
    denominator = np.linalg.norm(cov_X) * np.linalg.norm(cov_Y)
    
    if denominator == 0:
        return 0.0
        
    return numerator / denominator

def compute_avg_cross_domain_similarity(features, class_labels, domain_labels):
    """
    Iterates through classes and domains to compute average Feature Covariance Similarity.
    """
    unique_classes = np.unique(class_labels)
    unique_domains = np.unique(domain_labels)
    
    scores = []
    
    # Iterate over every class (Dog, Elephant...)
    for cls in unique_classes:
        # Get indices for this class
        cls_mask = (class_labels == cls)
        
        # Get features for this class separated by domain
        domain_feats = {}
        for dom in unique_domains:
            dom_mask = (domain_labels == dom) & cls_mask
            # Need enough samples to compute a stable covariance
            if np.sum(dom_mask) > 5: 
                domain_feats[dom] = features[dom_mask]
        
        # Calculate pairwise Similarity between domains for this class
        dom_keys = list(domain_feats.keys())
        for i in range(len(dom_keys)):
            for j in range(i + 1, len(dom_keys)):
                feat_A = domain_feats[dom_keys[i]]
                feat_B = domain_feats[dom_keys[j]]
                
                # IMPORTANT: Call the math helper function, NOT this function recursively
                score = feature_covariance_similarity(feat_A, feat_B)
                
                if not np.isnan(score):
                    scores.append(score)
                
    return np.mean(scores) if scores else 0.0

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    print("Loading PACS Dataset...")
    # We load the data as a flat list of (img, class, domain) tuples
    raw_dataset = get_pacs_loader(DATA_ROOT)
    
    # This simple dataset wrapper allows us to use PyTorch DataLoader
    class ListDataset(torch.utils.data.Dataset):
        def __init__(self, data): self.data = data
        def __len__(self): return len(self.data)
        def __getitem__(self, idx): return self.data[idx]
        
    dataset = ListDataset(raw_dataset)
    
    results_table = []

    print(f"\nStarting Analysis on {len(dataset)} images...")
    print("-" * 60)
    print(f"{'Method':<20} | {'Style Pred. (Lower=Better)':<25} | {'CKA Sim. (Higher=Better)':<25}")
    print("-" * 60)

    for method_name, checkpoint_path in CHECKPOINTS.items():
        if not os.path.exists(checkpoint_path):
            print(f"Skipping {method_name}: Checkpoint not found at {checkpoint_path}")
            continue
            
        # 1. Load Model
        model = ResNet50FeatureExtractor(checkpoint_path).to(DEVICE)
        
        # 2. Extract Features
        feats, classes, domains = extract_features(model, dataset)
        
        # 3. Calculate Metrics
        # Style Predictability (Linear Probe)
        style_pred_score = compute_style_predictability(feats, domains)
        
        # CKA Similarity
        cka_score = compute_avg_cross_domain_similarity(feats, classes, domains)
        
        # 4. Print Row
        print(f"{method_name:<20} | {style_pred_score:.4f}{' ':<19} | {cka_score:.4f}")
        
        results_table.append({
            "Method": method_name,
            "Style Pred": style_pred_score,
            "CKA": cka_score
        })

    print("-" * 60)
    print("\nAnalysis Complete. Copy these values into your LaTeX table.")