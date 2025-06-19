# filename: domainbed/domain_mappings.py

# ==============================================================================
# === CENTRAL REGISTRY =========================================================
# ==============================================================================
# This dictionary maps the official DomainBed dataset class names to the
# list of their original, full domain folder names.
#
# This serves as the single source of truth for your experiment setup. The
# FULL_ENV_NAMES variable in each dataset class in `datasets.py` should
# match the list defined here.
# ==============================================================================

DATASET_TO_DOMAINS_MAP = {
    "OfficeHome": ["Art", "Clipart", "Product", "Real World"],
    "PACS": ["art_painting", "cartoon", "photo", "sketch"],
    "VLCS": ["Caltech101", "LabelMe", "SUN09", "VOC2007"],
    "TerraIncognita": ["location_100", "location_38", "location_43", "location_46"],
    
    # Add any other datasets here as you support them.
    # For example:
    # "DomainNet": ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
}

# ==============================================================================
# === Obsolete Function (Kept for Reference) ===================================
# ==============================================================================

def get_training_domains(dataset_name, all_domain_folders, test_domain_name):
    """
    NOTE: This function is now considered obsolete for the ImageFolder-based
    datasets, as the more robust data pruning logic has been moved directly
    into the __init__ method of each dataset class in `datasets.py`.

    It is kept here for reference and for potential use with other types of
    dataset loaders that might not follow the ImageFolder structure.
    """
    if dataset_name not in DATASET_TO_DOMAINS_MAP:
        raise NotImplementedError(f"Dataset '{dataset_name}' is not registered in domain_mappings.py.")
    
    original_domains = DATASET_TO_DOMAINS_MAP[dataset_name]
    if test_domain_name not in original_domains:
        raise ValueError(f"Test domain '{test_domain_name}' is not one of the recognized original domains for dataset '{dataset_name}': {original_domains}")

    training_domains = []
    for domain in all_domain_folders:
        if domain != test_domain_name and test_domain_name not in domain:
            training_domains.append(domain)

    # 3. Verification printout
    print("--------------------------------------------------")
    print(f"   Dynamic Domain Selection for '{dataset_name}'")
    print(f"Test Domain: {test_domain_name}")
    print(f"Found {len(all_domain_folders)} total domains. Using {len(training_domains)} for training:")
    for name in sorted(training_domains):
         print(f"  - {name}")
    print("--------------------------------------------------")            
    return training_domains