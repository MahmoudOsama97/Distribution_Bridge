# filename: domainbed/domain_mappings.py

# Central registry mapping dataset names (as they appear in DomainBed)
# to their list of original, full domain folder names.
DATASET_TO_DOMAINS_MAP = {
    "OfficeHome": ["Art", "Clipart", "Product", "RealWorld"],
    "PACS": ["art_painting", "cartoon", "photo", "sketch"],
    "VLCS": ["CALTECH", "LABELME", "PASCAL", "SUN"],
    "TerraIncognita": ["location_38", "location_43", "location_46", "location_100"],
    # Add other datasets here as needed, e.g., "DomainNet"
}

def get_training_domains(dataset_name, all_domain_folders, test_domain_name):
    """
    Filters a list of all available domain folders to exclude the test domain
    and any synthetic domains derived from it. This function is now generic
    and works for any registered dataset.

    Args:
        dataset_name (str): The name of the dataset (e.g., "OfficeHome", "PACS").
        all_domain_folders (list[str]): Names of all subdirectories found.
        test_domain_name (str): The full name of the domain being used for testing.

    Returns:
        list[str]: A filtered list of domain names to be used for training.
    """
    # 1. Look up the official original domains for the given dataset
    if dataset_name not in DATASET_TO_DOMAINS_MAP:
        raise NotImplementedError(f"Dataset '{dataset_name}' is not registered in domain_mappings.py. Please add it to DATASET_TO_DOMAINS_MAP.")
    
    original_domains = DATASET_TO_DOMAINS_MAP[dataset_name]

    if test_domain_name not in original_domains:
        raise ValueError(f"Test domain '{test_domain_name}' is not one of the recognized original domains for dataset '{dataset_name}': {original_domains}")

    # 2. Filter out the test domain and its synthetic children
    training_domains = []
    for domain in all_domain_folders:
        # The core logic remains the same: exclude the domain if it IS the test domain
        # or if its name CONTAINS the test domain's name (for synthetic pairs).
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