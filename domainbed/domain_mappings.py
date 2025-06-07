# filename: domainbed/domain_mappings.py

import itertools

# Define the original domains for the OfficeHome dataset
ORIGINAL_DOMAINS = ["Art", "Clipart", "Product", "Real World"]

def get_training_domains(all_domain_folders, test_domain_name):
    """
    Filters the list of all available domain folders to exclude the test domain
    and any synthetic domains derived from it.

    Args:
        all_domain_folders (list[str]): A list of the names of all folders
                                        found in the data directory.
        test_domain_name (str): The name of the domain to be used for testing
                                (e.g., "Art").

    Returns:
        list[str]: A list of domain names to be used for training.
    """
    if test_domain_name not in ORIGINAL_DOMAINS:
        raise ValueError(f"Test domain '{test_domain_name}' is not one of the recognized original domains: {ORIGINAL_DOMAINS}")

    training_domains = []
    for domain in all_domain_folders:
        # Keep the domain if it's NOT the test domain AND it does not contain the test domain's name.
        # This elegantly excludes both the test domain itself (e.g., "Art") and any synthetic
        # domains derived from it (e.g., "SynDomain_Pair_Art_x_Clipart").
        if domain != test_domain_name and test_domain_name not in domain:
            training_domains.append(domain)

    # --- Verification printout ---
    print("--------------------------------------------------")
    print(f"       Dynamic Domain Selection Activated         ")
    print(f"Test Domain: {test_domain_name}")
    print(f"Found {len(all_domain_folders)} total domains. Using {len(training_domains)} for training:")
    for name in sorted(training_domains):
         print(f"  - {name}")
    print("--------------------------------------------------")
    
    return training_domains