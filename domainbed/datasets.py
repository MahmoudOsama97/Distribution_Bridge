# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset, ConcatDataset, Dataset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate
from .domain_mappings import get_training_domains


from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.fmow_dataset import FMoWDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Debug
    "Debug28",
    "Debug224",
    # Small images
    "ColoredMNIST",
    "RotatedMNIST",
    # Big images
    "VLCS",
    "PACS",
    "OfficeHome",
    "TerraIncognita",
    "DomainNet",
    "SVIRO",
    # WILDS datasets
    "WILDSCamelyon",
    "WILDSFMoW",
    # Spawrious datasets
    "SpawriousO2O_easy",
    "SpawriousO2O_medium",
    "SpawriousO2O_hard",
    "SpawriousM2M_easy",
    "SpawriousM2M_medium",
    "SpawriousM2M_hard",
]

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class Debug(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,))
                )
            )

class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ['0', '1', '2']

class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ['0', '1', '2']


class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            self.datasets.append(dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes


class ColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']

    def __init__(self, root, test_envs, hparams):
        super(ColoredMNIST, self).__init__(root, [0.1, 0.2, 0.9],
                                         self.color_dataset, (2, 28, 28,), 2)

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class RotatedMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']

    def __init__(self, root, test_envs, hparams):
        super(RotatedMNIST, self).__init__(root, [0, 15, 30, 45, 60, 75],
                                           self.rotate_dataset, (1, 28, 28,), 10)

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return TensorDataset(x, y)


# ==============================================================================
# === THE PARENT CLASS MUST BE MODIFIED TO ACCEPT A PRE-FILTERED LIST =========
# ==============================================================================

class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    # ADD `environments_to_load=None` to the constructor's signature
    def __init__(self, root, test_envs, augment, hparams, environments_to_load=None):
        super().__init__()

        # IF a specific list of environments is provided, USE IT.
        if environments_to_load is not None:
            environments = environments_to_load
            print(f"Loading a custom, pre-filtered set of {len(environments)} environments.")
        # OTHERWISE, fall back to the original behavior (scanning the directory).
        else:
            environments = [f.name for f in os.scandir(root) if f.is_dir()]
            environments = sorted(environments)
            print(f"Discovered {len(environments)} environments by scanning directory.")

        # ==================================================================
        # === ADD THIS LINE TO SAVE THE LIST OF LOADED DOMAINS =============
        # ==================================================================
        self.loaded_environments = environments
        # ==================================================================

        # The rest of the original __init__ method remains the same
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        augment_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(environments):
            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform
            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path, transform=env_transform)
            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)

# ==============================================================================
# === THE CHILD CLASSES NOW IMPLEMENT YOUR PRECISE PRUNING LOGIC ===============
# ==============================================================================

class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "R"]
    FULL_ENV_NAMES = ["Art", "Clipart", "Product", "Real World"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "office_home/")
        all_folders_in_dir = sorted([d for d in os.listdir(self.dir) if os.path.isdir(os.path.join(self.dir, d))])
        is_synthetic_setting = any("SynDomain" in folder for folder in all_folders_in_dir)

        if is_synthetic_setting:
            test_env_index = test_envs[0]
            test_domain_name = self.FULL_ENV_NAMES[test_env_index]

            # Logic: Load all original domains + only synthetic domains that are NOT related to the test domain.
            domains_to_load = []
            for folder in all_folders_in_dir:
                is_original = folder in self.FULL_ENV_NAMES
                is_unrelated_synthetic = "SynDomain" in folder and test_domain_name not in folder
                if is_original or is_unrelated_synthetic:
                    domains_to_load.append(folder)
            
            # Find the new index of the test domain within this pruned list.
            final_test_envs = [domains_to_load.index(test_domain_name)]
            
            print(f"--- Activating PRUNED synthetic data mode for OfficeHome ---")
            print(f"Test Domain: {test_domain_name}. New test index: {final_test_envs[0]}.")
            print(f"Loading a total of {len(domains_to_load)} domains: {domains_to_load}")

            # Pass the pruned list of domains and the new test index to the parent.
            super().__init__(self.dir, final_test_envs, hparams['data_augmentation'], hparams, environments_to_load=domains_to_load)
        else:
            print(f"--- Standard OfficeHome Mode Activated ---")
            super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class VLCS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["C", "L", "S", "V"]
    FULL_ENV_NAMES = ["Caltech101", "LabelMe", "SUN09", "VOC2007"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "vlcs/")
        all_folders_in_dir = sorted([d for d in os.listdir(self.dir) if os.path.isdir(os.path.join(self.dir, d))])
        is_synthetic_setting = any("SynDomain" in folder for folder in all_folders_in_dir)

        if is_synthetic_setting:
            test_env_index = test_envs[0]
            test_domain_name = self.FULL_ENV_NAMES[test_env_index]
            
            domains_to_load = []
            for folder in all_folders_in_dir:
                is_original = folder in self.FULL_ENV_NAMES
                is_unrelated_synthetic = "SynDomain" in folder and test_domain_name not in folder
                if is_original or is_unrelated_synthetic:
                    domains_to_load.append(folder)

            final_test_envs = [domains_to_load.index(test_domain_name)]
            
            print(f"--- Activating PRUNED synthetic data mode for VLCS ---")
            print(f"Test Domain: {test_domain_name}. New test index: {final_test_envs[0]}.")
            print(f"Loading a total of {len(domains_to_load)} domains: {domains_to_load}")

            super().__init__(self.dir, final_test_envs, hparams['data_augmentation'], hparams, environments_to_load=domains_to_load)
        else:
            print(f"--- Standard VLCS Mode Activated ---")
            super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "S"]
    FULL_ENV_NAMES = ["art_painting", "cartoon", "photo", "sketch"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "pacs/")
        all_folders_in_dir = sorted([d for d in os.listdir(self.dir) if os.path.isdir(os.path.join(self.dir, d))])
        is_synthetic_setting = any("SynDomain" in folder for folder in all_folders_in_dir)

        if is_synthetic_setting:
            test_env_index = test_envs[0]
            test_domain_name = self.FULL_ENV_NAMES[test_env_index]

            domains_to_load = []
            for folder in all_folders_in_dir:
                is_original = folder in self.FULL_ENV_NAMES
                is_unrelated_synthetic = "SynDomain" in folder and test_domain_name not in folder
                if is_original or is_unrelated_synthetic:
                    domains_to_load.append(folder)

            final_test_envs = [domains_to_load.index(test_domain_name)]

            print(f"--- Activating PRUNED synthetic data mode for PACS ---")
            print(f"Test Domain: {test_domain_name}. New test index: {final_test_envs[0]}.")
            print(f"Loading a total of {len(domains_to_load)} domains: {domains_to_load}")

            super().__init__(self.dir, final_test_envs, hparams['data_augmentation'], hparams, environments_to_load=domains_to_load)
        else:
            print(f"--- Standard PACS Mode Activated ---")
            super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class TerraIncognita(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["L100", "L38", "L43", "L46"]
    FULL_ENV_NAMES = ["location_100", "location_38", "location_43", "location_46"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "terra_incognita/")
        all_folders_in_dir = sorted([d for d in os.listdir(self.dir) if os.path.isdir(os.path.join(self.dir, d))])
        is_synthetic_setting = any("SynDomain" in folder for folder in all_folders_in_dir)

        if is_synthetic_setting:
            test_env_index = test_envs[0]
            test_domain_name = self.FULL_ENV_NAMES[test_env_index]

            domains_to_load = []
            for folder in all_folders_in_dir:
                is_original = folder in self.FULL_ENV_NAMES
                is_unrelated_synthetic = "SynDomain" in folder and test_domain_name not in folder
                if is_original or is_unrelated_synthetic:
                    domains_to_load.append(folder)

            final_test_envs = [domains_to_load.index(test_domain_name)]

            print(f"--- Activating PRUNED synthetic data mode for TerraIncognita ---")
            print(f"Test Domain: {test_domain_name}. New test index: {final_test_envs[0]}.")
            print(f"Loading a total of {len(domains_to_load)} domains: {domains_to_load}")

            super().__init__(self.dir, final_test_envs, hparams['data_augmentation'], hparams, environments_to_load=domains_to_load)
        else:
            print(f"--- Standard TerraIncognita Mode Activated ---")
            super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

# THIS IS THE NEWLY MODIFIED CLASS
class DomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 100
    N_STEPS = 10001
    ENVIRONMENTS = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
    FULL_ENV_NAMES = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "domain_net/")
        # ==================================================================
        # === ADD THIS LINE ================================================
        self.original_domain_names = self.FULL_ENV_NAMES
        # ==================================================================
        all_folders_in_dir = sorted([d for d in os.listdir(self.dir) if os.path.isdir(os.path.join(self.dir, d))])
        is_synthetic_setting = any("zSynDomain" in folder for folder in all_folders_in_dir)

        if is_synthetic_setting:
            test_env_index = test_envs[0]
            test_domain_name = self.FULL_ENV_NAMES[test_env_index]
            domains_to_load = []
            for folder in all_folders_in_dir:
                is_original = folder in self.FULL_ENV_NAMES
                is_unrelated_synthetic = "zSynDomain" in folder and test_domain_name not in folder
                if is_original or is_unrelated_synthetic:
                    domains_to_load.append(folder)
            final_test_envs = [domains_to_load.index(test_domain_name)]
            print(f"--- Activating PRUNED synthetic data mode for DomainNet ---")
            print(f"Test Domain: {test_domain_name}. New test index: {final_test_envs[0]}.")
            print(f"Loading a total of {len(domains_to_load)} domains: {domains_to_load}")
            super().__init__(self.dir, final_test_envs, hparams['data_augmentation'], hparams, environments_to_load=domains_to_load)
        else:
            print(f"--- Standard DomainNet Mode Activated ---")
            super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class SVIRO(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["aclass", "escape", "hilux", "i3", "lexus", "tesla", "tiguan", "tucson", "x5", "zoe"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "sviro/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class WILDSEnvironment:
    def __init__(
            self,
            wilds_dataset,
            metadata_name,
            metadata_value,
            transform=None):
        self.name = metadata_name + "_" + str(metadata_value)

        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_array = wilds_dataset.metadata_array
        subset_indices = torch.where(
            metadata_array[:, metadata_index] == metadata_value)[0]

        self.dataset = wilds_dataset
        self.indices = subset_indices
        self.transform = transform

    def __getitem__(self, i):
        x = self.dataset.get_input(self.indices[i])
        if type(x).__name__ != "Image":
            x = Image.fromarray(x)

        y = self.dataset.y_array[self.indices[i]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)


class WILDSDataset(MultipleDomainDataset):
    INPUT_SHAPE = (3, 224, 224)
    def __init__(self, dataset, metadata_name, test_envs, augment, hparams):
        super().__init__()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []

        for i, metadata_value in enumerate(
                self.metadata_values(dataset, metadata_name)):
            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            env_dataset = WILDSEnvironment(
                dataset, metadata_name, metadata_value, env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = dataset.n_classes

    def metadata_values(self, wilds_dataset, metadata_name):
        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_vals = wilds_dataset.metadata_array[:, metadata_index]
        return sorted(list(set(metadata_vals.view(-1).tolist())))


class WILDSCamelyon(WILDSDataset):
    ENVIRONMENTS = [ "hospital_0", "hospital_1", "hospital_2", "hospital_3",
            "hospital_4"]
    def __init__(self, root, test_envs, hparams):
        dataset = Camelyon17Dataset(root_dir=root)
        super().__init__(
            dataset, "hospital", test_envs, hparams['data_augmentation'], hparams)


class WILDSFMoW(WILDSDataset):
    ENVIRONMENTS = [ "region_0", "region_1", "region_2", "region_3",
            "region_4", "region_5"]
    def __init__(self, root, test_envs, hparams):
        dataset = FMoWDataset(root_dir=root)
        super().__init__(
            dataset, "region", test_envs, hparams['data_augmentation'], hparams)


## Spawrious base classes
class CustomImageFolder(Dataset):
    """
    A class that takes one folder at a time and loads a set number of images in a folder and assigns them a specific class
    """
    def __init__(self, folder_path, class_index, limit=None, transform=None):
        self.folder_path = folder_path
        self.class_index = class_index
        self.image_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
        if limit:
            self.image_paths = self.image_paths[:limit]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        label = torch.tensor(self.class_index, dtype=torch.long)
        return img, label

class SpawriousBenchmark(MultipleDomainDataset):
    ENVIRONMENTS = ["Test", "SC_group_1", "SC_group_2"]
    input_shape = (3, 224, 224)
    num_classes = 4
    class_list = ["bulldog", "corgi", "dachshund", "labrador"]

    def __init__(self, train_combinations, test_combinations, root_dir, augment=True, type1=False):
        self.type1 = type1
        train_datasets, test_datasets = self._prepare_data_lists(train_combinations, test_combinations, root_dir, augment)
        self.datasets = [ConcatDataset(test_datasets)] + train_datasets

    # Prepares the train and test data lists by applying the necessary transformations.
    def _prepare_data_lists(self, train_combinations, test_combinations, root_dir, augment):
        test_transforms = transforms.Compose([
            transforms.Resize((self.input_shape[1], self.input_shape[2])),
            transforms.transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        if augment:
            train_transforms = transforms.Compose([
                transforms.Resize((self.input_shape[1], self.input_shape[2])),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            train_transforms = test_transforms

        train_data_list = self._create_data_list(train_combinations, root_dir, train_transforms)
        test_data_list = self._create_data_list(test_combinations, root_dir, test_transforms)

        return train_data_list, test_data_list

    # Creates a list of datasets based on the given combinations and transformations.
    def _create_data_list(self, combinations, root_dir, transforms):
        data_list = []
        if isinstance(combinations, dict):
            
            # Build class groups for a given set of combinations, root directory, and transformations.
            for_each_class_group = []
            cg_index = 0
            for classes, comb_list in combinations.items():
                for_each_class_group.append([])
                for ind, location_limit in enumerate(comb_list):
                    if isinstance(location_limit, tuple):
                        location, limit = location_limit
                    else:
                        location, limit = location_limit, None
                    cg_data_list = []
                    for cls in classes:
                        path = os.path.join(root_dir, f"{0 if not self.type1 else ind}/{location}/{cls}")
                        data = CustomImageFolder(folder_path=path, class_index=self.class_list.index(cls), limit=limit, transform=transforms)
                        cg_data_list.append(data)
                    
                    for_each_class_group[cg_index].append(ConcatDataset(cg_data_list))
                cg_index += 1

            for group in range(len(for_each_class_group[0])):
                data_list.append(
                    ConcatDataset(
                        [for_each_class_group[k][group] for k in range(len(for_each_class_group))]
                    )
                )
        else:
            for location in combinations:
                path = os.path.join(root_dir, f"{0}/{location}/")
                data = ImageFolder(root=path, transform=transforms)
                data_list.append(data)

        return data_list
    
    
    # Buils combination dictionary for o2o datasets
    def build_type1_combination(self,group,test,filler):
        total = 3168
        counts = [int(0.97*total),int(0.87*total)]
        combinations = {}
        combinations['train_combinations'] = {
            ## correlated class
            ("bulldog",):[(group[0],counts[0]),(group[0],counts[1])],
            ("dachshund",):[(group[1],counts[0]),(group[1],counts[1])],
            ("labrador",):[(group[2],counts[0]),(group[2],counts[1])],
            ("corgi",):[(group[3],counts[0]),(group[3],counts[1])],
            ## filler
            ("bulldog","dachshund","labrador","corgi"):[(filler,total-counts[0]),(filler,total-counts[1])],
        }
        ## TEST
        combinations['test_combinations'] = {
            ("bulldog",):[test[0], test[0]],
            ("dachshund",):[test[1], test[1]],
            ("labrador",):[test[2], test[2]],
            ("corgi",):[test[3], test[3]],
        }
        return combinations

    # Buils combination dictionary for m2m datasets
    def build_type2_combination(self,group,test):
        total = 3168
        counts = [total,total]
        combinations = {}
        combinations['train_combinations'] = {
            ## correlated class
            ("bulldog",):[(group[0],counts[0]),(group[1],counts[1])],
            ("dachshund",):[(group[1],counts[0]),(group[0],counts[1])],
            ("labrador",):[(group[2],counts[0]),(group[3],counts[1])],
            ("corgi",):[(group[3],counts[0]),(group[2],counts[1])],
        }
        combinations['test_combinations'] = {
            ("bulldog",):[test[0], test[1]],
            ("dachshund",):[test[1], test[0]],
            ("labrador",):[test[2], test[3]],
            ("corgi",):[test[3], test[2]],
        }
        return combinations

## Spawrious classes for each Spawrious dataset 
class SpawriousO2O_easy(SpawriousBenchmark):
    def __init__(self, root_dir, test_envs, hparams):
        group = ["desert","jungle","dirt","snow"]
        test = ["dirt","snow","desert","jungle"]
        filler = "beach"
        combinations = self.build_type1_combination(group,test,filler)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir, hparams['data_augmentation'], type1=True)

class SpawriousO2O_medium(SpawriousBenchmark):
    def __init__(self, root_dir, test_envs, hparams):
        group = ['mountain', 'beach', 'dirt', 'jungle']
        test = ['jungle', 'dirt', 'beach', 'snow']
        filler = "desert"
        combinations = self.build_type1_combination(group,test,filler)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir, hparams['data_augmentation'], type1=True)

class SpawriousO2O_hard(SpawriousBenchmark):
    def __init__(self, root_dir, test_envs, hparams):
        group = ['jungle', 'mountain', 'snow', 'desert']
        test = ['mountain', 'snow', 'desert', 'jungle']
        filler = "beach"
        combinations = self.build_type1_combination(group,test,filler)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir, hparams['data_augmentation'], type1=True)

class SpawriousM2M_easy(SpawriousBenchmark):
    def __init__(self, root_dir, test_envs, hparams):
        group = ['desert', 'mountain', 'dirt', 'jungle']
        test = ['dirt', 'jungle', 'mountain', 'desert']
        combinations = self.build_type2_combination(group,test)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir, hparams['data_augmentation']) 

class SpawriousM2M_medium(SpawriousBenchmark):
    def __init__(self, root_dir, test_envs, hparams):
        group = ['beach', 'snow', 'mountain', 'desert']
        test = ['desert', 'mountain', 'beach', 'snow']
        combinations = self.build_type2_combination(group,test)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir, hparams['data_augmentation'])
        
class SpawriousM2M_hard(SpawriousBenchmark):
    ENVIRONMENTS = ["Test","SC_group_1","SC_group_2"]
    def __init__(self, root_dir, test_envs, hparams):
        group = ["dirt","jungle","snow","beach"]
        test = ["snow","beach","dirt","jungle"]
        combinations = self.build_type2_combination(group,test)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir, hparams['data_augmentation'])


# Helper class, likely already in datasets.py, but included for completeness.
# If it already exists, you don't need to add it again.
class SingleDomainImageFolder(ImageFolder):
    def __init__(self, root, transform, domain_name):
        super().__init__(root, transform)
        self.domain_name = domain_name
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, self.domain_name


