from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
import torch

class BinaryPetDataset(torch.utils.data.Dataset):
    def __init__(self, subset):
        self.subset = subset
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        img, mask = self.subset[idx]
        mask = (mask == 1).float()

        return img, mask

def get_transforms(img_size=128):
    transform_img = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])  
    
    transform_mask = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor(),
    ])
    return transform_img, transform_mask
    
def get_dataloader(
    data_dir="./dataset",
    img_size=256,
    batch_size=8,
    val_split=0.2,
    subset_size=None,
    num_workers=4,
    seed=42,
):
    generator = torch.Generator().manual_seed(seed)
    transform_img, transform_mask = get_transforms(img_size=img_size)

    dataset = datasets.OxfordIIITPet(
        root=data_dir,
        split="trainval",
        download=True,
        target_types="segmentation",
        transform=transform_img,
        target_transform=transform_mask
    )
    
    if subset_size is not None:
        subset = Subset(dataset, range(subset_size))
    else:
        subset = dataset
        
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_subset, val_subset = random_split(
        subset,
        [train_size, val_size],
        generator=generator
    )

    train_set = BinaryPetDataset(train_subset)
    val_set   = BinaryPetDataset(val_subset)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader

def get_test_dataloader(
    data_dir="./dataset",
    img_size=128,
    batch_size=8,
    num_workers=4,
):
    transform_img, transform_mask = get_transforms(img_size=img_size)

    test_dataset = datasets.OxfordIIITPet(
        root=data_dir,
        split="test",
        download=True,
        target_types="segmentation",
        transform=transform_img,
        target_transform=transform_mask
    )

    test_set = BinaryPetDataset(test_dataset)

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return test_loader