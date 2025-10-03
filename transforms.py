import torch
import albumentations as A

class TrainTransform:
    def __init__(self, means, stds):
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=means, std=stds),
        ],
        additional_targets={"mask": "mask"})

    def __call__(self, sample):
        image = sample["image"]
        mask = sample["mask"]

        image_np = image.permute(1, 2, 0).cpu().numpy()
        mask_np = mask.squeeze(0).cpu().numpy()

        augmented = self.transform(image=image_np, mask=mask_np)

        aug_image = augmented["image"]
        aug_mask = augmented["mask"]

        aug_image_t = torch.from_numpy(aug_image).permute(2, 0, 1)
        aug_mask_t  = torch.from_numpy(aug_mask).unsqueeze(0)

        sample["image"] = aug_image_t
        sample["mask"]  = aug_mask_t
        
        return sample

class TestValTransform:
    def __init__(self, means, stds):
        self.transform = A.Compose([
            A.Normalize(mean=means, std=stds)
        ],
        additional_targets={"mask": "mask"})

    def __call__(self, sample):
        image = sample["image"]
        mask = sample["mask"]

        image_np = image.permute(1, 2, 0).cpu().numpy()
        mask_np = mask.squeeze(0).cpu().numpy()

        augmented = self.transform(image=image_np, mask=mask_np)

        aug_image = augmented["image"]
        aug_mask = augmented["mask"]

        aug_image_t = torch.from_numpy(aug_image).permute(2, 0, 1)
        aug_mask_t  = torch.from_numpy(aug_mask).unsqueeze(0)

        sample["image"] = aug_image_t
        sample["mask"]  = aug_mask_t
        
        return sample