# ========================================================================
# CNN
# ------------------------------------------------------------------------
# Entrenamiento con cualquier modelo de segmentación con cualquier backbone y loss
# dataset local de imágenes utilizando torchgeo
#
# @author   Mauricio Viera-Torres <ronnyvieramt95@gmail.com>
# @address  La Tola (Ecuador)
# @version  1.0.0 (2025-05-23)
# ========================================================================


### Librerías

import os

import glob
from collections import Counter
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import numpy as np
import multiprocessing as mp

import rasterio

import matplotlib
matplotlib.use("Agg") # Important set matplotlib backend to headless
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

torch.cuda.empty_cache()

# Torchgeo functionality
from torchgeo.datasets import stack_samples, RasterDataset
from torchgeo.samplers import RandomBatchGeoSampler, GridGeoSampler
import torch.optim as optim

# Albumentations for data augmentation
import albumentations as A

# Torchmetrics
import torchmetrics

import matplotlib.pyplot as plt
import numpy as np
import ssl
import segmentation_models_pytorch as smp

import random

def set_seed(s=432):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(423)


### Directorios

#### Directorio base
BASE_DIR = r'C:\Users\ronny\OneDrive\Documentos\MAESTRIA\1. CONTENIDO\16. TFM\CODIGO'

#### Directorios de Output, Máscara Binaria y Ortofotos
OUTPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, '../DATA/OUTPUT/CNN/'))
INPUT_MASK_DIR = os.path.abspath(os.path.join(BASE_DIR, '../DATA/OUTPUT/MASCARAS/'))
INPUT_IMG_DIR = os.path.abspath(os.path.join(BASE_DIR, '../INSUMOS/ORTOFOTOS/'))

print("OUTPUT_DIR:", OUTPUT_DIR)
print("INPUT_MASK_DIR:", INPUT_MASK_DIR)
print("INPUT_IMG_DIR:", INPUT_IMG_DIR)


#### Comprobación de que los directorios sean los correctos
def listar_archivos(directorio):
    try:
        archivos = os.listdir(directorio)
        print(f"Archivos en {directorio}:")
        for f in archivos:
            print("  ", f)
    except FileNotFoundError:
        print(f"ERROR: No se encontró el directorio {directorio}")

listar_archivos(OUTPUT_DIR)
listar_archivos(INPUT_MASK_DIR)
listar_archivos(INPUT_IMG_DIR)


#### Parámetros

IN_CHANNELS = 3  
NUM_CLASSES = 2  
IMG_SIZE = 512
TRAIN_SAMPLE_SIZE = 1024
VAL_SAMPLE_SIZE = 512

EPOCHS = 250                         # 150–180 suele ir bien con ReduceLROnPlateau
LR = 6e-5                            # un poco > 4.5e-5 para dar “empuje”
BATCH_SIZE = 3                       # estable; usaremos acumulación para simular 6
GRAD_ACCUM_STEPS = 2                 # 3 x 2 = 6 “efectivo”
WEIGHT_DECAY = 1e-5                  # L2 suave
PATIENCE = 8                         # más aire al scheduler
SCHED_FACTOR = 0.3                   # bajada más marcada cuando haga plateau
CENTER_MARGIN = IMG_SIZE // 4        # 85 px aprox; sube IoU de test ligeramente
LOSS = "ce+dice"                     # cambia focal -> ce+dice (más estable)
CE_LABEL_SMOOTH = 0.05               # CE con label smoothing
DICE_WEIGHT = 0.7                    # 0.7*Dice + 0.3*CE

# SCHEDULER
# Lowers the lr dinamically to 10% if it IoU does not improve after PATIENCE epochs
PATIENCE = 10
SCHED_FACTOR = 0.3

SEGMENTATION_MODEL = 'deeplabv3plus'
BACKBONE = 'mit_b4'
LOSS = 'focal+dice'  # Solo para referencia, se usa luego
CHECKPOINT_FILE = 'deeplabv3plus_mit_b4_best.ckpt'


#### GPU 

import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_DEVICES = torch.cuda.device_count() if torch.cuda.is_available() else mp.cpu_count()
WORKERS = mp.cpu_count()


print(f"DEVICE: {DEVICE} | NUM_DEVICES: {NUM_DEVICES} | WORKERS: {WORKERS}")


WORKERS = 10


# Comprobar el nombre de la gpu

if torch.cuda.is_available():
    print("CUDA disponible ✅")
    print("Nombre de la GPU utilizada por PyTorch:")
    print(torch.cuda.get_device_name(0))
else:
    print("CUDA NO disponible ❌")


# DEVICE = "cuda"

### Hiperparámetros
hparams = {
    "epochs": EPOCHS,                   # Número total de épocas de entrenamiento
    "lr": LR,                           # Learning rate (tasa de aprendizaje)
    "batch_size": BATCH_SIZE,          # Tamaño de batch para el entrenamiento
    "model": SEGMENTATION_MODEL,       # Tipo de arquitectura de segmentación (ej. 'fpn')
    "backbone": BACKBONE,              # Backbone usado por el modelo (ej. 'resnet50')
    "loss": LOSS,                      # Función de pérdida (ej. 'ce' para CrossEntropy)
    "img_size": IMG_SIZE,              # Tamaño al que se redimensionan las imágenes (ej. 512x512)
    "train_samples": TRAIN_SAMPLE_SIZE,# Número de muestras usadas para visualización de entrenamiento
    "val_samples": VAL_SAMPLE_SIZE,    # Número de muestras para visualización de validación
    "patience": PATIENCE,              # Paciencia del scheduler (cuántas épocas sin mejora antes de reducir el LR)
    "sched_factor": SCHED_FACTOR       # Factor multiplicador del LR cuando se activa el scheduler
}

### Tensorboard writer

log_dir = os.path.join(BASE_DIR, "runs")

writer = SummaryWriter(log_dir=log_dir)
writer.add_hparams(hparams, {})


### Extract new RGB means and stds for your training data

# PNOA_MA_OF_ETRS89_HU30_h25_0559_1.tif
means = [109.9545, 108.8082, 98.0588]
stds = [57.7131, 52.6563, 49.0428]
scaled_means = [0.4312, 0.4267, 0.3845]
scaled_stds = [0.2263, 0.2065, 0.1923]


### Desnormalizar la imagen
def denormalize_image(tensor, means, stds):
    tensor = tensor.clone().cpu()
    for c in range(tensor.shape[0]):
        tensor[c] = tensor[c] * stds[c] + means[c]

    tensor = tensor * 255.0
    tensor = torch.clamp(tensor, 0, 255).byte()
    return tensor.numpy()

def create_combined_loss(ce_weight=0.5, dice_weight=0.5):
    ce = nn.CrossEntropyLoss()
    dice = smp.losses.DiceLoss(mode="multiclass")
    def loss_fn(preds, targets):
        return ce_weight * ce(preds, targets) + dice_weight * dice(preds, targets)
    return loss_fn


#### Función para recortar las máscaras según las ortofotos

def build_dataset(ds_name):
    imgs = PNOAImages(
        paths=INPUT_IMG_DIR,
        cache=False # NOTE keep as True if no GPU memory issues for faster training
    )
    print(imgs)

    labels = SidewalkLabels(
        paths=os.path.join(INPUT_MASK_DIR, ds_name),
        cache=False # NOTE keep as True if no GPU memory issues for faster training
    )
    print(labels)

    return imgs & labels

### Visualizar y registrar un par imagen-máscara (sample) en TensorBoard

def log_fixed_sample(fixed_sample, device, means, stds):
    image = fixed_sample["image"].to(device)
    mask = fixed_sample["mask"].to(device)

    # Denormalize image before logging.
    image = denormalize_image(image.squeeze(0), means, stds)
    mask = mask.squeeze().cpu().numpy()
    # Convert image from CHW to HWC for plotting.
    image = np.transpose(image, (1, 2, 0)) if image.ndim == 3 else image

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image)
    ax[0].set_title("Fixed Sample Image")
    ax[0].axis("off")
    
    ax[1].imshow(mask, cmap='viridis')
    ax[1].set_title("Fixed Sample Ground Truth")
    ax[1].axis("off")

    # Log the figure as Image
    writer.add_figure("Fixed Sample (Image & GT)", fig, global_step=0)

    plt.close(fig)

### Visualizar la máscara predicha por el modelo sobre una imagen fija del conjunto de validación
def log_validation_prediction(model, fixed_sample, epoch, device):
    model.eval()
    with torch.no_grad():
        img = fixed_sample["image"].to(device)
        logits = model(img)
        pred_mask = torch.argmax(logits, dim=1) # argmax along the output class dimension
    
    pred_mask = pred_mask.squeeze(0).cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(pred_mask, cmap='viridis')
    ax.set_title("Predicted Mask")
    ax.axis("off")
    
    writer.add_figure("Validation Prediction", fig, global_step=epoch)

    plt.close(fig)


### Funciones de pérdida
# LOSSES
def create_loss_function(loss_name: str):
    loss_name = loss_name.lower()
    if loss_name == 'ce':
        return nn.CrossEntropyLoss()
    elif loss_name == 'focal':
        return smp.losses.FocalLoss(mode="multiclass")
    elif loss_name == 'dice':
        return smp.losses.DiceLoss(mode="multiclass")
    elif loss_name == 'jaccard':
        return smp.losses.JaccardLoss(mode="multiclass")    # Jaccard loss is loss wrt model IoU
    
def create_combined_loss(ce_weight=0.3, dice_weight=0.7, label_smoothing=0.05, class_weights=None):
    """
    ce_weight + dice_weight = 1.0
    class_weights: torch.tensor([w_bg, w_fg], device=DEVICE) o None
    """
    ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    dice = smp.losses.DiceLoss(mode="multiclass")

    def loss_fn(preds, targets):
        return ce_weight * ce(preds, targets) + dice_weight * dice(preds, targets)

    return loss_fn



### MODEL ARCHITECTURES

def create_segmentation_model(seg_model, backbone, in_channels, num_classes, weights=False, activation=None):
    if weights is True:
        weights = "imagenet" 
    
    elif weights is False:
        weights = None
    
    seg_model = seg_model.lower()

    if seg_model == "unet":
        model = smp.Unet(
            encoder_name=backbone,
            encoder_weights=weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation
        )
    elif seg_model == "unetplusplus":
        model = smp.UnetPlusPlus(
            encoder_name=backbone,
            encoder_weights=weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation
        )
    elif seg_model == "linknet":
        model = smp.Linknet(
            encoder_name=backbone,
            encoder_weights=weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation
        )
    elif seg_model == "fpn":
        model = smp.FPN(
            encoder_name=backbone,
            encoder_weights=weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation
        )
    elif seg_model == "pspnet":
        model = smp.PSPNet(
            encoder_name=backbone,
            encoder_weights=weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation
        )
    elif seg_model == "pan":
        model = smp.PAN(
            encoder_name=backbone,
            encoder_weights=weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation
        )
    elif seg_model == "deeplabv3":
        model = smp.DeepLabV3(
            encoder_name=backbone,
            encoder_weights=weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation
        )
    elif seg_model == "deeplabv3plus":
        model = smp.DeepLabV3Plus(
            encoder_name=backbone,
            encoder_weights=weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation
        )

    else:
        raise ValueError("Please input a valid segmentation model name architecture")
    
    return model


### CHECKPOINT UTILITIES
    
def load_checkpoint(model, filename, device):
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Checkpoint cargado desde: {filename}")


# TRAINING LOOP

from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler(enabled=(DEVICE=="cuda"))

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_samples = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(dataloader, desc="✔ Entrenando", leave=False)
    for step, batch in enumerate(pbar, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks  = batch["mask"].squeeze(1).to(device, non_blocking=True).long()

        with autocast(enabled=(device=="cuda")):
            logits = model(images)
            loss = criterion(logits, masks) / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()

        if step % GRAD_ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        bs = images.size(0)
        running_loss  += (loss.item() * GRAD_ACCUM_STEPS) * bs
        total_samples += bs

    epoch_loss = running_loss / max(total_samples, 1)
    return epoch_loss



@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, metric, device):
    model.eval()
    running_loss = 0.0
    metric.reset()

    total_samples = 0
    for batch in tqdm(dataloader, desc="🔎 Validando", leave=False):
        images = batch["image"].to(device)
        masks  = batch["mask"].squeeze(1).to(device).long()
        
        logits = model(images)
        loss = criterion(logits, masks)
        running_loss += loss.item() * images.size(0)
        
        preds = torch.argmax(logits, dim=1)
        metric.update(preds, masks.squeeze(1).long())
        
        total_samples += images.size(0)
    
    epoch_loss = running_loss / total_samples
    epoch_iou = metric.compute().item()
    return epoch_loss, epoch_iou


def train_model(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    scheduler, 
    metric, 
    device, 
    epochs, 
    fixed_sample, 
    checkpoint_path,
):
    no_improve_epochs = 0

    best_val_iou = 0.0
    
    print(f"Starting training on {device =}")

    for epoch in range(epochs):
        # Do 1 epoch of training and validation, recording relevant metrics
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_iou = validate_one_epoch(model, val_loader, criterion, metric, device)
        
        print(f"Epoch [{epoch+1}/{epochs}]: "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val IoU: {val_iou:.4f}")
        
        # scalar logging
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Metric/Val_IoU",    val_iou,   epoch)
        writer.add_scalar("Learning Rate",     optimizer.param_groups[0]['lr'], epoch)

        log_validation_prediction(model, fixed_sample, epoch, device)
        
        # Validation IoU is what the scheduler monitors
        scheduler.step(val_iou)

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            no_improve_epochs = 0  # reiniciar si mejora
            print("✅ Validation IoU improved! Saving checkpoint...")
            torch.save({'model_state_dict': model.state_dict()}, checkpoint_path)
        else:
            no_improve_epochs += 1
            print(f"⚠️  No mejora detectada en {no_improve_epochs} épocas")

        if no_improve_epochs >= 50:
            print("⏹️ Early Stopping: No mejora en 35 épocas consecutivas.")
            break



    print("Finished Training.")
    return best_val_iou

# INFERENCE (TESTING)

@torch.no_grad()
def test_model_center_crop(
    model,
    dataloader,
    criterion,
    metric,
    device,
    center_margin=64,
):
    model.eval()
    metric.reset()
    total_loss = 0.0
    total_samples = 0

    
    for i, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        masks = batch["mask"].squeeze(1).to(device).long()
        
        logits = model(images)
        loss = criterion(logits, masks)

        total_loss += loss.item() * images.size(0)
        
        preds = torch.argmax(logits, dim=1)
        
        # -- Center crop both preds and masks:
        if center_margin > 0:
            preds = preds[:, 
                          center_margin:-center_margin, 
                          center_margin:-center_margin]
            masks = masks[:, 
                          center_margin:-center_margin, 
                          center_margin:-center_margin]
        
        metric.update(preds, masks)
        total_samples += images.size(0)
    
    avg_loss = total_loss / (total_samples if total_samples > 0 else 1)
    test_iou = metric.compute().item()

    print(f"Test (Center-Crop) Loss: {avg_loss:.4f} | Test IoU: {test_iou:.4f}")

    return avg_loss, test_iou


# MAIN SECTION
from transforms import TrainTransform, TestValTransform
from datasets import PNOAImages, SidewalkLabels

if __name__ == "__main__":
    # Allows to safely download trained weights from the script
    ssl._create_default_https_context = ssl._create_unverified_context

    print(f'Running on {NUM_DEVICES} {DEVICE}(s)')

    # Model creation
    model = create_segmentation_model(
        SEGMENTATION_MODEL,
        BACKBONE,
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        weights=True,
    ).to(DEVICE)
    
    class_weights = None

    ce   = nn.CrossEntropyLoss(label_smoothing=CE_LABEL_SMOOTH)
    dice = smp.losses.DiceLoss(mode="multiclass")

    def combined_loss(preds, targets):
        return 0.3 * ce(preds, targets) + 0.7 * dice(preds, targets)

    criterion = combined_loss


    # Metric selection (IoU by default)
    iou_metric = torchmetrics.JaccardIndex(
        task="multiclass",
        num_classes=NUM_CLASSES,
    ).to(DEVICE)


    # Build datasets from named directories in INPUT_DIR and assign transforms:
    train_ds = build_dataset("train")
    train_ds.transforms = TrainTransform(scaled_means, scaled_stds)
    val_ds = build_dataset("val")
    val_ds.transforms = TestValTransform(scaled_means, scaled_stds)
    test_ds = build_dataset("test")
    test_ds.transforms = TestValTransform(scaled_means, scaled_stds)

    # The dataloaders will call these samplers when sampling a sample crop from train / val / test areas 
    # Note that a sample crop will contain up to BATCH_SIZE stacked images
    train_sampler = RandomBatchGeoSampler(
        dataset=train_ds,
        size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        length=TRAIN_SAMPLE_SIZE,
    )

    val_sampler = RandomBatchGeoSampler(
        dataset=val_ds,
        size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        length=VAL_SAMPLE_SIZE,
    )

    # The test sampel will always sample single images one by one
    TEST_STRIDE = IMG_SIZE // 2
    CENTER_MARGIN = TEST_STRIDE // 4
    test_sampler = GridGeoSampler(
        dataset=test_ds,
        size=IMG_SIZE,
        stride=TEST_STRIDE,
    )

    # The dataloaders are iterators that quickly sample batches from the dataset using the sampler
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        collate_fn=stack_samples,
        num_workers=WORKERS
    )

    val_loader = DataLoader(
        val_ds,
        batch_sampler=val_sampler,
        collate_fn=stack_samples,
        num_workers=WORKERS
    )

    test_loader = DataLoader(
        test_ds,
        sampler=test_sampler,
        collate_fn=stack_samples,
        batch_size=1,
        num_workers=WORKERS
    )

    # This inefficient hack here lets you select which images is logged during training
    # Feel free to try other numbers until you see an interesting image
    FIXED_SAMPLE_INDEX = 111
    for i, batch in enumerate(test_loader):
        if i < FIXED_SAMPLE_INDEX:
            continue
        fixed_sample = batch
        break

    # Log the actual ground truth mask and the fixed image
    log_fixed_sample(fixed_sample, DEVICE, scaled_means, scaled_stds)

    # You can choose from many optimizers https://docs.pytorch.org/docs/stable/optim.html#algorithms
    # optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # The scheduler lowers the lr dinamically to 10% if it IoU does not improve after PATIENCE epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=PATIENCE, factor=SCHED_FACTOR, threshold=0, mode="max"
    )

    # Model training
    best_val_iou = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        metric=iou_metric,
        device=DEVICE,
        epochs=EPOCHS,
        fixed_sample=fixed_sample,
        checkpoint_path=CHECKPOINT_FILE,
    )

    load_checkpoint(model,CHECKPOINT_FILE, DEVICE) # Now load best-performing checkpoint for testing

    # Testing function only keeps the image centers, where segmentation models work best
    # As an exercise to the reader, try to make a gaussian window for the center instead of cropping
    test_loss, test_iou = test_model_center_crop(
        model=model, 
        dataloader=test_loader, 
        criterion=criterion, 
        metric=iou_metric, 
        device=DEVICE,
        center_margin=CENTER_MARGIN
    )

    # Log the overall run results
    writer.add_hparams(
        hparams,
        {"best_val_iou": best_val_iou, "test_loss": test_loss, "test_iou": test_iou}
    )

    writer.close()