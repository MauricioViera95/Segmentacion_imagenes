# ========================================================================
# CNN
# ------------------------------------------------------------------------
# Train any segmentation model with any backbone and loss on a
# local dataset of images using torchgeo
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
from tqdm import tqdm

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

### Directorios

#### Directorio base
BASE_DIR = r'C:/Users/ronny/OneDrive/Documentos/MAESTRIA/1. CONTENIDO/16. TFM/CODIGO'

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
EPOCHS = 30
IN_CHANNELS = 3  
NUM_CLASSES = 2  
IMG_SIZE = 512
TRAIN_SAMPLE_SIZE = 512
VAL_SAMPLE_SIZE = TRAIN_SAMPLE_SIZE // 2
PATIENCE = 6
SCHED_FACTOR = 0.1

#### GPU 

import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_DEVICES = torch.cuda.device_count() if torch.cuda.is_available() else mp.cpu_count()
# WORKERS = mp.cpu_count()
WORKERS = 3


print(f"DEVICE: {DEVICE} | NUM_DEVICES: {NUM_DEVICES} | WORKERS: {WORKERS}")

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
    "epochs": EPOCHS,
    "lr": None,                 # 🔄 Se actualizará en cada trial
    "batch_size": None,         # 🔄
    "model": None,              # 🔄
    "backbone": None,           # 🔄
    "loss": None,               # 🔄
    "img_size": IMG_SIZE,
    "train_samples": TRAIN_SAMPLE_SIZE,
    "val_samples": VAL_SAMPLE_SIZE,
    "patience": None,
    "sched_factor": None
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
def log_validation_prediction(model, fixed_sample, epoch, device, writer):
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
def create_loss_function(loss_name: str, alpha=0.5):
    loss_name = loss_name.lower()
    if loss_name == 'ce':
        return nn.CrossEntropyLoss()
    elif loss_name == 'focal':
        return smp.losses.FocalLoss(mode="multiclass")
    elif loss_name == 'dice':
        return smp.losses.DiceLoss(mode="multiclass")
    elif loss_name == 'jaccard':
        return smp.losses.JaccardLoss(mode="multiclass")    # Jaccard loss is loss wrt model IoU
    elif loss_name == 'ce+dice':
        ce = nn.CrossEntropyLoss()
        dice = smp.losses.DiceLoss(mode="multiclass")
        return lambda x, y: alpha * dice(x, y) + (1 - alpha) * ce(x, y)
    


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


#### Preguntar si la activación sigmoide se debería utilizar al ser un problema binario

### CHECKPOINT UTILITIES
    
def load_checkpoint(model, filename, device):
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Checkpoint cargado desde: {filename}")


# TRAINING LOOP

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_samples  = 0

    # Mostrar dispositivo una sola vez
    if device == 'cuda':
        print(f"🟢 Entrenando en: {device.upper()} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"🟡 Entrenando en: {device.upper()}")

    for batch in tqdm(dataloader):
        images = batch["image"].to(device)
        masks  = batch["mask"].squeeze(1).to(device).long()
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        
        bs = images.size(0)
        running_loss  += loss.item() * bs
        total_samples += bs
    
    epoch_loss = running_loss / total_samples
    return epoch_loss

@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, metric, device):
    model.eval()
    running_loss = 0.0
    metric.reset()

    total_samples = 0
    for batch in tqdm(dataloader, desc="Validating"):
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
    writer
):
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

        log_validation_prediction(model, fixed_sample, epoch, device, writer)
        
        # Validation IoU is what the scheduler monitors
        scheduler.step(val_iou)

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            print("Validation IoU improved! Saving checkpoint...")
            torch.save({'model_state_dict': model.state_dict()}, checkpoint_path)


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

    
    for i, batch in enumerate(tqdm(dataloader, desc="Testing")):
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




#### Búsqueda de hiperparámetros

import optuna
from torch.utils.tensorboard import SummaryWriter
from transforms import TrainTransform, TestValTransform
from datasets import PNOAImages, SidewalkLabels


def fix_seed(seed=42):
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def objective(trial):
    # Fijar semilla para garantizar que fixed_sample sea el mismo en todos los trials
    fix_seed(42)

    # Hiperparámetros a buscar
    # === Búsqueda de hiperparámetros enfocados en subir IoU sobre FPN+ResNet34 ===
    lr = trial.suggest_float("lr", 2e-5, 8e-5, log=True)  # centrado en torno a 5e-5
    loss_name = trial.suggest_categorical("loss", ["ce", "ce+dice", "focal"])
    alpha = trial.suggest_float("dice_weight", 0.4, 0.6)  # solo aplica para ce+dice
    batch_size = trial.suggest_categorical("batch_size", [2, 3])  # pequeño para mejorar generalización
    sched_patience = trial.suggest_int("sched_patience", 5, 7)
    sched_factor = trial.suggest_float("sched_factor", 0.08, 0.2)

    seg_model = "fpn"
    backbone = "resnet34"

    # Actualizar hparams
    hparams["lr"] = lr
    hparams["loss"] = loss_name
    hparams["batch_size"] = batch_size
    hparams["model"] = seg_model
    hparams["backbone"] = backbone
    hparams["patience"] = sched_patience
    hparams["sched_factor"] = sched_factor
    hparams["dice_weight"] = alpha

    # ==== TensorBoard writer único por trial ====
    trial_id = trial.number
    log_dir = os.path.join(BASE_DIR, f"runs/optuna_trial_{trial_id}")
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_hparams(hparams, {})

    # Modelo
    model = create_segmentation_model(seg_model, backbone, IN_CHANNELS, NUM_CLASSES, weights=True).to(DEVICE)
    criterion = create_loss_function(loss_name, alpha=alpha)
    metric = torchmetrics.JaccardIndex(task="multiclass", num_classes=NUM_CLASSES).to(DEVICE)

    # Datasets
    train_ds = build_dataset("train")
    train_ds.transforms = TrainTransform(scaled_means, scaled_stds)
    val_ds = build_dataset("val")
    val_ds.transforms = TestValTransform(scaled_means, scaled_stds)
    test_ds = build_dataset("test")
    test_ds.transforms = TestValTransform(scaled_means, scaled_stds)

    # Samplers
    train_sampler = RandomBatchGeoSampler(train_ds, size=IMG_SIZE, batch_size=batch_size, length=TRAIN_SAMPLE_SIZE)
    val_sampler = RandomBatchGeoSampler(val_ds, size=IMG_SIZE, batch_size=batch_size, length=VAL_SAMPLE_SIZE)
    test_sampler = GridGeoSampler(test_ds, size=IMG_SIZE, stride=IMG_SIZE // 2)

    # Loaders
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=stack_samples, num_workers=WORKERS)
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler, collate_fn=stack_samples, num_workers=WORKERS)
    test_loader = DataLoader(test_ds, sampler=test_sampler, batch_size=1, collate_fn=stack_samples, num_workers=WORKERS)

    # fixed_sample = next(iter(test_loader))
    FIXED_SAMPLE_INDEX = 111
    fixed_sample = None
    for i, batch in enumerate(test_loader):
        if i == FIXED_SAMPLE_INDEX:
            fixed_sample = batch
            break
    if fixed_sample is None:
        raise ValueError(f"❌ No se encontró una muestra con índice {FIXED_SAMPLE_INDEX} en el test_loader.")

    # Registrar imagen y GT al inicio del trial (una vez por trial)
    log_fixed_sample(fixed_sample, DEVICE, scaled_means, scaled_stds)
    # === Elegir una muestra fija (como en el código base) ===
    
    # Optimizador y scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=sched_patience,
        factor=sched_factor,
        mode="max"
    )

    # Entrenamiento
    best_val_iou = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        metric=metric,
        device=DEVICE,
        epochs=EPOCHS,
        fixed_sample=fixed_sample,
        checkpoint_path="optuna_trial.ckpt",
        writer=writer
    )

    # Evaluación final en test
    load_checkpoint(model, "optuna_trial.ckpt", DEVICE)
    test_loss, test_iou = test_model_center_crop(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        metric=metric,
        device=DEVICE,
        center_margin=IMG_SIZE // 8
    )

    # Registrar resultados finales del trial
    writer.add_hparams(hparams, {
        "val_iou": best_val_iou,
        "test_iou": test_iou,
        "test_loss": test_loss
    })
    writer.close()

    # Guardar resultados del trial
    trial.set_user_attr("val_iou", best_val_iou)
    trial.set_user_attr("test_iou", test_iou)
    trial.set_user_attr("test_loss", test_loss)

    return best_val_iou


### Ejecutar búsqueda

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=8)

    print("✅ Búsqueda finalizada")
    print("🎯 Mejor combinación:")
    print(study.best_trial)

    print("📊 Métricas del mejor trial:")
    print("  Val IoU :", study.best_trial.user_attrs["val_iou"])
    print("  Test IoU:", study.best_trial.user_attrs["test_iou"])
    print("  Test Loss:", study.best_trial.user_attrs["test_loss"])


# === Resumen de todos los trials ===
import pandas as pd

# Crear una lista con la info relevante
results = []
for t in study.trials:
    if t.state.name == "COMPLETE":
        results.append({
            "Trial": t.number,
            "Val IoU": round(t.user_attrs.get("val_iou", 0), 4),
            "Test IoU": round(t.user_attrs.get("test_iou", 0), 4),
            "Test Loss": round(t.user_attrs.get("test_loss", 0), 4),
            "LR": t.params.get("lr"),
            "Loss": t.params.get("loss"),
            "Dice α": t.params.get("dice_weight", None),
            "Batch": t.params.get("batch_size"),
            "Patience": t.params.get("sched_patience"),
            "Sched Factor": t.params.get("sched_factor"),
        })

# Crear DataFrame ordenado por Test IoU
df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by="Val IoU", ascending=False).reset_index(drop=True)

# Mostrar resumen ordenado
print("\n📊 Resumen de todos los trials ordenado por Test IoU:")
print(df_results.to_string(index=False))
