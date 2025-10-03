# ========================================================================
# INFERENCIA CNN (DeepLabV3+ MIT-B4) SOBRE VENTANA RÁSTER + EXPORTACIÓN SHP
# ========================================================================
#
# @author   Mauricio Viera-Torres <ronnyvieramt95@gmail.com>
# @address  La Tola (Ecuador)
# @version  1.0.0 (2025-05-23)
# ========================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import geopandas as gpd
from shapely.geometry import shape
import rasterio
from rasterio.windows import Window
from rasterio.features import shapes
import torch
import segmentation_models_pytorch as smp

# ---------------------------
# Directorios y rutas clave
# ---------------------------
BASE_DIR = r'C:\Users\ronny\OneDrive\Documentos\MAESTRIA\1. CONTENIDO\16. TFM\CODIGO'
OUTPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, '../DATA/OUTPUT/CNN/'))
INPUT_MASK_DIR = os.path.abspath(os.path.join(BASE_DIR, '../DATA/OUTPUT/MASCARAS/'))
INPUT_IMG_DIR = os.path.abspath(os.path.join(BASE_DIR, '../INSUMOS/ORTOFOTOS/'))

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Checkpoint en el directorio raíz del código
CHECKPOINT_FILE = os.path.join(BASE_DIR, 'deeplabv3plus_mit_b4_best.ckpt')

# ---------------------------
# Parámetros del modelo/datos
# ---------------------------
IN_CHANNELS = 3
NUM_CLASSES = 2
SEGMENTATION_MODEL = 'deeplabv3plus'
BACKBONE = 'mit_b4'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Medias y stds escaladas (0-1) usadas en entrenamiento
scaled_means = [0.4312, 0.4267, 0.3845]
scaled_stds  = [0.2263, 0.2065, 0.1923]

# ---------------------------
# Utilidades de modelo
# ---------------------------
def create_segmentation_model(seg_model, backbone, in_channels, num_classes, weights=True, activation=None):
    """
    Crea un modelo SMP de segmentación. weights=True => encoder preentrenado ImageNet.
    """
    enc_w = "imagenet" if weights else None
    seg_model = seg_model.lower()
    if seg_model == "deeplabv3plus":
        model = smp.DeepLabV3Plus(
            encoder_name=backbone,
            encoder_weights=enc_w,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation
        )
    else:
        raise ValueError("Esta integración está preparada para 'deeplabv3plus'.")
    return model

def load_checkpoint(model, filename, device):
    ckpt = torch.load(filename, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"✅ Checkpoint cargado desde: {filename}")

# ---------------------------
# Utilidades de inferencia/plot/SHP
# ---------------------------
@torch.no_grad()
def _normalize_rgb_tensor(x_np, means, stds, device):
    """
    x_np: np.ndarray [C,H,W] (C=3 RGB, rango 0-255).
    Devuelve tensor float32 normalizado [1,C,H,W] en device.
    """
    x = torch.from_numpy(x_np.astype(np.float32) / 255.0)  # [C,H,W] en 0-1
    for c in range(3):
        x[c] = (x[c] - means[c]) / stds[c]
    return x.unsqueeze(0).to(device)  # [1,C,H,W]

def _array_to_shapefile(class_array, transform, crs, shp_out,
                        class_map={0: "No permeable", 1: "Permeable"},
                        remove_holes=True, min_area=0.0):
    """
    Convierte un array 2D de clases (enteros) a polígonos y exporta a Shapefile.
    - min_area en unidades del CRS (si UTM ⇒ m²)
    """
    geoms, vals = [], []
    for geom, val in shapes(class_array.astype(np.int32), mask=None, transform=transform):
        if val is None:
            continue
        geom_shape = shape(geom)
        if remove_holes:
            geom_shape = geom_shape.buffer(0)  # limpia hoyos y self-intersections
        if min_area > 0 and geom_shape.area < min_area:
            continue
        geoms.append(geom_shape)
        vals.append(int(val))

    if not geoms:
        gpd.GeoDataFrame({"class_val": [], "class_name": []},
                         geometry=[], crs=crs).to_file(shp_out, driver="ESRI Shapefile")
        return

    gdf = gpd.GeoDataFrame(
        {"class_val": vals, "class_name": [class_map.get(v, f"Clase {v}") for v in vals]},
        geometry=geoms, crs=crs
    )
    gdf.to_file(shp_out, driver="ESRI Shapefile")

@torch.no_grad()
def _process_batch_tiles(model, tiles, coords, means, stds, prob_acc, count_acc, device, num_classes):
    batch_t = torch.cat([_normalize_rgb_tensor(t, means, stds, device) for t in tiles], dim=0)  # [B,3,tile,tile]
    logits = model(batch_t)  # [B,num_classes,tile,tile]
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    for i, (y0, y1, x0, x1) in enumerate(coords):
        h, w = y1 - y0, x1 - x0
        prob_acc[:, y0:y1, x0:x1] += probs[i, :, :h, :w]
        count_acc[y0:y1, x0:x1] += 1.0

@torch.no_grad()
def infer_window_cnn(
    model,
    raster_array,         # np.ndarray [C,H,W] con C>=3 (usamos RGB)
    means, stds,          # listas de 3 elementos (scaled_means/stds en 0-1)
    device=DEVICE,
    tile=512,
    stride=256,
    num_classes=NUM_CLASSES,
    batch_size=4
):
    """
    Inferencia por tiles con solape y promedio de probabilidades.
    Devuelve:
      - y_pred: np.ndarray [H,W] (clase entera)
      - prob_map: np.ndarray [num_classes, H, W] (probabilidades softmax)
    """
    C, H, W = raster_array.shape
    assert C >= 3, "Se requieren al menos 3 bandas (RGB)."

    rgb = raster_array[:3].astype(np.float32)

    prob_acc = np.zeros((num_classes, H, W), dtype=np.float32)
    count_acc = np.zeros((H, W), dtype=np.float32)

    tiles, coords = [], []
    for y0 in range(0, H, stride):
        for x0 in range(0, W, stride):
            y1 = min(y0 + tile, H)
            x1 = min(x0 + tile, W)
            tile_np = np.zeros((3, tile, tile), dtype=np.float32)
            crop = rgb[:, y0:y1, x0:x1]
            tile_np[:, :crop.shape[1], :crop.shape[2]] = crop

            tiles.append(tile_np)
            coords.append((y0, y1, x0, x1))

            if len(tiles) == batch_size:
                _process_batch_tiles(model, tiles, coords, means, stds, prob_acc, count_acc, device, num_classes)
                tiles, coords = [], []

    if tiles:
        _process_batch_tiles(model, tiles, coords, means, stds, prob_acc, count_acc, device, num_classes)

    count_acc[count_acc == 0] = 1.0
    prob_map = prob_acc / count_acc[None, :, :]
    y_pred = prob_map.argmax(axis=0).astype(np.uint8)
    return y_pred, prob_map

def visualizar_cnn_y_exportar(
    y_pred,                     # [H,W] clases 0/1
    raster_array,               # [C,H,W] para mostrar RGB
    clase_dict,                 # p.ej. {"Permeable":1, "No permeable":0}
    transform=None, crs=None,
    shp_out=None, min_area=0.0
):
    # Paleta/leyenda coherentes con clase_dict
    inv_map = {v: k for k, v in clase_dict.items()}  # 0/1 -> nombre
    class_labels = [inv_map.get(1, "Permeable"), inv_map.get(0, "No permeable")]
    class_colors = {inv_map.get(1, "Permeable"): "green",
                    inv_map.get(0, "No permeable"): "lightgray"}
    custom_cmap = ListedColormap([class_colors[cl] for cl in class_labels])
    legend_patches = [mpatches.Patch(color=class_colors[cl], label=cl) for cl in class_labels]

    # Figura compacta y legible
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.6), dpi=140)
    axs[0].imshow(y_pred, cmap=custom_cmap)
    axs[0].set_title("Clasificación CNN", fontsize=14, fontweight='bold', pad=8)
    axs[0].axis('off')
    axs[0].legend(handles=legend_patches, loc='lower left', fontsize=12, frameon=True, facecolor='white')

    rgb_img = np.transpose(raster_array[[0, 1, 2]], (1, 2, 0)).astype(np.uint8)
    axs[1].imshow(rgb_img)
    axs[1].set_title("Imagen RGB original", fontsize=14, fontweight='bold', pad=8)
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

    # Exportar a SHP si se solicita
    if shp_out is not None:
        if transform is None or crs is None:
            raise ValueError("Para exportar a SHP debes proporcionar 'transform' y 'crs'.")
        class_map = {val: key for key, val in clase_dict.items()}  # 0/1 -> nombre
        _array_to_shapefile(
            class_array=y_pred,
            transform=transform,
            crs=crs,
            shp_out=shp_out,
            class_map=class_map,
            remove_holes=True,
            min_area=min_area
        )

# ---------------------------
# Construcción y carga modelo
# ---------------------------
model = create_segmentation_model(
    SEGMENTATION_MODEL,
    BACKBONE,
    in_channels=IN_CHANNELS,
    num_classes=NUM_CLASSES,
    weights=True,           # encoder preentrenado
    activation=None
).to(DEVICE)

# Cargar checkpoint entrenado
if not os.path.exists(CHECKPOINT_FILE):
    raise FileNotFoundError(f"No se encontró el checkpoint: {CHECKPOINT_FILE}")
load_checkpoint(model, CHECKPOINT_FILE, DEVICE)
model.eval()

# ---------------------------
# Lectura de ventana del ráster
# ---------------------------
raster_name = "PNOA_MA_OF_ETRS89_HU30_h25_0559_1.tif"
raster_path = os.path.join(INPUT_IMG_DIR, raster_name)
if not os.path.exists(raster_path):
    raise FileNotFoundError(f"No se encontró el ráster: {raster_path}")

print(f"📄 Usando ráster: {raster_path}")

# Ventana (ajusta si necesitas)
col_off, row_off = 15000, 15000
width, height   = 1500, 1500
window = Window(col_off, row_off, width, height)

with rasterio.open(raster_path) as src:
    print(f"{src.count} bandas | Tamaño: {src.width} x {src.height}")
    raster_array   = src.read(window=window)        # (C,H,W)
    transform_win  = src.window_transform(window)   # transform georreferenciado de la ventana
    crs            = src.crs                        # CRS del ráster

print("ℹ️ raster_array shape:", raster_array.shape)

# ---------------------------
# Inferencia sobre la ventana
# ---------------------------
y_pred_cnn, prob_map = infer_window_cnn(
    model=model,
    raster_array=raster_array,
    means=scaled_means,
    stds=scaled_stds,
    device=DEVICE,
    tile=512,
    stride=256,
    num_classes=NUM_CLASSES,
    batch_size=4
)
print("✅ y_pred_cnn shape:", y_pred_cnn.shape)

# ---------------------------
# Visualización + exportación SHP
# ---------------------------
out_dir = os.path.join(OUTPUT_DIR, "pred_cnn")
os.makedirs(out_dir, exist_ok=True)
shp_out = os.path.join(out_dir, "clasificacion_cnn_window.shp")

visualizar_cnn_y_exportar(
    y_pred=y_pred_cnn,
    raster_array=raster_array,
    clase_dict={"Permeable": 1, "No permeable": 0},
    transform=transform_win,
    crs=crs,
    shp_out=shp_out,     # pon None si no quieres shapefile
    min_area=2.0         # m² si CRS métrico
)

print(f"✅ CNN: Shapefile exportado en: {shp_out}")

# ========================================================================
# INFERENCIA CNN (U-Net++ resnet152) SOBRE VENTANA RÁSTER + EXPORTACIÓN SHP
# ACTUALIZADO: invert_classes opcional
# ========================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import geopandas as gpd
from shapely.geometry import shape
import rasterio
from rasterio.windows import Window
from rasterio.features import shapes
import torch
import segmentation_models_pytorch as smp

# ---------------------------
# Directorios (según tu estructura)
# ---------------------------
BASE_DIR = r'C:\Users\ronny\OneDrive\Documentos\MAESTRIA\1. CONTENIDO\16. TFM\CODIGO'
OUTPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, '../DATA/OUTPUT/CNN/'))
INPUT_MASK_DIR = os.path.abspath(os.path.join(BASE_DIR, '../DATA/OUTPUT/MASCARAS/'))
INPUT_IMG_DIR  = os.path.abspath(os.path.join(BASE_DIR, '../INSUMOS/ORTOFOTOS/'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# Hiperparámetros y checkpoint (U-Net++ ResNet152)
# ---------------------------
IN_CHANNELS = 3
NUM_CLASSES = 2
SEGMENTATION_MODEL = 'unetplusplus'
BACKBONE = 'resnet152'
CHECKPOINT_FILE = os.path.join(BASE_DIR, 'unetplusplus_resnet152_best.ckpt')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Medias y stds escaladas (0-1) usadas en tu entrenamiento
scaled_means = [0.4312, 0.4267, 0.3845]
scaled_stds  = [0.2263, 0.2065, 0.1923]

# ---------------------------
# Modelo + checkpoint
# ---------------------------
def create_segmentation_model(seg_model, backbone, in_channels, num_classes, weights=True, activation=None):
    enc_w = "imagenet" if weights else None
    if seg_model.lower() == "unetplusplus":
        return smp.UnetPlusPlus(
            encoder_name=backbone,
            encoder_weights=enc_w,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation
        )
    raise ValueError("Esta integración está preparada para 'unetplusplus'.")

def load_checkpoint(model, filename, device):
    ckpt = torch.load(filename, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"✅ Checkpoint cargado desde: {filename}")

model = create_segmentation_model(SEGMENTATION_MODEL, BACKBONE, IN_CHANNELS, NUM_CLASSES, weights=True).to(DEVICE)
if not os.path.exists(CHECKPOINT_FILE):
    raise FileNotFoundError(f"No se encontró el checkpoint: {CHECKPOINT_FILE}")
load_checkpoint(model, CHECKPOINT_FILE, DEVICE)
model.eval()
for p in model.parameters(): p.requires_grad_(False)

# ---------------------------
# Utilidades inferencia/plot/SHP con invert_classes
# ---------------------------
@torch.no_grad()
def _normalize_rgb_tensor(x_np, means, stds, device):
    x_np = x_np.astype(np.float32)
    vmax = float(max(x_np[0].max(), x_np[1].max(), x_np[2].max()))
    if vmax <= 1.0: x = torch.from_numpy(x_np)
    elif vmax <= 255.0: x = torch.from_numpy(x_np / 255.0)
    else: x = torch.from_numpy(x_np / 65535.0)
    for c in range(3): x[c] = (x[c] - means[c]) / stds[c]
    return x.unsqueeze(0).to(device)

def _hann2d(h, w, eps=1e-6):
    wy = np.hanning(h); wx = np.hanning(w)
    w2d = np.outer(wy, wx).astype(np.float32)
    w2d = (w2d - w2d.min()) / (w2d.max() - w2d.min() + eps)
    return np.clip(w2d, 1e-3, None)

@torch.no_grad()
def _process_batch_tiles(model, tiles, coords, means, stds, prob_acc, count_acc, device, num_classes):
    B, tile = len(tiles), tiles[0].shape[1]
    weight_np = _hann2d(tile, tile)
    weight = torch.from_numpy(weight_np).to(device)

    batch_t = torch.cat([_normalize_rgb_tensor(t, means, stds, device) for t in tiles], dim=0)
    probs_sum = 0
    for aug in range(4):
        x = batch_t
        if aug == 1: x = torch.flip(x, dims=[-1])
        if aug == 2: x = torch.flip(x, dims=[-2])
        if aug == 3: x = torch.flip(x, dims=[-2, -1])
        logits = model(x)
        p = torch.softmax(logits, dim=1)
        if aug == 1: p = torch.flip(p, dims=[-1])
        if aug == 2: p = torch.flip(p, dims=[-2])
        if aug == 3: p = torch.flip(p, dims=[-2, -1])
        probs_sum += p
    probs = (probs_sum / 4.0) * weight[None, None, :, :]
    probs = probs.cpu().numpy()
    for i, (y0, y1, x0, x1) in enumerate(coords):
        h, w = y1 - y0, x1 - x0
        prob_acc[:, y0:y1, x0:x1] += probs[i, :, :h, :w]
        count_acc[y0:y1, x0:x1]   += weight_np[:h, :w]

@torch.no_grad()
def infer_window_cnn(
    model, raster_array, means, stds, device=DEVICE,
    tile=512, stride=256, num_classes=NUM_CLASSES, batch_size=4, invert_classes=False
):
    C, H, W = raster_array.shape
    assert C >= 3, "Se requieren al menos 3 bandas (RGB)."
    rgb = raster_array[:3].astype(np.float32)
    prob_acc = np.zeros((num_classes, H, W), dtype=np.float32)
    count_acc = np.zeros((H, W), dtype=np.float32)

    tiles, coords = [], []
    for y0 in range(0, H, stride):
        for x0 in range(0, W, stride):
            y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
            tile_np = np.zeros((3, tile, tile), dtype=np.float32)
            crop = rgb[:, y0:y1, x0:x1]
            tile_np[:, :crop.shape[1], :crop.shape[2]] = crop
            tiles.append(tile_np); coords.append((y0, y1, x0, x1))
            if len(tiles) == batch_size:
                _process_batch_tiles(model, tiles, coords, means, stds, prob_acc, count_acc, device, num_classes)
                tiles, coords = [], []
    if tiles:
        _process_batch_tiles(model, tiles, coords, means, stds, prob_acc, count_acc, device, num_classes)

    count_acc[count_acc == 0] = 1.0
    prob_map = prob_acc / count_acc[None, :, :]
    if invert_classes:
        prob_map = prob_map[::-1, ...]  # intercambia canal 0 y 1
    y_pred = prob_map.argmax(axis=0).astype(np.uint8)
    return y_pred, prob_map

def visualizar_cnn_y_exportar(
    y_pred, raster_array, clase_dict,
    transform=None, crs=None,
    shp_out=None, min_area=0.0, invert_classes=False
):
    if invert_classes:
        y_pred = 1 - y_pred  # asegura inversión también aquí si se desea

    inv_map = {v: k for k, v in clase_dict.items()}  # 0/1 -> nombre
    class_labels = [inv_map.get(1, "Permeable"), inv_map.get(0, "No permeable")]
    class_colors = {inv_map.get(1, "Permeable"): "green",
                    inv_map.get(0, "No permeable"): "lightgray"}
    custom_cmap = ListedColormap([class_colors[cl] for cl in class_labels])
    legend_patches = [mpatches.Patch(color=class_colors[cl], label=cl) for cl in class_labels]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5.6), dpi=140)
    axs[0].imshow(y_pred, cmap=custom_cmap)
    axs[0].set_title("Clasificación CNN (U-Net++)", fontsize=14, fontweight='bold', pad=8)
    axs[0].axis('off'); axs[0].legend(handles=legend_patches, loc='lower left', fontsize=12,
                                       frameon=True, facecolor='white')
    rgb_img = np.transpose(raster_array[[0, 1, 2]], (1, 2, 0)).astype(np.uint8)
    axs[1].imshow(rgb_img)
    axs[1].set_title("Imagen RGB original", fontsize=14, fontweight='bold', pad=8)
    axs[1].axis('off')
    plt.tight_layout(); plt.show()

    if shp_out is not None:
        if transform is None or crs is None:
            raise ValueError("Para exportar a SHP debes proporcionar 'transform' y 'crs'.")
        class_map = {val: key for key, val in clase_dict.items()}
        geoms, vals = [], []
        for geom, val in shapes(y_pred.astype(np.int32), mask=None, transform=transform):
            if val is None: continue
            geom_shape = shape(geom).buffer(0)
            if min_area > 0 and geom_shape.area < min_area:
                continue
            geoms.append(geom_shape); vals.append(int(val))
        gpd.GeoDataFrame(
            {"class_val": vals, "class_name": [class_map.get(v, f"Clase {v}") for v in vals]},
            geometry=geoms, crs=crs
        ).to_file(shp_out, driver="ESRI Shapefile")

# ---------------------------
# Inferencia en la MISMA ventana (y exportación SHP)
# ---------------------------
raster_name = "PNOA_MA_OF_ETRS89_HU30_h25_0559_1.tif"
raster_path = os.path.join(INPUT_IMG_DIR, raster_name)
if not os.path.exists(raster_path):
    raise FileNotFoundError(f"No se encontró el ráster: {raster_path}")

print(f"📄 Usando ráster: {raster_path}")
col_off, row_off = 15000, 15000
width, height   = 1500, 1500
window = Window(col_off, row_off, width, height)

with rasterio.open(raster_path) as src:
    print(f"{src.count} bandas | Tamaño: {src.width} x {src.height}")
    raster_array   = src.read(window=window)        # (C,H,W)
    transform_win  = src.window_transform(window)
    crs            = src.crs

print("ℹ️ raster_array shape:", raster_array.shape)

y_pred_cnn, prob_map = infer_window_cnn(
    model=model,
    raster_array=raster_array,
    means=scaled_means, stds=scaled_stds,
    device=DEVICE, tile=512, stride=256,
    num_classes=NUM_CLASSES, batch_size=4,
    invert_classes=True          
)

out_dir = os.path.join(OUTPUT_DIR, "pred_cnn_unetpp_resnet152")
os.makedirs(out_dir, exist_ok=True)
shp_out = os.path.join(out_dir, "clasificacion_cnn_window.shp")

visualizar_cnn_y_exportar(
    y_pred=y_pred_cnn,
    raster_array=raster_array,
    clase_dict={"Permeable": 1, "No permeable": 0},
    transform=transform_win, crs=crs,
    shp_out=shp_out, min_area=2.0,
    invert_classes=False          
)

print(f"✅ CNN (U-Net++): Shapefile exportado en: {shp_out}")
