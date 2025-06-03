import pyhdf
from pyhdf.SD import SD, SDC

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import sys 
import torch
from matplotlib import colormaps

sys.path.append('satvision-toa')

from threedcloud_svtoa import ABITransform, reverse_transform

def load_abi(abi_path):
    """Create a dict with ABI attributes used for creating chips."""
    ds = xr.open_dataset(abi_path)
    
    BOUND_SIZE = 1600
    LENGTH = 10848
    
    abi_dict = {
        'BOUND_SIZE': BOUND_SIZE, 
        'LENGTH': LENGTH,
        'abiLong': ds['Longitude'].to_numpy(),
        'abiLat': ds['Latitude'].to_numpy(),
    }
    
    # bounds of ABI lat/lon   
    abiLongB = abi_dict['abiLong'][
            BOUND_SIZE:LENGTH-BOUND_SIZE, 
            BOUND_SIZE:LENGTH-BOUND_SIZE]
    abiLatB = abi_dict['abiLat'][
            BOUND_SIZE:LENGTH-BOUND_SIZE, 
            BOUND_SIZE:LENGTH-BOUND_SIZE]
    
    # adjust NODATA values
    abiLongB[abiLongB == -999] = 10
    abiLatB[abiLatB == -999] = 10
    
    # update dict with updated bound arrays 
    abi_dict['abiLongB'] = abiLongB
    abi_dict['abiLatB'] = abiLatB
    
    # min/max values
    abi_dict['longMin'] = abiLongB.min()
    abi_dict['longMax'] = abiLongB.max()
    abi_dict['latMin'] = abiLatB.min()
    abi_dict['latMax'] = abiLatB.max()
    
    # create lat/lon slices in ABI data
    latSlice = abi_dict['abiLat'][:, 5424]
    latSlice = latSlice[18:-18]
    latSlice = latSlice[::-1]
    abi_dict['latSlice'] = latSlice
    
    longSlice = abi_dict['abiLong'][5424, :]
    longSlice = longSlice[18:-18]
    abi_dict['longSlice'] = longSlice
    
    return abi_dict


def gather_files(YYYY, DDD, HH, ROOT):
    """Glob-like file gathering in ROOT directory."""
    ABI_ = {
        "ROOT_PATH": None,

        "YYYY": None,
        "DDD": None,
        "HH": None,

        "00": [],
        "10": [],
        "15": [],
        "20": [],
        "30": [],
        "40": [],
        "45": [],
        "50": [],

        "L200": None,
        "L210": None,
        "L220": None,
        "L230": None,
        "L240": None,
        "L250": None,

        "everyten": False,
    }

    _ABI_PATH_ = ROOT + YYYY + "/" + DDD + "/" + HH
    
    # Find in folder
    for filename in os.listdir(_ABI_PATH_):
        if ABI_["ROOT_PATH"] == None:
            ABI_["ROOT_PATH"] = _ABI_PATH_
            ABI_["YYYY"] = filename[27:31]
            ABI_["DDD"] = filename[31:34]
            ABI_["HH"] = filename[34:36]
        MM = filename[36:38]
        if MM == "10":
            ABI_["everyten"] = True
        ABI_[f"{MM}"].append(filename)
    
    return ABI_


def get_L1B_L2(abipaths, l2path, YYYY, DDD, HH, ROOT):
    """Get Level-1B data from paths and datetime."""
    
    if len(abipaths) != 16:
        raise ImportError("This hour is bad")

    # Load each ABI channel image
    CHANNELS = []

    print("Loading Data")
    
    for file in abipaths:
        ds = xr.open_dataset(
            (ROOT + "/" + YYYY + "/" + DDD + "/" + HH + "/" + file))
        L1B = ds["Rad"].to_numpy()
        CHANNEL = int(file[19:21])
        CHANNELS.append((L1B, CHANNEL))

    # Sort channels
    CHANNELS.sort(key=lambda x: x[1])
    CHANNELS = [C[0] for C in CHANNELS]

    T = []
    
    # Resize all channels to have same shape
    for C in CHANNELS:
        S = C.shape[0] // 5424
        if S == 1:
            C = np.repeat(C, 2, axis=0)
            C = np.repeat(C, 2, axis=1)
        if S == 4:
            C = C[::2, ::2]
        T.append(C)

    CHANNELS = T

    # Create single image with all channels
    ABI = np.stack(CHANNELS, axis=2)

    return ABI


def create_chip(abi_dict, t, yy, ddn, lat, lon, ABI_ROOT):
    """Create a chip from given datetime and lat, lon input, 
    from ABI data. 
    """
    # Assert times are within min/max    
    if np.floor(t) < 12:
        raise ValueError("Times must be between 12-23")
        
    # Assert given lat/lon pair is within min/max    
    latMin, latMax = abi_dict['latMin'], abi_dict['latMax']
    longMin, longMax = abi_dict['longMin'], abi_dict['longMax']
    
    if lat < latMin or lat > latMax or lon < longMin or lon > longMax:
        raise ValueError(
            "Latitude and Longitude are too large or small.")
    
    # Search 1000x1000 area for best chip    
    AREA_SIZE = 1000
    
    # Retrieve our slice of ABI data from dict     
    latSlice, longSlice = abi_dict['latSlice'], abi_dict['longSlice']

    # Indices of lat/lon pair in ABI data
    lati = len(latSlice) - np.searchsorted(latSlice, lat) + 17
    loni = np.searchsorted(longSlice, lon) + 18
    
    # Retrieve ABI lat/lon values array     
    abiLat, abiLong = abi_dict['abiLat'], abi_dict['abiLong']
    
    # Calculate distance from ABI lat/lon vals from input    
    distances = np.abs(
            abiLat[lati-AREA_SIZE:lati+AREA_SIZE, 
                   loni-AREA_SIZE:loni+AREA_SIZE] - lat) + np.abs(
            abiLong[lati-AREA_SIZE:lati+AREA_SIZE, 
                    loni-AREA_SIZE:loni+AREA_SIZE] - lon)
    # Coords are min distance ABI lat/lon vals     
    coords = np.array(np.unravel_index(distances.argmin(), distances.shape))
    
    # Special case where coords are exactly twice area size - 1     
    if (coords[0] == 0 
            or coords[1] == 0 
            or coords[1] == 2*AREA_SIZE - 1 
            or coords[0] == 2*AREA_SIZE - 1):
        print("FALLBACK")
        distances = np.abs(abiLat - lat) + np.abs(abiLong - lon)
        coords = np.unravel_index(distances.argmin(), distances.shape)
    else: # general case where we start somewhere in middle
        coords[0] += lati - AREA_SIZE
        coords[1] += loni - AREA_SIZE
        
    # Retrieve our boundary constants from dict         
    BOUND_SIZE, LENGTH = abi_dict['BOUND_SIZE'], abi_dict['LENGTH']
    
    # Second bounding check for lat/lon (based on distances arr above)    
    if (coords[0] < BOUND_SIZE 
            or coords[1] < BOUND_SIZE 
            or coords[1] > LENGTH-BOUND_SIZE 
            or coords[0] > LENGTH-BOUND_SIZE):
        print(f'coords: {coords}')
        print(f'bound_size={BOUND_SIZE}')
        print(f'len-bound_size={LENGTH-BOUND_SIZE}')
        raise ValueError(
            "Generated chip coordinates are too close to the boundary.")

    hour = np.floor(t).astype(int)

    # Gather ABI file
    DATA = gather_files(str(yy), str(ddn), str(hour), ABI_ROOT)

    # Find the closest minute
    if DATA["everyten"]:
        minutes = np.round((t - np.floor(t)) * 6).astype(int) * 10
    else:
        minutes = np.round((t - np.floor(t)) * 4).astype(int) * 15

    # Shift hour/minute values, except if it would change the day
    if minutes == 60:
        if hour != 23:
            hour += 1
            minutes = 0
        else:
            if DATA["everyten"]:
                minutes = 50
            else:
                minutes = 45

    # Process minutes string            
    minutes = str(minutes)
    if minutes == "0":
        minutes = "00"

    # Collect ABI data from file    
    ABI = get_L1B_L2(
        DATA[minutes], DATA["L200"], DATA["YYYY"], DATA["DDD"], DATA["HH"], 
        ABI_ROOT)

    # Create 128x128 chip to return
    chip = ABI[coords[0]-64:coords[0]+64, coords[1]-64:coords[1]+64, :]
    
    # Rearrange chip's bands
    translation = [1, 2, 0, 4, 5, 6, 3, 8, 9, 10, 11, 13, 14, 15]
    chip = chip[..., translation]
    
    transform = ABITransform(img_size=128) 
    chip = transform(chip)
    chip = np.expand_dims(chip, axis=0)

    # Chip needs to be a tensor on cuda device for model inference
    chip = torch.from_numpy(chip).cuda();
    
    return chip, coords


def generate_transect_latlon(lat, lon):
    lat_list = generate_transect_single_coord(
        lat, 'lat')
    lon_list = generate_transect_single_coord(
        lon, 'lon')
    
    return lat_list, lon_list


def generate_transect_single_coord(coord, coord_type):
    """From user lat, lon pair, generate 91 transect lat, lon pairs."""
    coord_list = np.zeros(91)
    coord_list[45] = coord
    
    DELTA_LAT = -0.009673
    DELTA_LON = 0.002268
    
    if (coord_type == 'lat'):
        delta = DELTA_LAT
    else:
        delta = DELTA_LON
    
    for i in range(45):
        lower_val = coord - (delta * (45 - i))
        upper_val = coord + (delta * (i + 1))
        coord_list[i] = lower_val
        coord_list[i+46] = upper_val
        
    return coord_list


def pb_minmax_norm(img):
    """Normalize an image using per-band min/max."""
    normalized = np.zeros_like(img, dtype=float)

    for i in range(3):
        band = img[:,:,i]
        min_val = band.min()
        max_val = band.max()
        normalized[:,:,i] = (band - min_val) / (max_val - min_val)

    return normalized


def process_chip_viz(chip):
    """Process chip for visualization as an RGB image."""
    
    chip = chip.cpu().numpy().squeeze()

    image = reverse_transform(chip)
    
    red_coi = 0.9 
    green_coi = 0.45
    blue_coi = 0.65
    rgb_index = [1, 2, 0]

    rgb_image = np.stack((image[rgb_index[0], :, :]*red_coi,
                                image[rgb_index[1], :, :]*green_coi,
                                image[rgb_index[2], :, :]*blue_coi),
                                axis=-1)
    rgb_image = pb_minmax_norm(rgb_image*1.1)
    
    return rgb_image


def process_pred_viz(pred):
    """Process our model prediction (cloud mask) for visualization."""
    
    pred_binary = (pred > 0.5).float()
    pred_binary = pred_binary.cpu().numpy()
    pred_binary = np.rot90(pred_binary, k=1, axes=(2, 3)).squeeze()
    
    return pred_binary


def plot_rgb_chip_and_mask(chip, pred, lat, lon):
    """Input image, model prediction to create a side-by-side visualization."""
    
    rgb_image = process_chip_viz(chip)
    pred_binary = process_pred_viz(pred)
    
    num_predictions = 1
    
    fig, axes = plt.subplots(
        num_predictions, 2, figsize=(20, 6 * num_predictions), dpi=300)

    # Plot ABI input image as base layer
    axes[0].imshow(rgb_image)

    # Values to plot for transect
    #   x_values: 100 points, transect is midway across the 128x128 chip
    #   y_values: bottom to top of 128x128 chip, also 100 values
    x_values = np.linspace(128/2+9, 128/2-9, 100)  
    y_values = np.linspace(0, 128-1, 100)

    # Transect colormap
    cmap = colormaps['cool']
    colors = cmap(np.linspace(0, 1, len(x_values)))  # Map positions along the line to the colormap

    # Plot the transect with gradent color values
    for j in range(len(x_values)-1):
        axes[0].plot(
            [x_values[j], x_values[j+1]], 
            [y_values[j], y_values[j+1]], 
            color=colors[j], lw=2)

    axes[0].set_title(f'ABI image chip (channels [1, 2, 3])')
    axes[0].axis('on')

    # Plot model cloud mask prediction
    axes[1].matshow(pred_binary, cmap='viridis')  # First channel contains the binary mask
    axes[1].set_title(f'Predicted Mask')
    axes[1].axis('on')
    axes[1].invert_yaxis()
    axes[1].set_ylabel('Altitude (km)')
    axes[1].xaxis.set_ticks_position('bottom')
    
    # Get lat, lon pairs along the transect for plot legend
    lats, lons = generate_transect_latlon(lat, lon)
    
    # Calculate lat, lon pairs along 9 ticks
    num_ticks = 9
    idx = np.linspace(0, 91 - 1, num_ticks, dtype=int)
    idx_1 = idx[1:]
    fa = [f"Lat {lats[0]:.2f}\nLon {lons[0]:.2f}"]
    fa2 = np.vectorize(lambda x, y: f"{x:.2f}\n{y:.2f}")(lats[idx_1], lons[idx_1])
    fa = fa + list(fa2)
    
    # Add ticks to axis
    axes[1].set_xticks(idx)               # Set tick positions
    axes[1].set_xticklabels(fa, fontsize=8)
    axes[1].xaxis.set_ticks_position('bottom')  # Ensures ticks are at the bottom
    axes[1].xaxis.set_label_position('bottom')

    plt.tight_layout()
    plt.show()
    