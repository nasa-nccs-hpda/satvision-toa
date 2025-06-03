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

from threedcloud_svtoa import ABITransform

# load ABI data into a dict for later use
def load_abi(abi_path):
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