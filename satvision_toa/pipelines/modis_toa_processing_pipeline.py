import os
import sys
import tqdm
import dask
import satpy
import pathlib
import logging
import xarray as xr
import rioxarray as rxr
import multiprocessing
from osgeo import gdal
from pathlib import Path
from pyresample import geometry
from collections import ChainMap
from dask.diagnostics import ProgressBar

# trying to remove this dependency
# from core.model.SystemCommand import SystemCommand

dask.config.set(num_workers=8)
dask.config.set({"array.chunk-size": "1908MiB"})


class ModisSwathToGrid(object):

    # GDALWARP global command
    GDALWARP: str = 'gdalwarp -of GTIFF  -co ' + \
        'COMPRESS=LZW -tps -t_srs EPSG:4326'

    # GDALMERGE global command
    GDALMERGE: str = 'gdal_merge.py  -co COMPRESS=LZW -o'

    # READER global command
    READER: str = 'modis_l1b'

    # CACHE for day postprocessing
    CACHE_DAY_POST: str = '_day_filtered'

    def __init__(
                self,
                data_path: str = None,
                data_regex: str = None,
                output_dir: str = '.',
                bands: list = [
                    '1', '2', '3', '6', '7', '21',
                    '26', '27', '28', '29', '30',
                    '31', '32', '33'
                ],
                debug: bool = False,
                logger: str = None
            ) -> None:

        # set data filenames based on the input provided
        if data_path is not None:
            self.data_filenames = ddddddd
        elif data_regex is not None:
            self.data_filenames = XXXX
        else:
            sys.exit('Need to specify one of -d or -r for data selection.')

        # set data path and output directory
        self.data_path = data_path
        self.output_dir = output_dir

        # intermediate filename to save filtered output paths
        # self.filtered_output_filename = \
        #    self.outputDir / f"{str(dataPathName)}{self.CACHE_DAY_POST}"

        """
        # get the data path name
        dataPathName = self.dataPath.stem


        # set band ids
        self.bandIDs = bands

        # directory to store swaths
        self.swathDir = self.outputDir / '1-swaths'
        self.swathDir.mkdir(exist_ok=True)

        # directory to store mosaics
        self.mosaicDir = self.outputDir / '2-mosaic'
        self.mosaicDir.mkdir(exist_ok=True)

        # directory to store cache data
        self.cacheDir = self.outputDir / '3-cache'
        self.cacheDir.mkdir(exist_ok=True)

        # debug mode setup
        self.debug = debug

        # force writing of files
        self.force = False

        # setup logger
        self.logger = logger
        """

    # --------------------------------------------------------------------------
    # mosaic
    #
    # mosaic main process
    # --------------------------------------------------------------------------
    def mosaic(self) -> None:

        # output filename to store final mosaic
        output_filename = str(
            self.mosaicDir / f"{str(self.dataPath.stem)}-global.tif")
        self.logger.info(f'Selected output path: {output_filename}')

        # skip processing if file already exists
        if os.path.exists(output_filename) and not self.force:
            self.logger.info(f'Skipping {output_filename}, already exists.')
            return

        """
        # set default state before proceeding with processing
        filteredPaths = None

        self.logger.info(f'Intermediate filtered file {self.filteredOutPath}')

        # if the day filtered path does not exist
        if not self.filteredOutPath.exists():

            # return list of data paths in PosixPath type
            dataPaths = self._read_data_paths(self.dataPath)

            # dict with metadata from filename list
            self.logger.info(
                f'Processing metadata for {len(dataPaths)} files.')
            metaDataDict = self._getMetaData(dataPaths)

            # filter day night and view angle
            filteredPaths = self._filterDataPaths(metaDataDict)
            assert len(filteredPaths) > 0, \
                'No data files after filtering. ' + \
                'Consider adding more files or reducing the filters.'

            # make sure file path values are strings
            filteredPaths = list(map(str, filteredPaths))

            # save cache file with text files
            self._cachePathStrs(filteredPaths)
            self.logger.info(
                f'Saved {self.filteredOutPath} to cache files list.')

        # if we were able to cache files, proceed to read data and mosaic
        if filteredPaths is None:
            filteredPaths = self._read_data_paths(self.filteredOutPath)
            filteredPaths = list(map(str, filteredPaths))

        self.logger.info(
            f'Initializing satpy with {len(filteredPaths)} files.')

        # initialize satpy
        modisScene = satpy.Scene(
            filenames=filteredPaths, reader=self.READER)

        # filter_parameters
        # reader_kwargs={'mask_saturated': False}

        # too much output for now
        self.logger.info(
            modisScene.available_dataset_names(reader_name=self.READER))

        # too much output for now
        # self.logger.info(
        #    modisScene.available_dataset_ids(reader_name=self.READER))

        # load selected scenes
        modisScene.load(self.bandIDs)#, modifier=None)#, generate=False)

        self.logger.info(
            f'Resampling data files for {len(self.bandIDs)} bands.')

        # resample imagery
        resampledModisScene = modisScene.resample(
            self._getTargetArea(),
            #cache_dir=str(self.cacheDir),
            datasets=self.bandIDs,
            resampler='ewa'#'bucket_avg'#'nearest'#'bilinear'#'bucket_avg'#nearest'#ewa',#,#'nearest'  #'ewa',
        )
        self.logger.info('Done with resampling step.')

        # resampledModisScene.persist()

        # this step is a bit slower
        #with ProgressBar():
        #    resampledModisScene.save_datasets(
        #        base_dir=str(self.mosaicDir),
        #        filename="{name}.tif",
        #        datasets=self.bandIDs,
        #        dtype=np.float32,
        #        enhance=False
        #    )

        # this step is a bit faster
        with ProgressBar():
            resampledModisScene.save_datasets(
                filename=str(Path(output_filename).with_suffix('.nc')),
                datasets=self.bandIDs,
            )
        self.logger.info('Done with saving step.')

        # output bands, for now only the first 3
        output_bands = [f'CHANNEL_{i}' for i in self.bandIDs]
        rr = rxr.open_rasterio(str(Path(output_filename).with_suffix('.nc')))
        rr[output_bands].isel(band=0).rio.to_raster(
            output_filename, compress='LZW')

        # print(modisScene.all_dataset_ids())
        # print(modisScene.all_dataset_names())
        # print(resampledModisScene.all_dataset_ids())
        # print(resampledModisScene.all_dataset_names())

        # x.isel(band=0).rio.to_raster(
        # 'test_MOD021KM_2020_182-global-v2.tif',
        # compress='LZW')
        # x[['CHANNEL_1', 'CHANNEL_2', 'CHANNEL_3', 'CHANNEL_4']]

        self.logger.info(f"Done processing, saved {output_filename} file.")
        """
        return

    # --------------------------------------------------------------------------
    # _read_data_paths
    #
    # reads data paths
    # --------------------------------------------------------------------------
    def _read_data_paths(self, data_path: str, regex: bool = False) -> list:

        if not data_path.exists():
            raise FileNotFoundError(data_path)

        """
        with open(dataPath, 'r') as fh:

            dataPaths = []

            for line in list(fh.readlines()):

                dataFilename = pathlib.Path(line.strip())

                if not dataFilename.exists() and \
                        'HDF' not in str(dataFilename):

                    raise FileNotFoundError(dataFilename)

                dataPaths.append(dataFilename)

        return dataPaths
        """
    # --------------------------------------------------------------------------
    # _getMetaData
    #
    # get metadata from each path
    # --------------------------------------------------------------------------
    def _getMetaData(self, dataPaths: list) -> dict:
        metaDataDict = {}
        pool = multiprocessing.Pool(
            processes=multiprocessing.cpu_count() * 10)
        metaDataDict = pool.map(
            self._getMetaDataSubProcess, list(map(str, dataPaths)))
        return dict(ChainMap(*metaDataDict))

    # --------------------------------------------------------------------------
    # _getMetaDataField
    #
    # get single field from metadata, return None if not available
    # --------------------------------------------------------------------------
    def _getMetaDataField(
                self,
                dataPath: str,
                dataset: gdal.Dataset,
                dataField: str
            ) -> float:

        try:
            fieldValue = dataset.GetMetadata()[dataField]
            # print(dataset.GetMetadata())
        except KeyError:
            msg = f'{dataPath} did not have the metadata' + \
                f' key: {dataField}.'
            self.logger.warning(msg)
            fieldValue = None

        return fieldValue

    # --------------------------------------------------------------------------
    # _getMetaDataSubProcess
    #
    # get metadata from each path
    # --------------------------------------------------------------------------
    def _getMetaDataSubProcess(self, dataPath: str) -> dict:

        # nested dictionary for data path metadata
        metaDataDict = dict()

        # dictionary for metadata for each field
        fieldDataDict = dict()

        # subprocess for single data file processing
        dataSetBand1 = gdal.Open(str(dataPath))

        # get day night metadata
        fieldDataDict['DAYNIGHTFLAG'] = self._getMetaDataField(
            dataPath, dataSetBand1, 'DAYNIGHTFLAG')

        # get dummy view angle metadata
        # TODO: change to real value
        with xr.open_dataset(dataPath, engine='netcdf4') as file_hdf:
            fieldDataDict['SolarZenith'] = file_hdf['SolarZenith'].values.max()

        # fill metadata dict with nested dict
        metaDataDict[dataPath] = fieldDataDict

        # delete dataset band
        del dataSetBand1

        return metaDataDict

    # --------------------------------------------------------------------------
    # _filterDataPaths
    #
    # filter by specific arguments
    # --------------------------------------------------------------------------
    def _filterDataPaths(self, metadataDict: dict) -> list:
        dataPaths = []
        for key in tqdm.tqdm(metadataDict.keys()):

            # add AND to filter for zenith angle
            # Greenland - a lot of Lower solar zenith angles,
            # Greenland as an example (solar zenith greater than 72 no good)
            if metadataDict[key]['DAYNIGHTFLAG'] == "Day":# and \
                #        metadataDict[key]['SolarZenith'] <= 72:#72:
                dataPaths.append(key)
        return dataPaths

    # --------------------------------------------------------------------------
    # _cachePathStrs
    #
    # save cache file with filenames
    # --------------------------------------------------------------------------
    def _cachePathStrs(self, filteredPaths: list):
        filteredPathStrsNewLine = [dp + '\n' for dp in filteredPaths]
        with open(self.filteredOutPath, 'w') as fh:
            fh.writelines(filteredPathStrsNewLine)
        return

    # --------------------------------------------------------------------------
    # _getTargetArea
    #
    #
    # --------------------------------------------------------------------------
    def _getTargetArea(self):
        target_extent = (-180.0, -90.0, 180.0, 90.0)  # Global lat/lon extent
        # 1 km resolution (0.01 degrees)
        target_resolution = 0.0174532925199433
        target_rows = int(
            (target_extent[3] - target_extent[1]) / target_resolution)
        target_cols = int(
            (target_extent[2] - target_extent[0]) / target_resolution)
        target_area = geometry.AreaDefinition(
            "WGS84", "Custom", "Global WGS84 Grid", {
                'proj': 'latlong',
                'lon_0': 0,
                'ellps': 'WGS84',
                'datum': 'WGS84',
                'lat_ts': 0,
                'a': 6378137,
                'b': 6356752.3142
            }, target_cols, target_rows, target_extent)
        return target_area

    # --------------------------------------------------------------------------
    # _orthoOne
    #
    # run -> processStrips -> runOneStrip -> stripToToa -> orthoOne
    # --------------------------------------------------------------------------
    def _geolocateData(self, dataPath):

        cmd = f'{self.GDALWARP} {str(dataPath)}'

        output_name = str(dataPath).split('"')[1]\
            .split('/')[-1].replace('hdf', 'EV_1KM_RefSB.tif')

        output_path = self.swathDir / output_name

        self.logger.info(output_path)

        if output_path.exists():
            return output_path

        cmd = f'{cmd} {str(output_path)}'

        SystemCommand(cmd, logger=self.logger)

        return output_path

    # --------------------------------------------------------------------------
    # _mergeGeolocatedData
    #
    # run -> processStrips -> runOneStrip -> stripToToa -> orthoOne
    # --------------------------------------------------------------------------
    def _mergeGeolocatedData(self, geolocatedPaths):

        outputName = str(geolocatedPaths[0]).split(
            '/')[-1].replace('tif', 'mosaic.tif')

        outputPath = self.mosaicDir / outputName

        self.logger.info(outputPath)

        geolocatedPathsStr = ' '.join(
            [str(geolocatedPath) for geolocatedPath in geolocatedPaths])

        self.logger.info(geolocatedPathsStr)

        cmd = f'{self.GDALMERGE} {str(outputPath)} {geolocatedPathsStr}'

        SystemCommand(cmd, logger=self.logger)

        return
