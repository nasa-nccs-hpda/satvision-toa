import os
import sys
import logging
import pathlib
import argparse
from datetime import datetime
from satvision_toa.pipelines.modis_toa_processing_pipeline import \
    ModisSwathToGrid


# -----------------------------------------------------------------------------
# main
#
# python
# -----------------------------------------------------------------------------
def main():

    # Process command-line args.
    desc = 'Use this application to run MODIS mosaic of swaths.'

    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-dg',
                        '--debug',
                        action='store_true',
                        dest='debug',
                        help='Show extra output and write intermediate files')

    parser.add_argument('-d',
                        '--data-filename',
                        dest='data_filename',
                        required=False,
                        help='Path to text file containing TIFs' +
                        'to merge together in output directory')

    parser.add_argument('-r',
                        '--data-regex',
                        dest='data_regex',
                        required=False,
                        help='Regex to data paths (*.hdf) instead of text files')

    parser.add_argument('-b',
                        '--bands',
                        dest='bands',
                        nargs='*',
                        required=False,
                        default=[
                            '1', '2', '3', '6', '7', '21', '26', '27', '28',
                            '29', '30', '31', '32', '33'
                        ],
                        help='List of bands to perform composite')

    parser.add_argument('-o',
                        '--output-dir',
                        dest='output_dir',
                        default='.',
                        help='Output directory')

    args = parser.parse_args()

    # Create log directory if it does not exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s; %(levelname)s; %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(formatter)
    log_filename = f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.log'
    fh = logging.FileHandler(os.path.join(args.output_dir, log_filename))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)

    # start object of SwathToGrid
    pipeline = ModisSwathToGrid(
        data_path=args.data_filename,
        output_dir=args.output_dir,
        bands=args.bands,
        debug=args.debug
    )

    # perform mosaic of imagery
    pipeline.mosaic()

    return


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
