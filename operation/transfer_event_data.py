import argparse
from pathlib import Path
from astropy.time import Time
import solar_realtime_pipeline as sp

# Define the frequency bands only once if they do not change
BANDS = [
    '23MHz', '27MHz', '32MHz', '36MHz', '41MHz', 
    '46MHz', '50MHz', '55MHz', '59MHz', '64MHz', 
    '69MHz', '73MHz', '78MHz', '82MHz'
]

def parse_arguments():
    """
    Parses command-line arguments for downloading LWA data.
    """
    parser = argparse.ArgumentParser(description='Download LWA data')
    parser.add_argument('t_start', type=str, help='Start time in ISO format (e.g., 2024-02-04T21:00)')
    parser.add_argument('t_end', type=str, help='End time in ISO format (e.g., 2024-02-04T21:20)')
    parser.add_argument('--destdir', type=str, default='/nas6/ovro-lwa-data/',
                        help='Destination directory for the downloaded data')
    parser.add_argument('--interval', type=str, default='10s',
                        help='Download interval (e.g., 10s, 1min, 10min)')
    
    return parser.parse_args()

def download_data(t_start, t_end, destdir='/nas6/ovro-lwa-data/', interval='10s'):
    """
    Download LWA data for the given time range.
    """
    # Format the date string for the destination path
    datestr = t_start.split('T')[0].replace('-', '')
    
    destination_path = destdir + datestr + '/'
    Path(destination_path).mkdir(parents=True, exist_ok=True)  # Ensure the destination directory exists
    
    sp.download_timerange(Time(t_start), Time(t_end), 
                          file_path='slow', download_interval=interval, destination=destination_path, 
                          server='calim7', maxthread=5, bands=BANDS)

if __name__ == "__main__":

    """
    Example usage:
    python transfer_event_data.py 2024-02-04T21:00 2024-02-04T21:20

    need to add to the path
    export PYTHONPATH=$PYTHONPATH:/data1/pzhang/ovro-lwa-solar/operation/
    export PYTHONPATH=$PYTHONPATH:/data1/pzhang/ovro-lwa-solar/
    """

    args = parse_arguments()
    download_data(args.t_start, args.t_end, args.destdir, args.interval)
