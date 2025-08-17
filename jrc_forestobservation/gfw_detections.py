
from tools.utils import download_file
from pathlib import Path
import datetime

# Function to download data from GFW API
def download_gfw_data(gfw_extent, tile_name, output_dir="/mnt/hddarchive.nfs/amazonas_dir/Detections/gfw_raw"):

    gfw_api_url = f"https://data-api.globalforestwatch.org/dataset/gfw_integrated_alerts/latest/download/geotiff?grid=10/100000&tile_id={gfw_extent}&pixel_meaning=date_conf&x-api-key=2d60cd88-8348-4c0f-a6d5-bd9adb585a8c"
    output_path = Path(output_dir).joinpath(f"{gfw_extent}.tif")
    if not output_path.exists():
        download_file(gfw_api_url, output_path)
    return output_path

def decode_alert(alert_value_list):
    date_dict = {}
    for alert_value_list_item in alert_value_list:
        if alert_value_list_item == 0:
            continue

        days_since_base_date = int(float(str(alert_value_list_item)[1:]))

        # Calculate the actual date
        base_date = "2014-12-31"
        date = datetime.datetime.strptime(base_date, "%Y-%m-%d") + datetime.timedelta(days=days_since_base_date)
        if not date in date_dict.keys():
            date_dict[date] = [alert_value_list_item]
        else:
            date_dict[date].append(alert_value_list_item)

    return date_dict