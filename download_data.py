import argparse
import glob
import json
import os
import urllib
import urllib.request

import cv2 as cv
import flickrapi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from utils.argparse import get_args_parser


def main(args):
    # TODO: add as arguments
    folder_name = args.dataset
    keyword = " ".join(args.keyword)

    data_dir = f"data/{folder_name}/images/"
    os.makedirs(data_dir, exist_ok=True)

    # ADD API KEY HERE
    flickr = flickrapi.FlickrAPI(
        "ADD HERE", "ADD HERE", cache=True, format="parsed-json"
    )

    page_number = 1
    photo_id_to_metadata = {}

    exit_loop = False
    while True:
        print("page_number", page_number)
        photos_json_response = flickr.photos.search(
            text=keyword, per_page="500", extras="date_taken,url_c", page=page_number
        )

        for i, photo in enumerate(photos_json_response["photos"]["photo"]):
            photo_id = photo.get("id")
            url = photo.get("url_c")

            img_filename = f"{photo_id}.jpg"
            img_save_path = os.path.join(data_dir, img_filename)

            try:
                data = urllib.request.urlretrieve(url, img_save_path)
                photo_id_to_metadata[photo_id] = {
                    "datetaken": photo.get("datetaken"),
                }
            except TypeError:
                pass

            if len(photo_id_to_metadata) >= 3000:
                json_object = json.dumps(photo_id_to_metadata, indent=4)
                with open(f"data/{folder_name}.json", "w") as outfile:
                    outfile.write(json_object)
                exit_loop = True
                break

        page_number += 1

        json_object = json.dumps(photo_id_to_metadata, indent=4)
        with open(f"data/{folder_name}.json", "w") as outfile:
            outfile.write(json_object)

        if exit_loop:
            break

    print("[INFO] Done downloading images.")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Download Images", parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
