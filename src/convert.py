# https://www.lvisdataset.org/dataset

import glob
import os
import shutil
from collections import defaultdict
from urllib.parse import unquote, urlparse

import numpy as np
import supervisely as sly
from dataset_tools.convert import unpack_if_archive
from dotenv import load_dotenv
from supervisely.io.fs import (
    dir_exists,
    file_exists,
    get_file_ext,
    get_file_name,
    get_file_name_with_ext,
    get_file_size,
)
from supervisely.io.json import load_json_file
from tqdm import tqdm

import src.settings as s


def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:        
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path
    
def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count
    
def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:

    # project_name = "LVIS"
    train_images_path = "/home/grokhi/rawdata/lvis/train2017"
    val_images_path = "/home/grokhi/rawdata/lvis/val2017"
    test_images_path = "/home/grokhi/rawdata/lvis/test2017"
    train_json_path = "/home/grokhi/rawdata/lvis/lvis_v1_train.json"
    val_json_path = "/home/grokhi/rawdata/lvis/lvis_v1_val.json"
    test_challenge = "/home/grokhi/rawdata/lvis/lvis_v1_image_info_test_challenge.json"
    test_dev = "/home/grokhi/rawdata/lvis/lvis_v1_image_info_test_dev.json"
    batch_size = 30


    def create_ann(image_path):
        labels = []
        tags = []

        image_name = get_file_name_with_ext(image_path)
        image_shape = image_name_to_shape.get(image_name)
        if image_shape is None:
            image_np = sly.imaging.image.read(image_path)[:, :, 0]
            img_height = image_np.shape[0]
            img_wight = image_np.shape[1]
        else:
            img_height = image_shape[0]
            img_wight = image_shape[1]

        ann_data = image_name_to_ann_data[image_name]
        for curr_ann_data in ann_data:
            category_id = curr_ann_data[0]
            synset_value, def_value = category_to_synset_def[category_id]
            synset = sly.Tag(tag_synset, value=synset_value)
            dev = sly.Tag(tag_def, value=def_value)
            polygons_coords = curr_ann_data[1]
            for coords in polygons_coords:
                exterior = []
                for i in range(0, len(coords), 2):
                    exterior.append([int(coords[i + 1]), int(coords[i])])
                if len(exterior) < 3:
                    continue
                poligon = sly.Polygon(exterior)
                label_poly = sly.Label(poligon, idx_to_obj_class[category_id], tags=[synset, dev])
                labels.append(label_poly)

            bbox_coord = curr_ann_data[2]
            rectangle = sly.Rectangle(
                top=int(bbox_coord[1]),
                left=int(bbox_coord[0]),
                bottom=int(bbox_coord[1] + bbox_coord[3]),
                right=int(bbox_coord[0] + bbox_coord[2]),
            )
            label_rectangle = sly.Label(rectangle, idx_to_obj_class[category_id], tags=[synset, dev])
            labels.append(label_rectangle)

        exhaustive_value_list = image_to_exhaustive[image_name]
        if len(exhaustive_value_list) != 0:
            exhaustive_value = "".join(str(x) for x in exhaustive_value_list)
            exhaustive = sly.Tag(tag_exhaustive, value=exhaustive_value)
            tags.append(exhaustive)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=tags)


    tag_synset = sly.TagMeta("synset", sly.TagValueType.ANY_STRING)
    tag_def = sly.TagMeta("def", sly.TagValueType.ANY_STRING)
    tag_exhaustive = sly.TagMeta("not_exhaustive_category_ids", sly.TagValueType.ANY_STRING)

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(tag_metas=[tag_synset, tag_def, tag_exhaustive])
    api.project.update_meta(project.id, meta.to_json())

    ds_to_data = {
        "val2017": (val_images_path, val_json_path),
        "train2017": (train_images_path, train_json_path),
    }

    test_challenge_data_names = []
    test_challenge_data = load_json_file(test_challenge)
    for curr_image_info in test_challenge_data["images"]:
        image_name = curr_image_info["coco_url"].split("/")[-1]
        test_challenge_data_names.append(image_name)

    test_dev_data_names = []
    test_dev_data = load_json_file(test_dev)
    for curr_image_info in test_dev_data["images"]:
        image_name = curr_image_info["coco_url"].split("/")[-1]
        test_dev_data_names.append(image_name)


    test_ds_to_data = {"test challenge": test_challenge_data_names, "test dev": test_dev_data_names}

    idx_to_obj_class = {}

    for folder_name in ["train2017", "val2017"]:
        ds_name = "training set" if folder_name in "train2017" else "validation set"
        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)
        images_path = ds_to_data[folder_name][0]
        ann_path = ds_to_data[folder_name][1]
        image_id_to_name = {}
        image_name_to_ann_data = defaultdict(list)
        image_name_to_shape = {}
        category_to_synset_def = {}
        image_to_exhaustive = {}

        ann = load_json_file(ann_path)

        for curr_category in ann["categories"]:
            category_to_synset_def[curr_category["id"]] = (
                curr_category["synset"],
                curr_category["def"],
            )
            if idx_to_obj_class.get(curr_category["id"]) is None:
                obj_class = sly.ObjClass(curr_category["name"], sly.AnyGeometry)
                meta = meta.add_obj_class(obj_class)
                idx_to_obj_class[curr_category["id"]] = obj_class
        api.project.update_meta(project.id, meta.to_json())

        for curr_image_info in ann["images"]:
            image_name = curr_image_info["coco_url"].split("/")[-1]
            image_id_to_name[curr_image_info["id"]] = image_name
            image_name_to_shape[image_name] = (curr_image_info["height"], curr_image_info["width"])
            image_to_exhaustive[image_name] = curr_image_info["not_exhaustive_category_ids"]

        for curr_ann_data in ann["annotations"]:
            image_id = curr_ann_data["image_id"]
            image_name_to_ann_data[image_id_to_name[image_id]].append(
                [curr_ann_data["category_id"], curr_ann_data["segmentation"], curr_ann_data["bbox"]]
            )

        images_names = list(image_to_exhaustive.keys())

        progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

        for img_names_batch in sly.batched(images_names, batch_size=batch_size):
            images_pathes_batch = [
                os.path.join(images_path, image_name) for image_name in img_names_batch
            ]

            if folder_name == "val2017":
                for idx, im_path in enumerate(images_pathes_batch):
                    if file_exists(im_path):
                        continue
                    else:
                        im_path = os.path.join(train_images_path, get_file_name_with_ext(im_path))
                        images_pathes_batch[idx] = im_path

            img_infos = api.image.upload_paths(dataset.id, img_names_batch, images_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]

            anns_batch = [create_ann(image_path) for image_path in images_pathes_batch]
            api.annotation.upload_anns(img_ids, anns_batch)

            progress.iters_done_report(len(img_names_batch))


    for folder_name, images_names in test_ds_to_data.items():
        dataset = api.dataset.create(project.id, folder_name, change_name_if_conflict=True)

        progress = sly.Progress("Create dataset {}".format(folder_name), len(images_names))

        for img_names_batch in sly.batched(images_names, batch_size=batch_size):
            images_pathes_batch = [
                os.path.join(test_images_path, image_path) for image_path in img_names_batch
            ]

            img_infos = api.image.upload_paths(dataset.id, img_names_batch, images_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]

            progress.iters_done_report(len(img_names_batch))
    return project


