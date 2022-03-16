
import os, random, zipfile, tarfile
import supervisely as sly
import sly_globals as g
import gdown
from supervisely.io.json import load_json_file
from supervisely.io.fs import get_file_name, get_file_ext


def prepare_ann_data(anns_folder):

    ann_json = load_json_file(os.path.join(anns_folder, g.annotations_file_name))
    ann_shape = load_json_file(os.path.join(anns_folder, g.annotations_for_shape))

    for im_data in ann_json:
        g.image_name_to_polygon[im_data['ID']] = im_data['Label']['Leaf'][0] # all len(list) == 1
        g.image_name_to_classification[im_data['ID']] = im_data['Label']['classification']

    for data in ann_shape['images']:
        image_name = data['id']
        g.image_name_to_shape[image_name] = (data['height'], data['width'])


def create_ann(im_name):
    labels = []
    img_name = get_file_name(im_name)
    img_shape = g.image_name_to_shape[img_name]
    classification = g.image_name_to_classification[img_name]
    polygon_data = g.image_name_to_polygon[img_name]

    state = polygon_data['state']
    tag_state = sly.Tag(g.tag_meta_state, value=state)
    tag_classification = sly.Tag(g.tag_meta_classification, value=classification)

    geomerty = polygon_data['geometry']

    points = []
    for coords in geomerty:
        points.append(sly.PointLocation(coords['y'], coords['x']))

    polygon = sly.Polygon(points, interior=[])

    label = sly.Label(polygon, g.obj_class, tags=sly.TagCollection([tag_state, tag_classification]))
    labels.append(label)

    return sly.Annotation(img_size=img_shape, labels=labels)


def extract_zip(archive_path):
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, 'r') as archive:
            archive.extractall(g.work_dir_path)
    else:
        g.logger.warn('Archive cannot be unpacked {}'.format(g.arch_name))
        g.my_app.stop()


def extract_tar(archive_path, out_path):
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, 'r') as archive:
            archive.extractall(out_path)
    else:
        g.logger.warn('Archive cannot be unpacked {}'.format(g.arch_name))
        g.my_app.stop()


@g.my_app.callback("import_minne_apple")
@sly.timeit
def import_minne_apple(api: sly.Api, task_id, context, state, app_logger):

    gdown.download(g.coffee_leaves_url, g.archive_path, quiet=False)
    extract_zip()

    anns_folder = os.path.join(g.work_dir_path, g.annotation_folder_name)
    prepare_ann_data(anns_folder)

    images_arh_path = os.path.join(anns_folder, g.images_arh_name)
    extract_tar(images_arh_path, anns_folder)

    images_folder = os.path.join(anns_folder, g.images_folder_name)
    images_names = [item for item in os.listdir(images_folder) if get_file_ext(item) == g.images_ext]

    new_project = api.project.create(g.WORKSPACE_ID, g.project_name, change_name_if_conflict=True)
    api.project.update_meta(new_project.id, g.meta.to_json())

    new_dataset = api.dataset.create(new_project.id, g.dataset_name, change_name_if_conflict=True)

    sample_img_names = random.sample(images_names, g.sample_percent)

    progress = sly.Progress('Upload items', len(sample_img_names), app_logger)
    for img_batch in sly.batched(sample_img_names, batch_size=g.batch_size):

        img_pathes = [os.path.join(images_folder, name) for name in img_batch]
        img_infos = api.image.upload_paths(new_dataset.id, img_batch, img_pathes)
        img_ids = [im_info.id for im_info in img_infos]

        anns = [create_ann(img_name) for img_name in img_batch]
        api.annotation.upload_anns(img_ids, anns)

        progress.iters_done_report(len(img_batch))

    g.my_app.stop()


def main():
    sly.logger.info("Script arguments", extra={
        "TEAM_ID": g.TEAM_ID,
        "WORKSPACE_ID": g.WORKSPACE_ID
    })
    g.my_app.run(initial_events=[{"command": "import_minne_apple"}])


if __name__ == '__main__':
    sly.main_wrapper("main", main)