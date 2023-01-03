import os.path as op
import numpy as np
from glob import glob
import cv2

from dataloader.readers.reader_base import DatasetReaderBase, DriveManagerBase
import dataloader.data_util as tu
import utils.util_class as uc


class AnalyzeDriveManager(DriveManagerBase):
    def __init__(self, datapath, split):
        super().__init__(datapath, split)

    def list_drive_paths(self):
        return [op.join(self.datapath, "box_data")]

    def get_drive_name(self, drive_index):
        return f"drive{drive_index:02d}"


class AnalyzeReader(DatasetReaderBase):
    def __init__(self, drive_path, split, dataset_cfg):
        super().__init__(drive_path, split, dataset_cfg)

    def init_drive(self, drive_path, split):
        frame_names = glob(op.join(drive_path, "*.txt"))
        frame_names.sort()
        if split == "train":
            frame_names = frame_names[:-500]
        else:
            frame_names = frame_names[-500:]
        print("[AnalyzeReader.init_drive] # frames:", len(frame_names), "first:", frame_names[0])
        return frame_names

    def get_frame_name(self, index):
        return self.frame_names[index]

    def get_bboxes(self, index):
        """
        :return: bounding boxes in 'yxhw' format
        """
        label_file = self.frame_names[index]
        data = []
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                # ctgr_probs(k), occlusion_probs(n), objectness(1), x(1), y(1), h(1), w(1), l(1), z(1), ry(1)
                datum = np.array(line.strip("\n").split(" ")).astype(np.float32)
                if datum is not None:
                    data.append(datum)
        if not data:
            raise uc.MyExceptionToCatch("[get_data] empty data")
        data = np.array(data)
        return data

    def get_error(self, index):
        error_file = self.get_frame_name(index).replace("box_data", "error")
        data = []
        with open(error_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                # ctgr_probs(k), occlusion_probs(n), objectness(1), x(1), y(1), h(1), w(1), l(1), z(1), ry(1)
                datum = np.array(line.strip("\n").split(" ")).astype(np.float32)
                if datum is not None:
                    data.append(datum)
        if not data:
            raise uc.MyExceptionToCatch("[get_data] empty data")
        data = np.array(data)
        return data




# ==================================================
import config as cfg


def test_kitti_depth():
    print("===== start test_kitti_reader")
    dataset_cfg = cfg.Datasets.Kitti
    drive_mngr = KittiDriveManager(dataset_cfg.PATH, "train")
    drive_paths = drive_mngr.get_drive_paths()
    reader = KittiReader(drive_paths[0], "train", dataset_cfg)
    max_list = []
    for i in range(reader.num_frames()):
        image = reader.get_image(i)
        # intrinsic = reader.get_intrinsic(i)
        # extrinsic = reader.get_stereo_extrinsic(i)
        depth = reader.get_depth(i, image.shape)
        depth_view = apply_color_map(depth)
        # depth = np.repeat(depth, 3, axis=-1)
        # print("depth:", depth.shape, depth.dtype)
        # max_list.append(depth.max())
        # print(depth.max())
        # depth = depth / 80.
        # depth = 1 - depth
        # depth[depth >= 1] = 0
        # depth[depth < 0] = 0
        # depth = depth * 255
        # depth = np.asarray(depth, dtype=np.uint8)

        # valid_depth = norm_depth[np.where(norm_depth > 0)]
        # valid_depth = (1 - valid_depth) * 255
        # valid_depth = np.asarray(valid_depth, dtype=np.uint8)
        # norm_depth[norm_depth > 0] = valid_depth
        print("image:", image.shape)
        total_image = np.concatenate([image, depth_view], axis=0)
        # cv2.imshow("kitti_image", depth_view)
        cv2.imshow("kitti_depth", total_image)
        # test = op.join("/home/eagle/mun_workspace/22_paper/kitti/training/gt_depth", op.split(reader.frame_names[i])[-1])
        # test = op.join("/home/eagle/mun_workspace/22_paper/kitti/testing/gt_depth_npy", op.split(reader.frame_names[i])[-1].replace(".png", ".npy"))
        # np.save(op.join("/home/eagle/mun_workspace/22_paper/kitti/testing/gt_depth_npy", op.split(reader.frame_names[i])[-1].replace(".png", ".npy")), depth)
        # cv2.imwrite(op.join("/home/eagle/mun_workspace/22_paper/kitti/training/gt_depth_image", op.split(reader.frame_names[i])[-1]), depth_view)
        key = cv2.waitKey()
        if key == ord('q'):
            break
    # print(max(max_list))


def test_kitti_reader():
    print("===== start test_kitti_reader")
    dataset_cfg = cfg.Datasets.Kitti
    drive_mngr = KittiDriveManager(dataset_cfg.PATH, "train")
    drive_paths = drive_mngr.get_drive_paths()
    reader = KittiReader(drive_paths[0], "train", dataset_cfg)
    for i in range(reader.num_frames()):
        image = reader.get_image(i)
        bboxes2d, bboxes_3d, categories = reader.get_bboxes(i)
        print(f"frame {i}, bboxes:\n", bboxes2d)
        print(f"frame {i}, categories:\n", categories)
        # boxed_image = tu.draw_boxes(image, bboxes, dataset_cfg.CATEGORIES_TO_USE)
        # cv2.imshow("kitti", boxed_image)
        key = cv2.waitKey()
        if key == ord('q'):
            break
    print("!!! test_kitti_reader passed")


if __name__ == "__main__":
    test_kitti_reader()
