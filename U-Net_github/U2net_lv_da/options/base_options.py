import os.path as osp
import os


class parser(object):
    def __init__(self):
        self.name = "test"  # Expriment name
        self.image_folder = "/home/ml2/Desktop/Vscode/U-Net/U2net_levain_Dabhi/annotation_dataset/training/im"  # image folder path
        self.df_path = "/home/ml2/Desktop/Vscode/U-Net/U2net_levain_Dabhi/annotation_dataset/training/mask_annotations.csv"  # label csv path
        self.distributed = False  # True for multi gpu training
        self.isTrain = True

        self.fine_width = 192 * 4
        self.fine_height = 192 * 4

        # Mean std params
        self.mean = 0.5
        self.std = 0.5

        self.batchSize = 1 # 12
        self.nThreads = 2 # 3
        # self.max_dataset_size = float("inf")
        self.max_dataset_size = 1000

        self.serial_batches = False
        self.continue_train = False
        if self.continue_train:
            self.unet_checkpoint = "prev_checkpoints/cloth_segm.pth"

        # self.save_freq = 1000
        # self.print_freq = 10
        # self.image_log_freq = 100
        #
        # self.iter = 100000
        # self.lr = 0.0002
        # self.clip_grad = 5

        self.save_freq = 1000
        self.print_freq = 1000
        self.image_log_freq = 1000

        self.iter = 100000
        self.lr = 0.0002
        self.clip_grad = 5


        self.logs_dir = osp.join("logs", self.name)
        self.save_dir = osp.join("results", self.name)
