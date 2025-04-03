import gc
import logging
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from comet_ml import Experiment  # Import Comet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as TF
from skimage import io
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from torch.optim.lr_scheduler import ReduceLROnPlateau
from basics import f1_mae_torch
from data_loader_cache import (GOSColorJitter, GOSNormalize, GOSRandomCrop,
                               GOSRandomHFlip, GOSResize, create_dataloaders, GOSBinaryMask,
                               get_im_gt_name_dict)
# from isnet_model.isnet_attetion import ISNetDIS, ISNetGTEncoder, muti_loss_fusion
from isnet import ISNetDIS, ISNetGTEncoder, muti_loss_fusion

experiment = Experiment(
    api_key="77k6dTdORemDU6fcjRZGx9RPM",
    project_name="props_trial",
    workspace="isnet",
)
experiment.set_name("Experiment800")
experiment.log_other("description", "This experiment is testing model accuracy with new hyperparameters.")






# Configure logging
log_file_path = "saved_models/IS-Net/Training.log"
logging.basicConfig(
    filename=log_file_path, level=logging.INFO, format="%(levelname)s:%(message)s"
)
device = "cuda" if torch.cuda.is_available() else "cpu"

now = datetime.now()
timestamp = now.strftime("%Y-%m-%d %A %H:%M:%S")
# Log the date and time
logging.info(f"Logging started on: {timestamp}")

def get_gt_encoder(
    train_dataloaders,
    train_datasets,
    valid_dataloaders,
    valid_datasets,
    hypar,
    train_dataloaders_val,
    train_datasets_val,
):



    torch.manual_seed(hypar["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(hypar["seed"])
    print("define gt encoder ...")
    logging.info("define gt encoder ...")
    net = ISNetGTEncoder()
    if hypar["gt_encoder_model"] != "":
        model_path = hypar["model_path"] + "/" + hypar["gt_encoder_model"]
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(model_path))
            net.cuda()
        else:
            net.load_state_dict(torch.load(model_path, map_location="cpu"))
        print("gt encoder restored from the saved weights ...")
        logging.info("gt encoder restored from the saved weights ...")
        return net

    if torch.cuda.is_available():
        net.cuda()
    print("--- define optimizer for GT Encoder---")
    logging.info("--- define optimizer for GT Encoder---")
    optimizer = optim.Adam(
        net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0
    )
    # optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-3)



    model_path = hypar["model_path"]
    model_save_fre = hypar["model_save_fre"]
    max_ite = hypar["max_ite"]
    batch_size_train = hypar["batch_size_train"]
    batch_size_valid = hypar["batch_size_valid"]
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    ite_num = hypar["start_ite"]  # count the total iteration number
    ite_num4val = 0  #
    running_loss = 0.0  # count the toal loss
    running_tar_loss = 0.0  # count the target output loss
    last_f1 = [0 for x in range(len(valid_dataloaders))]
    train_num = train_datasets[0].__len__()
    net.train()
    start_last = time.time()
    gos_dataloader = train_dataloaders[0]
    epoch_num = hypar["max_epoch_num"]
    notgood_cnt = 0
    for epoch in range(epoch_num):  ## set the epoch num as 100000
        for i, data in enumerate(gos_dataloader):
            if ite_num >= max_ite:
                print("Training Reached the Maximal Iteration Number ", max_ite)
                logging.info("Training Reached the Maximal Iteration Nmuber {max_ite}")
                exit()

            # start_read = time.time()
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1
            labels = data["label"]
            if hypar["model_digit"] == "full":
                labels = labels.type(torch.FloatTensor)
            else:
                labels = labels.type(torch.HalfTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                labels_v = Variable(labels.cuda(), requires_grad=False)
            else:
                labels_v = Variable(labels, requires_grad=False)
            start_inf_loss_back = time.time()
            optimizer.zero_grad()
            ds, fs = net(labels_v)  # net(inputs_v)
            loss2, loss = net.compute_loss(ds, labels_v)
            # Loss function to be changed from here
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_tar_loss += loss2.item()
            # del outputs, loss
            del ds, loss2, loss
            end_inf_loss_back = time.time() - start_inf_loss_back

            print(
                "GT Encoder Training>>>"
                + model_path.split("/")[-1]
                + " - [epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f, time-per-iter: %3f s, time_read: %3f"
                % (
                    epoch + 1,
                    epoch_num,
                    (i + 1) * batch_size_train,
                    train_num,
                    ite_num,
                    running_loss / ite_num4val,
                    running_tar_loss / ite_num4val,
                    time.time() - start_last,
                    time.time() - start_last - end_inf_loss_back,
                )
            )

            logging.info(
                f"GT Encoder Training>>>{model_path.split('/')[-1]} - [epoch: {epoch + 1}/{epoch_num}, batch: {(i + 1) * batch_size_train}/{train_num}, ite: {ite_num}] train loss: {running_loss / ite_num4val:3f}, tar: {running_tar_loss / ite_num4val:3f}, time-per-iter: {time.time() - start_last:3f} s, time_read: {time.time() - start_last - end_inf_loss_back:3f}"
            )

            start_last = time.time()

            if ite_num % model_save_fre == 0:  # validate every 2000 iterations
                notgood_cnt += 1
                # net.eval()
                # tmp_f1, tmp_mae, val_loss, tar_loss, i_val, tmp_time = valid_gt_encoder(net, valid_dataloaders, valid_datasets, hypar, epoch)
                tmp_f1, tmp_mae, val_loss, tar_loss, i_val, tmp_time = valid_gt_encoder(
                    net, train_dataloaders_val, train_datasets_val, hypar, epoch
                )
                net.train()
                tmp_out = 0
                print("last_f1:", last_f1)
                logging.info(f"last_f1: {last_f1}")
                print("tmp_f1:", tmp_f1)
                logging.info(f"tmp_f1: {tmp_f1}")
                for fi in range(len(last_f1)):
                    if tmp_f1[fi] > last_f1[fi]:
                        tmp_out = 1
                print("tmp_out:", tmp_out)
                logging.info(f"tmp_out: {tmp_out}")
                if tmp_out:
                    notgood_cnt = 0
                    last_f1 = tmp_f1
                    tmp_f1_str = [str(round(f1x, 4)) for f1x in tmp_f1]
                    tmp_mae_str = [str(round(mx, 4)) for mx in tmp_mae]
                    maxf1 = "_".join(tmp_f1_str)
                    meanM = "_".join(tmp_mae_str)
                    # .cpu().detach().numpy()
                    model_name = (
                        "/GTENCODER-gpu_itr_"
                        + str(ite_num)
                        + "_traLoss_"
                        + str(np.round(running_loss / ite_num4val, 4))
                        + "_traTarLoss_"
                        + str(np.round(running_tar_loss / ite_num4val, 4))
                        + "_valLoss_"
                        + str(np.round(val_loss / (i_val + 1), 4))
                        + "_valTarLoss_"
                        + str(np.round(tar_loss / (i_val + 1), 4))
                        + "_maxF1_"
                        + maxf1
                        + "_mae_"
                        + meanM
                        + "_time_"
                        + str(
                            np.round(np.mean(np.array(tmp_time)) / batch_size_valid, 6)
                        )
                        + ".pth"
                    )
                    torch.save(net.state_dict(), model_path + model_name)

                running_loss = 0.0
                running_tar_loss = 0.0
                ite_num4val = 0
                if tmp_f1[0] > 0.99:
                    print("GT encoder is well-trained and obtained...")
                    logging.info("GT encoder is well-trained and obtained...")
                    return net
                if notgood_cnt >= hypar["early_stop"]:
                    print(
                        "No improvements in the last "
                        + str(notgood_cnt)
                        + " validation periods, so training stopped !"
                    )
                    logging.info(
                        f"No improvements in the last {notgood_cnt} validation periods, so training stopped!"
                    )
                    exit()
    print("Training Reaches The Maximum Epoch Number")
    logging.info("Training Reaches The Maximum Epoch Number")
    return net


def valid_gt_encoder(net, valid_dataloaders, valid_datasets, hypar, epoch=0):
    net.eval()
    print("Validating...")
    logging.info("Validating...")
    epoch_num = hypar["max_epoch_num"]
    val_loss = 0.0
    tar_loss = 0.0
    tmp_f1 = []
    tmp_mae = []
    tmp_time = []

    start_valid = time.time()
    for k in range(len(valid_dataloaders)):
        valid_dataloader = valid_dataloaders[k]
        valid_dataset = valid_datasets[k]
        val_num = valid_dataset.__len__()
        mybins = np.arange(0, 256)
        PRE = np.zeros((val_num, len(mybins) - 1))
        REC = np.zeros((val_num, len(mybins) - 1))
        F1 = np.zeros((val_num, len(mybins) - 1))
        MAE = np.zeros((val_num))

        val_cnt = 0.0
        i_val = None
        for i_val, data_val in enumerate(valid_dataloader):
            # imidx_val, inputs_val, labels_val, shapes_val = data_val['imidx'], data_val['image'], data_val['label'], data_val['shape']
            imidx_val, labels_val, shapes_val = (
                data_val["imidx"],
                data_val["label"],
                data_val["shape"],
            )

            if hypar["model_digit"] == "full":
                labels_val = labels_val.type(torch.FloatTensor)
            else:
                labels_val = labels_val.type(torch.HalfTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                labels_val_v = Variable(labels_val.cuda(), requires_grad=False)
            else:
                labels_val_v = Variable(labels_val, requires_grad=False)
            t_start = time.time()
            ds_val = net(labels_val_v)[0]
            t_end = time.time() - t_start
            tmp_time.append(t_end)
            # loss2_val, loss_val = muti_loss_fusion(ds_val, labels_val_v)  # loss function
            loss2_val, loss_val = net.compute_loss(ds_val, labels_val_v)
            # compute F measure
            for t in range(hypar["batch_size_valid"]):
                val_cnt = val_cnt + 1.0
                print("num of val: ", val_cnt)
                logging.info(f"num of val: {val_cnt}")
                i_test = imidx_val[t].data.numpy()

                pred_val = ds_val[0][t, :, :, :]  # B x 1 x H x W
                ## recover the prediction spatial size to the orignal image size
                pred_val = torch.squeeze(
                    F.upsample(
                        torch.unsqueeze(pred_val, 0),
                        (shapes_val[t][0], shapes_val[t][1]),
                        mode="bilinear",
                    )
                )
                ma = torch.max(pred_val)
                mi = torch.min(pred_val)
                pred_val = (pred_val - mi) / (ma - mi)  # max = 1
                # pred_val = normPRED(pred_val)

                gt = np.squeeze(
                    io.imread(valid_dataset.dataset["ori_gt_path"][i_test])
                )  # max = 255
                if gt.max() == 1:
                    gt = gt * 255
                with torch.no_grad():
                    gt = torch.tensor(gt).to(device)
                pre, rec, f1, mae = f1_mae_torch(
                    pred_val * 255, gt, valid_dataset, i_test, mybins, hypar
                )

                PRE[i_test, :] = pre
                REC[i_test, :] = rec
                F1[i_test, :] = f1
                MAE[i_test] = mae
            del ds_val, gt
            gc.collect()
            torch.cuda.empty_cache()
            # if(loss_val.data[0]>1):
            val_loss += loss_val.item()  # data[0]
            tar_loss += loss2_val.item()  # data[0]
            print(
                "[validating: %5d/%5d] val_ls:%f, tar_ls: %f, f1: %f, mae: %f, time: %f"
                % (
                    i_val,
                    val_num,
                    val_loss / (i_val + 1),
                    tar_loss / (i_val + 1),
                    np.amax(F1[i_test, :]),
                    MAE[i_test],
                    t_end,
                )
            )
            logging.info(
                f"[validating: {i_val:5d}/{val_num:5d}] val_ls:{val_loss / (i_val + 1):f}, tar_ls: {tar_loss / (i_val + 1):f}, f1: {np.amax(F1[i_test, :]):f}, mae: {MAE[i_test]:f}, time: {t_end:f}"
            )

            del loss2_val, loss_val

        print("============================")
        logging.info("============================")
        PRE_m = np.mean(PRE, 0)
        REC_m = np.mean(REC, 0)
        f1_m = (1 + 0.3) * PRE_m * REC_m / (0.3 * PRE_m + REC_m + 1e-8)
        # print('--------------:', np.mean(f1_m))
        tmp_f1.append(np.amax(f1_m))
        tmp_mae.append(np.mean(MAE))
        print("The max F1 Score: %f" % (np.max(f1_m)))
        logging.info(f"The max F1 Score: {np.max(f1_m):f}")

        print("MAE: ", np.mean(MAE))
        logging.info(f"MAE: {np.mean(MAE)}")
    # print('[epoch: %3d/%3d, ite: %5d] tra_ls: %3f, val_ls: %3f, tar_ls: %3f, maxf1: %3f, val_time: %6f'% (epoch + 1, epoch_num, ite_num, running_loss / ite_num4val, val_loss/val_cnt, tar_loss/val_cnt, tmp_f1[-1], time.time()-start_valid))

    return tmp_f1, tmp_mae, val_loss, tar_loss, i_val, tmp_time



def train(
    net,
    optimizer,
    train_dataloaders,
    train_datasets,
    valid_dataloaders,
    valid_datasets,
    hypar,
    train_dataloaders_val,
    train_datasets_val,
    
):  # model_path, model_save_fre, max_ite=1000000):

    experiment.log_parameters(hypar)

    best_train_loss = float("inf")
    best_val_loss = float("inf")
    loss_tracking_file = "saved_models/IS-Net/loss_tracking.txt"
    if not os.path.exists(os.path.dirname(loss_tracking_file)):
        os.makedirs(os.path.dirname(loss_tracking_file))

    if hypar["interm_sup"]:
        print("Get the gt encoder ...")
        logging.info("Get the gt encoder ...")
        featurenet = get_gt_encoder(
            train_dataloaders,
            train_datasets,
            valid_dataloaders,
            valid_datasets,
            hypar,
            train_dataloaders_val,
            train_datasets_val,
        )
        ## freeze the weights of gt encoder
        for param in featurenet.parameters():
            param.requires_grad = False

    model_path = hypar["model_path"]
    model_save_fre = hypar["model_save_fre"]
    max_ite = hypar["max_ite"]
    batch_size_train = hypar["batch_size_train"]
    batch_size_valid = hypar["batch_size_valid"]

    writer = SummaryWriter(log_dir=model_path)

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    ite_num = hypar["start_ite"]  # count the toal iteration number
    ite_num4val = 0  #
    running_loss = 0.0  # count the toal loss
    running_tar_loss = 0.0  # count the target output loss
    last_f1 = [0 for x in range(len(valid_dataloaders))]

    train_num = train_datasets[0].__len__()

    net.train()

    start_last = time.time()
    gos_dataloader = train_dataloaders[0]
    epoch_num = hypar["max_epoch_num"]
    notgood_cnt = 0

    interval_train_loss = 0.0
    interval_val_loss = 0.0
    interval_train_count = 0
    interval_val_count = 0
    avg_train_loss_list = []
    avg_val_loss_list = []

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=13, verbose=True)


    for epoch in range(epoch_num):  ## set the epoch num as 100000

        for i, data in enumerate(gos_dataloader):

            if ite_num >= max_ite:
                print("Training Reached the Maximal Iteration Number ", max_ite)
                logging.info(f"Training Reached the Maximal Iteration Number {max_ite}")

                exit()

            # start_read = time.time()
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            # get the inputs
            inputs, labels = data["image"], data["label"]

            if hypar["model_digit"] == "full":
                inputs = inputs.type(torch.FloatTensor)
                labels = labels.type(torch.FloatTensor)
            else:
                inputs = inputs.type(torch.HalfTensor)
                labels = labels.type(torch.HalfTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(
                    inputs.cuda(), requires_grad=False
                ), Variable(labels.cuda(), requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(
                    labels, requires_grad=False
                )

            # print("time lapse for data preparation: ", time.time()-start_read, ' s')

            # y zero the parameter gradients
            start_inf_loss_back = time.time()
            optimizer.zero_grad()

            if hypar["interm_sup"]:
                # forward + backward + optimize
                ds, dfs = net(inputs_v)
                _, fs = featurenet(labels_v)  ## extract the gt encodings
                loss2, loss = net.compute_loss_kl(ds, labels_v, dfs, fs, mode="MSE")
            else:
                # forward + backward + optimize
                ds, _ = net(inputs_v)
                loss2, loss = net.compute_loss(ds, labels_v)

            loss.backward()
            optimizer.step()

            

            # # print statistics
            running_loss += loss.item()
            running_tar_loss += loss2.item()

            experiment.log_metric("train_loss", running_loss / ite_num, step=ite_num)
            experiment.log_metric("train_target_loss", running_tar_loss / ite_num, step=ite_num)


            writer.add_scalar("Loss/Train", running_loss / ite_num4val, ite_num)

            interval_train_loss += loss.item()
            interval_train_count += 1

            # del outputs, loss
            del ds, loss2, loss
            end_inf_loss_back = time.time() - start_inf_loss_back

            print(
                ">>>"
                + model_path.split("/")[-1]
                + " - [epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f, time-per-iter: %3f s, time_read: %3f"
                % (
                    epoch + 1,
                    epoch_num,
                    (i + 1) * batch_size_train,
                    train_num,
                    ite_num,
                    running_loss / ite_num4val,
                    running_tar_loss / ite_num4val,
                    time.time() - start_last,
                    time.time() - start_last - end_inf_loss_back,
                )
            )
            logging.info(
                f">>>{model_path.split('/')[-1]} - [epoch: {epoch + 1:3d}/{epoch_num:3d}, batch: {(i + 1) * batch_size_train:5d}/{train_num:5d}, ite: {ite_num:d}] train loss: {running_loss / ite_num4val:3f}, tar: {running_tar_loss / ite_num4val:3f}, time-per-iter: {time.time() - start_last:3f} s, time_read: {time.time() - start_last - end_inf_loss_back:3f}"
            )

            start_last = time.time()

            current_train_loss = running_loss / ite_num4val
            if current_train_loss < best_train_loss:
                best_train_loss = current_train_loss

            if ite_num % model_save_fre == 0:  # validate every 2000 iterations
                notgood_cnt += 1
                net.eval()
                tmp_f1, tmp_mae, val_loss, tar_loss, i_val, tmp_time = valid(
                    net, valid_dataloaders, valid_datasets, hypar, epoch
                )
                current_val_loss = val_loss / (i_val + 1)
                net.train()  # resume

                experiment.log_metric("val_loss", val_loss/ (i_val + 1), step=ite_num)


                writer.add_scalar("Loss/Validation", val_loss / (i_val + 1), ite_num)

                f1_mean = np.mean(
                    tmp_f1
                )  # Assuming tmp_f1 is a list of F1 scores for each class
                writer.add_scalar("F1 Score/Validation", f1_mean, ite_num)
                experiment.log_metric("f1_score", f1_mean, step=ite_num)

                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    logging.info(f"Updated Best Validation Loss to: {best_val_loss}")

                interval_val_loss += val_loss
                interval_val_count += (
                    i_val + 1
                )  # Assuming i_val is the last index in validation loop

                avg_train_loss = (
                    interval_train_loss / interval_train_count
                    if interval_train_count > 0
                    else 0
                )
                avg_val_loss = (
                    interval_val_loss / interval_val_count
                    if interval_val_count > 0
                    else 0
                )
                avg_train_loss_list.append(avg_train_loss)
                avg_val_loss_list.append(avg_val_loss)
                interval_train_loss = 0.0
                interval_val_loss = 0.0
                interval_train_count = 0
                interval_val_count = 0


                
                tmp_out = 0
                print("last_f1:", last_f1)
                logging.info(f"last_f1: {last_f1}")

                print("tmp_f1:", tmp_f1)
                logging.info(f"tmp_f1: {tmp_f1}")

                for fi in range(len(last_f1)):
                    if tmp_f1[fi] > last_f1[fi]:
                        tmp_out = 1
                print("tmp_out:", tmp_out)
                logging.info(f"tmp_out: {tmp_out}")

                if tmp_out:
                    notgood_cnt = 0
                    last_f1 = tmp_f1
                    tmp_f1_str = [str(round(f1x, 4)) for f1x in tmp_f1]
                    tmp_mae_str = [str(round(mx, 4)) for mx in tmp_mae]
                    maxf1 = "_".join(tmp_f1_str)
                    meanM = "_".join(tmp_mae_str)

                    model_name = (
                        "/gpu_itr_"
                        + str(ite_num)
                        + "_traLoss_"
                        + str(np.round(running_loss / ite_num4val, 4))
                        + "_traTarLoss_"
                        + str(np.round(running_tar_loss / ite_num4val, 4))
                        + "_valLoss_"
                        + str(np.round(val_loss / (i_val + 1), 4))
                        + "_valTarLoss_"
                        + str(np.round(tar_loss / (i_val + 1), 4))
                        + "_maxF1_"
                        + maxf1
                        + "_mae_"
                        + meanM
                        + "_time_"
                        + str(
                            np.round(np.mean(np.array(tmp_time)) / batch_size_valid, 6)
                        )
                        + ".pth"
                    )
                    torch.save(net.state_dict(), model_path + model_name)
                    experiment.log_asset(model_path + model_name, file_name=model_name)


                    # model_name = "isnet" + ".pth"
                    # torch.save(net.state_dict(), model_path + model_name)

                    # log_file = os.path.join(model_path , "Loss_tracking.txt")
                    # with open(log_file , "a") as f:
                    #     f.write(f"Iteration: {ite_num}\n")
                    #     f.write(f"Training Loss: {np.round(running_loss / ite_num4val, 4)}\n")
                    #     f.write(f"Validation Loss: {np.round(val_loss / (i_val + 1), 4)}\n")
                    #     f.write(f"Max F1 Scores: {maxf1}\n")
                    #     f.write(f"MAE Scores: {meanM}\n")
                    #     f.write("="*50 + "\n")

                    f1_log_file = os.path.join(model_path, "f1_scores_checkpoint.txt")
                    with open(f1_log_file, "a") as f1_file:

                        f1_file.write(
                            f"Checkpoint Updated - Epoch: {epoch + 1}, Iteration: {ite_num}\n"
                        )
                        f1_file.write(f"F1 Scores: {', '.join(tmp_f1_str)}\n")
                        f1_file.write("=" * 30 + "\n")

                        

                running_loss = 0.0
                running_tar_loss = 0.0
                ite_num4val = 0

                scheduler.step(f1_mean)

                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar("Learning Rate", current_lr, ite_num)
                experiment.log_metric("Learning Rate", current_lr, step=ite_num)

                if notgood_cnt >= hypar["early_stop"]:
                    print(
                        "No improvements in the last "
                        + str(notgood_cnt)
                        + " validation periods, so training stopped !"
                    )
                    exit()

    writer.close()
    print("Training Reaches The Maximum Epoch Number")
    logging.info("Training Reaches The Maximum Epoch Number")


def valid(net, valid_dataloaders, valid_datasets, hypar, epoch=0):

    net.eval()
    print("Validating...")
    logging.info("Validating...")
    epoch_num = hypar["max_epoch_num"]

    val_loss = 0.0
    tar_loss = 0.0
    val_cnt = 0.0
    tmp_f1 = []
    tmp_mae = []
    tmp_time = []

    start_valid = time.time()

    for k in range(len(valid_dataloaders)):

        valid_dataloader = valid_dataloaders[k]
        valid_dataset = valid_datasets[k]

        val_num = valid_dataset.__len__()
        mybins = np.arange(0, 256)
        PRE = np.zeros((val_num, len(mybins) - 1))
        REC = np.zeros((val_num, len(mybins) - 1))
        F1 = np.zeros((val_num, len(mybins) - 1))
        MAE = np.zeros((val_num))

        for i_val, data_val in enumerate(valid_dataloader):
            val_cnt = val_cnt + 1.0
            imidx_val, inputs_val, labels_val, shapes_val = (
                data_val["imidx"],
                data_val["image"],
                data_val["label"],
                data_val["shape"],
            )

            if hypar["model_digit"] == "full":
                inputs_val = inputs_val.type(torch.FloatTensor)
                labels_val = labels_val.type(torch.FloatTensor)
            else:
                inputs_val = inputs_val.type(torch.HalfTensor)
                labels_val = labels_val.type(torch.HalfTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_val_v, labels_val_v = Variable(
                    inputs_val.cuda(), requires_grad=False
                ), Variable(labels_val.cuda(), requires_grad=False)
            else:
                inputs_val_v, labels_val_v = Variable(
                    inputs_val, requires_grad=False
                ), Variable(labels_val, requires_grad=False)

            t_start = time.time()
            ds_val = net(inputs_val_v)[0]
            t_end = time.time() - t_start
            tmp_time.append(t_end)

            loss2_val, loss_val = muti_loss_fusion(ds_val, labels_val_v)
            # loss2_val, loss_val = net.compute_loss(ds_val, labels_val_v)

            # compute F measure
            for t in range(hypar["batch_size_valid"]):
                i_test = imidx_val[t].data.numpy()

                pred_val = ds_val[0][t, :, :, :]
                pred_val = torch.squeeze(
                    F.upsample(
                        torch.unsqueeze(pred_val, 0),
                        (shapes_val[t][0], shapes_val[t][1]),
                        mode="bilinear",
                    )
                )
                ma = torch.max(pred_val)
                mi = torch.min(pred_val)
                pred_val = (pred_val - mi) / (ma - mi)

                if len(valid_dataset.dataset["ori_gt_path"]) != 0:
                    gt = np.squeeze(
                        io.imread(valid_dataset.dataset["ori_gt_path"][i_test])
                    )  # max = 255
                    if gt.max() == 1:
                        gt = gt * 255
                else:
                    gt = np.zeros((shapes_val[t][0], shapes_val[t][1]))
                with torch.no_grad():
                    gt = torch.tensor(gt).to(device)
                pre, rec, f1, mae = f1_mae_torch(
                    pred_val * 255, gt, valid_dataset, i_test, mybins, hypar
                )

                PRE[i_test, :] = pre
                REC[i_test, :] = rec
                F1[i_test, :] = f1
                MAE[i_test] = mae

                del ds_val, gt
                gc.collect()
                torch.cuda.empty_cache()
            val_loss += loss_val.item()  # data[0]
            tar_loss += loss2_val.item()  # data[0]

            print(
                "[validating: %5d/%5d] val_ls:%f, tar_ls: %f, f1: %f, mae: %f, time: %f"
                % (
                    i_val,
                    val_num,
                    val_loss / (i_val + 1),
                    tar_loss / (i_val + 1),
                    np.amax(F1[i_test, :]),
                    MAE[i_test],
                    t_end,
                )
            )
            logging.info(
                f"[validating: {i_val:5d}/{val_num:5d}] val_ls:{val_loss / (i_val + 1):f}, tar_ls: {tar_loss / (i_val + 1):f}, f1: {np.amax(F1[i_test, :]):f}, mae: {MAE[i_test]:f}, time: {t_end:f}"
            )

            del loss2_val, loss_val

        print("============================")
        logging.info("============================")
        PRE_m = np.mean(PRE, 0)
        REC_m = np.mean(REC, 0)
        f1_m = (1 + 0.3) * PRE_m * REC_m / (0.3 * PRE_m + REC_m + 1e-8)

        tmp_f1.append(np.amax(f1_m))
        tmp_mae.append(np.mean(MAE))

    return tmp_f1, tmp_mae, val_loss, tar_loss, i_val, tmp_time


class SaveTransformedImageAndMask(object):
    def __init__(
        self,
        save_dir_images,
        save_dir_masks,
        save_prefix="transformed_",
        save_format="jpg",
    ):
        self.save_dir_images = save_dir_images
        self.save_dir_masks = save_dir_masks
        self.save_prefix = save_prefix
        self.save_format = save_format
        os.makedirs(save_dir_images, exist_ok=True)
        os.makedirs(save_dir_masks, exist_ok=True)
        self.counter = 0

    def __call__(self, sample):
        image, mask = sample["image"], sample["label"]
        image_save_path = os.path.join(
            self.save_dir_images, f"{self.save_prefix}{self.counter}.{self.save_format}"
        )
        mask_save_path = os.path.join(
            self.save_dir_masks,
            f"{self.save_prefix}{self.counter}_mask.{self.save_format}",
        )
        save_image(image, image_save_path)
        print(f"Image saved to {image_save_path}")
        save_image(mask, mask_save_path)
        print(f"Mask saved to {mask_save_path}")
        self.counter += 1
        return sample


def main(train_datasets, valid_datasets, hypar):
    dataloaders_train = []
    dataloaders_valid = []
    

    if hypar["mode"] == "train":
        print("--- create training dataloader ---")
        logging.info("--- create training dataloader ---")

        train_nm_im_gt_list = get_im_gt_name_dict(train_datasets, flag="train")
        train_dataloaders, train_datasets = create_dataloaders(
            train_nm_im_gt_list,
            cache_size=hypar["cache_size"],
            cache_boost=hypar["cache_boost_train"],
            my_transforms=[
                # GOSRandomHFlip(),
                GOSRandomHFlip(prob=1),
                GOSColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, prob=0.2
                ),
                GOSBinaryMask(threshold=0.5),
                GOSNormalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0]),
                # SaveTransformedImageAndMask(
                # save_dir_images="sample_little/training",
                # save_dir_masks="sample_little/validation"),
                                                                             
            ],
            batch_size=hypar["batch_size_train"],
            shuffle=True,
        )

        train_dataloaders_val, train_datasets_val = create_dataloaders(
            train_nm_im_gt_list,
            cache_size=hypar["cache_size"],
            cache_boost=hypar["cache_boost_train"],
            my_transforms=[
                GOSNormalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0]),
            ],
            batch_size=hypar["batch_size_valid"],
            shuffle=True,
        )

        print(len(train_dataloaders), "train dataloaders created")
        logging.info(f"{len(train_dataloaders)} train dataloaders created")
    print("--- create valid dataloader ---")
    logging.info("--- create valid dataloader ---")

    # Build dataloader for validation or testing
    valid_nm_im_gt_list = get_im_gt_name_dict(valid_datasets, flag="valid")
    valid_dataloaders, valid_datasets = create_dataloaders(
        valid_nm_im_gt_list,
        cache_size=hypar["cache_size"],
        cache_boost=hypar["cache_boost_valid"],
        my_transforms=[
            GOSNormalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0]),
            GOSResize(hypar["input_size"]),
        ],
        batch_size=hypar["batch_size_valid"],
        shuffle=False,
    )

    print(len(valid_dataloaders), "valid dataloaders created")
    logging.info(f"{len(valid_dataloaders)} valid dataloaders created")

    # Step 2: Build model
    print("--- build model ---")
    logging.info("--- build model ---")
    net = hypar["model"]

    # Convert to half precision
    if hypar["model_digit"] == "half":
        net.half()
        for layer in net.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()

    if torch.cuda.is_available():
        net.cuda()

    # Restore model if specified
    if hypar["restore_model"] != "":
        print("restore model from:")
        logging.info("restore model from:")
        print(hypar["model_path"] + "/" + hypar["restore_model"])
        logging.info(f"{hypar['model_path']}/{hypar['restore_model']}")

        if torch.cuda.is_available():
            net.load_state_dict(
                torch.load(hypar["model_path"] + "/" + hypar["restore_model"]),
                strict=False,
            )
        else:
            net.load_state_dict(
                torch.load(
                    hypar["model_path"] + "/" + hypar["restore_model"],
                    map_location="cpu",
                ),
                strict=False,
            )


    # Step 3: Define optimizer
    print("--- define optimizer ---")
    logging.info("--- define optimizer ---")
    # optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    optimizer = optim.AdamW(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    # Step 4: Train or validate
    if hypar["mode"] == "train":
        train(
            net,
            optimizer,
            train_dataloaders,
            train_datasets,
            valid_dataloaders,
            valid_datasets,
            hypar,
            train_dataloaders_val,
            train_datasets_val,
            
        )
    else:
        valid(net, valid_dataloaders, valid_datasets, hypar)
    

"""Dataset path"""
if __name__ == "__main__":

    train_datasets, valid_datasets = [], []
    dataset_1, dataset_1 = {}, {}
    dataset_tr = {
        "name": "DIS5K-TR",
        "im_dir": "element_remove_256/training/im",  # Remove thresholding 
        "gt_dir": "element_remove_256/training/gt",  # Training Mask
        "im_ext": ".jpg",  
        "gt_ext": ".png",  
        "cache_dir": "../DIS5K-Cache/DIS-TR",
    }
    #CHECK EXPERINMENT NAME
    dataset_vd = {
        "name": "DIS5K-VD",
        "im_dir": "element_remove_256/validation/im",  # Remove thresholding 
        "gt_dir": "element_remove_256/validation/gt",  # Training Mask
        "im_ext": ".jpg",  
        "gt_ext": ".png",  
        "cache_dir": "../DIS5K-Cache/DIS-TR",
    }

    train_datasets = [dataset_tr]
    valid_datasets = [dataset_vd]
    hypar = {}

    hypar["mode"] = "train"

    hypar["interm_sup"] = False

    if hypar["mode"] == "train":
        hypar["valid_out_dir"] = ""
        hypar["model_path"] = "saved_models/IS-Net"      
        hypar["restore_model"] = "pretrained.pth"
        hypar["start_ite"] = 0
        hypar["gt_encoder_model"] = ""
    else:
        hypar["valid_out_dir"] = "demo_datasets/your_dataset_result"
        hypar["model_path"] = "saved_models/IS-Net"
        hypar["restore_model"] = ""

    hypar["model_digit"] = "full"
    hypar["seed"] = 0

    hypar["cache_size"] = [256,256]    #CHECK EXPERINMENT NAME
    hypar["cache_boost_train"] = False
    hypar["cache_boost_valid"] = False

    hypar["input_size"] = [256,256]
    hypar["crop_size"] = [256,256]
    hypar["random_flip_h"] = 1
    hypar["random_flip_v"] = 0

    print("building model...")
    logging.info("building model...")

    hypar["model"] = ISNetDIS()
    hypar["early_stop"] = 30
    hypar["model_save_fre"] = 20

    hypar["batch_size_train"] = 4   #CHECK EXPERINMENT NAME
    hypar["batch_size_valid"] = 1
    print("batch size: ", hypar["batch_size_train"])
    logging.info(f"batch size: {hypar['batch_size_train']}")

    hypar["max_ite"] = 1000000
    hypar["max_epoch_num"] = 1000000
    


    experiment.log_parameters({
    "train_dataset_name": dataset_tr["name"],
    "train_image_dir": dataset_tr["im_dir"],
    "train_gt_dir": dataset_tr["gt_dir"],
    "valid_image_dir": dataset_vd["im_dir"],
    "valid_gt_dir": dataset_vd["gt_dir"],

    "cache_size":hypar["cache_size"],
    "input_size":hypar["input_size"],
    "restore_model":hypar["restore_model"],
    "early_stop":hypar["early_stop"],
    "model_save_fre":hypar["model_save_fre"],
    "batch_size_train":hypar["batch_size_train"],
    "batch_size_valid":hypar["batch_size_valid"]
    })

    main(train_datasets, valid_datasets, hypar=hypar)
    experiment.end()  # End the Comet experiment


