import numpy as np
import pandas as pd
import torch
from keras.models import load_model
import os
from sgan.utils import relative_to_abs
import cv2 as cv
import skvideo.io
import sys
import pickle
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

new_model = load_model("lstm_hotel.h5")
def get_trajs(frame, step=10):

    obs_len = 8
    pred_len = 12
# step= 10
    trajs = {}
    data = pd.read_csv("/home/asyed/sgan/scripts/datasets/hotel/test/biwi_hotel.txt", sep="\t", header=None)
    data.columns = ["frameID", "pedID", "x", "y"]
    data.sort_values(by=["frameID", "pedID"])
    data.reset_index(drop=True)
# -1 because we include in selection
    seq_range = [frame - (obs_len - 1) * step, frame + pred_len * step]
    obs_range = [frame - (obs_len - 1) * step, frame]

    raw_obs_seq = data.loc[data["frameID"].between(obs_range[0], obs_range[1], inclusive=True)]
    raw_pred_seq = data.loc[data["frameID"].between(obs_range[1] + step, seq_range[1], inclusive=True)]
    peds_in_seq = raw_obs_seq.pedID.unique()
# print(peds_in_seq)
# curr_seq = np.zeros((len(peds_in_seq), 2, obs_len))
    curr_seq = np.zeros((len(peds_in_seq), 2, obs_len))

    curr_seq_rel = np.zeros((len(peds_in_seq), 2, obs_len))
    pred_curr_seq= np.zeros((len(peds_in_seq),2,pred_len))
# print(pred_curr_seq)
    id_list = []
    considered_ped = 0

    for ped_id in peds_in_seq:
        obs_ped_seq = raw_obs_seq.loc[raw_obs_seq.pedID == ped_id]
        pred_ped_seq = raw_pred_seq.loc[raw_pred_seq.pedID == ped_id]

        # pred_ped_seq= raw_pred_seq.loc[raw_pred_seq.pedID==ped_id]
        # print(obs_ped_seq)
        # seq has to have at least obs_len length
        if len(obs_ped_seq.frameID) == obs_len and len(pred_ped_seq.frameID) == pred_len:
            id_list.append(ped_id)

            # pred_ped_seq = raw_pred_seq.loc[raw_pred_seq.pedID == ped_id]
            # print(pred_ped_seq)
            trajs[ped_id] = {}

            obs_traj = obs_ped_seq[["x", "y"]].values.transpose()
            obs_traj_rel = np.zeros(obs_traj.shape)
            obs_traj_rel[:, 1:] = obs_traj[:, 1:] - obs_traj[:, :-1]

            pred_traj= pred_ped_seq[["x","y"]].values.transpose()

            pred_traj_rel= np.zeros(pred_traj.shape)
            pred_traj_rel[:, 1:] = pred_traj[:, 1:] - pred_traj[:, :-1]

            curr_seq[considered_ped, :, 0:obs_len] = obs_traj
            curr_seq_rel[considered_ped, :, 0:obs_len] = obs_traj_rel
            # print(obs_traj)


            trajs[ped_id]["obs"] = obs_traj.transpose()
            trajs[ped_id]["pred_gt"] = pred_ped_seq[["x", "y"]].values

            considered_ped += 1

    if considered_ped > 0:
        obs_list_tensor = torch.from_numpy(curr_seq[:considered_ped, :]).permute(2, 0, 1).float()
        obs_list_rel_tensor = torch.from_numpy(curr_seq_rel[:considered_ped, :]).permute(2, 0, 1)

        pred_list_tensor= torch.from_numpy(curr_seq[:considered_ped,:]).permute(2,0,1).float()


        #JUPYTER NOTE BOOK

        a = obs_list_rel_tensor.permute(1, 0, 2).numpy()
        predict_traj = new_model.predict(a)
        predict_traj1 = predict_traj
        predict_traj = torch.from_numpy(predict_traj).permute(1, 0, 2).float()
        pred_abs = relative_to_abs(predict_traj, obs_list_tensor[-1])
        pred_abs = pred_abs.permute(1, 0, 2)

        for i in range(considered_ped):
            ped_id = id_list[i]
            trajs[ped_id]["vanila_lstm"] = pred_abs[i]



        return trajs
    else:
        return None


# print(get_trajs(7000,10))
vid_path= "/home/asyed/trajectory_cnn/scenes_and_matrices/hotel.avi"
matrix_path= "/home/asyed/trajectory_cnn/scenes_and_matrices/hotel.txt"
color_dict = {"obs": (0, 0, 0),
              "pred_gt": (0, 250, 0), "vanila_lstm": (0, 0, 250)}


def world_to_img(world_coordinates, hom_matrix):
    scaled_trajs = []

    inv_matrix = np.linalg.inv(hom_matrix)

    # if several sequences
    if len(world_coordinates.shape) > 2:
        # easier to iterate over them
        world_coordinates = np.swapaxes(world_coordinates, 0, 1)

        for traj in world_coordinates:
            ones = np.ones((len(traj), 1))
            P = np.hstack((traj, ones))
            R = np.dot(inv_matrix, P.transpose()).transpose()
            y = (R[:, 0]/R[:, 2]).reshape(-1, 1)
            x = (R[:, 1]/R[:, 2]).reshape(-1, 1)
            scaled_trajs.append(np.hstack((x, y)))
    else:
        ones = np.ones((len(world_coordinates), 1))
        P = np.hstack((world_coordinates, ones))
        R = np.dot(inv_matrix, P.transpose())
        y = (R[0, :]/R[2, :]).reshape(-1, 1)
        x = (R[1, :]/R[2, :]).reshape(-1, 1)
        scaled_trajs.append(np.hstack((x, y)))
    return scaled_trajs

def img_to_world(input, matrix):
    return world_to_img(input, np.linalg.inv(matrix))

def get_frame(video_path, frame):
    cap = cv.VideoCapture(video_path)
    cap.set(cv.CAP_PROP_POS_FRAMES, frame)
    _, img = cap.read()
    return img

def print_to_img(trajs, video_path, matrix_path, frame):
    img = get_frame(video_path, frame)
    if trajs is not None:
        matrix = np.loadtxt(matrix_path, dtype=float)
        heigth, width, _ = img.shape

        scaled_trajs = {}
        for ped_id, ped in trajs.items():
            scaled_trajs[ped_id] = {}
            for traj_name, traj in ped.items():
                scaled_traj = []
                if traj.size != 0:
                    scaled_traj = world_to_img(traj, matrix)[0]
                scaled_trajs[ped_id][traj_name] = scaled_traj

        for ped_id, ped in scaled_trajs.items():
            for ped_seq_name, ped_sequence in ped.items():
                color = color_dict[ped_seq_name]
                if len(ped_sequence) > 0:
                    #draw pred_gt thicker if we can compute ade/fde on it
                    thick = 3 if ped_seq_name == "pred_gt" and len(ped_sequence) == 12 else 1

                    for index, point in enumerate(ped_sequence[:-1, :]):
                        real_pt_1 = tuple([int(round(x)) for x in point])
                        real_pt_2 = tuple([int(round(x)) for x in ped_sequence[index + 1]])
                        cv.line(img, real_pt_1, real_pt_2, color, thick)
    return img


data = pd.read_csv("/home/asyed/sgan/scripts/datasets/hotel/test/biwi_hotel.txt", sep="\t", header=None)
data.columns = ["frameID", "pedID", "x", "y"]
data.sort_values(by=["frameID", "pedID"])
data.reset_index(drop=True)
frameList = data.frameID.unique()
print(frameList)
max = frameList[-1]
writer = skvideo.io.FFmpegWriter("out/lstm10_hotel.mp4")
#
for frame_number in range(0, max, 10):
    if frame_number % 1000 == 0:
        print("Frame {}/{}".format(frame_number, max))

    trajs1 = None
    if frame_number in frameList:
        trajs1 = get_trajs(frame_number)
        # print(trajs)
        with open("trajec1", "wb+") as f:
             pickle.dump(trajs1, f)

    img = print_to_img(trajs1, vid_path, matrix_path, frame_number)

    writer.writeFrame(img)
print(img)


