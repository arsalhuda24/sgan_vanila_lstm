import numpy as np
from keras.layers import LSTM, Dense ,RepeatVector, Dropout, TimeDistributed
from keras import optimizers
from keras.models import Sequential
from sgan.data.trajectories import TrajectoryDataset
import matplotlib.pyplot as plt
from keras.models import load_model
from sgan.utils import relative_to_abs
import torch
import os
from sgan.losses import displacement_error, final_displacement_error
from keras.layers import embeddings

os.environ["CUDA_VISIBLE_DEVICES"]="1"

datasetss= TrajectoryDataset(data_dir="/home/asyed/sgan/scripts/datasets/univ/train" , obs_len=8, pred_len=12, skip=1, threshold=0.002,
        min_ped=1, delim='\t')

test= TrajectoryDataset(data_dir="/home/asyed/sgan/scripts/datasets/univ/test", obs_len=8, pred_len=12, skip=1, threshold=0.002,
        min_ped=1, delim='\t')


input_data= datasetss.obs_traj_rel.permute(0,2,1).numpy()
output_data= datasetss.pred_traj_rel.permute(0,2,1).numpy()

test_x= test.obs_traj_rel.permute(0,2,1).numpy()
test_y = test.pred_traj_rel.permute(0,2,1).numpy()

test_x_abs= test.obs_traj.permute(0,2,1).numpy()
test_y_abs=test.pred_traj.permute(0,2,1).numpy()
#
hidden_neurons= 128
# # # #
model= Sequential()
model.add(Dense(64, input_shape=(None,2),activation="linear"))
model.add(LSTM(input_shape=(8,64), output_dim= hidden_neurons, return_sequences= False))
model.add(RepeatVector(12))
model.add(LSTM( output_dim= hidden_neurons, return_sequences= True))
model.add(Dropout(0.3))
model.add(TimeDistributed(Dense(2,activation= "linear")))
optim= optimizers.adam(lr=0.001)
model.compile(loss="mse", optimizer=optim, metrics=["accuracy"])

model.fit(input_data, output_data,
                  validation_split=0.15,
                  epochs=50,
                  batch_size=64,
                  shuffle=True)
        # model.reset_states()



# history = model.fit(x, y, validation_data=(x_test, y_test), epochs=100, callbacks=callbacks_list)


model.save("lstm_univ_em.h5")
new_model= load_model("lstm_univ_em.h5")
pred=new_model.predict(test_x,batch_size=64)
predict= torch.from_numpy(pred)
predict= predict.permute(0,2,1)
prediction= predict.permute(0,2,1).numpy()

a=relative_to_abs(datasetss.obs_traj_rel.permute(2,0,1),datasetss.obs_traj.permute(2,0,1)[0])
# print(a)

# print(predict[1:3].permute(2,0,1))
# print(datasetss.pred_traj[1:3].permute(2,0,1)[0])
# print(datasetss.obs_traj[1])

# print(datasetss.obs_traj_rel[1:3].permute(2,0,1))
# print(datasetss.obs_traj[1:3].permute(2,0,1)[0])

abs_traj=relative_to_abs(predict.permute(2,0,1),test.pred_traj.permute(2,0,1)[0])
pred_traj= abs_traj.permute(1,0,2).numpy()
# print(pred_traj[1:4])
# print(datasetss.obs_traj[1:3])
# print(a)
# print(a.permute(1,0,2))

for i in range(110,115):
    plt.plot(pred_traj[i][:,0], pred_traj[i][:,1], "r+")
    plt.plot(test_x_abs[i][:,0], test_x_abs[i][:,1], "b+")
    plt.plot(test_y_abs[i][:,0], test_y_abs[i][:,1], "g+")

plt.show()
pred_traj_tensor=torch.from_numpy(pred_traj)
gt_tensor=torch.from_numpy(test_y_abs)
ade=displacement_error(pred_traj_tensor.permute(1,0,2),gt_tensor.permute(1,0,2),mode="sum")
# print(ade)
ade_res = (ade[0].detach()) / (len(pred_traj) * 12)
print(ade_res)
fde= final_displacement_error(pred_traj_tensor.permute(1,0,2)[-1],gt_tensor.permute(1,0,2)[-1],mode="sum")
fde_res= (fde[0].detach()) / (len(pred_traj))
# print(predict.permute(2,0,1))
# print(test.pred_traj.permute(2,0,1))
print(fde_res)

print(test_y_abs.shape)
print(pred_traj_tensor.shape)