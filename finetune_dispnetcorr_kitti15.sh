#dir_flyingthings3d=/media/qjc/D/data/FlyingThings3D
#dataset=flyingthings3d-tr
#root=$dir_flyingthings3d
#dataset_val=flyingthings3d-te
#root_val=$dir_flyingthings3d
dir_kitti=/media/qjc/D/data/kitti
dataset=kitti2015-tr
root=$dir_kitti
dataset_val=kitti2012-tr
root_val=$dir_kitti
net=dispnetcorr # dispnet/dispnetcorr/iresnet/gcnet/psmnet
path_model=./output/train_flyingthings3d-tr_dispnetcorr_supervised/model.pkl
loss_name=supervised # supervised/(common/depthmono/SsSMnet/Cap_ds_lr)[-mask]
bt=4


python main.py --mode finetune --net $net --loss_name $loss_name --batchsize $bt --updates 100000 \
               --path_model $path_model --lr 0.00002 --lr_adjust_start 20000 --lr_adjust_stride 100000 \
               --dataset $dataset --root $root --dataset_val $dataset_val --root_val $root_val


