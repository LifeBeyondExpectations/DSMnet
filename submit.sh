#dir_flyingthings3d=/media/qjc/D/data/FlyingThings3D
#root=$dir_flyingthings3d
#dataset=flyingthings3d-tr
dir_kitti=/media/qjc/D/data/kitti
root=$dir_kitti
dataset=kitti2015-te
net=dispnetcorr # dispnet/dispnetcorr/iresnet/gcnet
loss_name=depthmono-mask # supervised/(common/depthmono/SsSMnet/Cap_ds_lr)[-mask]
mode=train # train/finetune
dataset_train=kitti-raw # kitti-raw/kitti2015-tr/flyingthings3d-tr
flag_model=${mode}_${dataset_train}_${net}_${loss_name}
path_model=./output/$flag_model/model.pkl
bt=1


python main.py  --mode submit --net $net --batchsize $bt --dataset $dataset --root $root \
                   --path_model $path_model --flag_model $flag_model --flag_save true
