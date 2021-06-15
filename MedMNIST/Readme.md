train.py 为整个项目的主函数，输入发生了一些变化：

完成输入：

python train.py --data_name breastmnist --root medmnist --num_epoch 100 --download True --start_epoch 0 --lr 0.001 --batch_size 128 --model ResNet18

Default输入：

python train.py --data_name breastmnist --root medmnist --num_epoch 100 --download True

medmnist文件夹下的models.py为自主实现的残差网络

autoaugment.py为开源的自动数据增强方法，内部封装了三类策略

AutoAug_loaddata.py为自主编程的调用自动增强方法的接口函数

selfaug.py为自主编程实现的简单的数据增强方法

selfdcgan.py为自主实现的DCGAN数据增强方法

ganwrt.py为配套的输入输出转换函数

AI_Medmnist_Report为本次小作业的最终报告,正文部分大概6页多，加上目录附录参考文献有11页