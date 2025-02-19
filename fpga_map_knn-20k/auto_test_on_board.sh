#!/bin/sh
# 设置堆栈
ulimit -s unlimited

# 删除原来的结果文件
rm rps_fpga_result.txt
rm ../mmblk_sdcard/fpga_odom_knn_dataset/FPGA_ODOM_KNN_RESULT.txt
rm run_host.txt

# 设置参数
min_loop_num=1
max_loop_num=4

refe_set_num=126
ref_set_points_num=1
search_set_num=127
search_set_points_num=30
power_flag=0

# 运行代码
for ((ref_set_points_num=$min_loop_num; ref_set_points_num<=$max_loop_num; ref_set_points_num++))
do
    echo "running ./host.exe ../mmblk_sdcard/fpga_odom_knn_dataset/ $refe_set_num $ref_set_points_num $search_set_num $search_set_points_num $power_flag"
    ./host.exe ../mmblk_sdcard/fpga_odom_knn_dataset/ $refe_set_num $ref_set_points_num $search_set_num $search_set_points_num $power_flag >> run_host.txt
done

cp ../mmblk_sdcard/fpga_odom_knn_dataset/FPGA_ODOM_KNN_RESULT.txt ./