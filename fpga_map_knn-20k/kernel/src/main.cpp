#include <iostream>
#include <fstream>
#include <string>
#include <sstream> 
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cstdlib>// Header file needed to use rand
#include <ctime> // Header file needed to use time
#include <iomanip>
#undef __ARM_NEON__
#undef __ARM_NEON
#include "knn_odom.h"
#include "fpga_host_class.h"
#define __ARM_NEON__
#define __ARM_NEON

#define TEST_ALL

int main (int argc, char **argv) {

    /****************************************** 0. 变量定义 ******************************/
    // 输入点云 参考点集 及 搜索集
    My_PointXYZI KNN_reference_set[k_reference_set_size];
    My_PointXYZI KNN_query_set[k_query_set_size];

    int reference_set_num = 0;    // 参考数据集帧索引，可通过命令行设置
    int reference_point_num = 20;   // 参考数据集 点个数，单位 k ，可通过命令行设置
    int query_set_num = 0;        // 搜索数据集帧索引，可通过命令行设置
    int query_point_num = 1010;        // 搜索数据集帧索引，单位 k ，可通过命令行设置
    type_flag_hw sharp_flag = 0;    // 是 sharp 还是 flat 点; 
    int measure_power_flag = 0;     // 测量功耗 标志位

    std::string pointcloud_dir = "/home/sunhao/hyh/data/";

    std::cout << "Usage: ./LOAM_KNN_MAP [data_dir] [refe_label] [refe_size] [query_label] [query_size] [power_flag]" << std::endl;
    std::cout << "Default: ./LOAM_KNN_MAP /home/sunhao/LOAM_FPGA/fpga_map_knn/data/ 0 500 0 3 0" << std::endl;
    std::cout << "Avoid Segmentation fault (core dumped) : ulimit -s unlimited" << std::endl;

    // 读取命令行参数
    if(argc > 1)
        pointcloud_dir = argv[1];

    if(argc > 2)
        reference_set_num = atoi(argv[2]);

    if(argc > 3)
        reference_point_num = atoi(argv[3]);
    
    if(argc > 4)
        query_set_num = atoi(argv[4]);
    
    if(argc > 5)
        query_point_num = atoi(argv[5]);
    
    if(argc > 6)
        measure_power_flag = atoi(argv[6]);

    DEBUG_LOG("Input pointcloud dir: " << pointcloud_dir);

    /****************************************** 1. 从文件中读取点云数据 ******************************/
    std::string KNN_reference_set_file_name;
    std::string KNN_query_set_file_name;

    KNN_reference_set_file_name = pointcloud_dir + "dataset_map_" + std::to_string(reference_set_num) + "_" + std::to_string(reference_point_num) + "k_points.txt";
    KNN_query_set_file_name = pointcloud_dir + "dataset_query_" + std::to_string(query_set_num)  + "_" + std::to_string(query_point_num) + "k_points.txt";

    DEBUG_INFO("KNN_reference_set_file_name: " << KNN_reference_set_file_name);
    DEBUG_INFO("KNN_query_set_file_name: " << KNN_query_set_file_name);

    // 从点云文件读取点云
    int KNN_reference_set_size, KNN_query_set_size;
    DEBUG_LOG("reading KNN_reference_set ......");
    read_points_from_txt(KNN_reference_set_file_name, KNN_reference_set, KNN_reference_set_size);
    read_points_from_txt(KNN_query_set_file_name, KNN_query_set, KNN_query_set_size);
    DEBUG_LOG("before order KNN_query_set_size: " << KNN_query_set_size);

    DEBUG_LOG("KNN_query_set_size: " << KNN_query_set_size);
    DEBUG_LOG("KNN_reference_set_size: " << KNN_reference_set_size);

    /********************************************** 2. 建立 数据结构 *******************************************************/

TIMER_INIT(7);
TIMER_START(0);

    int ordered_DSVS[k_reference_set_size];
    int voxel_first_index_PL[k_voxels_number_max];
    int sub_voxel_flag_index_PL[k_voxels_number_max];
    int sub_voxel_first_index_PL[k_sub_voxel_number_max];
    int original_dataset_index[k_reference_set_size];

    inthw16 original_dataset_index16[voxel_points];

    My_PointXYZI_HW16 ordered_ref16_x[voxel_points];
    My_PointXYZI_HW16 ordered_ref16_y[voxel_points];
    My_PointXYZI_HW16 ordered_ref16_z[voxel_points];

    My_PointXYZI ordered_query[k_query_set_size];
    My_PointXYZI ordered_ref[k_reference_set_size];
    indexint index16[k_voxels_number_max];
    indexint subindex16[k_voxels_number_max];

    type_point_hw data_set_max_min_PL_xmin_hw;
    type_point_hw data_set_max_min_PL_ymin_hw;
    type_point_hw data_set_max_min_PL_zmin_hw;
    type_point_hw voxel_split_unit_PL_hw;
    voxel_int voxel_split_array_size_PL_x_array_size;
    voxel_int voxel_split_array_size_PL_y_array_size;
    voxel_int voxel_split_array_size_PL_z_array_size;
    voxel_int total_calculated_voxel_size;
    int packs; //16个点一组，packs是组的数量

    bool reorder_query_set = 1;
    
    setup_hardware_PL(KNN_reference_set, KNN_reference_set_size, data_set_max_min_PL_xmin_hw, data_set_max_min_PL_ymin_hw, data_set_max_min_PL_zmin_hw, voxel_split_unit_PL_hw, voxel_split_array_size_PL_x_array_size, voxel_split_array_size_PL_y_array_size, voxel_split_array_size_PL_z_array_size, total_calculated_voxel_size);
    
    DEBUG_INFO(data_set_max_min_PL_xmin_hw);
    DEBUG_INFO(data_set_max_min_PL_ymin_hw);
    DEBUG_INFO(data_set_max_min_PL_zmin_hw);
    DEBUG_INFO(voxel_split_unit_PL_hw);
    DEBUG_INFO(voxel_split_array_size_PL_x_array_size);
    DEBUG_INFO(voxel_split_array_size_PL_y_array_size);
    DEBUG_INFO(voxel_split_array_size_PL_z_array_size);
    DEBUG_INFO(total_calculated_voxel_size);

//query排序
    DSVS_build(original_dataset_index, KNN_query_set, KNN_query_set_size, sharp_flag, ordered_DSVS, ordered_query, voxel_first_index_PL, sub_voxel_flag_index_PL, sub_voxel_first_index_PL, reorder_query_set, ordered_ref16_x, ordered_ref16_y, ordered_ref16_z, original_dataset_index16, index16, subindex16, packs);

//reference排序
    reorder_query_set = 0;
    DSVS_build(original_dataset_index, KNN_reference_set, KNN_reference_set_size, sharp_flag, ordered_DSVS, ordered_ref, voxel_first_index_PL, sub_voxel_flag_index_PL, sub_voxel_first_index_PL, reorder_query_set, ordered_ref16_x, ordered_ref16_y, ordered_ref16_z, original_dataset_index16, index16, subindex16, packs);

    DEBUG_INFO(packs);

TIMER_STOP_ID(0);
DEBUG_TIME("Finish software building DSVS with reference points " << KNN_reference_set_size << " with " << TIMER_REPORT_MS(0) << " ms !" );
// getchar();

#ifdef USE_PCL_ON
TIMER_START(1);
    // 2.1 建立 kdtree
    nanoflann::KdTreeFLANN<PointT> kdtreeFromMap;
    PointCloud::Ptr KNN_reference_set_cloud(new PointCloud);
    transform_array_to_pointcloud(KNN_reference_set, KNN_reference_set_size, KNN_reference_set_cloud);
    // DEBUG_LOG("2 KNN_reference_set_size: " << KNN_reference_set_size << " KNN_reference_set_cloud.size() " << KNN_reference_set_cloud->points.size() );
    kdtreeFromMap.setInputCloud(KNN_reference_set_cloud);
TIMER_STOP_ID(1);
DEBUG_TIME("Finish building kdtree with reference points " << KNN_reference_set_size << " with " << TIMER_REPORT_MS(1) << " ms !" );
#endif

    /********************************************** 3. KNN 搜索 *******************************************************/

TIMER_START(2);    
    My_PointXYZI DSVS_query_result[k_query_set_size];

    My_PointXYZI_HW ordered_query_hw[k_query_set_size];
    count_uint DSVS_query_result_index_hw[k_query_set_size];
    type_dist_hw nearest_distance_hw[k_query_set_size];
    voxel_int sub_voxel_flag_index_PL_hw[k_voxels_number_max];

    for (int i = 0; i < KNN_query_set_size; i++)
    {
        ordered_query_hw[i].x = ordered_query[i].x;
        ordered_query_hw[i].y = ordered_query[i].y;
        ordered_query_hw[i].z = ordered_query[i].z;
    }

    for (int i = 0; i < k_voxels_number_max; i++)
    {
        sub_voxel_flag_index_PL_hw[i] = sub_voxel_flag_index_PL[i];
    }

#ifdef _USE_OPENCL_
    wrap_DSVS_search_hw(ordered_ref16_x, ordered_ref16_y, ordered_ref16_z, original_dataset_index16,
	 		ordered_query_hw, KNN_query_set_size,
	 		DSVS_query_result_index_hw, nearest_distance_hw,
            index16, sub_voxel_flag_index_PL_hw, subindex16);
#else
    
    DSVS_search_hw(ordered_ref16_x, ordered_ref16_y, ordered_ref16_z, original_dataset_index16,
	 		ordered_query_hw, KNN_query_set_size,
	 		DSVS_query_result_index_hw, nearest_distance_hw,
            index16, sub_voxel_flag_index_PL_hw, subindex16);

#endif
    for (int i = 0; i < KNN_query_set_size; i++)
    {
        DSVS_query_result[i] = KNN_reference_set[DSVS_query_result_index_hw[i]];
    }

TIMER_STOP_ID(2);
DEBUG_TIME("Finish search DSVS with query points " << KNN_query_set_size << " with " << TIMER_REPORT_MS(2) << " ms !" );
#ifdef USE_PCL_ON
    // kdtree 搜索
    My_PointXYZI KDTREE_query_result[k_query_set_size];
    kdtree_search(ordered_query, KNN_query_set_size, KNN_reference_set_cloud, kdtreeFromMap, KDTREE_query_result);  // 基于 kdtree 进行搜索。
#else
    //  brute force 搜索
    My_PointXYZI bf_KNN_query_result[k_query_set_size];
    brute_force_search(ordered_query, KNN_query_set_size, KNN_reference_set, KNN_reference_set_size, bf_KNN_query_result);  // 暴力 搜索。
#endif

    /********************************************** 4. KNN 结果比较 *******************************************************/
    int error_count = 0;
    float NN_true_ratio;
    
    // DEBUG_LOG("********************************** CPMPARE CLOSEST NN RESULT **********************************");
#ifdef USE_PCL_ON 
    compare_result(ordered_query, KNN_query_set_size, DSVS_query_result, KDTREE_query_result, error_count);
    DEBUG_INFO("image_KNN VS kdtree: " << ", error_count: " << error_count << " with true ratio = " << float(1-(float)error_count/(float)KNN_query_set_size));
#else
    compare_result(ordered_query, KNN_query_set_size, DSVS_query_result, bf_KNN_query_result, error_count);
    DEBUG_LOG("image_KNN VS brute-force: " << ", error_count: " << error_count << " with true ratio = " << float(1-(float)error_count/(float)KNN_query_set_size));
#endif
    NN_true_ratio = float(1-(float)error_count/(float)KNN_query_set_size);

    /********************************************** 5. 验证结果 *******************************************************/
    if (NN_true_ratio < 0.01)
    {
        DEBUG_INFO("NN_true_ratio : " << NN_true_ratio);
        fprintf(stdout, "*******************************************\n");
        fprintf(stdout, "FAIL: Output DOES NOT match the golden output\n");
        fprintf(stdout, "*******************************************\n");
        return 1;
    } 
    else 
    {
        DEBUG_INFO("NN_true_ratio: " << NN_true_ratio);
        fprintf(stdout, "*******************************************\n");
        fprintf(stdout, "PASS: The output matches the golden output!\n");
        fprintf(stdout, "*******************************************\n");
        return 0;
    }
}