
#pragma once

// 没有下面这两句，cosim 会报错。 /include/file.h:244:2: error: ‘__gmp_const’ does not name a type
#include <gmp.h> 
#define __gmp_const const


// 重要宏定义 通过宏定义来切换不同功能。
#define DEBUG			// 调试 宏定义
// #define RUN_SW			// 运行软件版本
// #define EST_POWER		// 测量功耗； 如果测量需要打开此宏定义。
// #define _USE_OPENCL_		// 编译文件中定义; 是否使用 OPENCL 
//#define USE_FLOAT		// 使用浮点数时打开，否则使用定点数。
#define USE_PCL_ON		// 使用 PCL 功能; 引用库， 调用 kdtree

// LOAM 本地切换区; 用于嵌入 LOAM 完整工程中
#define DEF_LOCAL_TEST
#ifdef __SYNTHESIS__
#undef DEBUG
#undef USE_PCL_ON
#endif

// 使用点云及kdtree库函数
#ifdef USE_PCL_ON
#include <nanoflann_pcl.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/kdtree.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#endif


#include <sys/time.h>
#include <iostream>

#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <cmath>

#include "timer.h"	// 计时头文件

// vitis hls 相关库文件
#include "ap_fixed.h"	//ap_fixed<18,6,AP_TRN_ZERO,AP_SAT>		<W,I,Q,O,N>
#include "ap_int.h"	//ap_int<N> or ap_uint<N>, 1<=N<=1024
#include "hls_math.h"	//data_t s = hls::sinf(angle);
#include "hls_stream.h"


// 测量功耗头文件
#ifdef EST_POWER
#include "power_monitor_104_api.h"
#endif

// 数据类型定义区

// #define USE_FLOAT

#ifdef USE_FLOAT
#define my_sqrt sqrt
typedef float type_point;

typedef float type_point_hw;
typedef float type_intensity_hw;
typedef float type_range_hw;
typedef float type_angle_hw;
typedef float type_radian_hw;
typedef float type_curvature_hw;
typedef float type_temp_hw;

typedef ap_uint<1> type_flag_hw;
typedef float type_dist_hw;		// 暂存的数据。。保存那些杂糅的数据。

#else
///*
#define my_sqrt hls::sqrt
typedef float type_point;

typedef ap_uint<1> type_flag_hw;

// 保留三位小数，也就是10位。整数也只给10位，正负 516。只用来表示点以及加减，不涉及其他的。
typedef ap_fixed<32, 10> type_point_hw; //209微降准确率

typedef ap_fixed<32, 20> type_dist_hw; //2816微降准确率,保存点之间距离

#endif

typedef ap_uint<5> uint_sub_voxel_size;
typedef ap_uint<23> count_uint;	//800+w 遍历参考集的点
typedef ap_uint<15> uint_query_size;	//16000*2 遍历查询集的点

typedef ap_uint<8> k_selection_int;	//256

typedef ap_uint<7> uint_128;	//128

typedef ap_uint<21> voxel_int;	//2097152 遍历voxel

typedef ap_int<22> voxel_sint;	//2097152*2 有负数，voxel间索引差

typedef ap_uint<6> uint_64;

typedef ap_uint<2> uint_4;

typedef ap_uint<32> indexint;

// LOAM 本地切换区; 用于嵌入 LOAM 完整工程中
#ifndef DEF_LOCAL_TEST
#include "LOAM_useful_functions.h"
#else

#ifdef USE_PCL_ON
	typedef pcl::PointXYZI PointType;
	typedef pcl::PointXYZI PointT;
	typedef pcl::PointCloud<PointT> PointCloud;
#endif

	#ifdef DEBUG
	#define DEBUG_LOG(x) std::cout << "[DEBUG] " << x << std::endl
	#define DEBUG_INFO(x) std::cout << "[INFO] " << x << std::endl
	#define DEBUG_ERROR(x) std::cout << "[ERROR] " << x << std::endl
	#define DEBUG_TIME(x) std::cout << "[TIME] " << x << std::endl
	#define DEBUG_OPERATION(x) x
	#define DEBUG_GETCHAR
	// #define DEBUG_GETCHAR getchar()
	#else
	#define DEBUG_LOG(x) /*x*/
	#define DEBUG_INFO(x) std::cout << "[INFO] " << x << std::endl
	#define DEBUG_ERROR(x) std::cout << "[ERROR] " << x << std::endl
	#define DEBUG_TIME(x) std::cout << "[TIME] " << x << std::endl
	#define DEBUG_GETCHAR 
	#define DEBUG_OPERATION(x)
	#endif
	typedef struct FourDimPoint {
		float x;
		float y;
		float z;
		float intensity;
	} My_PointXYZI;

#endif

// 硬件 点 数据类型
typedef struct FourDimPointHW {
	type_point_hw x;
	type_point_hw y;
	type_point_hw z;
    //type_intensity_hw intensity;
} My_PointXYZI_HW;

//16个点打包
typedef struct FourDimPointHW16 {
	type_point_hw p1;
	type_point_hw p2;
	type_point_hw p3;
	type_point_hw p4;
	type_point_hw p5;
	type_point_hw p6;
	type_point_hw p7;
	type_point_hw p8;
	type_point_hw p9;
	type_point_hw p10;
	type_point_hw p11;
	type_point_hw p12;
	type_point_hw p13;
	type_point_hw p14;
	type_point_hw p15;
	type_point_hw p16;
    //type_intensity_hw intensity;
} My_PointXYZI_HW16;

//16个点的索引打包
typedef struct int16 {
	int p1;
	int p2;
	int p3;
	int p4;
	int p5;
	int p6;
	int p7;
	int p8;
	int p9;
	int p10;
	int p11;
	int p12;
	int p13;
	int p14;
	int p15;
	int p16;
    //type_intensity_hw intensity;
} inthw16;

typedef struct MaxMin {
	type_point xmin;
	type_point xmax;
	type_point ymin;
	type_point ymax;
	type_point zmin;
	type_point zmax;
} My_MaxMin;



// 参数定义区
static const int k_data_max_value_abs = 500;

static const int k_reference_set_size = 7000000;
static const int k_query_set_size = 20000;
static const int k_query_set_size_max = 7296;
static const int k_query_set_size_min = 1000;

static const int k_axis_voxel_max = 500;			//the biggest size for splitting the axis  100*100*100 = 100 0000... need to be restrained
static const int k_voxels_number_max = 1500000; //max cell numebrs; =k_data_set_size/k_ideal_cell_size

static const int k_data_set_buffer_size = 2000;		//max size of dataset buffer

static const int k_dataset_buffer_gap = 300;			//buffer_gap = 1/3* k_data_set_buffer_size

static const int k_nearest_number_max = 1;		//K_MAX = 10//*********if change this, sum_index in search function need to be modified... 
static const int k_max_split_precision = 20;	//maximum of split_precision
static const int k_select_loop_num = 300;	// max number of searching voxels or sub-voxels

static const int k_near_voxel_size = 27;	//max numebr of neighboring voxels
static const int k_transform_neighbor_num = 48;		//influence the accuracy.. but also influence little to performance.
static const int k_over_thre_sub_voxel_num = 34;

static const int k_sub_voxel_x_size = 4;
static const int k_sub_voxel_y_size = 2;	//tobe_added_sub_hash_increment
static const int k_sub_voxel_z_size = 4;
static const int k_sub_voxel_size = 32;
static const int k_sub_voxel_number_max = 1600000;
static const float K_SQ_THRESHOLD = 1;			// KNN 阈值，确保小于此值的数据可以被搜索到
static const int K = 1;							// KNN 的 K。
static const int bundle = 16; //打包点的个数
static const int voxel_points = 347409; //todo
static const int bundlevoxel = 17;
static const int num16 = 3; //48/16

static const int pararead = 4; //流式从片外读进buffer的并行度

static const int pip = 8; //48个并行找knn的话fanout太大，频率上不去，分为48选8和8选1两层在两个周期内完成

// 软件函数区； 用于与 LOAM 完整工程对接。
#ifdef USE_PCL_ON

// 做 kdtree 搜索
void kdtree_search(My_PointXYZI* KNN_query_set, int KNN_query_set_size, PointCloud::Ptr kdtree_point_cloud, nanoflann::KdTreeFLANN<PointT> & kdtreeFromMap, My_PointXYZI* image_KNN_query_result);
// 将点云数组转成 pointcloud
void transform_array_to_pointcloud(My_PointXYZI* point_set, int point_set_size,  PointCloud::Ptr & out_point_cloud);

#endif

// 生成 KNN 测试数据集。即均匀生成不同数目的点云。
void generate_test_datasets(std::string input_file_name, std::string output_file_name, int required_point_size);
// 输出参数结果到外部文件。
void file_out_result(std::string file_dir, int reference_set_num, int reference_point_num, int query_set_num, int query_point_num, int measure_power_flag, 
                    double imageNN_SW_build_time, double imageNN_SW_search_time, double kdtree_build_time, double kd_tree_search_time, double imageNN_HW_build_time, double imageNN_HW_search_time, 
                    double imageNN_SW_build_power, double imageNN_SW_search_power, double kdtree_build_power, double kd_tree_search_power, double imageNN_HW_build_power, double imageNN_HW_search_power,
                    double image_kdtree_accuracy, double image_sw_hw_accuracy);
// 从 txt 读取点云到数组中
void read_points_from_txt(std::string file_name, My_PointXYZI* laserCloudInArray, int & point_size);

// 暴力搜索，做csim, cosim 的真值
void brute_force_search(My_PointXYZI* KNN_query_set, int KNN_query_set_size, My_PointXYZI* range_image_reference_set, int range_image_reference_set_size, My_PointXYZI* bf_KNN_query_result);

// 比较不同 KNN 结果。
void compare_result(My_PointXYZI* KNN_query_set, int KNN_query_set_size, My_PointXYZI* image_KNN_query_result, My_PointXYZI* bf_KNN_query_result, int & error_count);



void calculate_hash(My_PointXYZI* KNN_reference_set, int* data_set_hash, int KNN_reference_set_size);

void calculate_subhash(My_PointXYZI* KNN_reference_set, int* data_hash, int* data_set_sub_hash, int KNN_reference_set_size);

void count_hash(int* data_hash, int* count_voxel_size, int KNN_reference_set_size);

void cal_hash_first_index(int* first_index, int* count_voxel_size);

void subdivide_data_set(int* voxel_first_index_PL, int* sub_voxel_flag_index_PL, int& sub_voxel_first_index_sentry, int& bigger_voxel_number);

void calculate_split_data_set_hash(int KNN_reference_set_size, int* data_set_hash, int* data_set_sub_hash, int* sub_voxel_flag_index_PL, int* sub_voxel_size);

void cal_sub_voxel_first_index(int* sub_voxel_flag_index_PL, int* sub_voxel_first_index_PL, int* voxel_first_index_PL, int sub_voxel_first_index_sentry);

void cal_sub_voxel_first_index2(int* sub_voxel_first_index_PL, int* sub_voxel_size, int sub_voxel_first_index_sentry);

void reorder_data_set_reference(int* original_dataset_index, int KNN_reference_set_size, int* data_set_hash, int* sub_voxel_flag_index_PL, int* data_set_sub_hash, int* voxel_occupied_number, int* sub_voxel_occupied_number, int* voxel_first_index_PL, int* sub_voxel_first_index_PL, int* ordered_DSVS, My_PointXYZI* ordered_query, My_PointXYZI* KNN_reference_set);

void reorder_data_set(int* original_dataset_index, int KNN_reference_set_size, int* data_set_hash, int* voxel_occupied_number, int* query_set_first_index, int* ordered_DSVS, My_PointXYZI* ordered_query, My_PointXYZI* KNN_reference_set);

void reorder_query(int* original_dataset_index, int KNN_reference_set_size, int* data_set_hash, int* voxel_occupied_number, int* query_set_first_index, int* ordered_DSVS, My_PointXYZI* ordered_query, My_PointXYZI* KNN_reference_set);

float cal_dist(struct FourDimPointHW data1, struct FourDimPointHW data2);

void DSVS_build(int* original_dataset_index, My_PointXYZI* KNN_reference_set, int KNN_reference_set_size, type_flag_hw sharp_flag, int* ordered_DSVS, My_PointXYZI* ordered_query, int* voxel_first_index_PL, int* sub_voxel_flag_index_PL, int* sub_voxel_first_index_PL, bool reorder_query_set, My_PointXYZI_HW16* ordered_ref16_x, My_PointXYZI_HW16* ordered_ref16_y, My_PointXYZI_HW16* ordered_ref16_z, inthw16* original_dataset_index16, indexint* index16, indexint* subindex16, int& packs);

void initial_buffer(My_PointXYZI query_set, int* voxel_first_index_PL, My_PointXYZI* data_set, int* original_dataset_index, int& dataset_buffer_max_index_PL, int& dataset_buffer_min_index_PL);

void DSVS_search(My_PointXYZI* data_set, int data_set_size,
	int* original_dataset_index,
	My_PointXYZI* query_set, int query_set_size,
    int* query_result, float* nearest_distance,
	int* voxel_first_index_PL, int* sub_voxel_flag_index_PL, int* sub_voxel_first_index_PL, int dataset_buffer_max_index_PL, int dataset_buffer_min_index_PL);

void DSVS_search_hw(My_PointXYZI_HW16* ordered_ref16_x, My_PointXYZI_HW16* ordered_ref16_y, My_PointXYZI_HW16* ordered_ref16_z,
	inthw16* original_dataset_index16,
	My_PointXYZI_HW* query_set, int query_set_size,
    count_uint* query_result, type_dist_hw* nearest_distance,
	indexint* index16, voxel_int* sub_voxel_flag_index_PL, indexint* subindex16);

void get_min_max(My_PointXYZI* data_set, int data_set_size, type_point_hw& data_set_max_min_PL_xmin_hw, type_point_hw& data_set_max_min_PL_ymin_hw, type_point_hw& data_set_max_min_PL_zmin_hw);

void split_voxel_AABB(float input_split_unit, voxel_int& voxel_split_array_size_PL_x_array_size_hw, voxel_int& voxel_split_array_size_PL_y_array_size_hw, voxel_int& voxel_split_array_size_PL_z_array_size_hw, voxel_int& total_calculated_voxel_size_hw);

void setup_hardware_PL(My_PointXYZI* KNN_reference_set, int KNN_reference_set_size, type_point_hw& data_set_max_min_PL_xmin_hw, type_point_hw& data_set_max_min_PL_ymin_hw, type_point_hw& data_set_max_min_PL_zmin_hw, type_point_hw& voxel_split_unit_PL_hw, voxel_int& voxel_split_array_size_PL_x_array_size_hw, voxel_int& voxel_split_array_size_PL_y_array_size_hw, voxel_int& voxel_split_array_size_PL_z_array_size_hw, voxel_int& total_calculated_voxel_size_hw);
