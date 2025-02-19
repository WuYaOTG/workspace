#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "knn_odom.h"
#include "timer.h"

#ifdef _USE_OPENCL_
#include "xcl2.hpp"
#endif

typedef struct Power_Struct {
	double full_power_domain_power;
	double low_power_domain_power;
	double pl_domain_power;
} My_Power_Struct;

class FPGA_HOST{
    public:
        int fpga_host_init(std::string binaryFile);
        int fpga_host_allocate(int measure_power_flag_in);
        int fpga_host_run(My_PointXYZI_HW16* ordered_ref16_x, My_PointXYZI_HW16* ordered_ref16_y, My_PointXYZI_HW16* ordered_ref16_z, inthw16* original_dataset_index16,
	My_PointXYZI_HW* query_set, int query_set_size, count_uint* query_result, type_dist_hw* nearest_distance,
	indexint* index16, voxel_int* sub_voxel_flag_index_PL, indexint* subindex16);
        int fpga_host_deallocateMemory();
        int fpga_host_print_report();

        double get_run_time();

        int measure_static_power(int sleep_time, bool report);
        int start_measure_run_power();
        int end_measure_run_power(bool report);
        int report_run_power();
        int report_static_power();
    
    private:

        double run_kernel_time;
        double run_pure_kernel_time;
        double last_report_time[7];
        double total_run_energy, total_run_time, average_run_power;
        std::string kernel_record_file = "rps_fpga_result.txt";
	std::ofstream fout_fpga_record;	

        int measure_power_flag = 0;

#ifdef EST_POWER
 
        struct energy_sample* sample_static;
	struct energy_sample* sample_run;
        int sample_rate = 1;	// sample rate in miliseconds
	int debug_output_file = 1;
	struct timespec res_static, res_run;

        My_Power_Struct static_power = {0,0,0};
	My_Power_Struct run_power = {0,0,0};

#endif

#ifdef _USE_OPENCL_
        cl::Context m_context;
        cl::CommandQueue m_q;
        cl::Program m_prog;
        cl::Kernel krnl_DSVS;

        cl::Buffer buffer_ordered_ref16_x;
        cl::Buffer buffer_ordered_ref16_y;
        cl::Buffer buffer_ordered_ref16_z;
        cl::Buffer buffer_original_dataset_index16; 
        cl::Buffer buffer_query_set; 
        cl::Buffer buffer_query_result;
        cl::Buffer buffer_nearest_distance;
        cl::Buffer buffer_index16;
        cl::Buffer buffer_sub_voxel_flag_index_PL;
        cl::Buffer buffer_subindex16;
#endif
        My_PointXYZI_HW16 *ptr_ordered_ref16_x;
        My_PointXYZI_HW16 *ptr_ordered_ref16_y;
        My_PointXYZI_HW16 *ptr_ordered_ref16_z;
        inthw16 *ptr_original_dataset_index16; 
        My_PointXYZI_HW *ptr_query_set; 
        count_uint *ptr_query_result;
        type_dist_hw *ptr_nearest_distance;
        indexint* ptr_index16;
        voxel_int* ptr_sub_voxel_flag_index_PL;
        indexint* ptr_subindex16;
};

// void wrap_imageNN_all_hw(
//     int select_func, 
//     My_ImageNN * KNN_range_image, int range_image_reference_set_size, type_flag_hw sharp_flag, My_ImageNN * PCM_hw, int * index_PCM_hw, 
//     My_PointXYZI* KNN_query_set, int KNN_query_set_size, My_ImageNN* image_KNN_query_result, FPGA_HOST &fpga_host_ins);

void wrap_DSVS_search_hw(My_PointXYZI_HW16* ordered_ref16_x, My_PointXYZI_HW16* ordered_ref16_y, My_PointXYZI_HW16* ordered_ref16_z, inthw16* original_dataset_index16,
	My_PointXYZI_HW* query_set, int query_set_size, count_uint* query_result, type_dist_hw* nearest_distance,
	indexint* index16, voxel_int* sub_voxel_flag_index_PL, indexint* subindex16);