#pragma once

#include "fpga_host_class.h"
#include "knn_odom.h"

const int REPEAT_TIME = 10;
TIMER_INIT(7);

void wrap_DSVS_search_hw(My_PointXYZI_HW16* ordered_ref16_x, My_PointXYZI_HW16* ordered_ref16_y, My_PointXYZI_HW16* ordered_ref16_z, inthw16* original_dataset_index16,
	My_PointXYZI_HW* query_set, int query_set_size, count_uint* query_result, type_dist_hw* nearest_distance,
	indexint* index16, voxel_int* sub_voxel_flag_index_PL, indexint* subindex16)
{
#ifdef _USE_OPENCL_
    FPGA_HOST fpga_host_ins;
    DEBUG_LOG( "init the fpga" );
    fpga_host_ins.fpga_host_init("kernel.xclbin");
    DEBUG_LOG( "allocate the memory " );
    int measure_power_flag = 1;
    fpga_host_ins.fpga_host_allocate(measure_power_flag);
    DEBUG_LOG( "finished allocate the memory! " );

    fpga_host_ins.fpga_host_run(ordered_ref16_x, ordered_ref16_y, ordered_ref16_z, original_dataset_index16, query_set, query_set_size, query_result, nearest_distance,
	index16, sub_voxel_flag_index_PL, subindex16);

    DEBUG_LOG("to host print" );
    fpga_host_ins.fpga_host_print_report();

    fpga_host_ins.fpga_host_deallocateMemory();
#endif
}

int FPGA_HOST::fpga_host_init(std::string binaryFile) {

    for(int i = 0; i < 7; i++)
        last_report_time[i] = 0;

    TIMER_START(5);
#ifdef _USE_OPENCL_
    cl_int err;

    // get_xil_devices() is a utility API which will find the xilinx
    // platforms and will return list of devices connected to Xilinx platform
    auto devices = xcl::get_xil_devices();
    // read_binary_file() is a utility API which will load the binaryFile
    // and will return the pointer to file buffer.
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, m_context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err,
                  m_q = cl::CommandQueue(m_context, device,
                                         CL_QUEUE_PROFILING_ENABLE, &err));

        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        m_prog = cl::Program(m_context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    std::string kernel_name = "DSVS_search_hw";
    OCL_CHECK(err, krnl_DSVS = cl::Kernel(m_prog, kernel_name.c_str(), &err));

    // remove(kernel_record_file.c_str());	
    fout_fpga_record.open(kernel_record_file.c_str(), std::ios::app);
     std::cout << "open kernel_record_file~" << std::endl;
    if (fout_fpga_record.is_open() == false)
    {
        std::cout << "Cannot Open :" << kernel_record_file << " For writing output data...." << std::endl;
        std::cout << "Exiting...." << std::endl;
        return -1;
    }
    std::cout << "kernel_record_file: " << kernel_record_file << std::endl;
#endif
    TIMER_STOP(5);

    return 0;
}

int FPGA_HOST::fpga_host_allocate(int measure_power_flag_in) {
    TIMER_START(6);

    measure_power_flag = measure_power_flag_in;

    DEBUG_INFO("measure_power_flag: " << measure_power_flag);

    if(measure_power_flag > 0)
    {
        measure_static_power(0, false);
        report_static_power();
    }

#ifdef _USE_OPENCL_
DEBUG_LOG("before allocate buffer size.");
    cl_int err;
    OCL_CHECK(err, buffer_ordered_ref16_x = cl::Buffer(m_context, CL_MEM_READ_ONLY, sizeof(My_PointXYZI_HW16) * 34044, nullptr, &err));
    OCL_CHECK(err, buffer_ordered_ref16_y = cl::Buffer(m_context, CL_MEM_READ_ONLY, sizeof(My_PointXYZI_HW16) * 34044, nullptr, &err));
    OCL_CHECK(err, buffer_ordered_ref16_z = cl::Buffer(m_context, CL_MEM_READ_ONLY, sizeof(My_PointXYZI_HW16) * 34044, nullptr, &err));
    OCL_CHECK(err, buffer_original_dataset_index16 = cl::Buffer(m_context, CL_MEM_READ_ONLY, sizeof(inthw16) * 34044, nullptr, &err));
    OCL_CHECK(err, buffer_query_set = cl::Buffer(m_context, CL_MEM_READ_ONLY, sizeof(My_PointXYZI_HW) * 10000, nullptr, &err));
    OCL_CHECK(err, buffer_query_result = cl::Buffer(m_context, CL_MEM_WRITE_ONLY, sizeof(count_uint) * 10000, nullptr, &err));
    OCL_CHECK(err, buffer_nearest_distance = cl::Buffer(m_context, CL_MEM_WRITE_ONLY, sizeof(type_dist_hw) * 10000, nullptr, &err));
    OCL_CHECK(err, buffer_index16 = cl::Buffer(m_context, CL_MEM_READ_ONLY, sizeof(indexint) * 1303400, nullptr, &err));
    OCL_CHECK(err, buffer_sub_voxel_flag_index_PL = cl::Buffer(m_context, CL_MEM_READ_ONLY, sizeof(voxel_int) * 1303400, nullptr, &err));
    OCL_CHECK(err, buffer_subindex16 = cl::Buffer(m_context, CL_MEM_READ_ONLY, sizeof(indexint) * 8, nullptr, &err));

DEBUG_LOG("end allocate buffer size.");
#else

    ptr_ordered_ref16_x = (My_PointXYZI_HW16*)malloc(34044* sizeof(My_PointXYZI_HW16));
    ptr_ordered_ref16_y = (My_PointXYZI_HW16*)malloc(34044* sizeof(My_PointXYZI_HW16));
    ptr_ordered_ref16_z = (My_PointXYZI_HW16*)malloc(34044* sizeof(My_PointXYZI_HW16));
    ptr_original_dataset_index16 = (inthw16*)malloc(34044 * sizeof(inthw16));
    ptr_query_set = (My_PointXYZI_HW*)malloc(10000 * sizeof(My_PointXYZI_HW));
    ptr_query_result = (count_uint*)malloc(10000 * sizeof(count_uint));
    ptr_nearest_distance = (type_dist_hw*)malloc(10000 * sizeof(type_dist_hw));
    ptr_index16 = (indexint*)malloc(1303400 * sizeof(indexint));
    ptr_sub_voxel_flag_index_PL = (voxel_int*)malloc(1303400 * sizeof(voxel_int));
    ptr_subindex16 = (indexint*)malloc(8 * sizeof(indexint));

#endif
    TIMER_STOP(6);

    return 0;
}

int FPGA_HOST::fpga_host_run(My_PointXYZI_HW16* ordered_ref16_x, My_PointXYZI_HW16* ordered_ref16_y, My_PointXYZI_HW16* ordered_ref16_z, inthw16* original_dataset_index16,
	My_PointXYZI_HW* query_set, int query_set_size, count_uint* query_result, type_dist_hw* nearest_distance,
	indexint* index16, voxel_int* sub_voxel_flag_index_PL, indexint* subindex16)
{

TIMER_START(0);

TIMER_START(1);

    {
#ifdef _USE_OPENCL_
DEBUG_LOG("before allocate search pointer size.");
        ptr_ordered_ref16_x = (My_PointXYZI_HW16 *) m_q.enqueueMapBuffer (buffer_ordered_ref16_x , CL_TRUE , CL_MAP_WRITE , 0, sizeof(My_PointXYZI_HW16) * 34044);
        ptr_ordered_ref16_y = (My_PointXYZI_HW16 *) m_q.enqueueMapBuffer (buffer_ordered_ref16_y , CL_TRUE , CL_MAP_WRITE , 0, sizeof(My_PointXYZI_HW16) * 34044);
        ptr_ordered_ref16_z = (My_PointXYZI_HW16 *) m_q.enqueueMapBuffer (buffer_ordered_ref16_z , CL_TRUE , CL_MAP_WRITE , 0, sizeof(My_PointXYZI_HW16) * 34044);
        ptr_original_dataset_index16 = (inthw16 *) m_q.enqueueMapBuffer (buffer_original_dataset_index16 , CL_TRUE , CL_MAP_WRITE , 0, sizeof(inthw16) * 34044);
        ptr_query_set = (My_PointXYZI_HW *) m_q.enqueueMapBuffer (buffer_query_set , CL_TRUE , CL_MAP_WRITE , 0, sizeof(My_PointXYZI_HW) * 10000);
        ptr_query_result = (count_uint *) m_q.enqueueMapBuffer (buffer_query_result , CL_TRUE , CL_MAP_READ , 0, sizeof(count_uint) * 10000);
        ptr_nearest_distance = (type_dist_hw *) m_q.enqueueMapBuffer (buffer_nearest_distance , CL_TRUE , CL_MAP_READ , 0, sizeof(type_dist_hw) * 10000);
        ptr_index16 = (indexint *) m_q.enqueueMapBuffer (buffer_index16 , CL_TRUE , CL_MAP_WRITE , 0, sizeof(indexint) * 1303400);
        ptr_sub_voxel_flag_index_PL = (voxel_int *) m_q.enqueueMapBuffer (buffer_sub_voxel_flag_index_PL , CL_TRUE , CL_MAP_WRITE , 0, sizeof(voxel_int) * 1303400);
        ptr_subindex16 = (indexint *) m_q.enqueueMapBuffer (buffer_subindex16 , CL_TRUE , CL_MAP_WRITE , 0, sizeof(indexint) * 8);
DEBUG_LOG("end allocate search pointer size.");
#endif
        for(int i = 0; i < 34044; i++)
        {
            ptr_ordered_ref16_x[i] = ordered_ref16_x[i];
            ptr_ordered_ref16_y[i] = ordered_ref16_y[i];
            ptr_ordered_ref16_z[i] = ordered_ref16_z[i];
            //ptr_data_set[i].intensity = data_set[i].intensity;
        }
        for(int i = 0; i < 34044; i++)
        {
            ptr_original_dataset_index16[i] = original_dataset_index16[i];
        }
        for(int i = 0; i < 10000; i++)
        {
            ptr_query_set[i].x = query_set[i].x;
            ptr_query_set[i].y = query_set[i].y;
            ptr_query_set[i].z = query_set[i].z;
            //ptr_query_set[i].intensity = query_set[i].intensity;
        }
        for(int i = 0; i < 1303400; i++)
        {
            ptr_index16[i] = index16[i];
        }
        for(int i = 0; i < 1303400; i++)
        {
            ptr_sub_voxel_flag_index_PL[i] = sub_voxel_flag_index_PL[i];
        }
        for(int i = 0; i < 8; i++)
        {
            ptr_subindex16[i] = subindex16[i];
        }
    }

    if(measure_power_flag > 0)
        start_measure_run_power();

#ifdef _USE_OPENCL_
    
    cl_int err;
    int narg = 0;
    OCL_CHECK(err, err = krnl_DSVS.setArg(narg++, buffer_ordered_ref16_x));
    OCL_CHECK(err, err = krnl_DSVS.setArg(narg++, buffer_ordered_ref16_y));
    OCL_CHECK(err, err = krnl_DSVS.setArg(narg++, buffer_ordered_ref16_z));
    OCL_CHECK(err, err = krnl_DSVS.setArg(narg++, buffer_original_dataset_index16));
    OCL_CHECK(err, err = krnl_DSVS.setArg(narg++, buffer_query_set));
    OCL_CHECK(err, err = krnl_DSVS.setArg(narg++, query_set_size));
    OCL_CHECK(err, err = krnl_DSVS.setArg(narg++, buffer_query_result));
    OCL_CHECK(err, err = krnl_DSVS.setArg(narg++, buffer_nearest_distance));
    OCL_CHECK(err, err = krnl_DSVS.setArg(narg++, buffer_index16));
    OCL_CHECK(err, err = krnl_DSVS.setArg(narg++, buffer_sub_voxel_flag_index_PL));
    OCL_CHECK(err, err = krnl_DSVS.setArg(narg++, buffer_subindex16));
    /*
    OCL_CHECK(err, err = krnl_DSVS.setArg(narg++, data_set_max_min_PL_xmin));
    OCL_CHECK(err, err = krnl_DSVS.setArg(narg++, data_set_max_min_PL_ymin));
    OCL_CHECK(err, err = krnl_DSVS.setArg(narg++, data_set_max_min_PL_zmin));
    OCL_CHECK(err, err = krnl_DSVS.setArg(narg++, voxel_split_array_size_PL_x_array_size_PL));
    OCL_CHECK(err, err = krnl_DSVS.setArg(narg++, voxel_split_array_size_PL_y_array_size_PL));
    OCL_CHECK(err, err = krnl_DSVS.setArg(narg++, voxel_split_array_size_PL_x_array_size_PL));
    OCL_CHECK(err, err = krnl_DSVS.setArg(narg++, total_calculated_voxel_size_PL));
    OCL_CHECK(err, err = krnl_DSVS.setArg(narg++, packs_PL));
    */
    // Data will be migrated to kernel space
    m_q.enqueueMigrateMemObjects({buffer_ordered_ref16_x, buffer_ordered_ref16_y, buffer_ordered_ref16_z, buffer_original_dataset_index16, buffer_query_set, buffer_index16, buffer_sub_voxel_flag_index_PL, buffer_subindex16},0/* 0 means from host*/);

TIMER_STOP; 

DEBUG_INFO("kernel start------" );

TIMER_START(2);

for(int i = 0; i < REPEAT_TIME; i++)
{
    //Launch the Kernel
    m_q.enqueueTask(krnl_DSVS);
}
    // The result of the previous kernel execution will need to be retrieved in order to view the results. This call will transfer the data from FPGA to source_results vector
    m_q.enqueueMigrateMemObjects({buffer_query_result, buffer_nearest_distance}, CL_MIGRATE_MEM_OBJECT_HOST);

    m_q.finish();
 
#else
// 运行正常的非opencl的代码。即krnl函数即可。

#endif 

    if(measure_power_flag > 0)
    {
        end_measure_run_power(false);
        report_run_power();
    }

TIMER_STOP;

DEBUG_INFO("kernel end------");  

TIMER_START(3);

    {
        for(int i = 0; i < query_set_size; i++)
        {
            query_result[i] = ptr_query_result[i];
            nearest_distance[i] = ptr_nearest_distance[i];
        }
#ifdef _USE_OPENCL_
        m_q.enqueueUnmapMemObject(buffer_ordered_ref16_x , ptr_ordered_ref16_x);
        m_q.enqueueUnmapMemObject(buffer_ordered_ref16_y , ptr_ordered_ref16_y);
        m_q.enqueueUnmapMemObject(buffer_ordered_ref16_z , ptr_ordered_ref16_z);
        m_q.enqueueUnmapMemObject(buffer_original_dataset_index16 , ptr_original_dataset_index16);
        m_q.enqueueUnmapMemObject(buffer_query_set , ptr_query_set);
        m_q.enqueueUnmapMemObject(buffer_query_result , ptr_query_result);
        m_q.enqueueUnmapMemObject(buffer_nearest_distance , ptr_nearest_distance);
        m_q.enqueueUnmapMemObject(buffer_index16 , ptr_index16);
        m_q.enqueueUnmapMemObject(buffer_sub_voxel_flag_index_PL , ptr_sub_voxel_flag_index_PL);
        m_q.enqueueUnmapMemObject(buffer_subindex16 , ptr_subindex16);

        m_q.finish();
#endif
    }

TIMER_STOP;

TIMER_STOP_ID(0);
    return 0;
}


int FPGA_HOST::fpga_host_deallocateMemory() {

TIMER_START(4);

#ifdef _USE_OPENCL_
    // m_q.enqueueUnmapMemObject(buffer_KNN_range_image , ptr_KNN_range_image);
    // m_q.enqueueUnmapMemObject(buffer_PCM_hw , ptr_PCM_hw);
    // m_q.enqueueUnmapMemObject(buffer_index_PCM_hw , ptr_index_PCM_hw);
    // m_q.enqueueUnmapMemObject(buffer_KNN_query_set , ptr_KNN_query_set);
    // m_q.enqueueUnmapMemObject(buffer_image_KNN_query_result , ptr_image_KNN_query_result);

    // m_q.finish();

#else
    free(ptr_ordered_ref16_x);
    free(ptr_ordered_ref16_y);
    free(ptr_ordered_ref16_z);
    free(ptr_original_dataset_index16);
    free(ptr_query_set);
    free(ptr_query_result);
    free(ptr_nearest_distance);
    free(ptr_index16);
    free(ptr_sub_voxel_flag_index_PL);
    free(ptr_subindex16);

#endif

TIMER_STOP;

    return 0;
}

/****************************************************************
                    FPGA KMEANS PRINT REPORT()
 ***************************************************************/
int FPGA_HOST::fpga_host_print_report() {
    
    run_kernel_time = (TIMER_REPORT_MS(1) - last_report_time[1] + TIMER_REPORT_MS(2) - last_report_time[2] + TIMER_REPORT_MS(3) - last_report_time[3])/REPEAT_TIME;
    run_pure_kernel_time = (TIMER_REPORT_MS(2) - last_report_time[2])/REPEAT_TIME;

#ifdef DEBUG
    printf("------------------------------------------------------\n");
    printf("  Performance Summary                                 \n");
    printf("------------------------------------------------------\n");
    printf("  Device Initialization      : %12.4f ms\n", TIMER_REPORT_MS(5));
    printf("  Buffer Allocation          : %12.4f ms\n", TIMER_REPORT_MS(6));
    printf("------------------------------------------------------\n");
    printf("  Setting Input              : %12.4f ms\n", TIMER_REPORT_MS(1));
    printf("  Run kernel                 : %12.4f ms\n", TIMER_REPORT_MS(2));
    printf("  Output data                : %12.4f ms\n", TIMER_REPORT_MS(3));
    printf("  Deallocate Memory          : %12.4f ms\n", TIMER_REPORT_MS(4));
    printf("  Total imageNN Time         : %12.4f ms\n", TIMER_REPORT_MS(0));
    printf("  REPEAT_TIME                : %12d     \n", REPEAT_TIME);
    printf("------------------------------------------------------\n");
#endif

    for(int i = 0; i < 7; i++)
        last_report_time[i] = TIMER_REPORT_MS(i);

    DEBUG_INFO( "KERNEL RESULT: " << run_kernel_time<< " " << run_pure_kernel_time << " " << total_run_time*1000 << " " << total_run_energy << " " << average_run_power );
    fout_fpga_record << run_kernel_time << " " << run_pure_kernel_time << " " << total_run_time*1000 << " " << total_run_energy << " " << average_run_power  << std::endl;
    return 0;
}

double FPGA_HOST::get_run_time()
{
    return run_kernel_time;
}

int FPGA_HOST::measure_static_power(int sleep_time, bool report)
{
#ifdef EST_POWER
    sample_static = energy_meter_init(sample_rate, debug_output_file);  // sample rate in miliseconds
    energy_meter_start(sample_static);

    DEBUG_INFO( "sleeping...." );
    sleep(sleep_time);
    
    energy_meter_stop(sample_static);  	// stops sampling
    res_static = diff(sample_static->start_time, sample_static->stop_time);

    if(report)
    {
        DEBUG_INFO( "----------print static power----------" );
        energy_meter_printf(sample_static, stderr);  // print total results
    }

    energy_meter_destroy(sample_static);
#endif
    return 0;
}

int FPGA_HOST::start_measure_run_power()
{
#ifdef EST_POWER

    sample_run = energy_meter_init(sample_rate, debug_output_file);  // sample rate in miliseconds
    energy_meter_start(sample_run);

#endif
    return 0;
}

int FPGA_HOST::end_measure_run_power(bool report)
{
#ifdef EST_POWER

    energy_meter_stop(sample_run);  	// stops sampling
    res_run = diff(sample_run->start_time, sample_run->stop_time);
    total_run_time = (double)res_run.tv_sec+ (double)res_run.tv_nsec/1000000000.0;
    total_run_energy = 0.0;
	for (int i=0; i<NUM_SENSORS; i++) {
        total_run_energy += sample_run->energy[i];
    }
    average_run_power = total_run_energy/total_run_time;

    if(report)
    {
        DEBUG_INFO( "----------print run power----------" );
        energy_meter_printf(sample_run, stderr);  // print total results
    }
    
    energy_meter_destroy(sample_run);


 #endif
    
    return 0;
}

int FPGA_HOST::report_run_power()
{
#ifdef EST_POWER
    DEBUG_INFO( "----------print run power----------" );
    energy_meter_printf(sample_run, stderr);
#endif
    return 0;
}

int FPGA_HOST::report_static_power()
{
#ifdef EST_POWER
    DEBUG_INFO( "----------print static power----------" );
    energy_meter_printf(sample_static, stderr);  // print total results
#endif
    return 0;
}