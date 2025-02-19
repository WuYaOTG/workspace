#undef __ARM_NEON__
#undef __ARM_NEON
#include "knn_odom.h"
#define __ARM_NEON__
#define __ARM_NEON

static const voxel_int voxel_split_array_size_PL_x_array_size = 216;
static const voxel_int voxel_split_array_size_PL_y_array_size = 29;
static const voxel_int voxel_split_array_size_PL_z_array_size = 185;

static const voxel_int total_calculated_voxel_size = 1158840;

static const type_point_hw data_set_max_min_PL_xmin_hw = 75.0003;
static const type_point_hw data_set_max_min_PL_ymin_hw = -24.3433;
static const type_point_hw data_set_max_min_PL_zmin_hw = -137.805;

static const type_point_hw voxel_split_unit_PL_hw = 1;
static const type_point_hw voxel_split_unit_PL_hw4 = 0.25;

static const int packs = 34044;

#define SUB_VOXEL_SPLIT

static const My_PointXYZI_HW16 zerox = {0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};
static const My_PointXYZI_HW16 zeroy = {0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};
static const My_PointXYZI_HW16 zeroz = {0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};
static const inthw16 zeroindex = {0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};



void calculate_hash_stream_hw(hls::stream<My_PointXYZI_HW>& KNN_reference_set, hls::stream<My_PointXYZI_HW>& KNN_reference_set1, hls::stream<voxel_int>& data_set_hash, int KNN_reference_set_size, int* hash0)
{
    for (count_uint i = 0; i < KNN_reference_set_size; i++)
	{
#pragma HLS LOOP_TRIPCOUNT max=k_query_set_size_max min=k_query_set_size_min
        My_PointXYZI_HW temp = KNN_reference_set.read();
        
        //copy the point_data to data_x,y,z
        type_point_hw data_x = temp.x;
        type_point_hw data_y = temp.y;
        type_point_hw data_z = temp.z;
        voxel_int x_split_array_size = voxel_split_array_size_PL_x_array_size;
        voxel_int y_split_array_size = voxel_split_array_size_PL_y_array_size;
        voxel_int z_split_array_size = voxel_split_array_size_PL_z_array_size;
        //default x,y,z index as the max index
        voxel_int x_index;
        voxel_int y_index;
        voxel_int z_index;

        if (data_x <= data_set_max_min_PL_xmin_hw)
            x_index = 0;
        else
            x_index = (voxel_int)((data_x - data_set_max_min_PL_xmin_hw) / voxel_split_unit_PL_hw);

        if (x_index >= x_split_array_size)
            x_index = x_split_array_size - 1;

        if (data_y <= data_set_max_min_PL_ymin_hw)
            y_index = 0;
        else
            y_index = (voxel_int)((data_y - data_set_max_min_PL_ymin_hw) / voxel_split_unit_PL_hw);

        if (y_index >= y_split_array_size)
            y_index = y_split_array_size - 1;

        if (data_z <= data_set_max_min_PL_zmin_hw)
            z_index = 0;
        else
            z_index = (voxel_int)((data_z - data_set_max_min_PL_zmin_hw) / voxel_split_unit_PL_hw);

        if (z_index >= z_split_array_size)
            z_index = z_split_array_size - 1;

        //transform 3d index to a 1d index
        voxel_int data_hash = x_index * y_split_array_size * z_split_array_size + y_index * z_split_array_size + z_index;
        
        if (data_hash >= total_calculated_voxel_size)
            data_hash = (total_calculated_voxel_size - 1);
        if (data_hash < 0)
            data_hash = 0;
        
        if (i == 0)
        {
            hash0[0] = data_hash;
            DEBUG_INFO(data_hash);
        }

        data_set_hash.write(data_hash);
        KNN_reference_set1.write(temp);
    }
}



void input_src_hw(My_PointXYZI_HW* KNN_reference_set, hls::stream<My_PointXYZI_HW>& reference_input, int KNN_reference_set_size)
{ 
    for(uint_query_size i = 0; i < KNN_reference_set_size; i++)
    {
#pragma HLS LOOP_TRIPCOUNT max=k_query_set_size_max min=k_query_set_size_min
        reference_input.write(KNN_reference_set[i]);
    }
}

void input_reference_hw(My_PointXYZI_HW16* ordered_ref16_x, My_PointXYZI_HW16* ordered_ref16_y, My_PointXYZI_HW16* ordered_ref16_z, inthw16* original_dataset_index16, hls::stream<My_PointXYZI_HW16> ordered_ref16_xs[pararead], hls::stream<My_PointXYZI_HW16> ordered_ref16_ys[pararead], hls::stream<My_PointXYZI_HW16> ordered_ref16_zs[pararead], hls::stream<inthw16> original_dataset_index16s[pararead])
{ 
    int temp1;

    int y = packs % pararead;

    temp1 = packs - y;

    for(voxel_int i = 0; i < temp1; i+=pararead)
    {
#pragma HLS LOOP_TRIPCOUNT max=34044 min=34044
        for(int para_i = 0; para_i < pararead; para_i ++)
        {
#pragma HLS UNROLL

            ordered_ref16_xs[para_i].write(ordered_ref16_x[i+para_i]);
            ordered_ref16_ys[para_i].write(ordered_ref16_y[i+para_i]);
            ordered_ref16_zs[para_i].write(ordered_ref16_z[i+para_i]);
            original_dataset_index16s[para_i].write(original_dataset_index16[i+para_i]);
        }  
    }

    if (y != 0)
    {
        for(int para_i = 0; para_i < pararead; para_i ++)
        {
#pragma HLS UNROLL

            ordered_ref16_xs[para_i].write(zerox);
            ordered_ref16_ys[para_i].write(zeroy);
            ordered_ref16_zs[para_i].write(zeroz);
            original_dataset_index16s[para_i].write(zeroindex);
        }
    }

}

type_dist_hw cal_dist_hw(My_PointXYZI_HW data1, My_PointXYZI_HW data2)
{
	return ((data1.x - data2.x) * (data1.x - data2.x)
		+ (data1.y - data2.y) * (data1.y - data2.y)
		+ (data1.z - data2.z) * (data1.z - data2.z)
		);
}



//计算voxel中心点位置
void initial_hw(hls::stream<My_PointXYZI_HW>& reference_input, hls::stream<My_PointXYZI_HW>& reference_input1, hls::stream<voxel_int>& query_index1, hls::stream<voxel_int>& query_index2, hls::stream<type_point_hw>& local_grid_center_x, hls::stream<type_point_hw>& local_grid_center_y, hls::stream<type_point_hw>& local_grid_center_z, int query_set_size)
{
    for (uint_query_size i = 0; i < query_set_size; i++)
    {
#pragma HLS LOOP_TRIPCOUNT max=k_query_set_size_max min=k_query_set_size_min

    	My_PointXYZI_HW temp = reference_input.read();

    	voxel_int query_index_temp = query_index1.read();	// read query_i_copy

        voxel_int local_x_split_size = voxel_split_array_size_PL_x_array_size;
        voxel_int local_y_split_size = voxel_split_array_size_PL_y_array_size;
        voxel_int local_z_split_size = voxel_split_array_size_PL_z_array_size;

        voxel_int query_x_index = voxel_int(query_index_temp / local_y_split_size / local_z_split_size);
        voxel_int query_y_index = voxel_int((query_index_temp - query_x_index * local_y_split_size * local_z_split_size) / local_z_split_size);
        voxel_int query_z_index = query_index_temp - query_x_index * local_y_split_size * local_z_split_size - query_y_index * local_z_split_size;

        type_point_hw temp_x = (query_x_index + 0.5);
        type_point_hw temp_y = (query_y_index + 0.5);
        type_point_hw temp_z = (query_z_index + 0.5);
        type_point_hw local_grid_center_x_temp = data_set_max_min_PL_xmin_hw + temp_x * voxel_split_unit_PL_hw;
        type_point_hw local_grid_center_y_temp = data_set_max_min_PL_ymin_hw + temp_y * voxel_split_unit_PL_hw;
        type_point_hw local_grid_center_z_temp = data_set_max_min_PL_zmin_hw + temp_z * voxel_split_unit_PL_hw;

        local_grid_center_x.write(local_grid_center_x_temp);
        local_grid_center_y.write(local_grid_center_y_temp);
        local_grid_center_z.write(local_grid_center_z_temp);
        reference_input1.write(temp);
        query_index2.write(query_index_temp);
    }
}

//查表选neighbor voxel
void search_near_cells_hw(hls::stream<My_PointXYZI_HW>& reference_input, hls::stream<My_PointXYZI_HW>& reference_input1, hls::stream<voxel_int>& query_index2, hls::stream<voxel_int>& query_index3, hls::stream<count_uint>& valid_near_voxels, hls::stream<uint_4>& voxel_flag, hls::stream<type_point_hw>& local_grid_center_x, hls::stream<type_point_hw>& local_grid_center_y, hls::stream<type_point_hw>& local_grid_center_z, voxel_int* sub_voxel_flag_index_PL, int query_set_size)
{
    for (uint_query_size i = 0; i < query_set_size; i++)
	{
#pragma HLS LOOP_TRIPCOUNT max=k_query_set_size_max min=k_query_set_size_min

    	My_PointXYZI_HW temp = reference_input.read();
    	reference_input1.write(temp);

    	type_point_hw local_grid_center_x_temp = local_grid_center_x.read();
        type_point_hw local_grid_center_y_temp = local_grid_center_y.read();
        type_point_hw local_grid_center_z_temp = local_grid_center_z.read();

        voxel_int query_index_temp = query_index2.read();	// read query_i_copy
	    query_index3.write(query_index_temp);

//case1的7个voxel位置
        uint_sub_voxel_size loc_27[7];
//case2的16个voxel位置 
        uint_sub_voxel_size loc_17[16];
//case1的7个voxel索引
        voxel_sint to_be_added_index[7];
//case2的16个voxel索引
        voxel_sint to_be_added_index_17[16];
//query所在区域
        uint_sub_voxel_size loc = 0; // back 3 7 front 1 5
                                     //      2 6       0 4



//判断所在区域和neighbor voxel索引，共32种情况        
       if (temp.x >= local_grid_center_x_temp && temp.y >= local_grid_center_y_temp && temp.z >= local_grid_center_z_temp && temp.x >= local_grid_center_x_temp + voxel_split_unit_PL_hw4 && temp.y >= local_grid_center_y_temp && temp.z >= local_grid_center_z_temp + voxel_split_unit_PL_hw4) // x y z
        {
            to_be_added_index[0] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[1] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[2] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[3] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[4] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[5] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[6] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 0;

            to_be_added_index_17[0] = voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[1] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[2] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[3] = voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[4] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[5] = voxel_split_array_size_PL_z_array_size - 1;
            
            to_be_added_index_17[6] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[7] = 1;
            to_be_added_index_17[8] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[9] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[10] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[11] = -1;
            to_be_added_index_17[12] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;

            to_be_added_index_17[13] = -voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[14] = -voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[15] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;

            loc = 31;
            loc_27[0] = 19;
            loc_27[1] = 11;
            loc_27[2] = 7;
            loc_27[3] = 15;
            loc_27[4] = 5;
            loc_27[5] = 3;
            loc_27[6] = 1;

            loc_17[0] = 1;
            loc_17[1] = 2;
            loc_17[2] = 3;
            loc_17[3] = 4;
            loc_17[4] = 5;
            loc_17[5] = 7;

            loc_17[6] = 9;
            loc_17[7] = 10;
            loc_17[8] = 11;
            loc_17[9] = 12;
            loc_17[10] = 14;
            loc_17[11] = 16;
            loc_17[12] = 17;

            loc_17[13] = 19;
            loc_17[14] = 22;
            loc_17[15] = 23;
        }

        else if (temp.x >= local_grid_center_x_temp && temp.y >= local_grid_center_y_temp && temp.z >= local_grid_center_z_temp && temp.x >= local_grid_center_x_temp + voxel_split_unit_PL_hw4 && temp.y >= local_grid_center_y_temp && temp.z < local_grid_center_z_temp + voxel_split_unit_PL_hw4) // x y z
        {
            to_be_added_index[0] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[1] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[2] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[3] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[4] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[5] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[6] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 0;

            to_be_added_index_17[0] = voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[1] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[2] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[3] = voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[4] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[5] = voxel_split_array_size_PL_z_array_size - 1;
            
            to_be_added_index_17[6] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[7] = 1;
            to_be_added_index_17[8] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[9] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[10] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[11] = -1;
            to_be_added_index_17[12] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;

            to_be_added_index_17[13] = -voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[14] = -voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[15] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;

            loc = 30;
            loc_27[0] = 19;
            loc_27[1] = 11;
            loc_27[2] = 7;
            loc_27[3] = 15;
            loc_27[4] = 5;
            loc_27[5] = 3;
            loc_27[6] = 1;

            loc_17[0] = 1;
            loc_17[1] = 2;
            loc_17[2] = 3;
            loc_17[3] = 4;
            loc_17[4] = 5;
            loc_17[5] = 7;

            loc_17[6] = 9;
            loc_17[7] = 10;
            loc_17[8] = 11;
            loc_17[9] = 12;
            loc_17[10] = 14;
            loc_17[11] = 16;
            loc_17[12] = 17;

            loc_17[13] = 19;
            loc_17[14] = 22;
            loc_17[15] = 23;
        }

        else if (temp.x >= local_grid_center_x_temp && temp.y >= local_grid_center_y_temp && temp.z >= local_grid_center_z_temp && temp.x < local_grid_center_x_temp + voxel_split_unit_PL_hw4 && temp.y >= local_grid_center_y_temp && temp.z >= local_grid_center_z_temp + voxel_split_unit_PL_hw4) // x y z
        {
            to_be_added_index[0] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[1] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[2] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[3] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[4] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[5] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[6] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 0;

            to_be_added_index_17[0] = voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[1] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[2] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[3] = voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[4] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[5] = voxel_split_array_size_PL_z_array_size - 1;
            
            to_be_added_index_17[6] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[7] = 1;
            to_be_added_index_17[8] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[9] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[10] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[11] = -1;
            to_be_added_index_17[12] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;

            to_be_added_index_17[13] = -voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[14] = -voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[15] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;

            loc = 23;
            loc_27[0] = 19;
            loc_27[1] = 11;
            loc_27[2] = 7;
            loc_27[3] = 15;
            loc_27[4] = 5;
            loc_27[5] = 3;
            loc_27[6] = 1;

            loc_17[0] = 1;
            loc_17[1] = 2;
            loc_17[2] = 3;
            loc_17[3] = 4;
            loc_17[4] = 5;
            loc_17[5] = 7;

            loc_17[6] = 9;
            loc_17[7] = 10;
            loc_17[8] = 11;
            loc_17[9] = 12;
            loc_17[10] = 14;
            loc_17[11] = 16;
            loc_17[12] = 17;

            loc_17[13] = 19;
            loc_17[14] = 22;
            loc_17[15] = 23;
        }

        else if (temp.x >= local_grid_center_x_temp && temp.y >= local_grid_center_y_temp && temp.z >= local_grid_center_z_temp && temp.x < local_grid_center_x_temp + voxel_split_unit_PL_hw4 && temp.y >= local_grid_center_y_temp && temp.z < local_grid_center_z_temp + voxel_split_unit_PL_hw4) // x y z
        {
            to_be_added_index[0] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[1] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[2] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[3] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[4] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[5] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[6] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 0;

            to_be_added_index_17[0] = voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[1] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[2] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[3] = voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[4] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[5] = voxel_split_array_size_PL_z_array_size - 1;
            
            to_be_added_index_17[6] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[7] = 1;
            to_be_added_index_17[8] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[9] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[10] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[11] = -1;
            to_be_added_index_17[12] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;

            to_be_added_index_17[13] = -voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[14] = -voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[15] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;

            loc = 22;
            loc_27[0] = 19;
            loc_27[1] = 11;
            loc_27[2] = 7;
            loc_27[3] = 15;
            loc_27[4] = 5;
            loc_27[5] = 3;
            loc_27[6] = 1;

            loc_17[0] = 1;
            loc_17[1] = 2;
            loc_17[2] = 3;
            loc_17[3] = 4;
            loc_17[4] = 5;
            loc_17[5] = 7;

            loc_17[6] = 9;
            loc_17[7] = 10;
            loc_17[8] = 11;
            loc_17[9] = 12;
            loc_17[10] = 14;
            loc_17[11] = 16;
            loc_17[12] = 17;

            loc_17[13] = 19;
            loc_17[14] = 22;
            loc_17[15] = 23;
        }


//end1

        
        else if (temp.x < local_grid_center_x_temp && temp.y >= local_grid_center_y_temp && temp.z >= local_grid_center_z_temp && temp.x >= local_grid_center_x_temp - voxel_split_unit_PL_hw4 && temp.y >= local_grid_center_y_temp && temp.z >= local_grid_center_z_temp + voxel_split_unit_PL_hw4) // -x y z
        {
            to_be_added_index[0] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[1] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[2] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[3] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[4] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[5] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[6] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 0;

            to_be_added_index_17[0] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[1] = voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[2] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[3] = voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[4] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[5] = voxel_split_array_size_PL_z_array_size - 1;
            
            to_be_added_index_17[6] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[7] = 1;
            to_be_added_index_17[8] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[9] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[10] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[11] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;
            to_be_added_index_17[12] = -1;

            to_be_added_index_17[13] = -voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[14] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[15] = -voxel_split_array_size_PL_z_array_size;

            loc = 15;
            loc_27[0] = 20;
            loc_27[1] = 13;
            loc_27[2] = 9;
            loc_27[3] = 15;
            loc_27[4] = 5;
            loc_27[5] = 3;
            loc_27[6] = 2;

            loc_17[0] = 0;
            loc_17[1] = 1;
            loc_17[2] = 3;
            loc_17[3] = 4;
            loc_17[4] = 5;
            loc_17[5] = 7;

            loc_17[6] = 9;
            loc_17[7] = 10;
            loc_17[8] = 11;
            loc_17[9] = 12;
            loc_17[10] = 14;
            loc_17[11] = 15;
            loc_17[12] = 16;

            loc_17[13] = 19;
            loc_17[14] = 21;
            loc_17[15] = 22;
        }

        else if (temp.x < local_grid_center_x_temp && temp.y >= local_grid_center_y_temp && temp.z >= local_grid_center_z_temp && temp.x >= local_grid_center_x_temp - voxel_split_unit_PL_hw4 && temp.y >= local_grid_center_y_temp && temp.z < local_grid_center_z_temp + voxel_split_unit_PL_hw4) // -x y z
        {
            to_be_added_index[0] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[1] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[2] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[3] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[4] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[5] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[6] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 0;

            to_be_added_index_17[0] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[1] = voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[2] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[3] = voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[4] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[5] = voxel_split_array_size_PL_z_array_size - 1;
            
            to_be_added_index_17[6] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[7] = 1;
            to_be_added_index_17[8] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[9] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[10] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[11] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;
            to_be_added_index_17[12] = -1;

            to_be_added_index_17[13] = -voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[14] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[15] = -voxel_split_array_size_PL_z_array_size;

            loc = 14;
            loc_27[0] = 20;
            loc_27[1] = 13;
            loc_27[2] = 9;
            loc_27[3] = 15;
            loc_27[4] = 5;
            loc_27[5] = 3;
            loc_27[6] = 2;

            loc_17[0] = 0;
            loc_17[1] = 1;
            loc_17[2] = 3;
            loc_17[3] = 4;
            loc_17[4] = 5;
            loc_17[5] = 7;

            loc_17[6] = 9;
            loc_17[7] = 10;
            loc_17[8] = 11;
            loc_17[9] = 12;
            loc_17[10] = 14;
            loc_17[11] = 15;
            loc_17[12] = 16;

            loc_17[13] = 19;
            loc_17[14] = 21;
            loc_17[15] = 22;
        }

        else if (temp.x < local_grid_center_x_temp && temp.y >= local_grid_center_y_temp && temp.z >= local_grid_center_z_temp && temp.x < local_grid_center_x_temp - voxel_split_unit_PL_hw4 && temp.y >= local_grid_center_y_temp && temp.z >= local_grid_center_z_temp + voxel_split_unit_PL_hw4) // -x y z
        {
            to_be_added_index[0] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[1] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[2] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[3] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[4] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[5] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[6] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 0;

            to_be_added_index_17[0] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[1] = voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[2] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[3] = voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[4] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[5] = voxel_split_array_size_PL_z_array_size - 1;
            
            to_be_added_index_17[6] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[7] = 1;
            to_be_added_index_17[8] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[9] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[10] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[11] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;
            to_be_added_index_17[12] = -1;

            to_be_added_index_17[13] = -voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[14] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[15] = -voxel_split_array_size_PL_z_array_size;

            loc = 7;
            loc_27[0] = 20;
            loc_27[1] = 13;
            loc_27[2] = 9;
            loc_27[3] = 15;
            loc_27[4] = 5;
            loc_27[5] = 3;
            loc_27[6] = 2;

            loc_17[0] = 0;
            loc_17[1] = 1;
            loc_17[2] = 3;
            loc_17[3] = 4;
            loc_17[4] = 5;
            loc_17[5] = 7;

            loc_17[6] = 9;
            loc_17[7] = 10;
            loc_17[8] = 11;
            loc_17[9] = 12;
            loc_17[10] = 14;
            loc_17[11] = 15;
            loc_17[12] = 16;

            loc_17[13] = 19;
            loc_17[14] = 21;
            loc_17[15] = 22;
        }

        else if (temp.x < local_grid_center_x_temp && temp.y >= local_grid_center_y_temp && temp.z >= local_grid_center_z_temp && temp.x < local_grid_center_x_temp - voxel_split_unit_PL_hw4 && temp.y >= local_grid_center_y_temp && temp.z < local_grid_center_z_temp + voxel_split_unit_PL_hw4) // -x y z
        {
            to_be_added_index[0] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[1] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[2] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[3] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[4] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[5] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[6] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 0;

            to_be_added_index_17[0] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[1] = voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[2] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[3] = voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[4] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[5] = voxel_split_array_size_PL_z_array_size - 1;
            
            to_be_added_index_17[6] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[7] = 1;
            to_be_added_index_17[8] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[9] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[10] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[11] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;
            to_be_added_index_17[12] = -1;

            to_be_added_index_17[13] = -voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[14] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[15] = -voxel_split_array_size_PL_z_array_size;

            loc = 6;
            loc_27[0] = 20;
            loc_27[1] = 13;
            loc_27[2] = 9;
            loc_27[3] = 15;
            loc_27[4] = 5;
            loc_27[5] = 3;
            loc_27[6] = 2;

            loc_17[0] = 0;
            loc_17[1] = 1;
            loc_17[2] = 3;
            loc_17[3] = 4;
            loc_17[4] = 5;
            loc_17[5] = 7;

            loc_17[6] = 9;
            loc_17[7] = 10;
            loc_17[8] = 11;
            loc_17[9] = 12;
            loc_17[10] = 14;
            loc_17[11] = 15;
            loc_17[12] = 16;

            loc_17[13] = 19;
            loc_17[14] = 21;
            loc_17[15] = 22;
        }


//end2


        else if (temp.x >= local_grid_center_x_temp && temp.y < local_grid_center_y_temp && temp.z >= local_grid_center_z_temp && temp.x >= local_grid_center_x_temp + voxel_split_unit_PL_hw4 && temp.y < local_grid_center_y_temp && temp.z >= local_grid_center_z_temp + voxel_split_unit_PL_hw4) // x -y z
        {
            to_be_added_index[0] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[1] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[2] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[3] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[4] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[5] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[6] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 0;

            to_be_added_index_17[0] = -voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[1] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[2] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[3] = -voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[4] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[5] = -voxel_split_array_size_PL_z_array_size - 1;
            
            to_be_added_index_17[6] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[7] = 1;
            to_be_added_index_17[8] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[9] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[10] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[11] = -1;
            to_be_added_index_17[12] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;

            to_be_added_index_17[13] = voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[14] = voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[15] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;

            loc = 27;
            loc_27[0] = 21;
            loc_27[1] = 12;
            loc_27[2] = 7;
            loc_27[3] = 17;
            loc_27[4] = 5;
            loc_27[5] = 4;
            loc_27[6] = 1;

            loc_17[0] = 19;
            loc_17[1] = 20;
            loc_17[2] = 21;
            loc_17[3] = 22;
            loc_17[4] = 23;
            loc_17[5] = 25;

            loc_17[6] = 9;
            loc_17[7] = 10;
            loc_17[8] = 11;
            loc_17[9] = 12;
            loc_17[10] = 14;
            loc_17[11] = 16;
            loc_17[12] = 17;

            loc_17[13] = 1;
            loc_17[14] = 4;
            loc_17[15] = 5;
        }

        else if (temp.x >= local_grid_center_x_temp && temp.y < local_grid_center_y_temp && temp.z >= local_grid_center_z_temp && temp.x >= local_grid_center_x_temp + voxel_split_unit_PL_hw4 && temp.y < local_grid_center_y_temp && temp.z < local_grid_center_z_temp + voxel_split_unit_PL_hw4) // x -y z
        {
            to_be_added_index[0] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[1] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[2] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[3] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[4] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[5] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[6] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 0;

            to_be_added_index_17[0] = -voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[1] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[2] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[3] = -voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[4] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[5] = -voxel_split_array_size_PL_z_array_size - 1;
            
            to_be_added_index_17[6] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[7] = 1;
            to_be_added_index_17[8] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[9] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[10] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[11] = -1;
            to_be_added_index_17[12] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;

            to_be_added_index_17[13] = voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[14] = voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[15] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;

            loc = 26;
            loc_27[0] = 21;
            loc_27[1] = 12;
            loc_27[2] = 7;
            loc_27[3] = 17;
            loc_27[4] = 5;
            loc_27[5] = 4;
            loc_27[6] = 1;

            loc_17[0] = 19;
            loc_17[1] = 20;
            loc_17[2] = 21;
            loc_17[3] = 22;
            loc_17[4] = 23;
            loc_17[5] = 25;

            loc_17[6] = 9;
            loc_17[7] = 10;
            loc_17[8] = 11;
            loc_17[9] = 12;
            loc_17[10] = 14;
            loc_17[11] = 16;
            loc_17[12] = 17;

            loc_17[13] = 1;
            loc_17[14] = 4;
            loc_17[15] = 5;
        }

        else if (temp.x >= local_grid_center_x_temp && temp.y < local_grid_center_y_temp && temp.z >= local_grid_center_z_temp && temp.x < local_grid_center_x_temp + voxel_split_unit_PL_hw4 && temp.y < local_grid_center_y_temp && temp.z >= local_grid_center_z_temp + voxel_split_unit_PL_hw4) // x -y z
        {
            to_be_added_index[0] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[1] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[2] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[3] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[4] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[5] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[6] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 0;

            to_be_added_index_17[0] = -voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[1] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[2] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[3] = -voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[4] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[5] = -voxel_split_array_size_PL_z_array_size - 1;
            
            to_be_added_index_17[6] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[7] = 1;
            to_be_added_index_17[8] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[9] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[10] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[11] = -1;
            to_be_added_index_17[12] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;

            to_be_added_index_17[13] = voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[14] = voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[15] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;

            loc = 19;
            loc_27[0] = 21;
            loc_27[1] = 12;
            loc_27[2] = 7;
            loc_27[3] = 17;
            loc_27[4] = 5;
            loc_27[5] = 4;
            loc_27[6] = 1;

            loc_17[0] = 19;
            loc_17[1] = 20;
            loc_17[2] = 21;
            loc_17[3] = 22;
            loc_17[4] = 23;
            loc_17[5] = 25;

            loc_17[6] = 9;
            loc_17[7] = 10;
            loc_17[8] = 11;
            loc_17[9] = 12;
            loc_17[10] = 14;
            loc_17[11] = 16;
            loc_17[12] = 17;

            loc_17[13] = 1;
            loc_17[14] = 4;
            loc_17[15] = 5;
        }

        else if (temp.x >= local_grid_center_x_temp && temp.y < local_grid_center_y_temp && temp.z >= local_grid_center_z_temp && temp.x < local_grid_center_x_temp + voxel_split_unit_PL_hw4 && temp.y < local_grid_center_y_temp && temp.z < local_grid_center_z_temp + voxel_split_unit_PL_hw4) // x -y z
        {
            to_be_added_index[0] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[1] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[2] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[3] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[4] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[5] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[6] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 0;

            to_be_added_index_17[0] = -voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[1] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[2] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[3] = -voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[4] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[5] = -voxel_split_array_size_PL_z_array_size - 1;
            
            to_be_added_index_17[6] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[7] = 1;
            to_be_added_index_17[8] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[9] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[10] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[11] = -1;
            to_be_added_index_17[12] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;

            to_be_added_index_17[13] = voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[14] = voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[15] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;

            loc = 18;
            loc_27[0] = 21;
            loc_27[1] = 12;
            loc_27[2] = 7;
            loc_27[3] = 17;
            loc_27[4] = 5;
            loc_27[5] = 4;
            loc_27[6] = 1;

            loc_17[0] = 19;
            loc_17[1] = 20;
            loc_17[2] = 21;
            loc_17[3] = 22;
            loc_17[4] = 23;
            loc_17[5] = 25;

            loc_17[6] = 9;
            loc_17[7] = 10;
            loc_17[8] = 11;
            loc_17[9] = 12;
            loc_17[10] = 14;
            loc_17[11] = 16;
            loc_17[12] = 17;

            loc_17[13] = 1;
            loc_17[14] = 4;
            loc_17[15] = 5;
        }


//end3


        else if (temp.x >= local_grid_center_x_temp && temp.y >= local_grid_center_y_temp && temp.z < local_grid_center_z_temp && temp.x >= local_grid_center_x_temp + voxel_split_unit_PL_hw4 && temp.y >= local_grid_center_y_temp && temp.z >= local_grid_center_z_temp - voxel_split_unit_PL_hw4) // x y -z
        {
            to_be_added_index[0] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[1] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[2] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[3] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[4] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[5] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[6] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 0;

            to_be_added_index_17[5] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size - 1;
            to_be_added_index_17[0] = voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[1] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[2] = voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[3] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[4] = voxel_split_array_size_PL_z_array_size - 1;
            
            to_be_added_index_17[6] = 1;
            to_be_added_index_17[7] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[8] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[9] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[10] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;
            to_be_added_index_17[11] = -1;
            to_be_added_index_17[12] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;

            to_be_added_index_17[13] = -voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[14] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[15] = -voxel_split_array_size_PL_z_array_size - 1;

            loc = 29;
            loc_27[0] = 23;
            loc_27[1] = 11;
            loc_27[2] = 8;
            loc_27[3] = 16;
            loc_27[4] = 6;
            loc_27[5] = 3;
            loc_27[6] = 1;

            loc_17[0] = 1;
            loc_17[1] = 3;
            loc_17[2] = 4;
            loc_17[3] = 5;
            loc_17[4] = 7;
            loc_17[5] = 8;

            loc_17[6] = 10;
            loc_17[7] = 11;
            loc_17[8] = 12;
            loc_17[9] = 14;
            loc_17[10] = 15;
            loc_17[11] = 16;
            loc_17[12] = 17;

            loc_17[13] = 22;
            loc_17[14] = 23;
            loc_17[15] = 25;
        }

        else if (temp.x >= local_grid_center_x_temp && temp.y >= local_grid_center_y_temp && temp.z < local_grid_center_z_temp && temp.x >= local_grid_center_x_temp + voxel_split_unit_PL_hw4 && temp.y >= local_grid_center_y_temp && temp.z < local_grid_center_z_temp - voxel_split_unit_PL_hw4) // x y -z
        {
            to_be_added_index[0] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[1] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[2] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[3] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[4] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[5] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[6] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 0;

            to_be_added_index_17[5] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size - 1;
            to_be_added_index_17[0] = voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[1] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[2] = voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[3] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[4] = voxel_split_array_size_PL_z_array_size - 1;
            
            to_be_added_index_17[6] = 1;
            to_be_added_index_17[7] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[8] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[9] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[10] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;
            to_be_added_index_17[11] = -1;
            to_be_added_index_17[12] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;

            to_be_added_index_17[13] = -voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[14] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[15] = -voxel_split_array_size_PL_z_array_size - 1;

            loc = 28;
            loc_27[0] = 23;
            loc_27[1] = 11;
            loc_27[2] = 8;
            loc_27[3] = 16;
            loc_27[4] = 6;
            loc_27[5] = 3;
            loc_27[6] = 1;

            loc_17[0] = 1;
            loc_17[1] = 3;
            loc_17[2] = 4;
            loc_17[3] = 5;
            loc_17[4] = 7;
            loc_17[5] = 8;

            loc_17[6] = 10;
            loc_17[7] = 11;
            loc_17[8] = 12;
            loc_17[9] = 14;
            loc_17[10] = 15;
            loc_17[11] = 16;
            loc_17[12] = 17;

            loc_17[13] = 22;
            loc_17[14] = 23;
            loc_17[15] = 25;
        }

        else if (temp.x >= local_grid_center_x_temp && temp.y >= local_grid_center_y_temp && temp.z < local_grid_center_z_temp && temp.x < local_grid_center_x_temp + voxel_split_unit_PL_hw4 && temp.y >= local_grid_center_y_temp && temp.z >= local_grid_center_z_temp - voxel_split_unit_PL_hw4) // x y -z
        {
            to_be_added_index[0] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[1] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[2] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[3] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[4] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[5] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[6] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 0;

            to_be_added_index_17[5] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size - 1;
            to_be_added_index_17[0] = voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[1] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[2] = voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[3] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[4] = voxel_split_array_size_PL_z_array_size - 1;
            
            to_be_added_index_17[6] = 1;
            to_be_added_index_17[7] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[8] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[9] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[10] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;
            to_be_added_index_17[11] = -1;
            to_be_added_index_17[12] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;

            to_be_added_index_17[13] = -voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[14] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[15] = -voxel_split_array_size_PL_z_array_size - 1;

            loc = 21;
            loc_27[0] = 23;
            loc_27[1] = 11;
            loc_27[2] = 8;
            loc_27[3] = 16;
            loc_27[4] = 6;
            loc_27[5] = 3;
            loc_27[6] = 1;

            loc_17[0] = 1;
            loc_17[1] = 3;
            loc_17[2] = 4;
            loc_17[3] = 5;
            loc_17[4] = 7;
            loc_17[5] = 8;

            loc_17[6] = 10;
            loc_17[7] = 11;
            loc_17[8] = 12;
            loc_17[9] = 14;
            loc_17[10] = 15;
            loc_17[11] = 16;
            loc_17[12] = 17;

            loc_17[13] = 22;
            loc_17[14] = 23;
            loc_17[15] = 25;
        }

        else if (temp.x >= local_grid_center_x_temp && temp.y >= local_grid_center_y_temp && temp.z < local_grid_center_z_temp && temp.x < local_grid_center_x_temp + voxel_split_unit_PL_hw4 && temp.y >= local_grid_center_y_temp && temp.z < local_grid_center_z_temp - voxel_split_unit_PL_hw4) // x y -z
        {
            to_be_added_index[0] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[1] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[2] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[3] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[4] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[5] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[6] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 0;

            to_be_added_index_17[5] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size - 1;
            to_be_added_index_17[0] = voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[1] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[2] = voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[3] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[4] = voxel_split_array_size_PL_z_array_size - 1;
            
            to_be_added_index_17[6] = 1;
            to_be_added_index_17[7] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[8] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[9] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[10] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;
            to_be_added_index_17[11] = -1;
            to_be_added_index_17[12] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;

            to_be_added_index_17[13] = -voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[14] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[15] = -voxel_split_array_size_PL_z_array_size - 1;

            loc = 20;
            loc_27[0] = 23;
            loc_27[1] = 11;
            loc_27[2] = 8;
            loc_27[3] = 16;
            loc_27[4] = 6;
            loc_27[5] = 3;
            loc_27[6] = 1;

            loc_17[0] = 1;
            loc_17[1] = 3;
            loc_17[2] = 4;
            loc_17[3] = 5;
            loc_17[4] = 7;
            loc_17[5] = 8;

            loc_17[6] = 10;
            loc_17[7] = 11;
            loc_17[8] = 12;
            loc_17[9] = 14;
            loc_17[10] = 15;
            loc_17[11] = 16;
            loc_17[12] = 17;

            loc_17[13] = 22;
            loc_17[14] = 23;
            loc_17[15] = 25;
        }


//end4


        else if (temp.x < local_grid_center_x_temp && temp.y < local_grid_center_y_temp && temp.z >= local_grid_center_z_temp && temp.x >= local_grid_center_x_temp - voxel_split_unit_PL_hw4 && temp.y < local_grid_center_y_temp && temp.z >= local_grid_center_z_temp + voxel_split_unit_PL_hw4) // -x -y z
        {
            to_be_added_index[0] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[1] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[2] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[3] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[4] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[5] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[6] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 0;

            to_be_added_index_17[0] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[1] = -voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[2] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[3] = -voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[4] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[5] = -voxel_split_array_size_PL_z_array_size - 1;
            
            to_be_added_index_17[6] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[7] = 1;
            to_be_added_index_17[8] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[9] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[10] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[11] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;
            to_be_added_index_17[12] = -1;

            to_be_added_index_17[13] = voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[14] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[15] = voxel_split_array_size_PL_z_array_size;

            loc = 11;
            loc_27[0] = 22;
            loc_27[1] = 14;
            loc_27[2] = 9;
            loc_27[3] = 17;
            loc_27[4] = 5;
            loc_27[5] = 4;
            loc_27[6] = 2;

            loc_17[0] = 18;
            loc_17[1] = 19;
            loc_17[2] = 21;
            loc_17[3] = 22;
            loc_17[4] = 23;
            loc_17[5] = 25;

            loc_17[6] = 9;
            loc_17[7] = 10;
            loc_17[8] = 11;
            loc_17[9] = 12;
            loc_17[10] = 14;
            loc_17[11] = 15;
            loc_17[12] = 16;

            loc_17[13] = 1;
            loc_17[14] = 3;
            loc_17[15] = 4;
        }

        else if (temp.x < local_grid_center_x_temp && temp.y < local_grid_center_y_temp && temp.z >= local_grid_center_z_temp && temp.x >= local_grid_center_x_temp - voxel_split_unit_PL_hw4 && temp.y < local_grid_center_y_temp && temp.z < local_grid_center_z_temp + voxel_split_unit_PL_hw4) // -x -y z
        {
            to_be_added_index[0] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[1] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[2] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[3] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[4] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[5] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[6] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 0;

            to_be_added_index_17[0] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[1] = -voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[2] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[3] = -voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[4] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[5] = -voxel_split_array_size_PL_z_array_size - 1;
            
            to_be_added_index_17[6] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[7] = 1;
            to_be_added_index_17[8] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[9] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[10] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[11] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;
            to_be_added_index_17[12] = -1;

            to_be_added_index_17[13] = voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[14] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[15] = voxel_split_array_size_PL_z_array_size;

            loc = 10;
            loc_27[0] = 22;
            loc_27[1] = 14;
            loc_27[2] = 9;
            loc_27[3] = 17;
            loc_27[4] = 5;
            loc_27[5] = 4;
            loc_27[6] = 2;

            loc_17[0] = 18;
            loc_17[1] = 19;
            loc_17[2] = 21;
            loc_17[3] = 22;
            loc_17[4] = 23;
            loc_17[5] = 25;

            loc_17[6] = 9;
            loc_17[7] = 10;
            loc_17[8] = 11;
            loc_17[9] = 12;
            loc_17[10] = 14;
            loc_17[11] = 15;
            loc_17[12] = 16;

            loc_17[13] = 1;
            loc_17[14] = 3;
            loc_17[15] = 4;
        }

        else if (temp.x < local_grid_center_x_temp && temp.y < local_grid_center_y_temp && temp.z >= local_grid_center_z_temp && temp.x < local_grid_center_x_temp - voxel_split_unit_PL_hw4 && temp.y < local_grid_center_y_temp && temp.z >= local_grid_center_z_temp + voxel_split_unit_PL_hw4) // -x -y z
        {
            to_be_added_index[0] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[1] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[2] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[3] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[4] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[5] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[6] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 0;

            to_be_added_index_17[0] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[1] = -voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[2] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[3] = -voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[4] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[5] = -voxel_split_array_size_PL_z_array_size - 1;
            
            to_be_added_index_17[6] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[7] = 1;
            to_be_added_index_17[8] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[9] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[10] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[11] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;
            to_be_added_index_17[12] = -1;

            to_be_added_index_17[13] = voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[14] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[15] = voxel_split_array_size_PL_z_array_size;

            loc = 3;
            loc_27[0] = 22;
            loc_27[1] = 14;
            loc_27[2] = 9;
            loc_27[3] = 17;
            loc_27[4] = 5;
            loc_27[5] = 4;
            loc_27[6] = 2;

            loc_17[0] = 18;
            loc_17[1] = 19;
            loc_17[2] = 21;
            loc_17[3] = 22;
            loc_17[4] = 23;
            loc_17[5] = 25;

            loc_17[6] = 9;
            loc_17[7] = 10;
            loc_17[8] = 11;
            loc_17[9] = 12;
            loc_17[10] = 14;
            loc_17[11] = 15;
            loc_17[12] = 16;

            loc_17[13] = 1;
            loc_17[14] = 3;
            loc_17[15] = 4;
        }

        else if (temp.x < local_grid_center_x_temp && temp.y < local_grid_center_y_temp && temp.z >= local_grid_center_z_temp && temp.x < local_grid_center_x_temp - voxel_split_unit_PL_hw4 && temp.y < local_grid_center_y_temp && temp.z < local_grid_center_z_temp + voxel_split_unit_PL_hw4) // -x -y z
        {
            to_be_added_index[0] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[1] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[2] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[3] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[4] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[5] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[6] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 0;

            to_be_added_index_17[0] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[1] = -voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[2] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[3] = -voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[4] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[5] = -voxel_split_array_size_PL_z_array_size - 1;
            
            to_be_added_index_17[6] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[7] = 1;
            to_be_added_index_17[8] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[9] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[10] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[11] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;
            to_be_added_index_17[12] = -1;

            to_be_added_index_17[13] = voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[14] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[15] = voxel_split_array_size_PL_z_array_size;

            loc = 2;
            loc_27[0] = 22;
            loc_27[1] = 14;
            loc_27[2] = 9;
            loc_27[3] = 17;
            loc_27[4] = 5;
            loc_27[5] = 4;
            loc_27[6] = 2;

            loc_17[0] = 18;
            loc_17[1] = 19;
            loc_17[2] = 21;
            loc_17[3] = 22;
            loc_17[4] = 23;
            loc_17[5] = 25;

            loc_17[6] = 9;
            loc_17[7] = 10;
            loc_17[8] = 11;
            loc_17[9] = 12;
            loc_17[10] = 14;
            loc_17[11] = 15;
            loc_17[12] = 16;

            loc_17[13] = 1;
            loc_17[14] = 3;
            loc_17[15] = 4;
        }


//end5

        else if (temp.x >= local_grid_center_x_temp && temp.y < local_grid_center_y_temp && temp.z < local_grid_center_z_temp && temp.x >= local_grid_center_x_temp + voxel_split_unit_PL_hw4 && temp.y < local_grid_center_y_temp && temp.z >= local_grid_center_z_temp - voxel_split_unit_PL_hw4) // x -y -z
        {
            to_be_added_index[0] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[1] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[2] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[3] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[4] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[5] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[6] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 0;

            to_be_added_index_17[5] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size - 1;
            to_be_added_index_17[0] = -voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[1] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[2] = -voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[3] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[4] = -voxel_split_array_size_PL_z_array_size - 1;
            
            to_be_added_index_17[6] = 1;
            to_be_added_index_17[7] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[8] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[9] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[10] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;
            to_be_added_index_17[11] = -1;
            to_be_added_index_17[12] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;

            to_be_added_index_17[13] = voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[14] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[15] = voxel_split_array_size_PL_z_array_size - 1;

            loc = 25;
            loc_27[0] = 25;
            loc_27[1] = 12;
            loc_27[2] = 8;
            loc_27[3] = 18;
            loc_27[4] = 6;
            loc_27[5] = 4;
            loc_27[6] = 1;

            loc_17[0] = 19;
            loc_17[1] = 21;
            loc_17[2] = 22;
            loc_17[3] = 23;
            loc_17[4] = 25;
            loc_17[5] = 26;

            loc_17[6] = 10;
            loc_17[7] = 11;
            loc_17[8] = 12;
            loc_17[9] = 14;
            loc_17[10] = 15;
            loc_17[11] = 16;
            loc_17[12] = 17;

            loc_17[13] = 4;
            loc_17[14] = 5;
            loc_17[15] = 7;
        }

        else if (temp.x >= local_grid_center_x_temp && temp.y < local_grid_center_y_temp && temp.z < local_grid_center_z_temp && temp.x >= local_grid_center_x_temp + voxel_split_unit_PL_hw4 && temp.y < local_grid_center_y_temp && temp.z < local_grid_center_z_temp - voxel_split_unit_PL_hw4) // x -y -z
        {
            to_be_added_index[0] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[1] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[2] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[3] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[4] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[5] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[6] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 0;

            to_be_added_index_17[5] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size - 1;
            to_be_added_index_17[0] = -voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[1] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[2] = -voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[3] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[4] = -voxel_split_array_size_PL_z_array_size - 1;
            
            to_be_added_index_17[6] = 1;
            to_be_added_index_17[7] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[8] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[9] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[10] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;
            to_be_added_index_17[11] = -1;
            to_be_added_index_17[12] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;

            to_be_added_index_17[13] = voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[14] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[15] = voxel_split_array_size_PL_z_array_size - 1;

            loc = 24;
            loc_27[0] = 25;
            loc_27[1] = 12;
            loc_27[2] = 8;
            loc_27[3] = 18;
            loc_27[4] = 6;
            loc_27[5] = 4;
            loc_27[6] = 1;

            loc_17[0] = 19;
            loc_17[1] = 21;
            loc_17[2] = 22;
            loc_17[3] = 23;
            loc_17[4] = 25;
            loc_17[5] = 26;

            loc_17[6] = 10;
            loc_17[7] = 11;
            loc_17[8] = 12;
            loc_17[9] = 14;
            loc_17[10] = 15;
            loc_17[11] = 16;
            loc_17[12] = 17;

            loc_17[13] = 4;
            loc_17[14] = 5;
            loc_17[15] = 7;
        }

        else if (temp.x >= local_grid_center_x_temp && temp.y < local_grid_center_y_temp && temp.z < local_grid_center_z_temp && temp.x < local_grid_center_x_temp + voxel_split_unit_PL_hw4 && temp.y < local_grid_center_y_temp && temp.z >= local_grid_center_z_temp - voxel_split_unit_PL_hw4) // x -y -z
        {
            to_be_added_index[0] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[1] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[2] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[3] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[4] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[5] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[6] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 0;

            to_be_added_index_17[5] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size - 1;
            to_be_added_index_17[0] = -voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[1] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[2] = -voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[3] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[4] = -voxel_split_array_size_PL_z_array_size - 1;
            
            to_be_added_index_17[6] = 1;
            to_be_added_index_17[7] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[8] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[9] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[10] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;
            to_be_added_index_17[11] = -1;
            to_be_added_index_17[12] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;

            to_be_added_index_17[13] = voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[14] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[15] = voxel_split_array_size_PL_z_array_size - 1;

            loc = 17;
            loc_27[0] = 25;
            loc_27[1] = 12;
            loc_27[2] = 8;
            loc_27[3] = 18;
            loc_27[4] = 6;
            loc_27[5] = 4;
            loc_27[6] = 1;

            loc_17[0] = 19;
            loc_17[1] = 21;
            loc_17[2] = 22;
            loc_17[3] = 23;
            loc_17[4] = 25;
            loc_17[5] = 26;

            loc_17[6] = 10;
            loc_17[7] = 11;
            loc_17[8] = 12;
            loc_17[9] = 14;
            loc_17[10] = 15;
            loc_17[11] = 16;
            loc_17[12] = 17;

            loc_17[13] = 4;
            loc_17[14] = 5;
            loc_17[15] = 7;
        }

        else if (temp.x >= local_grid_center_x_temp && temp.y < local_grid_center_y_temp && temp.z < local_grid_center_z_temp && temp.x < local_grid_center_x_temp + voxel_split_unit_PL_hw4 && temp.y < local_grid_center_y_temp && temp.z < local_grid_center_z_temp - voxel_split_unit_PL_hw4) // x -y -z
        {
            to_be_added_index[0] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[1] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[2] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[3] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[4] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[5] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[6] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 0;

            to_be_added_index_17[5] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size - 1;
            to_be_added_index_17[0] = -voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[1] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[2] = -voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[3] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[4] = -voxel_split_array_size_PL_z_array_size - 1;
            
            to_be_added_index_17[6] = 1;
            to_be_added_index_17[7] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[8] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[9] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[10] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;
            to_be_added_index_17[11] = -1;
            to_be_added_index_17[12] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;

            to_be_added_index_17[13] = voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[14] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[15] = voxel_split_array_size_PL_z_array_size - 1;

            loc = 16;
            loc_27[0] = 25;
            loc_27[1] = 12;
            loc_27[2] = 8;
            loc_27[3] = 18;
            loc_27[4] = 6;
            loc_27[5] = 4;
            loc_27[6] = 1;

            loc_17[0] = 19;
            loc_17[1] = 21;
            loc_17[2] = 22;
            loc_17[3] = 23;
            loc_17[4] = 25;
            loc_17[5] = 26;

            loc_17[6] = 10;
            loc_17[7] = 11;
            loc_17[8] = 12;
            loc_17[9] = 14;
            loc_17[10] = 15;
            loc_17[11] = 16;
            loc_17[12] = 17;

            loc_17[13] = 4;
            loc_17[14] = 5;
            loc_17[15] = 7;
        }


//end6


        else if (temp.x < local_grid_center_x_temp && temp.y >= local_grid_center_y_temp && temp.z < local_grid_center_z_temp && temp.x >= local_grid_center_x_temp - voxel_split_unit_PL_hw4 && temp.y >= local_grid_center_y_temp && temp.z >= local_grid_center_z_temp - voxel_split_unit_PL_hw4) // -x y -z
        {
            to_be_added_index[0] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[1] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[2] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[3] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[4] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[5] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[6] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 0;

            to_be_added_index_17[5] = voxel_split_array_size_PL_z_array_size - 1;
            to_be_added_index_17[0] = voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[1] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[2] = voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[3] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[4] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size - 1;
            
            to_be_added_index_17[6] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[7] = 1;
            to_be_added_index_17[8] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[9] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[10] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;
            to_be_added_index_17[11] = -1;
            to_be_added_index_17[12] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;

            to_be_added_index_17[13] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[14] = -voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[15] = -voxel_split_array_size_PL_z_array_size - 1;

            loc = 13;
            loc_27[0] = 24;
            loc_27[1] = 13;
            loc_27[2] = 10;
            loc_27[3] = 16;
            loc_27[4] = 6;
            loc_27[5] = 3;
            loc_27[6] = 2;

            loc_17[0] = 1;
            loc_17[1] = 3;
            loc_17[2] = 4;
            loc_17[3] = 5;
            loc_17[4] = 6;
            loc_17[5] = 7;

            loc_17[6] = 9;
            loc_17[7] = 10;
            loc_17[8] = 12;
            loc_17[9] = 14;
            loc_17[10] = 15;
            loc_17[11] = 16;
            loc_17[12] = 17;

            loc_17[13] = 21;
            loc_17[14] = 22;
            loc_17[15] = 25;
        }

        else if (temp.x < local_grid_center_x_temp && temp.y >= local_grid_center_y_temp && temp.z < local_grid_center_z_temp && temp.x >= local_grid_center_x_temp - voxel_split_unit_PL_hw4 && temp.y >= local_grid_center_y_temp && temp.z < local_grid_center_z_temp - voxel_split_unit_PL_hw4) // -x y -z
        {
            to_be_added_index[0] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[1] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[2] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[3] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[4] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[5] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[6] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 0;

            to_be_added_index_17[5] = voxel_split_array_size_PL_z_array_size - 1;
            to_be_added_index_17[0] = voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[1] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[2] = voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[3] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[4] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size - 1;
            
            to_be_added_index_17[6] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[7] = 1;
            to_be_added_index_17[8] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[9] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[10] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;
            to_be_added_index_17[11] = -1;
            to_be_added_index_17[12] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;

            to_be_added_index_17[13] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[14] = -voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[15] = -voxel_split_array_size_PL_z_array_size - 1;

            loc = 12;
            loc_27[0] = 24;
            loc_27[1] = 13;
            loc_27[2] = 10;
            loc_27[3] = 16;
            loc_27[4] = 6;
            loc_27[5] = 3;
            loc_27[6] = 2;

            loc_17[0] = 1;
            loc_17[1] = 3;
            loc_17[2] = 4;
            loc_17[3] = 5;
            loc_17[4] = 6;
            loc_17[5] = 7;

            loc_17[6] = 9;
            loc_17[7] = 10;
            loc_17[8] = 12;
            loc_17[9] = 14;
            loc_17[10] = 15;
            loc_17[11] = 16;
            loc_17[12] = 17;

            loc_17[13] = 21;
            loc_17[14] = 22;
            loc_17[15] = 25;
        }

        else if (temp.x < local_grid_center_x_temp && temp.y >= local_grid_center_y_temp && temp.z < local_grid_center_z_temp && temp.x < local_grid_center_x_temp - voxel_split_unit_PL_hw4 && temp.y >= local_grid_center_y_temp && temp.z >= local_grid_center_z_temp - voxel_split_unit_PL_hw4) // -x y -z
        {
            to_be_added_index[0] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[1] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[2] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[3] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[4] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[5] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[6] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 0;

            to_be_added_index_17[5] = voxel_split_array_size_PL_z_array_size - 1;
            to_be_added_index_17[0] = voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[1] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[2] = voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[3] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[4] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size - 1;
            
            to_be_added_index_17[6] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[7] = 1;
            to_be_added_index_17[8] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[9] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[10] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;
            to_be_added_index_17[11] = -1;
            to_be_added_index_17[12] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;

            to_be_added_index_17[13] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[14] = -voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[15] = -voxel_split_array_size_PL_z_array_size - 1;

            loc = 5;
            loc_27[0] = 24;
            loc_27[1] = 13;
            loc_27[2] = 10;
            loc_27[3] = 16;
            loc_27[4] = 6;
            loc_27[5] = 3;
            loc_27[6] = 2;

            loc_17[0] = 1;
            loc_17[1] = 3;
            loc_17[2] = 4;
            loc_17[3] = 5;
            loc_17[4] = 6;
            loc_17[5] = 7;

            loc_17[6] = 9;
            loc_17[7] = 10;
            loc_17[8] = 12;
            loc_17[9] = 14;
            loc_17[10] = 15;
            loc_17[11] = 16;
            loc_17[12] = 17;

            loc_17[13] = 21;
            loc_17[14] = 22;
            loc_17[15] = 25;
        }

        else if (temp.x < local_grid_center_x_temp && temp.y >= local_grid_center_y_temp && temp.z < local_grid_center_z_temp && temp.x < local_grid_center_x_temp - voxel_split_unit_PL_hw4 && temp.y >= local_grid_center_y_temp && temp.z < local_grid_center_z_temp - voxel_split_unit_PL_hw4) // -x y -z
        {
            to_be_added_index[0] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[1] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[2] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[3] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[4] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[5] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[6] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 0;

            to_be_added_index_17[5] = voxel_split_array_size_PL_z_array_size - 1;
            to_be_added_index_17[0] = voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[1] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[2] = voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[3] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[4] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size - 1;
            
            to_be_added_index_17[6] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[7] = 1;
            to_be_added_index_17[8] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[9] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[10] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;
            to_be_added_index_17[11] = -1;
            to_be_added_index_17[12] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;

            to_be_added_index_17[13] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[14] = -voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[15] = -voxel_split_array_size_PL_z_array_size - 1;

            loc = 4;
            loc_27[0] = 24;
            loc_27[1] = 13;
            loc_27[2] = 10;
            loc_27[3] = 16;
            loc_27[4] = 6;
            loc_27[5] = 3;
            loc_27[6] = 2;

            loc_17[0] = 1;
            loc_17[1] = 3;
            loc_17[2] = 4;
            loc_17[3] = 5;
            loc_17[4] = 6;
            loc_17[5] = 7;

            loc_17[6] = 9;
            loc_17[7] = 10;
            loc_17[8] = 12;
            loc_17[9] = 14;
            loc_17[10] = 15;
            loc_17[11] = 16;
            loc_17[12] = 17;

            loc_17[13] = 21;
            loc_17[14] = 22;
            loc_17[15] = 25;
        }


//end7


        else if (temp.x < local_grid_center_x_temp && temp.y < local_grid_center_y_temp && temp.z < local_grid_center_z_temp && temp.x >= local_grid_center_x_temp - voxel_split_unit_PL_hw4 && temp.y < local_grid_center_y_temp && temp.z >= local_grid_center_z_temp - voxel_split_unit_PL_hw4) // -x -y -z
        {
            to_be_added_index[0] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[1] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[2] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[3] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[4] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[5] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[6] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 0;

            to_be_added_index_17[5] = -voxel_split_array_size_PL_z_array_size - 1;
            to_be_added_index_17[0] = -voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[1] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[2] = -voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[3] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[4] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size - 1;
            
            to_be_added_index_17[6] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[7] = 1;
            to_be_added_index_17[8] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[9] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[10] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;
            to_be_added_index_17[11] = -1;
            to_be_added_index_17[12] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;

            to_be_added_index_17[13] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[14] = voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[15] = voxel_split_array_size_PL_z_array_size - 1;

            loc = 9;
            loc_27[0] = 26;
            loc_27[1] = 14;
            loc_27[2] = 10;
            loc_27[3] = 18;
            loc_27[4] = 6;
            loc_27[5] = 4;
            loc_27[6] = 2;

            loc_17[0] = 19;
            loc_17[1] = 21;
            loc_17[2] = 22;
            loc_17[3] = 23;
            loc_17[4] = 24;
            loc_17[5] = 25;

            loc_17[6] = 9;
            loc_17[7] = 10;
            loc_17[8] = 12;
            loc_17[9] = 14;
            loc_17[10] = 15;
            loc_17[11] = 16;
            loc_17[12] = 17;

            loc_17[13] = 3;
            loc_17[14] = 4;
            loc_17[15] = 7;
        }

        else if (temp.x < local_grid_center_x_temp && temp.y < local_grid_center_y_temp && temp.z < local_grid_center_z_temp && temp.x >= local_grid_center_x_temp - voxel_split_unit_PL_hw4 && temp.y < local_grid_center_y_temp && temp.z < local_grid_center_z_temp - voxel_split_unit_PL_hw4) // -x -y -z
        {
            to_be_added_index[0] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[1] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[2] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[3] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[4] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[5] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[6] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 0;

            to_be_added_index_17[5] = -voxel_split_array_size_PL_z_array_size - 1;
            to_be_added_index_17[0] = -voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[1] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[2] = -voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[3] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[4] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size - 1;
            
            to_be_added_index_17[6] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[7] = 1;
            to_be_added_index_17[8] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[9] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[10] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;
            to_be_added_index_17[11] = -1;
            to_be_added_index_17[12] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;

            to_be_added_index_17[13] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[14] = voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[15] = voxel_split_array_size_PL_z_array_size - 1;

            loc = 8;
            loc_27[0] = 26;
            loc_27[1] = 14;
            loc_27[2] = 10;
            loc_27[3] = 18;
            loc_27[4] = 6;
            loc_27[5] = 4;
            loc_27[6] = 2;

            loc_17[0] = 19;
            loc_17[1] = 21;
            loc_17[2] = 22;
            loc_17[3] = 23;
            loc_17[4] = 24;
            loc_17[5] = 25;

            loc_17[6] = 9;
            loc_17[7] = 10;
            loc_17[8] = 12;
            loc_17[9] = 14;
            loc_17[10] = 15;
            loc_17[11] = 16;
            loc_17[12] = 17;

            loc_17[13] = 3;
            loc_17[14] = 4;
            loc_17[15] = 7;
        }

        else if (temp.x < local_grid_center_x_temp && temp.y < local_grid_center_y_temp && temp.z < local_grid_center_z_temp && temp.x < local_grid_center_x_temp - voxel_split_unit_PL_hw4 && temp.y < local_grid_center_y_temp && temp.z >= local_grid_center_z_temp - voxel_split_unit_PL_hw4) // -x -y -z
        {
            to_be_added_index[0] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[1] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[2] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[3] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[4] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[5] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[6] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 0;

            to_be_added_index_17[5] = -voxel_split_array_size_PL_z_array_size - 1;
            to_be_added_index_17[0] = -voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[1] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[2] = -voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[3] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[4] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size - 1;
            
            to_be_added_index_17[6] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[7] = 1;
            to_be_added_index_17[8] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[9] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[10] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;
            to_be_added_index_17[11] = -1;
            to_be_added_index_17[12] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;

            to_be_added_index_17[13] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[14] = voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[15] = voxel_split_array_size_PL_z_array_size - 1;

            loc = 1;
            loc_27[0] = 26;
            loc_27[1] = 14;
            loc_27[2] = 10;
            loc_27[3] = 18;
            loc_27[4] = 6;
            loc_27[5] = 4;
            loc_27[6] = 2;

            loc_17[0] = 19;
            loc_17[1] = 21;
            loc_17[2] = 22;
            loc_17[3] = 23;
            loc_17[4] = 24;
            loc_17[5] = 25;

            loc_17[6] = 9;
            loc_17[7] = 10;
            loc_17[8] = 12;
            loc_17[9] = 14;
            loc_17[10] = 15;
            loc_17[11] = 16;
            loc_17[12] = 17;

            loc_17[13] = 3;
            loc_17[14] = 4;
            loc_17[15] = 7;
        }

        else if (temp.x < local_grid_center_x_temp && temp.y < local_grid_center_y_temp && temp.z < local_grid_center_z_temp && temp.x < local_grid_center_x_temp - voxel_split_unit_PL_hw4 && temp.y < local_grid_center_y_temp && temp.z < local_grid_center_z_temp - voxel_split_unit_PL_hw4) // -x -y -z
        {
            to_be_added_index[0] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[1] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[2] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[3] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[4] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + -1;
            to_be_added_index[5] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[6] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 0;

            to_be_added_index_17[5] = -voxel_split_array_size_PL_z_array_size - 1;
            to_be_added_index_17[0] = -voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[1] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[2] = -voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[3] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[4] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size - 1;
            
            to_be_added_index_17[6] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index_17[7] = 1;
            to_be_added_index_17[8] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[9] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[10] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;
            to_be_added_index_17[11] = -1;
            to_be_added_index_17[12] = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1;

            to_be_added_index_17[13] = -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[14] = voxel_split_array_size_PL_z_array_size;
            to_be_added_index_17[15] = voxel_split_array_size_PL_z_array_size - 1;

            loc = 0;
            loc_27[0] = 26;
            loc_27[1] = 14;
            loc_27[2] = 10;
            loc_27[3] = 18;
            loc_27[4] = 6;
            loc_27[5] = 4;
            loc_27[6] = 2;

            loc_17[0] = 19;
            loc_17[1] = 21;
            loc_17[2] = 22;
            loc_17[3] = 23;
            loc_17[4] = 24;
            loc_17[5] = 25;

            loc_17[6] = 9;
            loc_17[7] = 10;
            loc_17[8] = 12;
            loc_17[9] = 14;
            loc_17[10] = 15;
            loc_17[11] = 16;
            loc_17[12] = 17;

            loc_17[13] = 3;
            loc_17[14] = 4;
            loc_17[15] = 7;
        }

        else
        {

        }
        
        // 13 14 12
        // 4 22 10 16
        // 11 17
        // 9 15     
        // 5 23
        // 3 21
        // 1 7
        // 19 25
        // 2
        // 0
        // 20
        // 18
        // 8
        // 6
        // 26
        // 24

//每个voxel与中心voxel的索引差

        // 13 14 voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size 12 -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size
        // 4 voxel_split_array_size_PL_z_array_size 22 -voxel_split_array_size_PL_z_array_size 10 1 16 -1
        // 11 voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 17 voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1
        // 9 -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 15 -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1    
        // 5 voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size 23 voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size
        // 3 -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size 21 -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size
        // 1 voxel_split_array_size_PL_z_array_size + 1 7 voxel_split_array_size_PL_z_array_size - 1
        // 19 -voxel_split_array_size_PL_z_array_size + 1 25 -voxel_split_array_size_PL_z_array_size - 1
        // 2 voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size + 1
        // 0 -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size + 1
        // 20 voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size + 1
        // 18 -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size + 1
        // 8 voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size - 1
        // 6 -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size - 1
        // 26 voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size - 1
        // 24 -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size - 1



//每个voxel与中心voxel的索引差
        voxel_sint to_be_added_index_table[27] = {0, voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size, -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size,
                                           voxel_split_array_size_PL_z_array_size, -voxel_split_array_size_PL_z_array_size, 1, -1,
                                           voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1, voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1,
                                           -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1, -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size -1,
                                           voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size, voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size,
                                           -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size, -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size,
                                           voxel_split_array_size_PL_z_array_size + 1, voxel_split_array_size_PL_z_array_size - 1, -voxel_split_array_size_PL_z_array_size + 1, -voxel_split_array_size_PL_z_array_size - 1,
                                           voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size + 1,
                                           -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size + 1,
                                           voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size + 1,
                                           -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size + 1,
                                           voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size - 1,
                                           -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + voxel_split_array_size_PL_z_array_size - 1,
                                           voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size - 1,
                                           -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size - voxel_split_array_size_PL_z_array_size - 1
                                          };



//对于32个可能的query所在区域，每种可能情况下27个voxel中最近的subvoxel索引
        uint_sub_voxel_size loc_table[27*32] = {0,1,2,3,4,5,6,7, 8,9,10,11,12,13,14,15, 16,17,18,19,20,21,22,23, 24,25,26,27,28,29,30,31,   0,1,2,3,4,5,6,7, 0,1,2,3,4,5,6,7, 0,1,2,3,4,5,6,7, 0,1,2,3,4,5,6,7,   24,25,26,27,28,29,30,31, 24,25,26,27,28,29,30,31, 24,25,26,27,28,29,30,31, 24,25,26,27,28,29,30,31,                               
                               0,1,2,3,0,1,2,3, 8,9,10,11,8,9,10,11, 16,17,18,19,16,17,18,19, 24,25,26,27,24,25,26,27,   4,5,6,7,4,5,6,7, 12,13,14,15,12,13,14,15, 20,21,22,23,20,21,22,23, 28,29,30,31,28,29,30,31,
                               0,0,0,0,4,4,4,4, 8,8,8,8,12,12,12,12, 16,16,16,16,20,20,20,20, 24,24,24,24,28,28,28,28,   3,3,3,3,7,7,7,7, 11,11,11,11,15,15,15,15, 19,19,19,19,23,23,23,23, 27,27,27,27,31,31,31,31,                       
                               0,0,0,0,4,4,4,4, 0,0,0,0,4,4,4,4, 0,0,0,0,4,4,4,4, 0,0,0,0,4,4,4,4,   3,3,3,3,7,7,7,7, 3,3,3,3,7,7,7,7, 3,3,3,3,7,7,7,7, 3,3,3,3,7,7,7,7,
                               24,24,24,24,28,28,28,28, 24,24,24,24,28,28,28,28, 24,24,24,24,28,28,28,28, 24,24,24,24,28,28,28,28,   27,27,27,27,31,31,31,31, 27,27,27,27,31,31,31,31, 27,27,27,27,31,31,31,31, 27,27,27,27,31,31,31,31,                               
                               0,1,2,3,0,1,2,3, 0,1,2,3,0,1,2,3, 0,1,2,3,0,1,2,3, 0,1,2,3,0,1,2,3,   4,5,6,7,4,5,6,7, 4,5,6,7,4,5,6,7, 4,5,6,7,4,5,6,7, 4,5,6,7,4,5,6,7,
                               24,25,26,27,24,25,26,27, 24,25,26,27,24,25,26,27, 24,25,26,27,24,25,26,27, 24,25,26,27,24,25,26,27,   28,29,30,31,28,29,30,31, 28,29,30,31,28,29,30,31, 28,29,30,31,28,29,30,31, 28,29,30,31,28,29,30,31,                               
                               0,0,0,0,0,0,0,0, 8,8,8,8,8,8,8,8, 16,16,16,16,16,16,16,16, 24,24,24,24,24,24,24,24,   3,3,3,3,3,3,3,3, 11,11,11,11,11,11,11,11, 19,19,19,19,19,19,19,19, 27,27,27,27,27,27,27,27,
                               4,4,4,4,4,4,4,4, 12,12,12,12,12,12,12,12, 20,20,20,20,20,20,20,20, 28,28,28,28,28,28,28,28,   7,7,7,7,7,7,7,7, 15,15,15,15,15,15,15,15, 23,23,23,23,23,23,23,23, 31,31,31,31,31,31,31,31,                               
                               0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,   24,24,24,24,24,24,24,24, 24,24,24,24,24,24,24,24, 24,24,24,24,24,24,24,24, 24,24,24,24,24,24,24,24,
                               4,4,4,4,4,4,4,4, 4,4,4,4,4,4,4,4, 4,4,4,4,4,4,4,4, 4,4,4,4,4,4,4,4,   28,28,28,28,28,28,28,28, 28,28,28,28,28,28,28,28, 28,28,28,28,28,28,28,28, 28,28,28,28,28,28,28,28,                              
                               3,3,3,3,3,3,3,3, 3,3,3,3,3,3,3,3, 3,3,3,3,3,3,3,3, 3,3,3,3,3,3,3,3,   27,27,27,27,27,27,27,27, 27,27,27,27,27,27,27,27, 27,27,27,27,27,27,27,27, 27,27,27,27,27,27,27,27,
                               7,7,7,7,7,7,7,7, 7,7,7,7,7,7,7,7, 7,7,7,7,7,7,7,7, 7,7,7,7,7,7,7,7,   31,31,31,31,31,31,31,31, 31,31,31,31,31,31,31,31, 31,31,31,31,31,31,31,31, 31,31,31,31,31,31,31,31
                              };

//2*2*2=8
/*
        uint_sub_voxel_size loc_table[27*8] = {0,1,2,3,4,5,6,7, 0,1,2,3,0,1,2,3, 4,5,6,7,4,5,6,7,
                               0,1,0,1,4,5,4,5, 2,3,2,3,6,7,6,7, 0,0,2,2,4,4,6,6, 1,1,3,3,5,5,7,7,
                               0,0,2,2,0,0,2,2, 1,1,3,3,1,1,3,3,
                               4,4,6,6,4,4,6,6, 5,5,7,7,5,5,7,7,
                               0,1,0,1,0,1,0,1, 2,3,2,3,2,3,2,3,
                               4,5,4,5,4,5,4,5, 6,7,6,7,6,7,6,7,
                               0,0,0,0,4,4,4,4, 1,1,1,1,5,5,5,5,
                               2,2,2,2,6,6,6,6, 3,3,3,3,7,7,7,7,
                               0,0,0,0,0,0,0,0, 4,4,4,4,4,4,4,4,
                               2,2,2,2,2,2,2,2, 6,6,6,6,6,6,6,6,
                               1,1,1,1,1,1,1,1, 5,5,5,5,5,5,5,5,
                               3,3,3,3,3,3,3,3, 7,7,7,7,7,7,7,7
                              };
*/

//备用
/*
        uint_sub_voxel_size sub_table[18*32] = {0,1,8,9, 4,5,12,13, 2,10,18,17,16, 6,14,22,21,20,   0,1,2,8,9,10, 4,5,6,12,13,14, 16,17,18,20,21,22,   1,2,3,9,10,11, 5,6,7,13,14,15, 17,18,19,21,22,23,   2,3,10,11, 6,7,14,15, 1,9,17,18,19, 5,13,21,22,23,
        0,1,8,9, 4,5,12,13, 2,10,18,17,16, 6,14,22,21,20,   0,1,2,8,9,10, 4,5,6,12,13,14, 16,17,18,20,21,22,   1,2,3,9,10,11, 5,6,7,13,14,15, 17,18,19,21,22,23,   2,3,10,11, 6,7,14,15, 1,9,17,18,19, 5,13,21,22,23,
        0,1,8,9,16,17, 4,5,12,13,20,21, 2,10,18,6,14,22,   0,1,2,8,9,10,16,17,18, 4,5,6,12,13,14,20,21,22,   1,2,3,9,10,11,17,18,19, 5,6,7,13,14,15,21,22,23,   2,3,10,11,18,19, 6,7,14,15,22,23, 1,9,17,5,13,21,
        0,1,8,9,16,17, 4,5,12,13,20,21, 2,10,18,6,14,22,   0,1,2,8,9,10,16,17,18, 4,5,6,12,13,14,20,21,22,   1,2,3,9,10,11,17,18,19, 5,6,7,13,14,15,21,22,23,   2,3,10,11,18,19, 6,7,14,15,22,23, 1,9,17,5,13,21,
        8,9,16,17,24,25, 12,13,20,21,28,29, 10,18,26,14,22,30,   8,9,10,16,17,18,24,25,26, 12,13,14,20,21,22,28,29,30,   9,10,11,17,18,19,25,26,27, 13,14,15,21,22,23,29,30,31,   10,11,18,19,26,27, 14,15,22,23,30,31, 9,17,25,13,21,29,
        8,9,16,17,24,25, 12,13,20,21,28,29, 10,18,26,14,22,30,   8,9,10,16,17,18,24,25,26, 12,13,14,20,21,22,28,29,30,   9,10,11,17,18,19,25,26,27, 13,14,15,21,22,23,29,30,31,   10,11,18,19,26,27, 14,15,22,23,30,31, 9,17,25,13,21,29,
        16,17,24,25, 20,21,28,29, 8,9,10,18,26, 12,13,14,22,30,   16,17,18,24,25,26, 20,21,22,28,29,30, 8,9,10,12,13,14,   17,18,19,25,26,27, 21,22,23,29,30,31, 9,10,11,13,14,15,   18,19,26,27, 22,23,30,31, 11,10,9,17,25, 15,14,13,21,29,
        16,17,24,25, 20,21,28,29, 8,9,10,18,26, 12,13,14,22,30,   16,17,18,24,25,26, 20,21,22,28,29,30, 8,9,10,12,13,14,   17,18,19,25,26,27, 21,22,23,29,30,31, 9,10,11,13,14,15,   18,19,26,27, 22,23,30,31, 11,10,9,17,25, 15,14,13,21,29
        };
*/

        voxel_sint candidate_region_index = query_index_temp;

        voxel_int sub_voxel_flag = sub_voxel_flag_index_PL[candidate_region_index];

//中心voxel未分成subvoxel
        if (sub_voxel_flag == k_sub_voxel_number_max)
        {

            valid_near_voxels.write(candidate_region_index);
            voxel_flag.write(1);

            for (uint_sub_voxel_size ii = 0; ii < 16; ii++)
            {
                candidate_region_index = query_index_temp + to_be_added_index_17[ii];
                //candidate_region_index = query_index_temp + to_be_added_index_table[ii];

                if (candidate_region_index > total_calculated_voxel_size or candidate_region_index < 0)
                {
                    valid_near_voxels.write(total_calculated_voxel_size);
                    voxel_flag.write(1);
                }
                else
                {
                    sub_voxel_flag = sub_voxel_flag_index_PL[candidate_region_index];
                    //周边voxel未分成subvoxel
                    if (sub_voxel_flag == k_sub_voxel_number_max)
                    {

                        valid_near_voxels.write(candidate_region_index);
                        voxel_flag.write(1);

                    }
                    //周边voxel分成了subvoxel
                    else
                    {
                        //voxel_int candidate_search_sub_voxel_index = sub_voxel_flag + loc_table[32*loc_17[ii]+loc];
                        voxel_int candidate_search_sub_voxel_index = sub_voxel_flag + loc_table[32*ii+loc];
                        valid_near_voxels.write(candidate_search_sub_voxel_index);
                        voxel_flag.write(0);
       
                    }
                }
            }
//中心voxel分成了subvoxel
        }
        else
        {

            for (uint_64 paral_ins_index = 0; paral_ins_index < 10; paral_ins_index++) //todo
            {

                valid_near_voxels.write(sub_voxel_flag + paral_ins_index);
                //valid_near_voxels.write(sub_voxel_flag + sub_table[32*loc + paral_ins_index]);
                voxel_flag.write(2);
            }

            for (uint_sub_voxel_size ii = 0; ii < 7; ii++)
            {
                candidate_region_index = query_index_temp + to_be_added_index[ii];

                if (candidate_region_index > total_calculated_voxel_size or candidate_region_index < 0)
                {
                    valid_near_voxels.write(total_calculated_voxel_size);
                    voxel_flag.write(1);
                }
                else
                {
                    sub_voxel_flag = sub_voxel_flag_index_PL[candidate_region_index];
                    //周边voxel未分成subvoxel
                    if (sub_voxel_flag == k_sub_voxel_number_max)
                    {

                        valid_near_voxels.write(candidate_region_index);
                        voxel_flag.write(3);

                    }
                    //周边voxel分成了subvoxel
                    else
                    {
                        voxel_int candidate_search_sub_voxel_index = sub_voxel_flag + loc_table[32*loc_27[ii]+loc];
                        
                        valid_near_voxels.write(candidate_search_sub_voxel_index);
                        voxel_flag.write(2);
              
                    }
                }
            }
        }

    }
}



My_PointXYZI_HW16 data_set_buffer_x[k_data_set_buffer_size];
My_PointXYZI_HW16 data_set_buffer_y[k_data_set_buffer_size];
My_PointXYZI_HW16 data_set_buffer_z[k_data_set_buffer_size];
inthw16 original_data_index_buffer[k_data_set_buffer_size];

//取neighbor voxel中的点
void get_point_hw(hls::stream<My_PointXYZI_HW>& reference_input, hls::stream<My_PointXYZI_HW>& reference_input1, hls::stream<voxel_int>& query_index4, hls::stream<voxel_int>& query_index5, hls::stream<count_uint>& valid_near_voxels, hls::stream<uint_4>& voxel_flag, hls::stream<My_PointXYZI_HW16> ordered_ref16_xs[pararead], hls::stream<My_PointXYZI_HW16> ordered_ref16_ys[pararead], hls::stream<My_PointXYZI_HW16> ordered_ref16_zs[pararead], hls::stream<inthw16> original_dataset_index16s[pararead], indexint* index16, indexint* subindex16, int query_set_size, hls::stream<My_PointXYZI_HW16> current_data_point_x[num16], hls::stream<My_PointXYZI_HW16> current_data_point_y[num16], hls::stream<My_PointXYZI_HW16> current_data_point_z[num16], hls::stream<inthw16> original_dataset_index_temp[num16], int* hash0)
{
//#pragma HLS ARRAY_PARTITION dim=1 factor=60 type=cyclic variable=original_data_index_buffer
//#pragma HLS ARRAY_PARTITION dim=1 factor=60 type=cyclic variable=data_set_buffer

    voxel_int first_query_voxel_first_index = hash0[0]; // todo

    DEBUG_INFO(hash0[0]);

    voxel_int first;
    
    if (hash0[0] != 0)
    {
        first = index16[first_query_voxel_first_index](22,0);
    }
    else
    {
        first = 0;
    }

    DEBUG_INFO(first);

    int dataset_buffer_max_index_PL = 0;
    int dataset_buffer_min_index_PL = 0;

    if (first - 0.5*k_data_set_buffer_size < 0)
    {
        dataset_buffer_max_index_PL = dataset_buffer_min_index_PL = 0;
    }
    else
    {
        dataset_buffer_max_index_PL = dataset_buffer_min_index_PL = first - 0.5*k_data_set_buffer_size;
    }

    int y = dataset_buffer_max_index_PL % pararead;

    dataset_buffer_max_index_PL = dataset_buffer_max_index_PL - y;

    dataset_buffer_min_index_PL = dataset_buffer_max_index_PL;

//first query前的数据
    for (voxel_int i = 0; i < dataset_buffer_max_index_PL; i+=pararead)
    {
#pragma HLS LOOP_TRIPCOUNT max=k_query_set_size_max min=k_query_set_size_min
        if (dataset_buffer_max_index_PL < packs)
		{
            for(int para_i = 0; para_i < pararead; para_i ++)
            {
    #pragma HLS UNROLL
                data_set_buffer_x[0] = ordered_ref16_xs[para_i].read();
                data_set_buffer_y[0] = ordered_ref16_ys[para_i].read();
                data_set_buffer_z[0] = ordered_ref16_zs[para_i].read();
                original_data_index_buffer[0] = original_dataset_index16s[para_i].read();
            }
		}
		else break;
    }

//初始化buffer
    for (voxel_int i = 0; i < k_data_set_buffer_size; i+=pararead)
	{

        count_uint buffer_new_index = dataset_buffer_max_index_PL % k_data_set_buffer_size;		//transform index from dataset to dataset_buffer

		if (dataset_buffer_max_index_PL < packs)
		{
			for(int para_i = 0; para_i < pararead; para_i ++)
            {
    #pragma HLS UNROLL
                data_set_buffer_x[buffer_new_index+para_i] = ordered_ref16_xs[para_i].read();
                data_set_buffer_y[buffer_new_index+para_i] = ordered_ref16_ys[para_i].read();
                data_set_buffer_z[buffer_new_index+para_i] = ordered_ref16_zs[para_i].read();
                original_data_index_buffer[buffer_new_index+para_i] = original_dataset_index16s[para_i].read();
                dataset_buffer_max_index_PL = dataset_buffer_max_index_PL + 1;
            }
		}
		else break;
	}

    

    get_point_hw_label11:for (voxel_int i = 0; i < query_set_size; i++)
    {
#pragma HLS LOOP_TRIPCOUNT max=k_query_set_size_max min=k_query_set_size_min

        get_point_label0:for (uint_64 count_ij = 0; count_ij < bundlevoxel; count_ij++)
        {
#pragma HLS loop_flatten
#pragma HLS pipeline II=1

            indexint max = 0;
            if (count_ij == 0)
            {
                My_PointXYZI_HW temp = reference_input.read();
                reference_input1.write(temp);

                voxel_int query_index_temp = query_index4.read();	// read query_i_copy
                query_index5.write(query_index_temp);
            }

            count_uint current_search_hash_local;
            uint_4 voxel_flag_local;

            current_search_hash_local = valid_near_voxels.read();
            voxel_flag_local = voxel_flag.read();

            if (current_search_hash_local == total_calculated_voxel_size)
            {
                current_data_point_x[0].write(zerox);
                current_data_point_y[0].write(zeroy);
                current_data_point_z[0].write(zeroz);
                original_dataset_index_temp[0].write(zeroindex);

                current_data_point_x[1].write(zerox);
                current_data_point_y[1].write(zeroy);
                current_data_point_z[1].write(zeroz);
                original_dataset_index_temp[1].write(zeroindex);

                current_data_point_x[2].write(zerox);
                current_data_point_y[2].write(zeroy);
                current_data_point_z[2].write(zeroz);
                original_dataset_index_temp[2].write(zeroindex);

            }
            else
            {
                //从周边voxel中取点
                if (voxel_flag_local == 1 | voxel_flag_local == 3)
                {

                    if (index16[current_search_hash_local](31,29) != 0)
                    {
                        if (max < index16[current_search_hash_local](22,0))
                        {
                            max = index16[current_search_hash_local](22,0);
                        }
                        voxel_int data_set_buffer_index = index16[current_search_hash_local](22,0) % k_data_set_buffer_size;
                        
                        //该voxel有1-16个点，共1组
                        if (index16[current_search_hash_local](31,29) == 1)
                        {
                            current_data_point_x[0].write(data_set_buffer_x[data_set_buffer_index]);
                            current_data_point_y[0].write(data_set_buffer_y[data_set_buffer_index]);
                            current_data_point_z[0].write(data_set_buffer_z[data_set_buffer_index]);
                            original_dataset_index_temp[0].write(original_data_index_buffer[data_set_buffer_index]);
                            current_data_point_x[1].write(zerox);
                            current_data_point_y[1].write(zeroy);
                            current_data_point_z[1].write(zeroz);
                            original_dataset_index_temp[1].write(zeroindex);
                            current_data_point_x[2].write(zerox);
                            current_data_point_y[2].write(zeroy);
                            current_data_point_z[2].write(zeroz);
                            original_dataset_index_temp[2].write(zeroindex);
                        }
                        //该voxel有17-32个点，共2组
                        else if (index16[current_search_hash_local](31,29) == 2)
                        {
                            current_data_point_x[0].write(data_set_buffer_x[data_set_buffer_index]);
                            current_data_point_y[0].write(data_set_buffer_y[data_set_buffer_index]);
                            current_data_point_z[0].write(data_set_buffer_z[data_set_buffer_index]);
                            original_dataset_index_temp[0].write(original_data_index_buffer[data_set_buffer_index]);
                            current_data_point_x[1].write(data_set_buffer_x[data_set_buffer_index+1]);
                            current_data_point_y[1].write(data_set_buffer_y[data_set_buffer_index+1]);
                            current_data_point_z[1].write(data_set_buffer_z[data_set_buffer_index+1]);
                            original_dataset_index_temp[1].write(original_data_index_buffer[data_set_buffer_index+1]);
                            current_data_point_x[2].write(zerox);
                            current_data_point_y[2].write(zeroy);
                            current_data_point_z[2].write(zeroz);
                            original_dataset_index_temp[2].write(zeroindex);
                        }
                        //该voxel有33-48个点，共3组
                        else if (index16[current_search_hash_local](31,29) == 3)
                        {
                            current_data_point_x[0].write(data_set_buffer_x[data_set_buffer_index]);
                            current_data_point_y[0].write(data_set_buffer_y[data_set_buffer_index]);
                            current_data_point_z[0].write(data_set_buffer_z[data_set_buffer_index]);
                            original_dataset_index_temp[0].write(original_data_index_buffer[data_set_buffer_index]);
                            current_data_point_x[1].write(data_set_buffer_x[data_set_buffer_index+1]);
                            current_data_point_y[1].write(data_set_buffer_y[data_set_buffer_index+1]);
                            current_data_point_z[1].write(data_set_buffer_z[data_set_buffer_index+1]);
                            original_dataset_index_temp[1].write(original_data_index_buffer[data_set_buffer_index+1]);
                            current_data_point_x[2].write(data_set_buffer_x[data_set_buffer_index+2]);
                            current_data_point_y[2].write(data_set_buffer_y[data_set_buffer_index+2]);
                            current_data_point_z[2].write(data_set_buffer_z[data_set_buffer_index+2]);
                            original_dataset_index_temp[2].write(original_data_index_buffer[data_set_buffer_index+2]);
                        }
                        else
                        {
                            current_data_point_x[0].write(zerox);
                            current_data_point_y[0].write(zeroy);
                            current_data_point_z[0].write(zeroz);
                            original_dataset_index_temp[0].write(zeroindex);
                            current_data_point_x[1].write(zerox);
                            current_data_point_y[1].write(zeroy);
                            current_data_point_z[1].write(zeroz);
                            original_dataset_index_temp[1].write(zeroindex);
                            current_data_point_x[2].write(zerox);
                            current_data_point_y[2].write(zeroy);
                            current_data_point_z[2].write(zeroz);
                            original_dataset_index_temp[2].write(zeroindex);
                        }
                    }
                    else
                    {
                        current_data_point_x[0].write(zerox);
                        current_data_point_y[0].write(zeroy);
                        current_data_point_z[0].write(zeroz);
                        original_dataset_index_temp[0].write(zeroindex);
                        current_data_point_x[1].write(zerox);
                        current_data_point_y[1].write(zeroy);
                        current_data_point_z[1].write(zeroz);
                        original_dataset_index_temp[1].write(zeroindex);
                        current_data_point_x[2].write(zerox);
                        current_data_point_y[2].write(zeroy);
                        current_data_point_z[2].write(zeroz);
                        original_dataset_index_temp[2].write(zeroindex);
                    }
                }
                //从周边subvoxel中取点
                else
                {
                    if (subindex16[current_search_hash_local](31,29) != 0)
                    {
                        if (max < subindex16[current_search_hash_local](22,0))
                        {
                            max = subindex16[current_search_hash_local](22,0);
                        }
                        voxel_int data_set_buffer_index = subindex16[current_search_hash_local](22,0) % k_data_set_buffer_size;
                        
                        //该subvoxel有1-16个点，共1组
                        if (subindex16[current_search_hash_local](31,29) == 1)
                        {
                            current_data_point_x[0].write(data_set_buffer_x[data_set_buffer_index]);
                            current_data_point_y[0].write(data_set_buffer_y[data_set_buffer_index]);
                            current_data_point_z[0].write(data_set_buffer_z[data_set_buffer_index]);
                            original_dataset_index_temp[0].write(original_data_index_buffer[data_set_buffer_index]);
                            current_data_point_x[1].write(zerox);
                            current_data_point_y[1].write(zeroy);
                            current_data_point_z[1].write(zeroz);
                            original_dataset_index_temp[1].write(zeroindex);
                            current_data_point_x[2].write(zerox);
                            current_data_point_y[2].write(zeroy);
                            current_data_point_z[2].write(zeroz);
                            original_dataset_index_temp[2].write(zeroindex);
                        }
                        //该subvoxel有17-32个点，共2组
                        else if (subindex16[current_search_hash_local](31,29) == 2)
                        {
                            current_data_point_x[0].write(data_set_buffer_x[data_set_buffer_index]);
                            current_data_point_y[0].write(data_set_buffer_y[data_set_buffer_index]);
                            current_data_point_z[0].write(data_set_buffer_z[data_set_buffer_index]);
                            original_dataset_index_temp[0].write(original_data_index_buffer[data_set_buffer_index]);
                            current_data_point_x[1].write(data_set_buffer_x[data_set_buffer_index+1]);
                            current_data_point_y[1].write(data_set_buffer_y[data_set_buffer_index+1]);
                            current_data_point_z[1].write(data_set_buffer_z[data_set_buffer_index+1]);
                            original_dataset_index_temp[1].write(original_data_index_buffer[data_set_buffer_index+1]);
                            current_data_point_x[2].write(zerox);
                            current_data_point_y[2].write(zeroy);
                            current_data_point_z[2].write(zeroz);
                            original_dataset_index_temp[2].write(zeroindex);
                        }
                        //该voxel有33-48个点，共3组
                        else if (subindex16[current_search_hash_local](31,29) == 3)
                        {
                            current_data_point_x[0].write(data_set_buffer_x[data_set_buffer_index]);
                            current_data_point_y[0].write(data_set_buffer_y[data_set_buffer_index]);
                            current_data_point_z[0].write(data_set_buffer_z[data_set_buffer_index]);
                            original_dataset_index_temp[0].write(original_data_index_buffer[data_set_buffer_index]);
                            current_data_point_x[1].write(data_set_buffer_x[data_set_buffer_index+1]);
                            current_data_point_y[1].write(data_set_buffer_y[data_set_buffer_index+1]);
                            current_data_point_z[1].write(data_set_buffer_z[data_set_buffer_index+1]);
                            original_dataset_index_temp[1].write(original_data_index_buffer[data_set_buffer_index+1]);
                            current_data_point_x[2].write(data_set_buffer_x[data_set_buffer_index+2]);
                            current_data_point_y[2].write(data_set_buffer_y[data_set_buffer_index+2]);
                            current_data_point_z[2].write(data_set_buffer_z[data_set_buffer_index+2]);
                            original_dataset_index_temp[2].write(original_data_index_buffer[data_set_buffer_index+2]);
                        }
                        else
                        {
                            current_data_point_x[0].write(zerox);
                            current_data_point_y[0].write(zeroy);
                            current_data_point_z[0].write(zeroz);
                            original_dataset_index_temp[0].write(zeroindex);
                            current_data_point_x[1].write(zerox);
                            current_data_point_y[1].write(zeroy);
                            current_data_point_z[1].write(zeroz);
                            original_dataset_index_temp[1].write(zeroindex);
                            current_data_point_x[2].write(zerox);
                            current_data_point_y[2].write(zeroy);
                            current_data_point_z[2].write(zeroz);
                            original_dataset_index_temp[2].write(zeroindex);
                        }
                    }
                    else
                    {
                        current_data_point_x[0].write(zerox);
                        current_data_point_y[0].write(zeroy);
                        current_data_point_z[0].write(zeroz);
                        original_dataset_index_temp[0].write(zeroindex);
                        current_data_point_x[1].write(zerox);
                        current_data_point_y[1].write(zeroy);
                        current_data_point_z[1].write(zeroz);
                        original_dataset_index_temp[1].write(zeroindex);
                        current_data_point_x[2].write(zerox);
                        current_data_point_y[2].write(zeroy);
                        current_data_point_z[2].write(zeroz);
                        original_dataset_index_temp[2].write(zeroindex);
                    }
                }
            }

            

            //更新buffer
            if (dataset_buffer_max_index_PL < packs && max > (dataset_buffer_min_index_PL + dataset_buffer_max_index_PL)/2 + k_dataset_buffer_gap)
            {
                    count_uint buffer_new_index = dataset_buffer_max_index_PL % k_data_set_buffer_size;		//transform index from dataset to dataset_buffer
                        
                        for(int para_i = 0; para_i < pararead; para_i ++)
                        {
        #pragma HLS UNROLL
                            data_set_buffer_x[buffer_new_index+para_i] = ordered_ref16_xs[para_i].read();
                            data_set_buffer_y[buffer_new_index+para_i] = ordered_ref16_ys[para_i].read();
                            data_set_buffer_z[buffer_new_index+para_i] = ordered_ref16_zs[para_i].read();
                            original_data_index_buffer[buffer_new_index+para_i] = original_dataset_index16s[para_i].read();
                            dataset_buffer_max_index_PL = dataset_buffer_max_index_PL + 1;
                            dataset_buffer_min_index_PL = dataset_buffer_min_index_PL + 1;
                        }
            }
        }
    }

    for (voxel_int i = dataset_buffer_max_index_PL; i < packs; i+=pararead)
    {
#pragma HLS LOOP_TRIPCOUNT max=k_query_set_size_max min=k_query_set_size_min

            for(int para_i = 0; para_i < pararead; para_i ++)
            {
    #pragma HLS UNROLL
                data_set_buffer_x[0] = ordered_ref16_xs[para_i].read();
                data_set_buffer_y[0] = ordered_ref16_ys[para_i].read();
                data_set_buffer_z[0] = ordered_ref16_zs[para_i].read();
                original_data_index_buffer[0] = original_dataset_index16s[para_i].read();
            }
    }

}

//把上个function中16个1组的点整理成48个点并行select knn
void unpack(hls::stream<My_PointXYZI_HW>& reference_input, hls::stream<voxel_int>& query_index4, hls::stream<voxel_int>& query_index5, hls::stream<type_dist_hw> candidate_distance[k_transform_neighbor_num], hls::stream<count_uint> original_dataset_index_local[k_transform_neighbor_num], hls::stream<My_PointXYZI_HW16> current_data_point_x[num16], hls::stream<My_PointXYZI_HW16> current_data_point_y[num16], hls::stream<My_PointXYZI_HW16> current_data_point_z[num16], hls::stream<inthw16> original_dataset_index_temp[num16], int query_set_size)
{
    My_PointXYZI_HW16 current_data_point_xtemp;
    My_PointXYZI_HW16 current_data_point_ytemp;
    My_PointXYZI_HW16 current_data_point_ztemp;
    My_PointXYZI_HW current_data_point[16];
    inthw16 original_dataset_index;
    My_PointXYZI_HW temp;

    for (voxel_int i = 0; i < query_set_size; i++)
    {
#pragma HLS LOOP_TRIPCOUNT max=k_query_set_size_max min=k_query_set_size_min

        for (uint_64 count_i = 0; count_i < bundlevoxel; count_i++)
        {
#pragma HLS PIPELINE

            if (count_i == 0)
            {
                temp = reference_input.read();

    	        voxel_int query_index_temp = query_index4.read();	// read query_i_copy
	            query_index5.write(query_index_temp);
            }

            for (uint_64 count3 = 0; count3 < num16; count3++)
            {

                current_data_point_xtemp = current_data_point_x[count3].read();
                current_data_point_ytemp = current_data_point_y[count3].read();
                current_data_point_ztemp = current_data_point_z[count3].read();
                original_dataset_index = original_dataset_index_temp[count3].read();

                current_data_point[0].x = current_data_point_xtemp.p1;
                current_data_point[0].y = current_data_point_ytemp.p1;
                current_data_point[0].z = current_data_point_ztemp.p1;
                candidate_distance[16*count3+0].write(cal_dist_hw(temp, current_data_point[0]));
                current_data_point[1].x = current_data_point_xtemp.p2;
                current_data_point[1].y = current_data_point_ytemp.p2;
                current_data_point[1].z = current_data_point_ztemp.p2;
                candidate_distance[16*count3+1].write(cal_dist_hw(temp, current_data_point[1]));
                current_data_point[2].x = current_data_point_xtemp.p3;
                current_data_point[2].y = current_data_point_ytemp.p3;
                current_data_point[2].z = current_data_point_ztemp.p3;
                candidate_distance[16*count3+2].write(cal_dist_hw(temp, current_data_point[2]));
                current_data_point[3].x = current_data_point_xtemp.p4;
                current_data_point[3].y = current_data_point_ytemp.p4;
                current_data_point[3].z = current_data_point_ztemp.p4;
                candidate_distance[16*count3+3].write(cal_dist_hw(temp, current_data_point[3]));

                current_data_point[4].x = current_data_point_xtemp.p5;
                current_data_point[4].y = current_data_point_ytemp.p5;
                current_data_point[4].z = current_data_point_ztemp.p5;
                candidate_distance[16*count3+4].write(cal_dist_hw(temp, current_data_point[4]));
                current_data_point[5].x = current_data_point_xtemp.p6;
                current_data_point[5].y = current_data_point_ytemp.p6;
                current_data_point[5].z = current_data_point_ztemp.p6;
                candidate_distance[16*count3+5].write(cal_dist_hw(temp, current_data_point[5]));
                current_data_point[6].x = current_data_point_xtemp.p7;
                current_data_point[6].y = current_data_point_ytemp.p7;
                current_data_point[6].z = current_data_point_ztemp.p7;
                candidate_distance[16*count3+6].write(cal_dist_hw(temp, current_data_point[6]));
                current_data_point[7].x = current_data_point_xtemp.p8;
                current_data_point[7].y = current_data_point_ytemp.p8;
                current_data_point[7].z = current_data_point_ztemp.p8;
                candidate_distance[16*count3+7].write(cal_dist_hw(temp, current_data_point[7]));

                current_data_point[8].x = current_data_point_xtemp.p9;
                current_data_point[8].y = current_data_point_ytemp.p9;
                current_data_point[8].z = current_data_point_ztemp.p9;
                candidate_distance[16*count3+8].write(cal_dist_hw(temp, current_data_point[8]));
                current_data_point[9].x = current_data_point_xtemp.p10;
                current_data_point[9].y = current_data_point_ytemp.p10;
                current_data_point[9].z = current_data_point_ztemp.p10;
                candidate_distance[16*count3+9].write(cal_dist_hw(temp, current_data_point[9]));
                current_data_point[10].x = current_data_point_xtemp.p11;
                current_data_point[10].y = current_data_point_ytemp.p11;
                current_data_point[10].z = current_data_point_ztemp.p11;
                candidate_distance[16*count3+10].write(cal_dist_hw(temp, current_data_point[10]));
                current_data_point[11].x = current_data_point_xtemp.p12;
                current_data_point[11].y = current_data_point_ytemp.p12;
                current_data_point[11].z = current_data_point_ztemp.p12;
                candidate_distance[16*count3+11].write(cal_dist_hw(temp, current_data_point[11]));

                current_data_point[12].x = current_data_point_xtemp.p13;
                current_data_point[12].y = current_data_point_ytemp.p13;
                current_data_point[12].z = current_data_point_ztemp.p13;
                candidate_distance[16*count3+12].write(cal_dist_hw(temp, current_data_point[12]));
                current_data_point[13].x = current_data_point_xtemp.p14;
                current_data_point[13].y = current_data_point_ytemp.p14;
                current_data_point[13].z = current_data_point_ztemp.p14;
                candidate_distance[16*count3+13].write(cal_dist_hw(temp, current_data_point[13]));
                current_data_point[14].x = current_data_point_xtemp.p15;
                current_data_point[14].y = current_data_point_ytemp.p15;
                current_data_point[14].z = current_data_point_ztemp.p15;
                candidate_distance[16*count3+14].write(cal_dist_hw(temp, current_data_point[14]));
                current_data_point[15].x = current_data_point_xtemp.p16;
                current_data_point[15].y = current_data_point_ytemp.p16;
                current_data_point[15].z = current_data_point_ztemp.p16;
                candidate_distance[16*count3+15].write(cal_dist_hw(temp, current_data_point[15]));

                original_dataset_index_local[16*count3+0].write(original_dataset_index.p1);
                original_dataset_index_local[16*count3+1].write(original_dataset_index.p2);
                original_dataset_index_local[16*count3+2].write(original_dataset_index.p3);
                original_dataset_index_local[16*count3+3].write(original_dataset_index.p4);

                original_dataset_index_local[16*count3+4].write(original_dataset_index.p5);
                original_dataset_index_local[16*count3+5].write(original_dataset_index.p6);
                original_dataset_index_local[16*count3+6].write(original_dataset_index.p7);
                original_dataset_index_local[16*count3+7].write(original_dataset_index.p8);

                original_dataset_index_local[16*count3+8].write(original_dataset_index.p9);
                original_dataset_index_local[16*count3+9].write(original_dataset_index.p10);
                original_dataset_index_local[16*count3+10].write(original_dataset_index.p11);
                original_dataset_index_local[16*count3+11].write(original_dataset_index.p12);

                original_dataset_index_local[16*count3+12].write(original_dataset_index.p13);
                original_dataset_index_local[16*count3+13].write(original_dataset_index.p14);
                original_dataset_index_local[16*count3+14].write(original_dataset_index.p15);
                original_dataset_index_local[16*count3+15].write(original_dataset_index.p16);
            
            }
        }
    }
}



//48个点并行算knn
void select_knn_hw(hls::stream<count_uint>& query_result_s, hls::stream<type_dist_hw>& nearest_distance_s, hls::stream<type_dist_hw> candidate_distance_s[k_transform_neighbor_num], hls::stream<count_uint> original_dataset_index[k_transform_neighbor_num], int query_set_size, hls::stream<voxel_int>& query_index5)
{
    
    count_uint nearest_index_PL_local[k_nearest_number_max];
    type_dist_hw nearest_distance_PL_local[k_nearest_number_max];

    count_uint original_dataset_index_local[k_transform_neighbor_num];
    type_dist_hw candidate_distance[k_transform_neighbor_num];
    k_selection_int cmp_array[k_nearest_number_max + k_transform_neighbor_num];

    type_dist_hw candidate_distance_8[pip];
    count_uint original_dataset_index_local_8[pip];

    

    for (voxel_int i = 0; i < query_set_size; i++)
    {
#pragma HLS LOOP_TRIPCOUNT max=k_query_set_size_max min=k_query_set_size_min        

        

    loop_dataflow_cal_knn:
        for (uint_64 count_i = 0; count_i < bundlevoxel; count_i++)
        {
#pragma HLS loop_flatten
#pragma HLS pipeline II=1

            for (int i = 0; i < pip; i++)
            {
                original_dataset_index_local_8[i] = 0;
            }
            if (count_i == 0)
            {
                voxel_int query_index_temp = query_index5.read();	// read query_i_copy

            loop_reset_result_array:

                nearest_index_PL_local[0] = 0;
                nearest_distance_PL_local[0] = 100;
            }

            for (uint_128 cache_temp_index = 0; cache_temp_index < k_transform_neighbor_num; cache_temp_index++)
            {
                candidate_distance[cache_temp_index] = candidate_distance_s[cache_temp_index].read();
                original_dataset_index_local[cache_temp_index] = original_dataset_index[cache_temp_index].read();
            }

			cmp_array[0] = 0;

			for (uint_128 parallel_i = 0; parallel_i < k_transform_neighbor_num; parallel_i++)
			{
				cmp_array[parallel_i + 1 ] = -1;

				for (uint_128 parallel_j = 0; parallel_j < k_transform_neighbor_num; parallel_j++)
				{
					if (candidate_distance[parallel_i] > candidate_distance[parallel_j] || ( candidate_distance[parallel_i] == candidate_distance[parallel_j] && parallel_i >= parallel_j ))
						cmp_array[parallel_i + 1 ] ++;
				}

				if (candidate_distance[parallel_i] >= nearest_distance_PL_local[0])
					cmp_array[parallel_i + 1 ] ++;
				else cmp_array[0] ++;

			}

            //48选8
            for (uint_128 parallel_i = 0; parallel_i < pip; parallel_i++)
			{

            	for (uint_128 parallel_j = 0; parallel_j < k_transform_neighbor_num/pip; parallel_j++)
			    {
                    if (cmp_array[parallel_i*k_transform_neighbor_num/pip + parallel_j + 1 ] < 1)
                    {
                        candidate_distance_8[parallel_i] = candidate_distance[parallel_i*k_transform_neighbor_num/pip + parallel_j];
                        original_dataset_index_local_8[parallel_i] = original_dataset_index_local[parallel_i*k_transform_neighbor_num/pip + parallel_j];
                    }
                }
			}

            //8选1
            for (uint_128 parallel_i = 0; parallel_i < pip; parallel_i++)
			{

                if (original_dataset_index_local_8[parallel_i] != 0)
                {
					nearest_distance_PL_local[0] = candidate_distance_8[parallel_i];
					nearest_index_PL_local[0] = original_dataset_index_local_8[parallel_i];
                }

			}

            if (count_i == bundlevoxel - 1)
            {
                query_result_s.write(nearest_index_PL_local[0]);

                nearest_distance_s.write(nearest_distance_PL_local[0]);
            }
        } //while()    

    }
    
}

void out(hls::stream<count_uint>& query_result_s, hls::stream<type_dist_hw>& nearest_distance_s, count_uint* query_result, type_dist_hw* nearest_distance, int query_set_size)
{

    for(int query_i = 0; query_i < query_set_size; query_i++)
    {
#pragma HLS pipeline II=1
#pragma HLS LOOP_TRIPCOUNT max=k_query_set_size_max min=k_query_set_size_min
        query_result[query_i] = query_result_s.read();
        nearest_distance[query_i] = nearest_distance_s.read();
    }
}

void DSVS_search_hw(My_PointXYZI_HW16* ordered_ref16_x, My_PointXYZI_HW16* ordered_ref16_y, My_PointXYZI_HW16* ordered_ref16_z,
	inthw16* original_dataset_index16,
	My_PointXYZI_HW* query_set, int query_set_size,
    count_uint* query_result, type_dist_hw* nearest_distance,
    indexint* index16, voxel_int* sub_voxel_flag_index_PL, indexint* subindex16)
{
// #pragma HLS INTERFACE mode=s_axilite bundle=CTRL port=dataset_buffer_min_index_PL
// #pragma HLS INTERFACE mode=s_axilite bundle=CTRL port=dataset_buffer_max_index_PL
// #pragma HLS INTERFACE mode=s_axilite bundle=CTRL port=query_set_size
// #pragma HLS INTERFACE mode=s_axilite bundle=CTRL port=data_set_size
// #pragma HLS INTERFACE mode=s_axilite bundle=CTRL port=return
#pragma HLS INTERFACE mode=m_axi bundle=gmem1 depth=0 port=subindex16
#pragma HLS INTERFACE mode=m_axi bundle=gmem6 depth=1158840 port=sub_voxel_flag_index_PL
#pragma HLS INTERFACE mode=m_axi bundle=gmem1 depth=1158840 port=index16
#pragma HLS INTERFACE mode=m_axi bundle=gmem6 depth=10000 port=nearest_distance
#pragma HLS INTERFACE mode=m_axi bundle=gmem2 depth=10000 port=query_result
#pragma HLS INTERFACE mode=m_axi bundle=gmem2 depth=10000 port=query_set
#pragma HLS INTERFACE mode=m_axi bundle=gmem5 depth=34044 port=original_dataset_index16
#pragma HLS INTERFACE mode=m_axi bundle=gmem0 depth=34044 port=ordered_ref16_x
#pragma HLS INTERFACE mode=m_axi bundle=gmem3 depth=34044 port=ordered_ref16_y
#pragma HLS INTERFACE mode=m_axi bundle=gmem4 depth=34044 port=ordered_ref16_z
#pragma HLS DATAFLOW

    hls::stream<My_PointXYZI_HW> reference_input;

    hls::stream<My_PointXYZI_HW16> ordered_ref16_xs[pararead];

    hls::stream<My_PointXYZI_HW16> ordered_ref16_ys[pararead];

    hls::stream<My_PointXYZI_HW16> ordered_ref16_zs[pararead];

    hls::stream<inthw16> original_dataset_index16s[pararead];

    hls::stream<count_uint> valid_near_voxels;

    hls::stream<uint_4> voxel_flag;

    hls::stream<type_dist_hw> candidate_distance[k_transform_neighbor_num];

    hls::stream<count_uint> original_dataset_index_local[k_transform_neighbor_num];

    hls::stream<voxel_int> query_index1;

    hls::stream<voxel_int> query_index2;

    hls::stream<voxel_int> query_index3;

    hls::stream<voxel_int> query_index4;

    hls::stream<voxel_int> query_index5;

    hls::stream<type_point_hw> local_grid_center_x;

    hls::stream<type_point_hw> local_grid_center_y;

    hls::stream<type_point_hw> local_grid_center_z;

    hls::stream<My_PointXYZI_HW> reference_input1;

    hls::stream<My_PointXYZI_HW> reference_input2;

    hls::stream<My_PointXYZI_HW> reference_input3;

    hls::stream<My_PointXYZI_HW> reference_input4;

    hls::stream<My_PointXYZI_HW16> current_data_point_x[num16];

    hls::stream<My_PointXYZI_HW16> current_data_point_y[num16];

    hls::stream<My_PointXYZI_HW16> current_data_point_z[num16];

    hls::stream<inthw16> original_dataset_index_temp[num16];

    hls::stream<count_uint> query_result_s;

    hls::stream<type_dist_hw> nearest_distance_s;

    //#pragma HLS STREAM variable=reference_input depth=1000

    #pragma HLS STREAM variable=reference_input1 depth=10000

    #pragma HLS STREAM variable=reference_input2 depth=10000

    #pragma HLS STREAM variable=reference_input3 depth=20

    #pragma HLS STREAM variable=reference_input4 depth=20

    //#pragma HLS STREAM variable=valid_near_voxels depth=40000

    //#pragma HLS STREAM variable=voxel_flag depth=40000

    // #pragma HLS STREAM variable=candidate_neighbors depth=k_transform_neighbor_num*2

    //#pragma HLS STREAM variable=candidate_distance depth=k_transform_neighbor_num*2000

    //#pragma HLS STREAM variable=original_dataset_index_local depth=k_transform_neighbor_num*2000

    #pragma HLS STREAM variable=query_index1 depth=10000

    #pragma HLS STREAM variable=query_index2 depth=10000

    #pragma HLS STREAM variable=query_index3 depth=20

    #pragma HLS STREAM variable=query_index4 depth=20

    //#pragma HLS STREAM variable=query_index5 depth=1000

    #pragma HLS STREAM variable=local_grid_center_x depth=10000

    #pragma HLS STREAM variable=local_grid_center_y depth=10000

    #pragma HLS STREAM variable=local_grid_center_z depth=10000

    //#pragma HLS STREAM variable=current_data_point_x depth=5000

    //#pragma HLS STREAM variable=current_data_point_y depth=5000

    //#pragma HLS STREAM variable=current_data_point_z depth=5000

    //#pragma HLS STREAM variable=original_dataset_index_temp depth=5000

    #pragma HLS STREAM variable=ordered_ref16_xs depth=8

    #pragma HLS STREAM variable=ordered_ref16_ys depth=8

    #pragma HLS STREAM variable=ordered_ref16_zs depth=8

    #pragma HLS STREAM variable=original_dataset_index16s depth=8

    int hash0[1];

    input_src_hw(query_set, reference_input, query_set_size);
    
    calculate_hash_stream_hw(reference_input, reference_input1, query_index1, query_set_size, hash0);

    input_reference_hw(ordered_ref16_x, ordered_ref16_y, ordered_ref16_z, original_dataset_index16, ordered_ref16_xs, ordered_ref16_ys, ordered_ref16_zs, original_dataset_index16s);

    initial_hw(reference_input1, reference_input2, query_index1, query_index2, local_grid_center_x, local_grid_center_y, local_grid_center_z, query_set_size);

    search_near_cells_hw(reference_input2, reference_input3, query_index2, query_index3, valid_near_voxels, voxel_flag, local_grid_center_x, local_grid_center_y, local_grid_center_z, sub_voxel_flag_index_PL, query_set_size);

    get_point_hw(reference_input3, reference_input4, query_index3, query_index4, valid_near_voxels, voxel_flag, ordered_ref16_xs, ordered_ref16_ys, ordered_ref16_zs, original_dataset_index16s, index16, subindex16, query_set_size, current_data_point_x, current_data_point_y, current_data_point_z, original_dataset_index_temp, hash0);

    unpack(reference_input4, query_index4, query_index5, candidate_distance, original_dataset_index_local, current_data_point_x, current_data_point_y, current_data_point_z, original_dataset_index_temp, query_set_size);

    select_knn_hw(query_result_s, nearest_distance_s, candidate_distance, original_dataset_index_local, query_set_size, query_index5);

    out(query_result_s, nearest_distance_s, query_result, nearest_distance, query_set_size);
}


