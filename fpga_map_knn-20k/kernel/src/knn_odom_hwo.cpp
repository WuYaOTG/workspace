#undef __ARM_NEON__
#undef __ARM_NEON
#include "knn_odom.h"
#define __ARM_NEON__
#define __ARM_NEON

static voxel_int voxel_split_array_size_PL_x_array_size;
static voxel_int voxel_split_array_size_PL_y_array_size;
static voxel_int voxel_split_array_size_PL_z_array_size;

static voxel_int total_calculated_voxel_size;

static type_point_hw data_set_max_min_PL_xmin_hw;
static type_point_hw data_set_max_min_PL_ymin_hw;
static type_point_hw data_set_max_min_PL_zmin_hw;

static type_point_hw voxel_split_unit_PL_hw;
static type_point_hw voxel_split_unit_PL_hw4;

static int packs;

#define SUB_VOXEL_SPLIT



void calculate_hash_stream_hw(hls::stream<My_PointXYZI_HW>& KNN_reference_set, hls::stream<My_PointXYZI_HW>& KNN_reference_set1, hls::stream<voxel_int>& data_set_hash, int KNN_reference_set_size)
{
    for (count_uint i = 0; i < KNN_reference_set_size; i++)
	{
#pragma HLS LOOP_TRIPCOUNT max=k_query_set_size_max min=k_query_set_size_min
        My_PointXYZI_HW temp = KNN_reference_set.read();
        KNN_reference_set1.write(temp);
        
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
            DEBUG_INFO(data_hash);
        }

        data_set_hash.write(data_hash);
    }
}



void input_src_hw(My_PointXYZI_HW* KNN_reference_set, hls::stream<My_PointXYZI_HW>& reference_input, int KNN_reference_set_size)
{ 
    for(uint_query_size i = 0; i < KNN_reference_set_size; i++)
    {
#pragma HLS LOOP_TRIPCOUNT max=k_query_set_size_max min=k_query_set_size_min
        My_PointXYZI_HW temp_image_pixel = KNN_reference_set[i];
        My_PointXYZI_HW temp_image_pixel_hw;
        temp_image_pixel_hw.x = temp_image_pixel.x;
        temp_image_pixel_hw.y = temp_image_pixel.y;
        temp_image_pixel_hw.z = temp_image_pixel.z;
        reference_input.write(temp_image_pixel_hw);
    }
}

void input_reference_hw(My_PointXYZI_HW16* ordered_ref16_x, My_PointXYZI_HW16* ordered_ref16_y, My_PointXYZI_HW16* ordered_ref16_z, inthw16* original_dataset_index16, hls::stream<My_PointXYZI_HW16> ordered_ref16_xs[pararead], hls::stream<My_PointXYZI_HW16> ordered_ref16_ys[pararead], hls::stream<My_PointXYZI_HW16> ordered_ref16_zs[pararead], hls::stream<inthw16> original_dataset_index16s[pararead], int KNN_reference_set_size)
{ 
    My_PointXYZI_HW16 temp_image_pixelx;
    My_PointXYZI_HW16 temp_image_pixely;
    My_PointXYZI_HW16 temp_image_pixelz;

    inthw16 temp;

    //voxel_int first_query_voxel_first_index = 277022; // todo

    //voxel_int first = index16[first_query_voxel_first_index](22,0);

    //DEBUG_INFO(first);

    voxel_int first = 10153;
    int temp1;
    if (first - 0.5*k_data_set_buffer_size < 0)
    {
        temp1 = 0;
    }
    else
    {
        temp1 = first - 0.5*k_data_set_buffer_size;
    }

    for(voxel_int i = temp1; i < KNN_reference_set_size; i+=pararead)
    {
#pragma HLS LOOP_TRIPCOUNT max=34044 min=34044
        for(int para_i = 0; para_i < pararead; para_i ++)
        {
#pragma HLS pipeline II=1
            temp_image_pixelx = ordered_ref16_x[i+para_i];
            temp_image_pixely = ordered_ref16_y[i+para_i];
            temp_image_pixelz = ordered_ref16_z[i+para_i];
            temp = original_dataset_index16[i+para_i];

            //My_PointXYZI_HW temp_image_pixel_hw;
            //temp_image_pixel_hw.x = temp_image_pixel.x;
            //temp_image_pixel_hw.y = temp_image_pixel.y;
            //temp_image_pixel_hw.z = temp_image_pixel.z;

            ordered_ref16_xs[para_i].write(temp_image_pixelx);
            ordered_ref16_ys[para_i].write(temp_image_pixely);
            ordered_ref16_zs[para_i].write(temp_image_pixelz);
            original_dataset_index16s[para_i].write(temp);
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



void initial_hw(hls::stream<My_PointXYZI_HW>& reference_input, hls::stream<My_PointXYZI_HW>& reference_input1, hls::stream<voxel_int>& query_index1, hls::stream<voxel_int>& query_index2, hls::stream<type_point_hw>& local_grid_center_x, hls::stream<type_point_hw>& local_grid_center_y, hls::stream<type_point_hw>& local_grid_center_z, int query_set_size)
{
    for (uint_query_size i = 0; i < query_set_size; i++)
    {
#pragma HLS LOOP_TRIPCOUNT max=k_query_set_size_max min=k_query_set_size_min

    	My_PointXYZI_HW temp = reference_input.read();
    	reference_input1.write(temp);

    	voxel_int query_index_temp = query_index1.read();	// read query_i_copy
	    query_index2.write(query_index_temp);

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
    }
}

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

        uint_sub_voxel_size loc_27[7];
        uint_sub_voxel_size loc_17[16];
        voxel_sint to_be_added_index[7];
        voxel_sint to_be_added_index_17[16];
        uint_sub_voxel_size loc = 0; // back 3 7 front 1 5
                                     //      2 6       0 4



        //search candidate voxels.

        //2*2*2=8
        /*
        if (temp.x >= local_grid_center_x_temp && temp.y >= local_grid_center_y_temp && temp.z >= local_grid_center_z_temp) // x y z
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

            loc = 7;
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
        
        else if (temp.x < local_grid_center_x_temp && temp.y >= local_grid_center_y_temp && temp.z >= local_grid_center_z_temp) // -x y z
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

            loc = 3;
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
        else if (temp.x >= local_grid_center_x_temp && temp.y < local_grid_center_y_temp && temp.z >= local_grid_center_z_temp) // x -y z
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

            loc = 5;
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
        else if (temp.x >= local_grid_center_x_temp && temp.y >= local_grid_center_y_temp && temp.z < local_grid_center_z_temp) // x y -z
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

            loc = 6;
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
        else if (temp.x < local_grid_center_x_temp && temp.y < local_grid_center_y_temp && temp.z >= local_grid_center_z_temp) // -x -y z
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

            loc = 1;
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
        else if (temp.x >= local_grid_center_x_temp && temp.y < local_grid_center_y_temp && temp.z < local_grid_center_z_temp) // x -y -z
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

            loc = 4;
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
        else if (temp.x < local_grid_center_x_temp && temp.y >= local_grid_center_y_temp && temp.z < local_grid_center_z_temp) // -x y -z
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

            loc = 2;
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
        else if (temp.x < local_grid_center_x_temp && temp.y < local_grid_center_y_temp && temp.z < local_grid_center_z_temp) // -x -y -z
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
            DEBUG_INFO(20230601);
        }
        */

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
            //DEBUG_INFO(20230601);
        }
        
        //DEBUG_INFO(i);
        //DEBUG_INFO(loc);
        
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

        uint_sub_voxel_size sub_table[18*32] = {0,1,8,9, 4,5,12,13, 2,10,18,17,16, 6,14,22,21,20,   0,1,2,8,9,10, 4,5,6,12,13,14, 16,17,18,20,21,22,   1,2,3,9,10,11, 5,6,7,13,14,15, 17,18,19,21,22,23,   2,3,10,11, 6,7,14,15, 1,9,17,18,19, 5,13,21,22,23,
        0,1,8,9, 4,5,12,13, 2,10,18,17,16, 6,14,22,21,20,   0,1,2,8,9,10, 4,5,6,12,13,14, 16,17,18,20,21,22,   1,2,3,9,10,11, 5,6,7,13,14,15, 17,18,19,21,22,23,   2,3,10,11, 6,7,14,15, 1,9,17,18,19, 5,13,21,22,23,
        0,1,8,9,16,17, 4,5,12,13,20,21, 2,10,18,6,14,22,   0,1,2,8,9,10,16,17,18, 4,5,6,12,13,14,20,21,22,   1,2,3,9,10,11,17,18,19, 5,6,7,13,14,15,21,22,23,   2,3,10,11,18,19, 6,7,14,15,22,23, 1,9,17,5,13,21,
        0,1,8,9,16,17, 4,5,12,13,20,21, 2,10,18,6,14,22,   0,1,2,8,9,10,16,17,18, 4,5,6,12,13,14,20,21,22,   1,2,3,9,10,11,17,18,19, 5,6,7,13,14,15,21,22,23,   2,3,10,11,18,19, 6,7,14,15,22,23, 1,9,17,5,13,21,
        8,9,16,17,24,25, 12,13,20,21,28,29, 10,18,26,14,22,30,   8,9,10,16,17,18,24,25,26, 12,13,14,20,21,22,28,29,30,   9,10,11,17,18,19,25,26,27, 13,14,15,21,22,23,29,30,31,   10,11,18,19,26,27, 14,15,22,23,30,31, 9,17,25,13,21,29,
        8,9,16,17,24,25, 12,13,20,21,28,29, 10,18,26,14,22,30,   8,9,10,16,17,18,24,25,26, 12,13,14,20,21,22,28,29,30,   9,10,11,17,18,19,25,26,27, 13,14,15,21,22,23,29,30,31,   10,11,18,19,26,27, 14,15,22,23,30,31, 9,17,25,13,21,29,
        16,17,24,25, 20,21,28,29, 8,9,10,18,26, 12,13,14,22,30,   16,17,18,24,25,26, 20,21,22,28,29,30, 8,9,10,12,13,14,   17,18,19,25,26,27, 21,22,23,29,30,31, 9,10,11,13,14,15,   18,19,26,27, 22,23,30,31, 11,10,9,17,25, 15,14,13,21,29,
        16,17,24,25, 20,21,28,29, 8,9,10,18,26, 12,13,14,22,30,   16,17,18,24,25,26, 20,21,22,28,29,30, 8,9,10,12,13,14,   17,18,19,25,26,27, 21,22,23,29,30,31, 9,10,11,13,14,15,   18,19,26,27, 22,23,30,31, 11,10,9,17,25, 15,14,13,21,29
        };

        voxel_sint candidate_region_index = query_index_temp;

        
        if (candidate_region_index > total_calculated_voxel_size or candidate_region_index < 0)
        {
            //DEBUG_INFO(2023041101);
            //DEBUG_INFO(candidate_region_index);
        }
        

        voxel_int sub_voxel_flag = sub_voxel_flag_index_PL[candidate_region_index];

        if (sub_voxel_flag == k_sub_voxel_number_max)
        {

            valid_near_voxels.write(candidate_region_index);
            voxel_flag.write(1);

            //bool flagm = 0;
            for (uint_sub_voxel_size ii = 0; ii < 16; ii++)
            {
                candidate_region_index = query_index_temp + to_be_added_index_17[ii];
                //candidate_region_index = query_index_temp + to_be_added_index_table[ii];
                if (candidate_region_index > total_calculated_voxel_size or candidate_region_index < 0)
                {
                    //DEBUG_INFO(2023041102);
                    //DEBUG_INFO(candidate_region_index);
                    valid_near_voxels.write(total_calculated_voxel_size);
                    voxel_flag.write(1);
                    //flagm = 1;
                }
                else
                {
                    sub_voxel_flag = sub_voxel_flag_index_PL[candidate_region_index];
                    if (sub_voxel_flag == k_sub_voxel_number_max)
                    {

                        valid_near_voxels.write(candidate_region_index);
                        voxel_flag.write(1);

                    }
                    else
                    {

                        //voxel_int candidate_search_sub_voxel_index = sub_voxel_flag + loc_table[32*loc_17[ii]+loc];
                        voxel_int candidate_search_sub_voxel_index = sub_voxel_flag + loc_table[32*ii+loc];
                        valid_near_voxels.write(candidate_search_sub_voxel_index);
                        voxel_flag.write(0);
       
                    }
                }
            }
            //if (flagm == 1)
            //{
                //valid_near_voxels.write(7000000 + 1);
                //voxel_flag.write(1);
            //}
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
                    //DEBUG_INFO(2023041103);
                    //DEBUG_INFO(candidate_region_index);
                    valid_near_voxels.write(total_calculated_voxel_size);
                    voxel_flag.write(1);
                }
                else
                {
                    sub_voxel_flag = sub_voxel_flag_index_PL[candidate_region_index];

                    if (sub_voxel_flag == k_sub_voxel_number_max)
                    {

                        valid_near_voxels.write(candidate_region_index);
                        voxel_flag.write(3);

                    }
                    else
                    {
                        voxel_int candidate_search_sub_voxel_index = sub_voxel_flag + loc_table[32*loc_27[ii]+loc];
                        
                        valid_near_voxels.write(candidate_search_sub_voxel_index);
                        voxel_flag.write(2);
              
                    }
                }
            }
            //valid_near_voxels.write(7000000 + 1);
            //voxel_flag.write(1);
        }
       
        //valid_near_voxels.write(k_reference_set_size + 1);
        //voxel_flag.write(1);
    }
}

/*
void search_candidate_neighbors_hw(hls::stream<My_PointXYZI_HW>& reference_input, hls::stream<My_PointXYZI_HW>& reference_input1, hls::stream<voxel_int>& query_index3, hls::stream<voxel_int>& query_index4, hls::stream<count_uint>& valid_near_voxels, hls::stream<uint_4>& voxel_flag, hls::stream<count_uint> candidate_neighbors[k_transform_neighbor_num], count_uint* voxel_first_index_PL, count_uint* sub_voxel_first_index_PL, int query_set_size)
{
    //int c0612 = 0;
    for (uint_query_size i = 0; i < query_set_size; i++)
	{
#pragma HLS LOOP_TRIPCOUNT max=k_query_set_size_max min=k_query_set_size_min

    	My_PointXYZI_HW temp = reference_input.read();
    	reference_input1.write(temp);

        voxel_int query_index_temp = query_index3.read();	// read query_i_copy
	    query_index4.write(query_index_temp);

        //uint_64 candidate_neighbors_count_local = 0;

	    count_uint over_threshold_num = 0;
	    count_uint over_threshold_sub_voxel[k_over_thre_sub_voxel_num];
        
        loop_valid_voxels:
        for(uint_64 region_i = 0; region_i < 18; region_i++)
        {
            count_uint current_search_hash_local;
            uint_4 voxel_flag_local;

            current_search_hash_local = valid_near_voxels.read();
            voxel_flag_local = voxel_flag.read();

            //if ((voxel_flag_local == 2 | voxel_flag_local == 3) &  region_i == 15)
            if (current_search_hash_local != k_reference_set_size + 1)
            {

            {

                //{

                    //if near voxel
                    if (voxel_flag_local == 1 | voxel_flag_local == 3)
                    {
                        count_uint current_search_hash = current_search_hash_local;
                        if (current_search_hash >= 0 && current_search_hash < total_calculated_voxel_size)
                        {

                            count_uint hash_start_index = voxel_first_index_PL[current_search_hash];
                            count_uint current_voxel_size = voxel_first_index_PL[current_search_hash + 1] - hash_start_index; //size of current voxel.

                            if (current_voxel_size > 0)
                            {
                                for (uint_64 temp_index = 0; temp_index < k_transform_neighbor_num; temp_index++)
                                {
                                    if (temp_index < current_voxel_size)
                                    {
                                        count_uint candidate_neighbor_index = hash_start_index + temp_index;
                                        candidate_neighbors[temp_index].write(candidate_neighbor_index);
                                        //c0612 = c0612 + 1;
                                    }
                                    else
                                    {
                                        candidate_neighbors[temp_index].write(k_reference_set_size);
                                    }
                                }
                                //candidate_neighbors_count_local = candidate_neighbors_count_local + 1;
                            }
                        }
                    }
    #ifdef SUB_VOXEL_SPLIT
                    else
                    {

                        count_uint current_search_sub_hash = current_search_hash_local;
                        //if (current_search_sub_hash < k_sub_voxel_number_max-1)	//remove sub_voxel out of range
                        //{
                            count_uint sub_hash_start_index = sub_voxel_first_index_PL[current_search_sub_hash];
                            count_uint current_sub_voxel_size;

                            if(sub_voxel_first_index_PL[current_search_sub_hash + 1] > sub_hash_start_index)
                            {
                                current_sub_voxel_size = sub_voxel_first_index_PL[current_search_sub_hash + 1] - sub_hash_start_index; //size of current voxel.
                                for (uint_64 temp_index = 0; temp_index < k_transform_neighbor_num; temp_index++)
                                {
                                    if (temp_index < current_sub_voxel_size)
                                    {
                                        count_uint candidate_neighbor_index = sub_hash_start_index + temp_index;
                                        candidate_neighbors[temp_index].write(candidate_neighbor_index);
                                        //c0612 = c0612 + 1;
                                    }
                                    else
                                    {
                                        candidate_neighbors[temp_index].write(k_reference_set_size);
                                    }
                                }

                                //candidate_neighbors_count_local = candidate_neighbors_count_local + 1;
                            //else
                            //    current_sub_voxel_size = 0;

                            //if (candidate_neighbors_count_local < 50 - 1 && current_sub_voxel_size > 0) //todo
                            //{
                                //if (current_sub_voxel_size >= k_transform_neighbor_num)
                                //{
                                    //if(over_threshold_num < k_over_thre_sub_voxel_num)
                                    //if(over_threshold_num < 34)
                                    //{
                                        //over_threshold_sub_voxel[over_threshold_num] = current_search_sub_hash;
                                        //over_threshold_num = over_threshold_num + 1;
                                    //}
                                //}
                                //for (uint_64 temp_index = 0; temp_index < k_transform_neighbor_num; temp_index++)
                                //{
                                    //if (temp_index < current_sub_voxel_size)
                                    //{
                                        //count_uint candidate_neighbor_index = sub_hash_start_index + temp_index;
                                        //candidate_neighbors[temp_index].write(candidate_neighbor_index);
                                    //}
                                    //else
                                    //{
                                        //candidate_neighbors[temp_index].write(k_reference_set_size);
                                    //}
                                //}

                                //Candidate_neighbors_count_local = candidate_neighbors_count_local + 1;
                            }
                        //}

                        //}
                    }
    #endif
                }
            }	//if !empty()
            else
            {
                break;
            }
        }	// for valid_voxels

        /*
        //for over threshold sub-voxel
        for(count_uint over_sub_index = 0; over_sub_index < over_threshold_num; over_sub_index++)
        {
            count_uint current_search_sub_hash = over_threshold_sub_voxel[over_sub_index];
            //int sub_hash_start_index = sub_voxel_first_index_PL[current_search_sub_hash] + k_transform_neighbor_num;
            count_uint sub_hash_start_index = sub_voxel_first_index_PL[current_search_sub_hash] + 60;
            count_uint current_sub_voxel_size = sub_voxel_first_index_PL[current_search_sub_hash + 1] - sub_hash_start_index; //size of current voxel.

            //if (candidate_neighbors_count_local < k_select_loop_num - 1 && current_sub_voxel_size > 0)
            if (candidate_neighbors_count_local < 300 - 1 && current_sub_voxel_size > 0)
            {
                //for (int temp_index = 0; temp_index < k_transform_neighbor_num; temp_index++)
                for (count_uint temp_index = 0; temp_index < 60; temp_index++)
                {
                    /*
                    ###
                    if (temp_index == k_transform_neighbor_num - 1)
                    {
                        if (current_sub_voxel_size >= k_transform_neighbor_num - 1)
                        {
                            //candidate_neighbors[k_transform_neighbor_num - 1 + candidate_neighbors_count_local*k_transform_neighbor_num + i*k_transform_neighbor_num*k_select_loop_num] = k_transform_neighbor_num - 1;	//valid data's number
                            candidate_neighbors[k_transform_neighbor_num - 1].write(k_transform_neighbor_num - 1);
                        }
                        else
                            //candidate_neighbors[k_transform_neighbor_num - 1 + candidate_neighbors_count_local*k_transform_neighbor_num + i*k_transform_neighbor_num*k_select_loop_num] = current_sub_voxel_size;	//valid data's number
                            candidate_neighbors[k_transform_neighbor_num - 1].write(current_sub_voxel_size);
                    }
                    else
                    {
                        if (temp_index < current_sub_voxel_size)
                        {
                            int candidate_neighbor_index = sub_hash_start_index + temp_index;
                            //candidate_neighbors[temp_index + candidate_neighbors_count_local*k_transform_neighbor_num + i*k_transform_neighbor_num*k_select_loop_num] = candidate_neighbor_index;
                            candidate_neighbors[temp_index].write(candidate_neighbor_index);
                        }
                        else
                        {
                            //candidate_neighbors[temp_index + candidate_neighbors_count_local*k_transform_neighbor_num + i*k_transform_neighbor_num*k_select_loop_num] = k_reference_set_size;
                            candidate_neighbors[temp_index].write(k_reference_set_size);
                        }
                    }
                    ###
                    */
                    /*
                    if (temp_index < current_sub_voxel_size)
                    {
                        count_uint candidate_neighbor_index = sub_hash_start_index + temp_index;
                        //candidate_neighbors[temp_index + candidate_neighbors_count_local*k_transform_neighbor_num + i*k_transform_neighbor_num*k_select_loop_num] = candidate_neighbor_index;
                        candidate_neighbors[temp_index].write(candidate_neighbor_index);
                    }
                    else
                    {
                        //candidate_neighbors[temp_index + candidate_neighbors_count_local*k_transform_neighbor_num + i*k_transform_neighbor_num*k_select_loop_num] = k_reference_set_size;
                        //candidate_neighbors[temp_index].write(k_reference_set_size);
                        candidate_neighbors[temp_index].write(7000000);
                    }
                    //
                }
            }

            candidate_neighbors_count_local = candidate_neighbors_count_local + 1;

        }
        

        search_candidate_neighbors_hw_label0:for (uint_64 temp_index = 0; temp_index < k_transform_neighbor_num; temp_index++)
        {
#pragma HLS UNROLL
            if (temp_index == k_transform_neighbor_num - 1)
                candidate_neighbors[temp_index].write(0);
            else
                candidate_neighbors[temp_index].write(k_reference_set_size + 1);
        }
        //candidate_neighbors_count_local = candidate_neighbors_count_local + 1;
        //if (candidate_neighbors_count_local > 17)
        //{
            //DEBUG_INFO(candidate_neighbors_count_local);
        //}
    }
    //DEBUG_INFO(c0612);
}
*/



My_PointXYZI_HW16 data_set_buffer_x[k_data_set_buffer_size];
My_PointXYZI_HW16 data_set_buffer_y[k_data_set_buffer_size];
My_PointXYZI_HW16 data_set_buffer_z[k_data_set_buffer_size];
inthw16 original_data_index_buffer[k_data_set_buffer_size];

/*
void initial_buffer(My_PointXYZI_HW query_set, int* voxel_first_index_PL, My_PointXYZI_HW* data_set, int* original_dataset_index, int& dataset_buffer_max_index_PL, int& dataset_buffer_min_index_PL)
{

    //int dataset_buffer_max_index_PL = 0;
	//int dataset_buffer_min_index_PL = 0;

    My_PointXYZI first_query_point[1];
	first_query_point[0].x =  query_set.x;
	first_query_point[0].y =  query_set.y;
	first_query_point[0].z =  query_set.z;
    int first_query_hash[1];
    calculate_hash(first_query_point, first_query_hash, 1);
	int first_query_voxel_first_index = voxel_first_index_PL[first_query_hash[0]];

    if (first_query_voxel_first_index - 0.5*k_data_set_buffer_size < 0)
    {
        dataset_buffer_max_index_PL = dataset_buffer_min_index_PL = 0;
    }
    else
    {
        dataset_buffer_max_index_PL = dataset_buffer_min_index_PL = first_query_voxel_first_index - 0.5*k_data_set_buffer_size;
    }

    //DEBUG_INFO(origin_data_set[0].x);
    for (int i = 0; i < k_data_set_buffer_size; i++)
	{
		//DEBUG_INFO(origin_data_set[0].x);
        //DEBUG_INFO(i);
        int buffer_new_index = dataset_buffer_max_index_PL % k_data_set_buffer_size;		//transform index from dataset to dataset_buffer
        //DEBUG_INFO(buffer_new_index);
		if (dataset_buffer_max_index_PL < k_reference_set_size)
		{
			data_set_buffer[buffer_new_index].x = data_set[dataset_buffer_max_index_PL].x;
			data_set_buffer[buffer_new_index].y = data_set[dataset_buffer_max_index_PL].y;
			data_set_buffer[buffer_new_index].z = data_set[dataset_buffer_max_index_PL].z;
			original_data_index_buffer[buffer_new_index] = original_dataset_index[dataset_buffer_max_index_PL];
			dataset_buffer_max_index_PL = dataset_buffer_max_index_PL + 1;
		}
		else break;
	}
}
*/


void get_point_hw(hls::stream<My_PointXYZI_HW>& reference_input, hls::stream<My_PointXYZI_HW>& reference_input1, hls::stream<voxel_int>& query_index4, hls::stream<voxel_int>& query_index5, hls::stream<count_uint>& valid_near_voxels, hls::stream<uint_4>& voxel_flag, hls::stream<My_PointXYZI_HW16> ordered_ref16_xs[pararead], hls::stream<My_PointXYZI_HW16> ordered_ref16_ys[pararead], hls::stream<My_PointXYZI_HW16> ordered_ref16_zs[pararead], hls::stream<inthw16> original_dataset_index16s[pararead], inthw16* original_dataset_index16, indexint* index16, indexint* subindex16, int query_set_size, int dataset_buffer_max_index_PL, int dataset_buffer_min_index_PL, int data_set_size, hls::stream<My_PointXYZI_HW16> current_data_point_x[num16], hls::stream<My_PointXYZI_HW16> current_data_point_y[num16], hls::stream<My_PointXYZI_HW16> current_data_point_z[num16], hls::stream<inthw16> original_dataset_index_temp[num16])
{
//#pragma HLS ARRAY_PARTITION dim=1 factor=60 type=cyclic variable=original_data_index_buffer
//#pragma HLS ARRAY_PARTITION dim=1 factor=60 type=cyclic variable=data_set_buffer
    //int c0513 = 0;
    //int c0618 = 0;

    voxel_int first_query_voxel_first_index = 277022; // todo

    voxel_int first = index16[first_query_voxel_first_index](22,0);

    DEBUG_INFO(first);

    if (first - 0.5*k_data_set_buffer_size < 0)
    {
        dataset_buffer_max_index_PL = dataset_buffer_min_index_PL = 0;
    }
    else
    {
        dataset_buffer_max_index_PL = dataset_buffer_min_index_PL = first - 0.5*k_data_set_buffer_size;
    }

    //DEBUG_INFO(origin_data_set[0].x);
    for (voxel_int i = 0; i < k_data_set_buffer_size; i+=pararead)
	{
		//DEBUG_INFO(origin_data_set[0].x);
        //DEBUG_INFO(i);
        count_uint buffer_new_index = dataset_buffer_max_index_PL % k_data_set_buffer_size;		//transform index from dataset to dataset_buffer
        //DEBUG_INFO(buffer_new_index);
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

    /*
    for (voxel_int i = 0; i < query_set_size; i++)
    {
#pragma HLS LOOP_TRIPCOUNT max=k_query_set_size_max min=k_query_set_size_min

    	//bool zero = 0;
        //int iii;
        My_PointXYZI_HW temp = reference_input.read();
    	reference_input1.write(temp);

    	voxel_int query_index_temp = query_index4.read();	// read query_i_copy
	    query_index5.write(query_index_temp);
        //My_PointXYZI_HW16 current_data_point_x[3*bundlevoxel];
        //My_PointXYZI_HW16 current_data_point_y[3*bundlevoxel];
        //My_PointXYZI_HW16 current_data_point_z[3*bundlevoxel];
        //inthw16 original_dataset_index_temp[3*bundlevoxel];
        My_PointXYZI_HW16 zerox;
        My_PointXYZI_HW16 zeroy;
        My_PointXYZI_HW16 zeroz;
        inthw16 zeroindex;

        zerox.p1 = 0;
        zeroy.p1 = 0;
        zeroz.p1 = 0;
        zerox.p2 = 0;
        zeroy.p2 = 0;
        zeroz.p2 = 0;
        zerox.p3 = 0;
        zeroy.p3 = 0;
        zeroz.p3 = 0;
        zerox.p4 = 0;
        zeroy.p4 = 0;
        zeroz.p4 = 0;

        zerox.p5 = 0;
        zeroy.p5 = 0;
        zeroz.p5 = 0;
        zerox.p6 = 0;
        zeroy.p6 = 0;
        zeroz.p6 = 0;
        zerox.p7 = 0;
        zeroy.p7 = 0;
        zeroz.p7 = 0;
        zerox.p8 = 0;
        zeroy.p8 = 0;
        zeroz.p8 = 0;

        zerox.p9 = 0;
        zeroy.p9 = 0;
        zeroz.p9 = 0;
        zerox.p10 = 0;
        zeroy.p10 = 0;
        zeroz.p10 = 0;
        zerox.p11 = 0;
        zeroy.p11 = 0;
        zeroz.p11 = 0;
        zerox.p12 = 0;
        zeroy.p12 = 0;
        zeroz.p12 = 0;

        zerox.p13 = 0;
        zeroy.p13 = 0;
        zeroz.p13 = 0;
        zerox.p14 = 0;
        zeroy.p14 = 0;
        zeroz.p14 = 0;
        zerox.p15 = 0;
        zeroy.p15 = 0;
        zeroz.p15 = 0;
        zerox.p16 = 0;
        zeroy.p16 = 0;
        zeroz.p16 = 0;

        zeroindex.p1 = 0;
        zeroindex.p2 = 0;
        zeroindex.p3 = 0;
        zeroindex.p4 = 0;
        zeroindex.p5 = 0;
        zeroindex.p6 = 0;
        zeroindex.p7 = 0;
        zeroindex.p8 = 0;
        zeroindex.p9 = 0;
        zeroindex.p10 = 0;
        zeroindex.p11 = 0;
        zeroindex.p12 = 0;
        zeroindex.p13 = 0;
        zeroindex.p14 = 0;
        zeroindex.p15 = 0;
        zeroindex.p16 = 0;

        //k_selection_int cc = 0;
        get_point_label0:for (uint_64 count_ij = 0; count_ij < bundlevoxel + 1; count_ij++)
        {
//#pragma HLS PIPELINE off
#pragma HLS PIPELINE
            count_uint current_search_hash_local;
            uint_4 voxel_flag_local;

            current_search_hash_local = valid_near_voxels.read();
            voxel_flag_local = voxel_flag.read();

            //DEBUG_INFO(current_search_hash_local);
            //DEBUG_INFO(i);
            //if ((voxel_flag_local == 2 | voxel_flag_local == 3) &  region_i == 15)
            if (current_search_hash_local == k_reference_set_size + 1)
            {
                break;
            }
            else
            {
            	//DEBUG_INFO(114514);
                if (voxel_flag_local == 1 | voxel_flag_local == 3)
                {
//#pragma HLS UNROLL

                        //int data_set_buffer_index = current_neighbor_index % k_data_set_buffer_size;
                        //int data_set_buffer_index = 1;
                        //int data_set_buffer_index = temp_index;
                        //My_PointXYZI current_data_point = data_set_buffer[data_set_buffer_index];
                        //original_dataset_index_local[temp_index].write(original_data_index_buffer[data_set_buffer_index]);
                    
                    //DEBUG_INFO(114514);
                    if (index16[current_search_hash_local] != k_reference_set_size)
                    {
                        if (index16[current_search_hash_local](31,29) == 1)
                        {
                            current_data_point_x[0].write(ordered_ref16_x[index16[current_search_hash_local](22,0)]);
                            current_data_point_y[0].write(ordered_ref16_y[index16[current_search_hash_local](22,0)]);
                            current_data_point_z[0].write(ordered_ref16_z[index16[current_search_hash_local](22,0)]);
                            original_dataset_index_temp[0].write(original_dataset_index16[index16[current_search_hash_local](22,0)]);
                            current_data_point_x[1].write(zerox);
                            current_data_point_y[1].write(zeroy);
                            current_data_point_z[1].write(zeroz);
                            original_dataset_index_temp[1].write(zeroindex);
                            current_data_point_x[2].write(zerox);
                            current_data_point_y[2].write(zeroy);
                            current_data_point_z[2].write(zeroz);
                            original_dataset_index_temp[2].write(zeroindex);
                            current_data_point_x[3].write(zerox);
                            current_data_point_y[3].write(zeroy);
                            current_data_point_z[3].write(zeroz);
                            original_dataset_index_temp[3].write(zeroindex);
                            //cc = cc + 4;
                            //DEBUG_INFO(114515);

                        }
                        else if (index16[current_search_hash_local](31,29) == 2)
                        {
                            current_data_point_x[0].write(ordered_ref16_x[index16[current_search_hash_local](22,0)]);
                            current_data_point_y[0].write(ordered_ref16_y[index16[current_search_hash_local](22,0)]);
                            current_data_point_z[0].write(ordered_ref16_z[index16[current_search_hash_local](22,0)]);
                            original_dataset_index_temp[0].write(original_dataset_index16[index16[current_search_hash_local](22,0)]);
                            current_data_point_x[1].write(ordered_ref16_x[index16[current_search_hash_local](22,0)+1]);
                            current_data_point_y[1].write(ordered_ref16_y[index16[current_search_hash_local](22,0)+1]);
                            current_data_point_z[1].write(ordered_ref16_z[index16[current_search_hash_local](22,0)+1]);
                            original_dataset_index_temp[1].write(original_dataset_index16[index16[current_search_hash_local](22,0)+1]);
                            current_data_point_x[2].write(zerox);
                            current_data_point_y[2].write(zeroy);
                            current_data_point_z[2].write(zeroz);
                            original_dataset_index_temp[2].write(zeroindex);
                            current_data_point_x[3].write(zerox);
                            current_data_point_y[3].write(zeroy);
                            current_data_point_z[3].write(zeroz);
                            original_dataset_index_temp[3].write(zeroindex);
                            //cc = cc + 4;
                            //DEBUG_INFO(114516);

                        }
                        else if (index16[current_search_hash_local](31,29) == 3)
                        {
                            current_data_point_x[0].write(ordered_ref16_x[index16[current_search_hash_local](22,0)]);
                            current_data_point_y[0].write(ordered_ref16_y[index16[current_search_hash_local](22,0)]);
                            current_data_point_z[0].write(ordered_ref16_z[index16[current_search_hash_local](22,0)]);
                            original_dataset_index_temp[0].write(original_dataset_index16[index16[current_search_hash_local](22,0)]);
                            current_data_point_x[1].write(ordered_ref16_x[index16[current_search_hash_local](22,0)+1]);
                            current_data_point_y[1].write(ordered_ref16_y[index16[current_search_hash_local](22,0)+1]);
                            current_data_point_z[1].write(ordered_ref16_z[index16[current_search_hash_local](22,0)+1]);
                            original_dataset_index_temp[1].write(original_dataset_index16[index16[current_search_hash_local](22,0)+1]);
                            current_data_point_x[2].write(ordered_ref16_x[index16[current_search_hash_local](22,0)+2]);
                            current_data_point_y[2].write(ordered_ref16_y[index16[current_search_hash_local](22,0)+2]);
                            current_data_point_z[2].write(ordered_ref16_z[index16[current_search_hash_local](22,0)+2]);
                            original_dataset_index_temp[2].write(original_dataset_index16[index16[current_search_hash_local](22,0)+2]);
                            current_data_point_x[3].write(zerox);
                            current_data_point_y[3].write(zeroy);
                            current_data_point_z[3].write(zeroz);
                            original_dataset_index_temp[3].write(zeroindex);
                            //cc = cc + 4;
                            //DEBUG_INFO(114517);

                        }
                        else if (index16[current_search_hash_local](31,29) == 4)
                        {
                            current_data_point_x[0].write(ordered_ref16_x[index16[current_search_hash_local](22,0)]);
                            current_data_point_y[0].write(ordered_ref16_y[index16[current_search_hash_local](22,0)]);
                            current_data_point_z[0].write(ordered_ref16_z[index16[current_search_hash_local](22,0)]);
                            original_dataset_index_temp[0].write(original_dataset_index16[index16[current_search_hash_local](22,0)]);
                            current_data_point_x[1].write(ordered_ref16_x[index16[current_search_hash_local](22,0)+1]);
                            current_data_point_y[1].write(ordered_ref16_y[index16[current_search_hash_local](22,0)+1]);
                            current_data_point_z[1].write(ordered_ref16_z[index16[current_search_hash_local](22,0)+1]);
                            original_dataset_index_temp[1].write(original_dataset_index16[index16[current_search_hash_local](22,0)+1]);
                            current_data_point_x[2].write(ordered_ref16_x[index16[current_search_hash_local](22,0)+2]);
                            current_data_point_y[2].write(ordered_ref16_y[index16[current_search_hash_local](22,0)+2]);
                            current_data_point_z[2].write(ordered_ref16_z[index16[current_search_hash_local](22,0)+2]);
                            original_dataset_index_temp[2].write(original_dataset_index16[index16[current_search_hash_local](22,0)+2]);
                            current_data_point_x[3].write(ordered_ref16_x[index16[current_search_hash_local](22,0)+3]);
                            current_data_point_y[3].write(ordered_ref16_y[index16[current_search_hash_local](22,0)+3]);
                            current_data_point_z[3].write(ordered_ref16_z[index16[current_search_hash_local](22,0)+3]);
                            original_dataset_index_temp[3].write(original_dataset_index16[index16[current_search_hash_local](22,0)+3]);
                            //cc = cc + 4;
                            //DEBUG_INFO(114518);

                        }
                        else
                        {
                            //DEBUG_INFO(114514);
                        }

                        //original_dataset_index_local[temp_index].write(original_dataset_index16[current_neighbor_index]);
                        //candidate_distance[temp_index].write(cal_dist_hw(temp, current_data_point));
                        //c0513 = c0513 + 1;
                        /*
                        if (current_neighbor_index >= dataset_buffer_min_index_PL && current_neighbor_index < dataset_buffer_max_index_PL)
                        {
                            candidate_distance[temp_index].write(cal_dist(temp, current_data_point));
                        }
                        else
                        {
                            //DEBUG_INFO(123456789);
                            candidate_distance[temp_index].write(100);
                        }
                        
                        //zero = 1;
                    }
                    //else
                    //{
                    //    zero = 0;
                    //}
                }
                else
                {
                    if (subindex16[current_search_hash_local] != k_reference_set_size)
                    {
                        if (subindex16[current_search_hash_local](31,29) == 1)
                        {
                            current_data_point_x[0].write(ordered_ref16_x[subindex16[current_search_hash_local](22,0)]);
                            current_data_point_y[0].write(ordered_ref16_y[subindex16[current_search_hash_local](22,0)]);
                            current_data_point_z[0].write(ordered_ref16_z[subindex16[current_search_hash_local](22,0)]);
                            original_dataset_index_temp[0].write(original_dataset_index16[subindex16[current_search_hash_local](22,0)]);
                            current_data_point_x[1].write(zerox);
                            current_data_point_y[1].write(zeroy);
                            current_data_point_z[1].write(zeroz);
                            original_dataset_index_temp[1].write(zeroindex);
                            current_data_point_x[2].write(zerox);
                            current_data_point_y[2].write(zeroy);
                            current_data_point_z[2].write(zeroz);
                            original_dataset_index_temp[2].write(zeroindex);
                            current_data_point_x[3].write(zerox);
                            current_data_point_y[3].write(zeroy);
                            current_data_point_z[3].write(zeroz);
                            original_dataset_index_temp[3].write(zeroindex);
                            //cc = cc + 4;
                        }
                        else if (subindex16[current_search_hash_local](31,29) == 2)
                        {
                            current_data_point_x[0].write(ordered_ref16_x[subindex16[current_search_hash_local](22,0)]);
                            current_data_point_y[0].write(ordered_ref16_y[subindex16[current_search_hash_local](22,0)]);
                            current_data_point_z[0].write(ordered_ref16_z[subindex16[current_search_hash_local](22,0)]);
                            original_dataset_index_temp[0].write(original_dataset_index16[subindex16[current_search_hash_local](22,0)]);
                            current_data_point_x[1].write(ordered_ref16_x[subindex16[current_search_hash_local](22,0)+1]);
                            current_data_point_y[1].write(ordered_ref16_y[subindex16[current_search_hash_local](22,0)+1]);
                            current_data_point_z[1].write(ordered_ref16_z[subindex16[current_search_hash_local](22,0)+1]);
                            original_dataset_index_temp[1].write(original_dataset_index16[subindex16[current_search_hash_local](22,0)+1]);
                            current_data_point_x[2].write(zerox);
                            current_data_point_y[2].write(zeroy);
                            current_data_point_z[2].write(zeroz);
                            original_dataset_index_temp[2].write(zeroindex);
                            current_data_point_x[3].write(zerox);
                            current_data_point_y[3].write(zeroy);
                            current_data_point_z[3].write(zeroz);
                            original_dataset_index_temp[3].write(zeroindex);
                            //cc = cc + 4;
                        }
                        else if (subindex16[current_search_hash_local](31,29) == 3)
                        {
                            current_data_point_x[0].write(ordered_ref16_x[subindex16[current_search_hash_local](22,0)]);
                            current_data_point_y[0].write(ordered_ref16_y[subindex16[current_search_hash_local](22,0)]);
                            current_data_point_z[0].write(ordered_ref16_z[subindex16[current_search_hash_local](22,0)]);
                            original_dataset_index_temp[0].write(original_dataset_index16[subindex16[current_search_hash_local](22,0)]);
                            current_data_point_x[1].write(ordered_ref16_x[subindex16[current_search_hash_local](22,0)+1]);
                            current_data_point_y[1].write(ordered_ref16_y[subindex16[current_search_hash_local](22,0)+1]);
                            current_data_point_z[1].write(ordered_ref16_z[subindex16[current_search_hash_local](22,0)+1]);
                            original_dataset_index_temp[1].write(original_dataset_index16[subindex16[current_search_hash_local](22,0)+1]);
                            current_data_point_x[2].write(ordered_ref16_x[subindex16[current_search_hash_local](22,0)+2]);
                            current_data_point_y[2].write(ordered_ref16_y[subindex16[current_search_hash_local](22,0)+2]);
                            current_data_point_z[2].write(ordered_ref16_z[subindex16[current_search_hash_local](22,0)+2]);
                            original_dataset_index_temp[2].write(original_dataset_index16[subindex16[current_search_hash_local](22,0)+2]);
                            current_data_point_x[3].write(zerox);
                            current_data_point_y[3].write(zeroy);
                            current_data_point_z[3].write(zeroz);
                            original_dataset_index_temp[3].write(zeroindex);
                            //cc = cc + 4;
                        }
                        else if (subindex16[current_search_hash_local](31,29) == 4)
                        {
                            current_data_point_x[0].write(ordered_ref16_x[subindex16[current_search_hash_local](22,0)]);
                            current_data_point_y[0].write(ordered_ref16_y[subindex16[current_search_hash_local](22,0)]);
                            current_data_point_z[0].write(ordered_ref16_z[subindex16[current_search_hash_local](22,0)]);
                            original_dataset_index_temp[0].write(original_dataset_index16[subindex16[current_search_hash_local](22,0)]);
                            current_data_point_x[1].write(ordered_ref16_x[subindex16[current_search_hash_local](22,0)+1]);
                            current_data_point_y[1].write(ordered_ref16_y[subindex16[current_search_hash_local](22,0)+1]);
                            current_data_point_z[1].write(ordered_ref16_z[subindex16[current_search_hash_local](22,0)+1]);
                            original_dataset_index_temp[1].write(original_dataset_index16[subindex16[current_search_hash_local](22,0)+1]);
                            current_data_point_x[2].write(ordered_ref16_x[subindex16[current_search_hash_local](22,0)+2]);
                            current_data_point_y[2].write(ordered_ref16_y[subindex16[current_search_hash_local](22,0)+2]);
                            current_data_point_z[2].write(ordered_ref16_z[subindex16[current_search_hash_local](22,0)+2]);
                            original_dataset_index_temp[2].write(original_dataset_index16[subindex16[current_search_hash_local](22,0)+2]);
                            current_data_point_x[3].write(ordered_ref16_x[subindex16[current_search_hash_local](22,0)+3]);
                            current_data_point_y[3].write(ordered_ref16_y[subindex16[current_search_hash_local](22,0)+3]);
                            current_data_point_z[3].write(ordered_ref16_z[subindex16[current_search_hash_local](22,0)+3]);
                            original_dataset_index_temp[3].write(original_dataset_index16[subindex16[current_search_hash_local](22,0)+3]);
                            //cc = cc + 4;
                        }
                        else
                        {
                            //DEBUG_INFO(114514);
                        }

                        //original_dataset_index_local[temp_index].write(original_dataset_index16[current_neighbor_index]);
                        //candidate_distance[temp_index].write(cal_dist_hw(temp, current_data_point));
                        //c0513 = c0513 + 1;
                        /*
                        if (current_neighbor_index >= dataset_buffer_min_index_PL && current_neighbor_index < dataset_buffer_max_index_PL)
                        {
                            candidate_distance[temp_index].write(cal_dist(temp, current_data_point));
                        }
                        else
                        {
                            //DEBUG_INFO(123456789);
                            candidate_distance[temp_index].write(100);
                        }
                        
                        //zero = 1;
                    }
                }
            }
        }
        */

    My_PointXYZI_HW16 zerox;
    My_PointXYZI_HW16 zeroy;
    My_PointXYZI_HW16 zeroz;
    inthw16 zeroindex;
    //inthw16 oneindex;

    zerox.p1 = 0;
    zeroy.p1 = 0;
    zeroz.p1 = 0;
    zerox.p2 = 0;
    zeroy.p2 = 0;
    zeroz.p2 = 0;
    zerox.p3 = 0;
    zeroy.p3 = 0;
    zeroz.p3 = 0;
    zerox.p4 = 0;
    zeroy.p4 = 0;
    zeroz.p4 = 0;

    zerox.p5 = 0;
    zeroy.p5 = 0;
    zeroz.p5 = 0;
    zerox.p6 = 0;
    zeroy.p6 = 0;
    zeroz.p6 = 0;
    zerox.p7 = 0;
    zeroy.p7 = 0;
    zeroz.p7 = 0;
    zerox.p8 = 0;
    zeroy.p8 = 0;
    zeroz.p8 = 0;

    zerox.p9 = 0;
    zeroy.p9 = 0;
    zeroz.p9 = 0;
    zerox.p10 = 0;
    zeroy.p10 = 0;
    zeroz.p10 = 0;
    zerox.p11 = 0;
    zeroy.p11 = 0;
    zeroz.p11 = 0;
    zerox.p12 = 0;
    zeroy.p12 = 0;
    zeroz.p12 = 0;

    zerox.p13 = 0;
    zeroy.p13 = 0;
    zeroz.p13 = 0;
    zerox.p14 = 0;
    zeroy.p14 = 0;
    zeroz.p14 = 0;
    zerox.p15 = 0;
    zeroy.p15 = 0;
    zeroz.p15 = 0;
    zerox.p16 = 0;
    zeroy.p16 = 0;
    zeroz.p16 = 0;

    zeroindex.p1 = 0;
    zeroindex.p2 = 0;
    zeroindex.p3 = 0;
    zeroindex.p4 = 0;
    zeroindex.p5 = 0;
    zeroindex.p6 = 0;
    zeroindex.p7 = 0;
    zeroindex.p8 = 0;
    zeroindex.p9 = 0;
    zeroindex.p10 = 0;
    zeroindex.p11 = 0;
    zeroindex.p12 = 0;
    zeroindex.p13 = 0;
    zeroindex.p14 = 0;
    zeroindex.p15 = 0;
    zeroindex.p16 = 0;

/*
    oneindex.p1 = k_reference_set_size + 1;
    oneindex.p2 = 0;
    oneindex.p3 = 0;
    oneindex.p4 = 0;
    oneindex.p5 = 0;
    oneindex.p6 = 0;
    oneindex.p7 = 0;
    oneindex.p8 = 0;
    oneindex.p9 = 0;
    oneindex.p10 = 0;
    oneindex.p11 = 0;
    oneindex.p12 = 0;
    oneindex.p13 = 0;
    oneindex.p14 = 0;
    oneindex.p15 = 0;
    oneindex.p16 = 0;
*/

    get_point_hw_label11:for (voxel_int i = 0; i < query_set_size; i++)
    {
#pragma HLS LOOP_TRIPCOUNT max=k_query_set_size_max min=k_query_set_size_min

        get_point_label0:for (uint_64 count_ij = 0; count_ij < bundlevoxel; count_ij++)
        {
//#pragma HLS PIPELINE off
#pragma HLS loop_flatten
#pragma HLS pipeline II=1
            //int bug = 0;
            indexint max = 0;
            indexint min = 0;
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

            //DEBUG_INFO(current_search_hash_local);
            //DEBUG_INFO(i);
            //if ((voxel_flag_local == 2 | voxel_flag_local == 3) &  region_i == 15)
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

                //bug = bug + 1;
                //break;
            }
            else
            {
            	//DEBUG_INFO(114514);
                if (voxel_flag_local == 1 | voxel_flag_local == 3)
                {
//#pragma HLS UNROLL

                        //int data_set_buffer_index = current_neighbor_index % k_data_set_buffer_size;
                        //int data_set_buffer_index = 1;
                        //int data_set_buffer_index = temp_index;
                        //My_PointXYZI current_data_point = data_set_buffer[data_set_buffer_index];
                        //original_dataset_index_local[temp_index].write(original_data_index_buffer[data_set_buffer_index]);
                    
                    //DEBUG_INFO(114514);
                    if (index16[current_search_hash_local] != k_reference_set_size)
                    {
                        if (min > index16[current_search_hash_local](22,0))
                        {
                            min = index16[current_search_hash_local](22,0);
                        }
                        if (max < index16[current_search_hash_local](22,0))
                        {
                            max = index16[current_search_hash_local](22,0);
                        }
                        voxel_int data_set_buffer_index = index16[current_search_hash_local](22,0) % k_data_set_buffer_size;
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
                            //bug = bug + 1;
                            //current_data_point_x[3].write(zerox);
                            //current_data_point_y[3].write(zeroy);
                            //current_data_point_z[3].write(zeroz);
                            //original_dataset_index_temp[3].write(zeroindex);
                            //cc = cc + 4;
                            //DEBUG_INFO(114515);

                        }
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
                            //bug = bug + 1;
                            //current_data_point_x[3].write(zerox);
                            //current_data_point_y[3].write(zeroy);
                            //current_data_point_z[3].write(zeroz);
                            //original_dataset_index_temp[3].write(zeroindex);
                            //cc = cc + 4;
                            //DEBUG_INFO(114516);

                        }
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
                            //bug = bug + 1;
                            //current_data_point_x[3].write(zerox);
                            //current_data_point_y[3].write(zeroy);
                            //current_data_point_z[3].write(zeroz);
                            //original_dataset_index_temp[3].write(zeroindex);
                            //cc = cc + 4;
                            //DEBUG_INFO(114517);

                        }
                        /*
                        else if (index16[current_search_hash_local](31,29) == 4)
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
                            current_data_point_x[3].write(data_set_buffer_x[data_set_buffer_index+3]);
                            current_data_point_y[3].write(data_set_buffer_y[data_set_buffer_index+3]);
                            current_data_point_z[3].write(data_set_buffer_z[data_set_buffer_index+3]);
                            original_dataset_index_temp[3].write(original_data_index_buffer[data_set_buffer_index+3]);
                            //cc = cc + 4;
                            //DEBUG_INFO(114518);

                        }
                        */
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
                            //bug = bug + 1;
                            //DEBUG_INFO(114514);
                        }

                        //original_dataset_index_local[temp_index].write(original_dataset_index16[current_neighbor_index]);
                        //candidate_distance[temp_index].write(cal_dist_hw(temp, current_data_point));
                        //c0513 = c0513 + 1;
                        /*
                        if (current_neighbor_index >= dataset_buffer_min_index_PL && current_neighbor_index < dataset_buffer_max_index_PL)
                        {
                            candidate_distance[temp_index].write(cal_dist(temp, current_data_point));
                        }
                        else
                        {
                            //DEBUG_INFO(123456789);
                            candidate_distance[temp_index].write(100);
                        }
                        */
                        //zero = 1;
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
                    if (subindex16[current_search_hash_local] != k_reference_set_size)
                    {
                        if (subindex16[current_search_hash_local](31,29) == 1)
                        {
                            current_data_point_x[0].write(data_set_buffer_x[subindex16[current_search_hash_local](22,0)]);
                            current_data_point_y[0].write(data_set_buffer_y[subindex16[current_search_hash_local](22,0)]);
                            current_data_point_z[0].write(data_set_buffer_z[subindex16[current_search_hash_local](22,0)]);
                            original_dataset_index_temp[0].write(original_data_index_buffer[subindex16[current_search_hash_local](22,0)]);
                            current_data_point_x[1].write(zerox);
                            current_data_point_y[1].write(zeroy);
                            current_data_point_z[1].write(zeroz);
                            original_dataset_index_temp[1].write(zeroindex);
                            current_data_point_x[2].write(zerox);
                            current_data_point_y[2].write(zeroy);
                            current_data_point_z[2].write(zeroz);
                            original_dataset_index_temp[2].write(zeroindex);
                            //bug = bug + 1;
                            //current_data_point_x[3].write(zerox);
                            //current_data_point_y[3].write(zeroy);
                            //current_data_point_z[3].write(zeroz);
                            //original_dataset_index_temp[3].write(zeroindex);
                            //cc = cc + 4;
                        }
                        else if (subindex16[current_search_hash_local](31,29) == 2)
                        {
                            current_data_point_x[0].write(data_set_buffer_x[subindex16[current_search_hash_local](22,0)]);
                            current_data_point_y[0].write(data_set_buffer_y[subindex16[current_search_hash_local](22,0)]);
                            current_data_point_z[0].write(data_set_buffer_z[subindex16[current_search_hash_local](22,0)]);
                            original_dataset_index_temp[0].write(original_data_index_buffer[subindex16[current_search_hash_local](22,0)]);
                            current_data_point_x[1].write(data_set_buffer_x[subindex16[current_search_hash_local](22,0)+1]);
                            current_data_point_y[1].write(data_set_buffer_y[subindex16[current_search_hash_local](22,0)+1]);
                            current_data_point_z[1].write(data_set_buffer_z[subindex16[current_search_hash_local](22,0)+1]);
                            original_dataset_index_temp[1].write(original_data_index_buffer[subindex16[current_search_hash_local](22,0)+1]);
                            current_data_point_x[2].write(zerox);
                            current_data_point_y[2].write(zeroy);
                            current_data_point_z[2].write(zeroz);
                            original_dataset_index_temp[2].write(zeroindex);
                            //bug = bug + 1;
                            //current_data_point_x[3].write(zerox);
                            //current_data_point_y[3].write(zeroy);
                            //current_data_point_z[3].write(zeroz);
                            //original_dataset_index_temp[3].write(zeroindex);
                            //cc = cc + 4;
                        }
                        else if (subindex16[current_search_hash_local](31,29) == 3)
                        {
                            current_data_point_x[0].write(data_set_buffer_x[subindex16[current_search_hash_local](22,0)]);
                            current_data_point_y[0].write(data_set_buffer_y[subindex16[current_search_hash_local](22,0)]);
                            current_data_point_z[0].write(data_set_buffer_z[subindex16[current_search_hash_local](22,0)]);
                            original_dataset_index_temp[0].write(original_data_index_buffer[subindex16[current_search_hash_local](22,0)]);
                            current_data_point_x[1].write(data_set_buffer_x[subindex16[current_search_hash_local](22,0)+1]);
                            current_data_point_y[1].write(data_set_buffer_y[subindex16[current_search_hash_local](22,0)+1]);
                            current_data_point_z[1].write(data_set_buffer_z[subindex16[current_search_hash_local](22,0)+1]);
                            original_dataset_index_temp[1].write(original_data_index_buffer[subindex16[current_search_hash_local](22,0)+1]);
                            current_data_point_x[2].write(data_set_buffer_x[subindex16[current_search_hash_local](22,0)+2]);
                            current_data_point_y[2].write(data_set_buffer_y[subindex16[current_search_hash_local](22,0)+2]);
                            current_data_point_z[2].write(data_set_buffer_z[subindex16[current_search_hash_local](22,0)+2]);
                            original_dataset_index_temp[2].write(original_data_index_buffer[subindex16[current_search_hash_local](22,0)+2]);
                            //bug = bug + 1;
                            //current_data_point_x[3].write(zerox);
                            //current_data_point_y[3].write(zeroy);
                            //current_data_point_z[3].write(zeroz);
                            //original_dataset_index_temp[3].write(zeroindex);
                            //cc = cc + 4;
                        }
                        /*
                        else if (subindex16[current_search_hash_local](31,29) == 4)
                        {
                            current_data_point_x[0].write(data_set_buffer_x[subindex16[current_search_hash_local](22,0)]);
                            current_data_point_y[0].write(data_set_buffer_y[subindex16[current_search_hash_local](22,0)]);
                            current_data_point_z[0].write(data_set_buffer_z[subindex16[current_search_hash_local](22,0)]);
                            original_dataset_index_temp[0].write(original_data_index_buffer[subindex16[current_search_hash_local](22,0)]);
                            current_data_point_x[1].write(data_set_buffer_x[subindex16[current_search_hash_local](22,0)+1]);
                            current_data_point_y[1].write(data_set_buffer_y[subindex16[current_search_hash_local](22,0)+1]);
                            current_data_point_z[1].write(data_set_buffer_z[subindex16[current_search_hash_local](22,0)+1]);
                            original_dataset_index_temp[1].write(original_data_index_buffer[subindex16[current_search_hash_local](22,0)+1]);
                            current_data_point_x[2].write(data_set_buffer_x[subindex16[current_search_hash_local](22,0)+2]);
                            current_data_point_y[2].write(data_set_buffer_y[subindex16[current_search_hash_local](22,0)+2]);
                            current_data_point_z[2].write(data_set_buffer_z[subindex16[current_search_hash_local](22,0)+2]);
                            original_dataset_index_temp[2].write(original_data_index_buffer[subindex16[current_search_hash_local](22,0)+2]);
                            current_data_point_x[3].write(data_set_buffer_x[subindex16[current_search_hash_local](22,0)+3]);
                            current_data_point_y[3].write(data_set_buffer_y[subindex16[current_search_hash_local](22,0)+3]);
                            current_data_point_z[3].write(data_set_buffer_z[subindex16[current_search_hash_local](22,0)+3]);
                            original_dataset_index_temp[3].write(original_data_index_buffer[subindex16[current_search_hash_local](22,0)+3]);
                            //cc = cc + 4;
                        }
                        */
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
                            //bug = bug + 1;
                            //DEBUG_INFO(114514);
                        }

                        //original_dataset_index_local[temp_index].write(original_dataset_index16[current_neighbor_index]);
                        //candidate_distance[temp_index].write(cal_dist_hw(temp, current_data_point));
                        //c0513 = c0513 + 1;
                        /*
                        if (current_neighbor_index >= dataset_buffer_min_index_PL && current_neighbor_index < dataset_buffer_max_index_PL)
                        {
                            candidate_distance[temp_index].write(cal_dist(temp, current_data_point));
                        }
                        else
                        {
                            //DEBUG_INFO(123456789);
                            candidate_distance[temp_index].write(100);
                        }
                        */
                        //zero = 1;
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

            if (dataset_buffer_max_index_PL < packs && max > (dataset_buffer_min_index_PL + dataset_buffer_max_index_PL)/2 + k_dataset_buffer_gap)
            {
                for (voxel_int j = 0; j < k_dataset_buffer_gap; j+=pararead)
                {
    //#pragma HLS UNROLL factor=12
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
                            dataset_buffer_min_index_PL = dataset_buffer_min_index_PL + 1;
                            
                        }
                    }
                    else break;

                }
            }
            //DEBUG_INFO(bug);
            //if (bug!=17)
            //{
                //DEBUG_INFO(114514114);
            //}
        }

    }

}

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

    	//bool b = 0;
        //int iii;
        
        //k_selection_int cc;
        

        //cc = c.read();
        //ccc.write(cc);
        //DEBUG_INFO(cc);
        //DEBUG_INFO(i);
        for (uint_64 count_i = 0; count_i < bundlevoxel; count_i++)
        {
#pragma HLS PIPELINE

            if (count_i == 0)
            {
                temp = reference_input.read();
    	        //reference_input1.write(temp);

    	        voxel_int query_index_temp = query_index4.read();	// read query_i_copy
	            query_index5.write(query_index_temp);
            }

            //if (b == 1)
            //{
                //for (uint_64 count_j = 0; count_j < 16; count_j++)
                //{
                    //candidate_distance[16*count3+count_j].write(30000+1);
                    //original_dataset_index_local[16*count3+count_j].write(1);
                //}
                
                //break;
            //} 
            for (uint_64 count3 = 0; count3 < num16; count3++)
            {

                current_data_point_xtemp = current_data_point_x[count3].read();
                current_data_point_ytemp = current_data_point_y[count3].read();
                current_data_point_ztemp = current_data_point_z[count3].read();
                original_dataset_index = original_dataset_index_temp[count3].read();
                //DEBUG_INFO(current_data_point_xtemp.p1);
                //if (original_dataset_index.p1 == k_reference_set_size + 1)
                //{
                    //b = 1;
                    //current_data_point_xtemp = current_data_point_x[1].read();
                    //current_data_point_ytemp = current_data_point_y[1].read();
                    //current_data_point_ztemp = current_data_point_z[1].read();
                    //original_dataset_index = original_dataset_index_temp[1].read();
                    //current_data_point_xtemp = current_data_point_x[2].read();
                    //current_data_point_ytemp = current_data_point_y[2].read();
                    //current_data_point_ztemp = current_data_point_z[2].read();
                    //original_dataset_index = original_dataset_index_temp[2].read();
                    //current_data_point_xtemp = current_data_point_x[3].read();
                    //current_data_point_ytemp = current_data_point_y[3].read();
                    //current_data_point_ztemp = current_data_point_z[3].read();
                    //original_dataset_index = original_dataset_index_temp[3].read();
                    //break;
                //}

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
        //for (uint_128 j = 0; j < k_transform_neighbor_num; j++)
        //{
            //candidate_distance[j].write(0);
            //original_dataset_index_local[j].write(k_reference_set_size + 1);
        //}
    }
}

/*
void get_point_hw(hls::stream<My_PointXYZI_HW>& reference_input, hls::stream<voxel_int>& query_index4, hls::stream<voxel_int>& query_index5, hls::stream<count_uint> candidate_neighbors[k_transform_neighbor_num], hls::stream<type_dist_hw> candidate_distance[k_transform_neighbor_num], hls::stream<count_uint> original_dataset_index_local[k_transform_neighbor_num], My_PointXYZI_HW16* ordered_ref16_x, My_PointXYZI_HW16* ordered_ref16_y, My_PointXYZI_HW16* ordered_ref16_z, inthw16* original_dataset_index16, int query_set_size, int dataset_buffer_max_index_PL, int dataset_buffer_min_index_PL, int data_set_size)
{
//#pragma HLS ARRAY_PARTITION dim=1 factor=60 type=cyclic variable=original_data_index_buffer
//#pragma HLS ARRAY_PARTITION dim=1 factor=60 type=cyclic variable=data_set_buffer
    //int c0513 = 0;
    for (voxel_int i = 0; i < query_set_size; i++)
    {
#pragma HLS LOOP_TRIPCOUNT max=k_query_set_size_max min=k_query_set_size_min

    	My_PointXYZI_HW temp = reference_input.read();
    	//reference_input1.write(temp);

    	voxel_int query_index_temp = query_index4.read();	// read query_i_copy
	    query_index5.write(query_index_temp);

        get_point_label0:for (k_selection_int count_i = 0; count_i < 18; count_i++)
        {
//#pragma HLS PIPELINE off
#pragma HLS PIPELINE
            count_uint candidate_neighbors_local1[k_transform_neighbor_num];

            get_point_label1:for (uint_64 cache_temp_index = 0; cache_temp_index < k_transform_neighbor_num; cache_temp_index++)
            {

                candidate_neighbors_local1[cache_temp_index] = candidate_neighbors[cache_temp_index].read();
            }

            if (candidate_neighbors_local1[0] == k_reference_set_size + 1)
            {
                //exit_break_flag = true;
                //DEBUG_INFO(count_i);
            	get_point_label2:for (uint_64 temp_index = 0; temp_index < k_transform_neighbor_num; temp_index++)
                {
                    candidate_distance[temp_index].write(30000 + 1);
                    original_dataset_index_local[temp_index].write(k_reference_set_size + 1);
                }
                break;
            }
            else
            {
            	get_point_label3:for (uint_64 temp_index = 0; temp_index < k_transform_neighbor_num; temp_index++)
                {
//#pragma HLS UNROLL


    count_uint current_neighbor_index;
                    current_neighbor_index = candidate_neighbors_local1[temp_index];

                    if (current_neighbor_index >= 0 && current_neighbor_index <= data_set_size)
                    {
                        //int data_set_buffer_index = current_neighbor_index % k_data_set_buffer_size;
                        //int data_set_buffer_index = 1;
                        //int data_set_buffer_index = temp_index;
                        //My_PointXYZI current_data_point = data_set_buffer[data_set_buffer_index];
                        //original_dataset_index_local[temp_index].write(original_data_index_buffer[data_set_buffer_index]);
                        My_PointXYZI_HW current_data_point;
                        current_data_point.x = data_set_x[current_neighbor_index];
                        current_data_point.y = data_set_y[current_neighbor_index];
                        current_data_point.z = data_set_z[current_neighbor_index];
                        original_dataset_index_local[temp_index].write(original_dataset_index[current_neighbor_index]);
                        candidate_distance[temp_index].write(cal_dist_hw(temp, current_data_point));
                        //c0513 = c0513 + 1;
                        /*
                        if (current_neighbor_index >= dataset_buffer_min_index_PL && current_neighbor_index < dataset_buffer_max_index_PL)
                        {
                            candidate_distance[temp_index].write(cal_dist(temp, current_data_point));
                        }
                        else
                        {
                            //DEBUG_INFO(123456789);
                            candidate_distance[temp_index].write(100);
                        }
                        
                    }
                    else
                    {
                        candidate_distance[temp_index].write(30000);
                        original_dataset_index_local[temp_index].write(k_reference_set_size);
                    }
                }
            }
        }

        get_point_label0:while (dataset_buffer_max_index_PL < data_set_size && voxel_first_index_PL[query_index_temp] > (dataset_buffer_min_index_PL + dataset_buffer_max_index_PL)/2 + k_dataset_buffer_gap)
        {
#pragma HLS LOOP_TRIPCOUNT avg=1 max=1 min=1
        loop_buffer_dataset:
            for (int j = 0; j < k_dataset_buffer_gap; j++)
            {
//#pragma HLS UNROLL factor=12

                if (dataset_buffer_max_index_PL < data_set_size)
                {
                    int buffer_new_index = dataset_buffer_max_index_PL % k_data_set_buffer_size;		//transform index from dataset to dataset_buffer

                    data_set_buffer[buffer_new_index].x = data_set[dataset_buffer_max_index_PL].x;
                    data_set_buffer[buffer_new_index].y = data_set[dataset_buffer_max_index_PL].y;
                    data_set_buffer[buffer_new_index].z = data_set[dataset_buffer_max_index_PL].z;
                    original_data_index_buffer[buffer_new_index] = original_dataset_index[dataset_buffer_max_index_PL];
                    dataset_buffer_max_index_PL = dataset_buffer_max_index_PL + 1;
                    dataset_buffer_min_index_PL = dataset_buffer_min_index_PL + 1;
                }
                else break;

            }

        }

    }
    //DEBUG_INFO(2023040529);
    //DEBUG_INFO(dataset_buffer_max_index_PL);
    //DEBUG_INFO(dataset_buffer_min_index_PL);
    //DEBUG_INFO(c0513);
}
*/



void select_knn_hw(count_uint* query_result, type_dist_hw* nearest_distance, hls::stream<type_dist_hw> candidate_distance_s[k_transform_neighbor_num], hls::stream<count_uint> original_dataset_index[k_transform_neighbor_num], int query_set_size, hls::stream<voxel_int>& query_index5)
{
    count_uint nearest_index_PL_local[k_nearest_number_max];
    type_dist_hw nearest_distance_PL_local[k_nearest_number_max];

    count_uint original_dataset_index_local[k_transform_neighbor_num];
    type_dist_hw candidate_distance[k_transform_neighbor_num];
    k_selection_int cmp_array[k_nearest_number_max + k_transform_neighbor_num];

    for (voxel_int i = 0; i < query_set_size; i++)
    {
#pragma HLS LOOP_TRIPCOUNT max=k_query_set_size_max min=k_query_set_size_min
        

        

    loop_dataflow_cal_knn:
        for (uint_64 count_i = 0; count_i < bundlevoxel; count_i++)
        {
#pragma HLS loop_flatten
#pragma HLS pipeline II=1

            //if (num16*count_i >= cc)
            //{
                //for (uint_64 count_j = 0; count_j < 16; count_j++)
                //{
                    //candidate_distance[16*count3+count_j].write(30000+1);
                    //original_dataset_index_local[16*count3+count_j].write(1);
                //}
                //break;
            //} 

            if (count_i == 0)
            {
                voxel_int query_index_temp = query_index5.read();	// read query_i_copy
                //k_selection_int cc = ccc.read();

            loop_reset_result_array:

                nearest_index_PL_local[0] = 0;
                nearest_distance_PL_local[0] = 100;
            }

            for (uint_128 cache_temp_index = 0; cache_temp_index < k_transform_neighbor_num; cache_temp_index++)
            {
                candidate_distance[cache_temp_index] = candidate_distance_s[cache_temp_index].read();
                original_dataset_index_local[cache_temp_index] = original_dataset_index[cache_temp_index].read();
                //DEBUG_INFO(candidate_distance[cache_temp_index]);
            }

            //if (original_dataset_index_local[0] == k_reference_set_size + 1)
            //{
                //DEBUG_INFO(230617);
                
                //break;
            //}
            //else
            //{
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

				if (cmp_array[0] < 1 && cmp_array[0] != 0)
				{
					nearest_distance_PL_local[cmp_array[0]] = nearest_distance_PL_local[0];
					nearest_index_PL_local[cmp_array[0]] = nearest_index_PL_local[0];
				}

				//move the element in candidate_distance
				for (uint_128 parallel_i = 0; parallel_i < k_transform_neighbor_num; parallel_i++)
				{
					if (cmp_array[parallel_i + 1 ] < 1)
					{
						nearest_distance_PL_local[cmp_array[parallel_i + 1 ]] = candidate_distance[parallel_i];
						nearest_index_PL_local[cmp_array[parallel_i + 1 ]] = original_dataset_index_local[parallel_i];
					}
				}
                if (count_i == bundlevoxel - 1)
                {
                    query_result[i] = nearest_index_PL_local[0];

                    nearest_distance[i] = nearest_distance_PL_local[0];
                }
        } //while()

        

    }
}



void DSVS_search_hw(My_PointXYZI_HW16* ordered_ref16_x, My_PointXYZI_HW16* ordered_ref16_y, My_PointXYZI_HW16* ordered_ref16_z, int data_set_size,
	inthw16* original_dataset_index16,
	My_PointXYZI_HW* query_set, int query_set_size,
    count_uint* query_result, type_dist_hw* nearest_distance,
    indexint* index16, voxel_int* sub_voxel_flag_index_PL, indexint* subindex16, int dataset_buffer_max_index_PL, int dataset_buffer_min_index_PL)
{
// #pragma HLS INTERFACE mode=s_axilite bundle=CTRL port=dataset_buffer_min_index_PL
// #pragma HLS INTERFACE mode=s_axilite bundle=CTRL port=dataset_buffer_max_index_PL
// #pragma HLS INTERFACE mode=s_axilite bundle=CTRL port=query_set_size
// #pragma HLS INTERFACE mode=s_axilite bundle=CTRL port=data_set_size
// #pragma HLS INTERFACE mode=s_axilite bundle=CTRL port=return
#pragma HLS INTERFACE mode=m_axi bundle=gmem1 depth=624294 port=subindex16
#pragma HLS INTERFACE mode=m_axi bundle=gmem6 depth=1303400 port=sub_voxel_flag_index_PL
#pragma HLS INTERFACE mode=m_axi bundle=gmem1 depth=1303400 port=index16
#pragma HLS INTERFACE mode=m_axi bundle=gmem2 depth=7296 port=nearest_distance
#pragma HLS INTERFACE mode=m_axi bundle=gmem2 depth=7296 port=query_result
#pragma HLS INTERFACE mode=m_axi bundle=gmem2 depth=7296 port=query_set
#pragma HLS INTERFACE mode=m_axi bundle=gmem5 depth=598347 port=original_dataset_index16
#pragma HLS INTERFACE mode=m_axi bundle=gmem0 depth=598347 port=ordered_ref16_x
#pragma HLS INTERFACE mode=m_axi bundle=gmem3 depth=598347 port=ordered_ref16_y
#pragma HLS INTERFACE mode=m_axi bundle=gmem4 depth=598347 port=ordered_ref16_z
#pragma HLS DATAFLOW
    ///*
    hls::stream<My_PointXYZI_HW> reference_input;

    hls::stream<My_PointXYZI_HW16> ordered_ref16_xs[pararead];

    hls::stream<My_PointXYZI_HW16> ordered_ref16_ys[pararead];

    hls::stream<My_PointXYZI_HW16> ordered_ref16_zs[pararead];

    hls::stream<inthw16> original_dataset_index16s[pararead];

    hls::stream<count_uint> valid_near_voxels;

    hls::stream<uint_4> voxel_flag;

    //hls::stream<count_uint> candidate_neighbors[k_transform_neighbor_num];

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

    //hls::stream<k_selection_int> c;

    //hls::stream<k_selection_int> ccc;

    //#pragma HLS STREAM variable=reference_input depth=40

    #pragma HLS STREAM variable=reference_input1 depth=20

    //#pragma HLS STREAM variable=reference_input2 depth=40

    #pragma HLS STREAM variable=reference_input3 depth=20

    //#pragma HLS STREAM variable=reference_input4 depth=40

    //#pragma HLS STREAM variable=valid_near_voxels depth=40

    //#pragma HLS STREAM variable=voxel_flag depth=40

    //#pragma HLS STREAM variable=candidate_neighbors depth=k_transform_neighbor_num*2

    //#pragma HLS STREAM variable=candidate_distance depth=k_transform_neighbor_num*2

    //#pragma HLS STREAM variable=original_dataset_index_local depth=k_transform_neighbor_num*2

    //#pragma HLS STREAM variable=query_index1 depth=40

    //#pragma HLS STREAM variable=query_index2 depth=40

    #pragma HLS STREAM variable=query_index3 depth=20

    //#pragma HLS STREAM variable=query_index4 depth=40

    //#pragma HLS STREAM variable=query_index5 depth=40

    //#pragma HLS STREAM variable=local_grid_center_x depth=40

    //#pragma HLS STREAM variable=local_grid_center_y depth=40

    //#pragma HLS STREAM variable=local_grid_center_z depth=40

    //#pragma HLS STREAM variable=current_data_point_x depth=40

    //#pragma HLS STREAM variable=current_data_point_y depth=40

    //#pragma HLS STREAM variable=current_data_point_z depth=40

    //#pragma HLS STREAM variable=original_dataset_index_temp depth=40

    data_set_max_min_PL_xmin_hw = 75.0003;
    data_set_max_min_PL_ymin_hw = -24.3433;
    data_set_max_min_PL_zmin_hw = -137.805;

    voxel_split_unit_PL_hw = 1;

    voxel_split_array_size_PL_x_array_size = 216;
    voxel_split_array_size_PL_y_array_size = 29;
    voxel_split_array_size_PL_z_array_size = 185;

    total_calculated_voxel_size = 1158840;

    //data_set_max_min_PL_xmin_hw = data_set_max_min_PL_xmin_hw_PL;
    //data_set_max_min_PL_ymin_hw = data_set_max_min_PL_ymin_hw_PL;
    //data_set_max_min_PL_zmin_hw = data_set_max_min_PL_zmin_hw_PL;

    //voxel_split_unit_PL_hw = voxel_split_unit_PL_hw_PL;
    voxel_split_unit_PL_hw4 = 0.25;

    packs = 34044;

    //voxel_split_array_size_PL_x_array_size = voxel_split_array_size_PL_x_array_size_PL;
    //voxel_split_array_size_PL_y_array_size = voxel_split_array_size_PL_y_array_size_PL;
    //voxel_split_array_size_PL_z_array_size = voxel_split_array_size_PL_z_array_size_PL;

    //total_calculated_voxel_size = total_calculated_voxel_size_PL;



    /*
    DEBUG_INFO(data_set_max_min_PL_xmin_hw);
    DEBUG_INFO(data_set_max_min_PL_ymin_hw);
    DEBUG_INFO(data_set_max_min_PL_zmin_hw);
    DEBUG_INFO(voxel_split_unit_PL_hw);
    DEBUG_INFO(voxel_split_array_size_PL_x_array_size);
    DEBUG_INFO(voxel_split_array_size_PL_y_array_size);
    DEBUG_INFO(voxel_split_array_size_PL_z_array_size);
    DEBUG_INFO(total_calculated_voxel_size);
    */

    //int dataset_buffer_max_index_PL = 0;
    //int dataset_buffer_min_index_PL = 0;

    input_src_hw(query_set, reference_input, query_set_size);

    input_reference_hw(ordered_ref16_x, ordered_ref16_y, ordered_ref16_z, original_dataset_index16, ordered_ref16_xs, ordered_ref16_ys, ordered_ref16_zs, original_dataset_index16s, packs);

    calculate_hash_stream_hw(reference_input, reference_input1, query_index1, query_set_size);

    initial_hw(reference_input1, reference_input2, query_index1, query_index2, local_grid_center_x, local_grid_center_y, local_grid_center_z, query_set_size);

    //initial_buffer(query_set[0], voxel_first_index_PL, data_set, original_dataset_index, dataset_buffer_max_index_PL, dataset_buffer_min_index_PL);

    search_near_cells_hw(reference_input2, reference_input3, query_index2, query_index3, valid_near_voxels, voxel_flag, local_grid_center_x, local_grid_center_y, local_grid_center_z, sub_voxel_flag_index_PL, query_set_size);

    //search_candidate_neighbors_hw(reference_input3, reference_input4, query_index3, query_index4, valid_near_voxels, voxel_flag, candidate_neighbors, voxel_first_index_PL, sub_voxel_first_index_PL, query_set_size);
    //DEBUG_INFO(202306271);
    get_point_hw(reference_input3, reference_input4, query_index3, query_index4, valid_near_voxels, voxel_flag, ordered_ref16_xs, ordered_ref16_ys, ordered_ref16_zs, original_dataset_index16s, original_dataset_index16, index16, subindex16, query_set_size, dataset_buffer_max_index_PL, dataset_buffer_min_index_PL, data_set_size, current_data_point_x, current_data_point_y, current_data_point_z, original_dataset_index_temp);
//DEBUG_INFO(202306272);
    unpack(reference_input4, query_index4, query_index5, candidate_distance, original_dataset_index_local, current_data_point_x, current_data_point_y, current_data_point_z, original_dataset_index_temp, query_set_size);
//DEBUG_INFO(202306273);
    select_knn_hw(query_result, nearest_distance, candidate_distance, original_dataset_index_local, query_set_size, query_index5);
//DEBUG_INFO(202306274);
}


