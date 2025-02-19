#undef __ARM_NEON__
#undef __ARM_NEON
#include "knn_odom.h"
#define __ARM_NEON__
#define __ARM_NEON

static float data_set_max_min_PL_xmin;
static float data_set_max_min_PL_xmax;
static float data_set_max_min_PL_ymin;
static float data_set_max_min_PL_ymax;
static float data_set_max_min_PL_zmin;
static float data_set_max_min_PL_zmax;

static float voxel_split_unit_PL;

static int voxel_split_array_size_PL_x_array_size;
static int voxel_split_array_size_PL_y_array_size;
static int voxel_split_array_size_PL_z_array_size;

static int total_calculated_voxel_size;
static float sub_voxel_split_unit_PL_x;
static float sub_voxel_split_unit_PL_y;
static float sub_voxel_split_unit_PL_z;

static float square_input_precision_PL;

static int k_sub_voxel_threshold;

#define SUB_VOXEL_SPLIT
#ifdef USE_PCL_ON

// 做 kdtree 搜索
void kdtree_search(My_PointXYZI* KNN_query_set, int KNN_query_set_size, PointCloud::Ptr kdtree_point_cloud, nanoflann::KdTreeFLANN<PointT> & kdtreeFromMap, My_PointXYZI* image_KNN_query_result)
{
    TIMER_INIT(5);
    TIMER_START(2);

    for(int query_i = 0; query_i < KNN_query_set_size; query_i++)
    {
        My_PointXYZI query_point = KNN_query_set[query_i];
        My_PointXYZI this_query_KNN_result;
        
        PointT kd_query_point;
        kd_query_point.x = query_point.x; 
        kd_query_point.y = query_point.y;
        kd_query_point.z = query_point.z;
        kd_query_point.intensity = query_point.intensity;

        std::vector<int> pointSearchInd;		//最邻近搜索的最近点
        std::vector<float> pointSearchSqDis;    //最邻近搜索的最近点到该点的距离

        kdtreeFromMap.nearestKSearch(kd_query_point, K, pointSearchInd, pointSearchSqDis);

        int closestPointInd = pointSearchInd[0];

        this_query_KNN_result.x = kdtree_point_cloud->points[closestPointInd].x;
        this_query_KNN_result.y = kdtree_point_cloud->points[closestPointInd].y;
        this_query_KNN_result.z = kdtree_point_cloud->points[closestPointInd].z;
        this_query_KNN_result.intensity = kdtree_point_cloud->points[closestPointInd].intensity;

        image_KNN_query_result[query_i] = this_query_KNN_result;
    }
    TIMER_STOP_ID(2);

    DEBUG_TIME("Finished kdtree_search with query points " << KNN_query_set_size << " with " << TIMER_REPORT_MS(2) << " ms !" );
}

// 将点云数组转成pointcloud
void transform_array_to_pointcloud(My_PointXYZI* point_set, int point_set_size, PointCloud::Ptr & out_point_cloud)
{
    out_point_cloud.reset(new PointCloud);
    for(int i = 0; i < point_set_size; i++)
    {
        PointT p;
        p.x = point_set[i].x;
        p.y = point_set[i].y;
        p.z = point_set[i].z;
        p.intensity = point_set[i].intensity;
        out_point_cloud->points.push_back(p);
    }
    DEBUG_LOG("point_set_size: " << point_set_size << " out_point_cloud.size() " << out_point_cloud->points.size() );
}

#endif

// 生成 KNN 测试数据集。即均匀生成不同数目的点云。
void generate_test_datasets(std::string input_file_name, std::string output_file_name, int required_point_size)
{

    My_PointXYZI laserCloudInArray[k_reference_set_size];
    int point_size;
    std::ifstream file(input_file_name);
    if(!file)
    {
        std::cout << "error! fails read pointcloud txt file! "<< std::endl;
        point_size = 0;
    }
    else{
        std::string line;   // 读到的每一行数据
        std::string read_element;   // 每一行按空格拆分后的单个数据
        float line_data[6]; // 一行最多6个数据
        int line_i=0;   // 行数
        
        while(getline(file,line))   // 读取一行数据，读入line中
        {
            std::istringstream line_stream(line);   // 分割line中数据，转化成数据流
            int line_element_count = 0;
            while(line_stream >> read_element)  // 对于每一个数据，写入 line_data 中
            {
                line_data[line_element_count] = atof(read_element.data());
                line_element_count++;
            }

            {
                laserCloudInArray[line_i].x = line_data[0];
                laserCloudInArray[line_i].y = line_data[1];
                laserCloudInArray[line_i].z = line_data[2];
                laserCloudInArray[line_i].intensity = line_data[3];
            }

            line_i = line_i + 1;
        }
        point_size = line_i;
    }

    if(required_point_size >= point_size)
    {
        std::cout << "ERROR: required_point_size >= point_size: " << required_point_size << " >= " << point_size << std::endl;
    }
    else
    {
        // 1. 随机生成数据。。

        // 2. 均匀间隔生成数据
        std::ofstream required_dataset_out(output_file_name, std::ios::ate);
        int out_points_count = 0;
        float sample_gap = float(point_size) / float(required_point_size);
        // sample_gap = float(point_size-sample_gap*5) / float(required_point_size);
        float i = 0;
        int prev_floor_i = -1;
        for(i = 0; i < point_size; i+=sample_gap)
        {
            int floor_i = int(i);
            if(floor_i != prev_floor_i)
            {
                required_dataset_out << laserCloudInArray[floor_i].x << " " << laserCloudInArray[floor_i].y << " " << laserCloudInArray[floor_i].z << " " << laserCloudInArray[floor_i].intensity << std::endl;
                out_points_count++;
            }
            prev_floor_i = floor_i;
        }
        required_dataset_out.close();
        std::cout << "required_point_size: " << required_point_size << " out_points_count: " << out_points_count << " sample_gap: " << sample_gap << " final_i: " << i << std::endl;

    }

}

// 输出参数结果到外部文件。
void file_out_result(std::string file_dir, int reference_set_num, int reference_point_num, int query_set_num, int query_point_num, int measure_power_flag, 
                    double imageNN_SW_build_time, double imageNN_SW_search_time, double kdtree_build_time, double kd_tree_search_time, double imageNN_HW_build_time, double imageNN_HW_search_time, 
                    double imageNN_SW_build_power, double imageNN_SW_search_power, double kdtree_build_power, double kd_tree_search_power, double imageNN_HW_build_power, double imageNN_HW_search_power,
                    double image_kdtree_accuracy, double image_sw_hw_accuracy)
{
    std::ofstream fout;
    fout.open(file_dir+"/FPGA_ODOM_KNN_RESULT.txt", std::ios::app);
    if (!fout)
    {
        std::cout << "fail to open "<< file_dir << "/FPGA_ODOM_KNN_RESULT.txt" << std::endl;
    }
    else
    {
        std::cout << "Open "<< file_dir << "/FPGA_ODOM_KNN_RESULT.txt Successful." << std::endl;
        fout << reference_set_num << " " << reference_point_num << " " << query_set_num << " " << query_point_num << " " << measure_power_flag << " ";
        fout << imageNN_SW_build_time << " " << imageNN_SW_search_time << " " << kdtree_build_time << " " << kd_tree_search_time << " " << imageNN_HW_build_time << " " << imageNN_HW_search_time << " ";
        fout << imageNN_SW_build_power << " " << imageNN_SW_search_power << " " << kdtree_build_power << " " << kd_tree_search_power << " " << imageNN_HW_build_power << " " << imageNN_HW_search_power << " ";
        fout << image_kdtree_accuracy << " " << image_sw_hw_accuracy << " " ;

        fout << std::endl;
    }
    fout.close();
}

// 从 txt 读取点云到数组中
void read_points_from_txt(std::string file_name, My_PointXYZI* laserCloudInArray, int & point_size)
{
    std::ifstream file(file_name);
    if(!file)
    {
        std::cout << "!!! error! fails read pointcloud txt file from" << file_name << std::endl;
        point_size = 0;
    }
    else{
        std::string line;   // 读到的每一行数据
        std::string read_element;   // 每一行按空格拆分后的单个数据
        float line_data[6]; // 一行最多6个数据
        int line_i=0;   // 行数
        
        while(getline(file,line))   // 读取一行数据，读入line中
        {
            std::istringstream line_stream(line);   // 分割line中数据，转化成数据流
            int line_element_count = 0;
            while(line_stream >> read_element)  // 对于每一个数据，写入 line_data 中
            {
                line_data[line_element_count] = atof(read_element.data());
                line_element_count++;
            }

            {
                laserCloudInArray[line_i].x = line_data[0];
                laserCloudInArray[line_i].y = line_data[1];
                laserCloudInArray[line_i].z = line_data[2];
                laserCloudInArray[line_i].intensity = line_data[3];
            }

            line_i = line_i + 1;

        }

        point_size = line_i;

    }
}



/*
	Get the bound of x,y,z axis
	Input: dataset with normal bits size
	Output: x_max,x_min,y_max,y_min,z_min,z_max
 */
void get_min_max(My_PointXYZI* data_set, int data_set_size, type_point_hw& data_set_max_min_PL_xmin_hw, type_point_hw& data_set_max_min_PL_ymin_hw, type_point_hw& data_set_max_min_PL_zmin_hw)
{
	float x_max, x_min, y_max, y_min, z_max, z_min;

	x_max = y_max = z_max = -k_data_max_value_abs;
	x_min = y_min = z_min = k_data_max_value_abs;

loop_cache_and_min_max:
	for (int i = 0; i < data_set_size; i++)
	{
		float data_x = data_set[i].x;
		float data_y = data_set[i].y;
		float data_z = data_set[i].z;

		//compare x
		if (x_max < data_x)
			x_max = data_x;
		if (x_min > data_x)
			x_min = data_x;


		//compare y
		if (y_max < data_y)
			y_max = data_y;
		if (y_min > data_y)
			y_min = data_y;


		//compare z
		if (z_max < data_z)
			z_max = data_z;
		if (z_min > data_z)
			z_min = data_z;

	}
    
	//store these min max in a structure for convenient usage
	//My_MaxMin data_set_max_min;
	data_set_max_min_PL_xmin = x_min;
	data_set_max_min_PL_xmax = x_max;
	data_set_max_min_PL_ymin = y_min;
	data_set_max_min_PL_ymax = y_max;
	data_set_max_min_PL_zmin = z_min;
	data_set_max_min_PL_zmax = z_max;

    data_set_max_min_PL_xmin_hw = x_min;
    data_set_max_min_PL_ymin_hw = y_min;
    data_set_max_min_PL_zmin_hw = z_min;
}

/*  wyt
    input_split_unit：用户指定的分割单元大小（即每个体素的边长）。
    voxel_split_array_size_PL_x_array_size_hw y z 表示三轴上的体素分割数量。
    total_calculated_voxel_size_hw：表示总的体素数量。
    data_set_max_min_PL_xmax ymax zmax 表示三维空间的边界范围（AABB 的最小值和最大值）。
*/

void split_voxel_AABB(float input_split_unit, voxel_int& voxel_split_array_size_PL_x_array_size_hw, voxel_int& voxel_split_array_size_PL_y_array_size_hw, voxel_int& voxel_split_array_size_PL_z_array_size_hw, voxel_int& total_calculated_voxel_size_hw)
{
//init
	float input_split_unit_local = input_split_unit;
	voxel_split_unit_PL = input_split_unit;

//)限制分割单元的范围
	if (input_split_unit_local > k_max_split_precision)
		input_split_unit_local = k_max_split_precision;
	if (input_split_unit_local <= 0)
		input_split_unit_local = 0.08;

//计算每个轴上的分割数量
	int x_split_size = ((data_set_max_min_PL_xmax - data_set_max_min_PL_xmin) / input_split_unit_local) + 1;	//
	int y_split_size = ((data_set_max_min_PL_ymax - data_set_max_min_PL_ymin) / input_split_unit_local) + 1;
	int z_split_size = ((data_set_max_min_PL_zmax - data_set_max_min_PL_zmin) / input_split_unit_local) + 1;

//限制分割数量的最大值
	if (x_split_size > k_axis_voxel_max)
		x_split_size = k_axis_voxel_max;
	if (y_split_size > k_axis_voxel_max)
		y_split_size = k_axis_voxel_max;
	if (z_split_size > k_axis_voxel_max)
		z_split_size = k_axis_voxel_max;
	//if the calculated size is too big, then the total_..may be minus number
//计算总的体素数量
	float total_sub_spaces = ((x_split_size) * (y_split_size) * (z_split_size));	//????wo.....

	//ensure the number of sub-spaces won't exceed the k_sub_region_max
	//todo: consider set a boundary and remove the points in dataset out the boundary
//计算子体素的分割单元大小，用于进一步细分体素。	
    sub_voxel_split_unit_PL_x = voxel_split_unit_PL / k_sub_voxel_x_size;//这是预设好的静态变量
    sub_voxel_split_unit_PL_y = voxel_split_unit_PL / k_sub_voxel_y_size;
    sub_voxel_split_unit_PL_z = voxel_split_unit_PL / k_sub_voxel_z_size;

	//when know the split unit and range, we can calculate the each split point and store it in a array.
	//stored data range: [min+true_split_precise, min + split_size*true_split_precise]
	//Use comparator "<" to calculate the index of the sub-spaces. so the first data ought to be min+true_split_precise

	//the split array range [min,max];(contain the min and max)
//将计算的分割数量和总的体素数量赋值给输出参数。
	voxel_split_array_size_PL_x_array_size = x_split_size;
	voxel_split_array_size_PL_y_array_size = y_split_size;
	voxel_split_array_size_PL_z_array_size = z_split_size;

    voxel_split_array_size_PL_x_array_size_hw = x_split_size;
	voxel_split_array_size_PL_y_array_size_hw = y_split_size;
	voxel_split_array_size_PL_z_array_size_hw = z_split_size;

    total_calculated_voxel_size = voxel_split_array_size_PL_x_array_size * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
    total_calculated_voxel_size_hw = total_calculated_voxel_size;

// 错误检查
	if(x_split_size <=0 || x_split_size > k_axis_voxel_max)
	    std::cout << "error, x_split_size <=0 || x_split_size > k_axis_voxel_max" << std::endl;

#ifdef DEBUG_MODE
	std::cout << std::endl;
	std::cout << "in ExSplitVoxelPrecise" << std::endl;
	std::cout << "x_split_size: " << x_split_size << " y_split_size: " << y_split_size << " z_split_size: " << z_split_size << std::endl;
	std::cout << "voxel_split_unit_PL: " << voxel_split_unit_PL << std::endl;

#endif
}

/*  wyt
    初始化硬件参数
    KNN_reference_set：点云数据数组，表示参考点集。
    KNN_reference_set_size：点云数据的大小（点的数量）。
    data_set_max_min_PL_xmin_hw：用于存储计算结果的引用参数。
*/
void setup_hardware_PL(My_PointXYZI* KNN_reference_set, int KNN_reference_set_size, type_point_hw& data_set_max_min_PL_xmin_hw, type_point_hw& data_set_max_min_PL_ymin_hw, type_point_hw& data_set_max_min_PL_zmin_hw, type_point_hw& voxel_split_unit_PL_hw, voxel_int& voxel_split_array_size_PL_x_array_size_hw, voxel_int& voxel_split_array_size_PL_y_array_size_hw, voxel_int& voxel_split_array_size_PL_z_array_size_hw, voxel_int& total_calculated_voxel_size_hw)
{
//调用 get_min_max 函数，计算点云数据在三轴上的最小值和最大值，并将结果存储在 data_set_max_min_PL_xmin_hw y z
    get_min_max(KNN_reference_set, KNN_reference_set_size, data_set_max_min_PL_xmin_hw, data_set_max_min_PL_ymin_hw, data_set_max_min_PL_zmin_hw);

//分割大小
    float input_split_unit = 1;

//调用函数，根据默认的分割单元大小和点云的边界范围，计算 X、Y、Z 轴上的体素分割数量以及总的体素数量，并将结果存储在传入的引用参数中。
    split_voxel_AABB(input_split_unit, voxel_split_array_size_PL_x_array_size_hw, voxel_split_array_size_PL_y_array_size_hw, voxel_split_array_size_PL_z_array_size_hw, total_calculated_voxel_size_hw);

//如果分割单元大小 input_split_unit 大于 0.1，则计算其平方值并存储在 square_input_precision_PL 中。
//否则，使用全局变量 voxel_split_unit_PL 的平方值。
    if(input_split_unit > 0.1)
		square_input_precision_PL = input_split_unit * input_split_unit;
	else
		square_input_precision_PL = voxel_split_unit_PL * voxel_split_unit_PL;

//将 X 轴的体素分割数量 voxel_split_array_size_PL_x_array_size 赋值给 k_sub_voxel_threshold
    k_sub_voxel_threshold = voxel_split_array_size_PL_x_array_size;

//如果k_sub_voxel_threshold >= k_transform_neighbor_num则将其设置为k_transform_neighbor_num - 1确保不超过预设的最大值
    if(k_sub_voxel_threshold >= k_transform_neighbor_num)
	{
		k_sub_voxel_threshold = k_transform_neighbor_num - 1;
    }

//设置硬件分割单元
    voxel_split_unit_PL_hw = 1;
}

/*  wyt
    为点云数据中的每个点计算一个哈希值，用于将三维空间中的点映射到一维
    KNN_reference_set：点云数据数组，表示参考点集。
    KNN_reference_set_size：点云数据的大小（点的数量）。
    data_set_hash：更新每个点的哈希值的数组。这是输出
*/
void calculate_hash(My_PointXYZI* KNN_reference_set, int* data_set_hash, int KNN_reference_set_size)
{
    for (int i = 0; i < KNN_reference_set_size; i++)
	{
        //copy the point_data to data_x,y,z
        float data_x = KNN_reference_set[i].x;
        float data_y = KNN_reference_set[i].y;
        float data_z = KNN_reference_set[i].z;
//获取XYZ轴上的体素分割数量。
        int x_split_array_size = voxel_split_array_size_PL_x_array_size;
        int y_split_array_size = voxel_split_array_size_PL_y_array_size;
        int z_split_array_size = voxel_split_array_size_PL_z_array_size;
        //default x,y,z index as the max index
        int x_index;
        int y_index;
        int z_index;
//计算xyz三轴索引
        if (data_x <= data_set_max_min_PL_xmin)
            x_index = 0;
        else
            x_index = (int)((data_x - data_set_max_min_PL_xmin) / voxel_split_unit_PL);

        if (x_index >= x_split_array_size)
            x_index = x_split_array_size - 1;

        if (data_y <= data_set_max_min_PL_ymin)
            y_index = 0;
        else
            y_index = (int)((data_y - data_set_max_min_PL_ymin) / voxel_split_unit_PL);

        if (y_index >= y_split_array_size)
            y_index = y_split_array_size - 1;

        if (data_z <= data_set_max_min_PL_zmin)
            z_index = 0;
        else
            z_index = (int)((data_z - data_set_max_min_PL_zmin) / voxel_split_unit_PL);

        if (z_index >= z_split_array_size)
            z_index = z_split_array_size - 1;

        //transform 3d index to a 1d index
//_index * y_split_array_size * z_split_array_size + y_index * z_split_array_size + z_index 这是个公式确保每个体素有一个唯一的哈希值
        int data_hash = x_index * y_split_array_size * z_split_array_size + y_index * z_split_array_size + z_index;
//确保hash合法        
        if (data_hash >= total_calculated_voxel_size)
            data_hash = (total_calculated_voxel_size - 1);
        if (data_hash < 0)
            data_hash = 0;
//存储
        data_set_hash[i] = data_hash;
    }
}

/*  wyt
    hls优化calculate_hash
    KNN_reference_set：输入流，表示点云数据。
    KNN_reference_set1：输出流，用于传递点云数据。
    data_set_hash：输出流，用于存储每个点的哈希值。
    KNN_reference_set_size：点云数据的大小（点的数量）。
    功能类似calculate_hash
 */
void calculate_hash_stream(hls::stream<My_PointXYZI>& KNN_reference_set, hls::stream<My_PointXYZI>& KNN_reference_set1, hls::stream<int>& data_set_hash, int KNN_reference_set_size)
{
    for (int i = 0; i < KNN_reference_set_size; i++)
	{
#pragma HLS LOOP_TRIPCOUNT max=k_query_set_size_max min=k_query_set_size_min
        My_PointXYZI temp = KNN_reference_set.read();
        KNN_reference_set1.write(temp);
        
        //copy the point_data to data_x,y,z
        float data_x = temp.x;
        float data_y = temp.y;
        float data_z = temp.z;
        int x_split_array_size = voxel_split_array_size_PL_x_array_size;
        int y_split_array_size = voxel_split_array_size_PL_y_array_size;
        int z_split_array_size = voxel_split_array_size_PL_z_array_size;
        //default x,y,z index as the max index
        int x_index;
        int y_index;
        int z_index;

        if (data_x <= data_set_max_min_PL_xmin)
            x_index = 0;
        else
            x_index = (int)((data_x - data_set_max_min_PL_xmin) / voxel_split_unit_PL);

        if (x_index >= x_split_array_size)
            x_index = x_split_array_size - 1;

        if (data_y <= data_set_max_min_PL_ymin)
            y_index = 0;
        else
            y_index = (int)((data_y - data_set_max_min_PL_ymin) / voxel_split_unit_PL);

        if (y_index >= y_split_array_size)
            y_index = y_split_array_size - 1;

        if (data_z <= data_set_max_min_PL_zmin)
            z_index = 0;
        else
            z_index = (int)((data_z - data_set_max_min_PL_zmin) / voxel_split_unit_PL);

        if (z_index >= z_split_array_size)
            z_index = z_split_array_size - 1;

        //transform 3d index to a 1d index
        int data_hash = x_index * y_split_array_size * z_split_array_size + y_index * z_split_array_size + z_index;
        
        if (data_hash >= total_calculated_voxel_size)
            data_hash = (total_calculated_voxel_size - 1);
        if (data_hash < 0)
            data_hash = 0;
        
        data_set_hash.write(data_hash);
    }
}

/*  wyt
    计算子哈希，映射子体素
    KNN_reference_set：点云数据数组，表示参考点集。
    data_hash：每个点的体素哈希值数组。
    KNN_reference_set_size：点云数据的大小（点的数量）。
    data_set_sub_hash：更新每个点的子哈希值的数组。这是输出
*/
void calculate_subhash(My_PointXYZI* KNN_reference_set, int* data_hash, int* data_set_sub_hash, int KNN_reference_set_size)
{
    for (int i = 0; i < KNN_reference_set_size; i++)
	{
//计算三轴体素索引
        int x_gain = voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size;
        int x_index = data_hash[i] / x_gain;

        int y_gain = voxel_split_array_size_PL_z_array_size;
        int y_index = (data_hash[i] - x_index * x_gain) / y_gain;
        int z_index = data_hash[i] - x_index * x_gain - y_index * y_gain;
//计算当前体素的边界范围
        My_MaxMin voxel_max_min_temp;
        //real_split_precision_PL = split_unit_PL_parm.x; .y .z
        voxel_max_min_temp.xmin = data_set_max_min_PL_xmin + (voxel_split_unit_PL * x_index);
        voxel_max_min_temp.xmax = voxel_max_min_temp.xmin + voxel_split_unit_PL;
        voxel_max_min_temp.ymin = data_set_max_min_PL_ymin + (voxel_split_unit_PL * y_index);
        voxel_max_min_temp.ymax = voxel_max_min_temp.ymin + voxel_split_unit_PL;
        voxel_max_min_temp.zmin = data_set_max_min_PL_zmin + (voxel_split_unit_PL * z_index);
        voxel_max_min_temp.zmax = voxel_max_min_temp.zmin + voxel_split_unit_PL;
//计算当前体素的边界范围
        int x_sub_index, y_sub_index, z_sub_index;

        if (KNN_reference_set[i].x <= voxel_max_min_temp.xmin)
            x_sub_index = 0;
        else
            x_sub_index = (uint_sub_voxel_size)((KNN_reference_set[i].x - voxel_max_min_temp.xmin) / sub_voxel_split_unit_PL_x);

        if (x_sub_index >= k_sub_voxel_x_size)
            x_sub_index = k_sub_voxel_x_size - 1;

        if (KNN_reference_set[i].y <= voxel_max_min_temp.ymin)
            y_sub_index = 0;
        else
            y_sub_index = (uint_sub_voxel_size)((KNN_reference_set[i].y - voxel_max_min_temp.ymin) / sub_voxel_split_unit_PL_y);

        if (y_sub_index >= k_sub_voxel_y_size)
            y_sub_index = k_sub_voxel_y_size - 1;

        if (KNN_reference_set[i].z <= voxel_max_min_temp.zmin)
            z_sub_index = 0;
        else
            z_sub_index = (uint_sub_voxel_size)((KNN_reference_set[i].z - voxel_max_min_temp.zmin) / sub_voxel_split_unit_PL_z);

        if (z_sub_index >= k_sub_voxel_z_size)
            z_sub_index = k_sub_voxel_z_size - 1;

        //transform 3d index to a 1d index
        uint_sub_voxel_size data_sub_hash = x_sub_index * k_sub_voxel_y_size * k_sub_voxel_z_size + y_sub_index * k_sub_voxel_z_size + z_sub_index;

        if (data_sub_hash >= k_sub_voxel_size){
            data_sub_hash = (k_sub_voxel_size - 1);
        }
        if (data_sub_hash < 0)
            data_sub_hash = 0;

        data_set_sub_hash[i] = data_sub_hash;
    }

}
/*  wyt
    将点云数据从数组传输到 HLS 流，硬件加速
    KNN_reference_set：点云数据数组，表示参考点集。
    KNN_reference_set_size：点云数据的大小（点的数量）。
    reference_input：输出流，用于存储传输的点云数据。
*/
void input_src(My_PointXYZI* KNN_reference_set, hls::stream<My_PointXYZI>& reference_input, int KNN_reference_set_size)
{ 
    for(int i = 0; i < KNN_reference_set_size; i++)
    {
#pragma HLS LOOP_TRIPCOUNT max=k_query_set_size_max min=k_query_set_size_min
        My_PointXYZI temp_image_pixel = KNN_reference_set[i];   //读取点云数据，存到temp
        My_PointXYZI temp_image_pixel_hw;                       // 复制点的坐标，存入temp
        temp_image_pixel_hw.x = temp_image_pixel.x;
        temp_image_pixel_hw.y = temp_image_pixel.y;
        temp_image_pixel_hw.z = temp_image_pixel.z;
        reference_input.write(temp_image_pixel_hw);             //将临时变量temp写入输出流reference_input以便后续处理。
    }
}

//计算两点距离
float cal_dist(My_PointXYZI data1, My_PointXYZI data2)
{
	return ((data1.x - data2.x) * (data1.x - data2.x)
		+ (data1.y - data2.y) * (data1.y - data2.y)
		+ (data1.z - data2.z) * (data1.z - data2.z)
		);
}



/*  暴力搜索
    KNN_query_set：查询点集数组，类型为 My_PointXYZI
    KNN_query_set_size：查询点集的大小（点的数量）。
    range_image_reference_set：参考点集数组类型为 My_PointXYZI
    range_image_reference_set_size：参考点集的大小（点的数量）。
    bf_KNN_query_result：每个查询点的最近邻点结果数组。这是输出
*/
void brute_force_search(My_PointXYZI* KNN_query_set, int KNN_query_set_size, My_PointXYZI* range_image_reference_set, int range_image_reference_set_size, My_PointXYZI* bf_KNN_query_result)
{
//初始化计时器
    TIMER_INIT(5);
    TIMER_START(0);

    for(int query_i = 0; query_i < KNN_query_set_size; query_i++)
    {
//获取当前查询点，初始化 this_query_KNN_result和最小距离 current_min_dis
        My_PointXYZI query_point = KNN_query_set[query_i];
        My_PointXYZI this_query_KNN_result;
        this_query_KNN_result.x = 0; this_query_KNN_result.y = 0;  this_query_KNN_result.z = 0;  this_query_KNN_result.intensity = 0; 

        float current_min_dis = 10000;
//遍历参考点
        for(int i = 0; i < range_image_reference_set_size; i++)
        {
//计算distance
            float this_dis = sqrt( 
                (query_point.x- range_image_reference_set[i].x) * (query_point.x- range_image_reference_set[i].x) + 
                (query_point.y- range_image_reference_set[i].y) * (query_point.y- range_image_reference_set[i].y) + 
                (query_point.z- range_image_reference_set[i].z) * (query_point.z- range_image_reference_set[i].z) 
            );
//更新最近邻点
            if(this_dis < current_min_dis)
            {
                this_query_KNN_result.x = range_image_reference_set[i].x;
                this_query_KNN_result.y = range_image_reference_set[i].y;
                this_query_KNN_result.z = range_image_reference_set[i].z;
                this_query_KNN_result.intensity = range_image_reference_set[i].intensity;
                current_min_dis = this_dis;
            }
        }
//存储最近邻点结果
        bf_KNN_query_result[query_i] = this_query_KNN_result;
    }
    int count = 0;
//停止计时器并输出耗时
    TIMER_STOP_ID(0);

    DEBUG_TIME("Finished brute force search with query points " << KNN_query_set_size << " with " << TIMER_REPORT_MS(0) << " ms !" );
}

/*  比较结果。用于验证新的最近邻搜索算法（new_query_result）与基准算法（gt_query_result）的一致性。
    KNN_query_set：查询点集数组
    KNN_query_set_size：查询点集数量
    new_query_result：新的最近邻搜索结果数组
    gt_query_result：基准（Ground Truth）最近邻搜索结果数组 ？这是什么
    error_count：统计误差数量，这是输出
*/
void compare_result(My_PointXYZI* KNN_query_set, int KNN_query_set_size, My_PointXYZI* new_query_result, My_PointXYZI* gt_query_result, int & error_count)
{
    int count = 0;
    for(int i = 0; i < KNN_query_set_size; i++)
    {
//比较差值，如果任一坐标差值超过阈值0.1，则认为存在误差。
        if(fabs(gt_query_result[i].x - new_query_result[i].x) > 0.1 || 
            fabs(gt_query_result[i].y - new_query_result[i].y) > 0.1 || 
            fabs(gt_query_result[i].z - new_query_result[i].z) > 0.1 )
        {
            count = count + 1;
//分别计算查询点KNN_query_set 与 gt_query_result 、 new_query_result的距离，dis<1 认为存在误差
            float dis_gt = sqrt( 
                (KNN_query_set[i].x- gt_query_result[i].x) * (KNN_query_set[i].x- gt_query_result[i].x) + 
                (KNN_query_set[i].y- gt_query_result[i].y) * (KNN_query_set[i].y- gt_query_result[i].y) + 
                (KNN_query_set[i].z- gt_query_result[i].z) * (KNN_query_set[i].z- gt_query_result[i].z) 
            );
            float dis_new = sqrt( 
                (KNN_query_set[i].x- new_query_result[i].x) * (KNN_query_set[i].x- new_query_result[i].x) + 
                (KNN_query_set[i].y- new_query_result[i].y) * (KNN_query_set[i].y- new_query_result[i].y) + 
                (KNN_query_set[i].z- new_query_result[i].z) * (KNN_query_set[i].z- new_query_result[i].z) 
            );
            
            /*
            DEBUG_LOG("error KNN result when i = " << i << " with gt_dis = " << dis_gt << " and new_dis = " << dis_new << " " << std::endl
            << " with query point: " << KNN_query_set[i].x << " " << KNN_query_set[i].y << " " << KNN_query_set[i].z << " "  << std::endl
            << " with gt_result: " << gt_query_result[i].x << " " << gt_query_result[i].y << " " << gt_query_result[i].z << " "  << std::endl
            << " with image_KNN_result: " << new_query_result[i].x << " " << new_query_result[i].y << " " << new_query_result[i].z 
             << " ");
            */

            if(dis_gt < 1)
                error_count ++;
        }
    }
    DEBUG_INFO(count);
}

/*
    data_hash：哈希表数组，存储每个点的哈希值（哈希桶索引）
    count_voxel_size：统计每个哈希桶中点数的数组
    KNN_reference_set_size：参考点集的数量

    data_hash[i] = 5，表示第 i 个点属于哈希桶 5。
    count_voxel_size[5] = 42，表示哈希桶 5 中已经有 42 个点。 执行 count_voxel_size[data_hash[i]] = count_voxel_size[data_hash[i]] + 1; 后：
    count_voxel_size[5] 的值变为 43，表示哈希桶 5 中现在有 43 个点。
*/
void count_hash(int* data_hash, int* count_voxel_size, int KNN_reference_set_size)
{
    for (int i = 0; i < KNN_reference_set_size; i++)
	{
        if (count_voxel_size[data_hash[i]] < 10000)
        {
                count_voxel_size[data_hash[i]] = count_voxel_size[data_hash[i]] + 1;
        }
        else
        {
    #ifdef ESSENTIAL_INFO
            std::cout << "error occurs, too many points in a cell, over 10000" << std::endl;
    #endif
        }
    }
}

/*  wyt 计算每个哈希桶（voxel）中第一个点的索引，并将结果存储在 first_index 数组中
    count_voxel_size：一个数组，记录每个哈希桶中的点数。
    count：哈希桶的总数。
    first_index：输出数组，存储每个哈希桶中第一个点的索引。
*/
void cal_hash_first_index(int* first_index, int* count_voxel_size, int count)
{
    first_index[0] = 0;
    for (int i = 1; i < count; i++)
	{
 // 当前哈希桶的起始索引 = 前一个哈希桶的起始索引 + 前一个哈希桶中的点数
        first_index[i] = first_index[i - 1] + count_voxel_size[i - 1];
    }
}

/*  将超过阈值的哈希桶分为子哈希桶，但是没超阈值的为什么要给k_sub_voxel_number_max，不懂
    voxel_first_index_PL：input，哈希桶中第一个点的索引
    sub_voxel_first_index_sentry：引用变量，用于跟踪下一个子哈希桶的起始索引。随函数更新
    bigger_voxel_number：引用变量，用于记录点数超过阈值的哈希桶的数量。随函数更新
    sub_voxel_flag_index_PL output，更新记录每个哈希桶的子哈希桶的起始索引。
*/
void subdivide_data_set(int* voxel_first_index_PL, int* sub_voxel_flag_index_PL, int& sub_voxel_first_index_sentry, int& bigger_voxel_number)
{
    for (int i = 0; i < total_calculated_voxel_size; i++)
	{
		int temp_cell_size = voxel_first_index_PL[i + 1] - voxel_first_index_PL[i];//计算当前哈希桶中的点数

		if (temp_cell_size >= k_sub_voxel_threshold)//k_sub_voxel_threshold是静态变量 ，若超过阈值
        {
            sub_voxel_flag_index_PL[i] = sub_voxel_first_index_sentry;//记录当前哈希桶的子哈希桶的起始索引
			sub_voxel_first_index_sentry += (k_sub_voxel_size + 1);	//one more sub_voxel to using index indicate cell_size//更新子哈希桶的起始索引，为下一个子哈希桶预留空间
			bigger_voxel_number = bigger_voxel_number + 1;//更新超过阈值的哈希桶的数量
		}
		else
		{
			sub_voxel_flag_index_PL[i] = k_sub_voxel_number_max;//k_sub_voxel_number_max .h文件中的全局变量，表不需要分
		}
	}
//处理最后一个哈希桶的特殊情况，total_calculated_voxel_size是上面定义的全局变量
	sub_voxel_flag_index_PL[total_calculated_voxel_size - 1] = sub_voxel_first_index_sentry; //todo?
}

/*  更新子哈希桶索引
    KNN_reference_set_size：数据集中点的总数。
    data_set_hash：每个点的哈希值，即它所属的哈希桶索引
    data_set_sub_hash：每个点的子哈希值
    sub_voxel_flag_index_PL：存储每个哈希桶的子哈希桶的起始索引（如果哈希桶被细分）或 k_sub_voxel_number_max（如果哈希桶未被细分）。
    sub_voxel_size：记录每个子哈希桶中的点数。output
*/
void calculate_split_data_set_hash(int KNN_reference_set_size, int* data_set_hash, int* data_set_sub_hash, int* sub_voxel_flag_index_PL, int* sub_voxel_size)
{
    for (int i = 0; i < KNN_reference_set_size; i++)
	{

		int data_hash = data_set_hash[i];//获取当前点的哈希值
		int sub_voxel_flag_index_temp = sub_voxel_flag_index_PL[data_hash];//获取当前哈希桶的子哈希桶起始索引

		//if the voxel was sub-split
		if (sub_voxel_flag_index_temp < k_sub_voxel_number_max)
		{
			int data_sub_hash = data_set_sub_hash[i];
			//calculate the size of each sub-voxel 更新输出，对应子哈希桶的大小
			sub_voxel_size[sub_voxel_flag_index_temp + data_sub_hash] = sub_voxel_size[sub_voxel_flag_index_temp + data_sub_hash] + 1;
		}
	}
}

/*  根据哈希桶的细分状态，记录子哈希桶的起始索引
    sub_voxel_flag_index_PL：每个哈希桶的子哈希桶起始索引（被细分）或k_sub_voxel_number_max（未被细分）。
    sub_voxel_first_index_PL：子哈希桶的起始索引
    voxel_first_index_PL：每个哈希桶的起始索引
    sub_voxel_first_index_sentry：子哈希桶的起始索引的哨兵值（通常用于初始化或边界检查）。
*/  
void cal_sub_voxel_first_index(int* sub_voxel_flag_index_PL, int* sub_voxel_first_index_PL, int* voxel_first_index_PL, int sub_voxel_first_index_sentry)
{
    for (int i = 0; i < total_calculated_voxel_size; i++)
	{
		int sub_voxel_flag_index_temp = sub_voxel_flag_index_PL[i];//获取当前哈希桶的子哈希桶起始索引
		if (sub_voxel_flag_index_temp < k_sub_voxel_number_max)//如果被细分
		{
            //将哈希桶 i 的起始索引赋值给子哈希桶的起始索引。个人理解：类似于树
			sub_voxel_first_index_PL[sub_voxel_flag_index_temp] = voxel_first_index_PL[i];
		}
	}
}

/*
    sub_voxel_first_index_PL：子哈希桶的起始索引 输出，根据函数变化
    sub_voxel_size：子哈希桶的大小
    sub_voxel_first_index_sentry：子哈希桶的起始索引的哨兵值（通常用于边界检查）。？？？
*/
void cal_sub_voxel_first_index2(int* sub_voxel_first_index_PL, int* sub_voxel_size, int sub_voxel_first_index_sentry)
{
    for (int i = 1; i < sub_voxel_first_index_sentry; i++)
	{
		if (sub_voxel_first_index_PL[i] == 0)//如果当前子哈希桶的起始索引未设置（即值为 0）
		{
			sub_voxel_first_index_PL[i] = sub_voxel_first_index_PL[i - 1] + sub_voxel_size[i - 1];//算当前子哈希桶的起始索引
		}
	}
}

/*  根据哈希桶和子哈希桶的结构，重新排序数据集并生成有序的索引和点云数据。更像把一堆书按照某种规则（如书名首字母）整理，但不需要保证每个书架的书是连续排列的。
    original_dataset_index：存储原始数据集的索引。
    KNN_reference_set_size：KNN 参考集的大小。
    data_set_hash：每个点的哈希桶索引。
    sub_voxel_flag_index_PL：子哈希桶是否被分割的标志位。
    data_set_sub_hash：子哈希桶索引。
    voxel_occupied_number：哈希桶中已占用的点数。
    sub_voxel_occupied_number：子哈希桶中已占用的点数。
    voxel_first_index_PL：哈希桶的起始索引。
    sub_voxel_first_index_PL：子哈希桶的起始索引。
    ordered_DSVS：重新排序后的索引。Output
    My_PointXYZI* ordered_ref：重新排序后的点云数据。Output
    My_PointXYZI* KNN_reference_set：原始 KNN 参考集。Output
*/
void reorder_data_set_reference(int* original_dataset_index, int KNN_reference_set_size, int* data_set_hash, int* sub_voxel_flag_index_PL, int* data_set_sub_hash, int* voxel_occupied_number, int* sub_voxel_occupied_number, int* voxel_first_index_PL, int* sub_voxel_first_index_PL, int* ordered_DSVS, My_PointXYZI* ordered_ref, My_PointXYZI* KNN_reference_set)
{
    for (int i = 0; i < KNN_reference_set_size; i++)
	{
		int current_voxel_hash = data_set_hash[i];//获取当前点的哈希索引
		int insert_index;

		int sub_voxel_flag_index_temp = sub_voxel_flag_index_PL[current_voxel_hash];
		//if has sub-voxel
		if (sub_voxel_flag_index_temp < k_sub_voxel_number_max)
		{
			int current_sub_voxel_index = data_set_sub_hash[i] + sub_voxel_flag_index_temp;//计算当前子哈希桶的全局索引，当前索引+起始索引
			//voxel_first_index + sub_voxel_index + sub_voxel_occupied_num.
			int sub_voxel_occupied_number_temp = sub_voxel_occupied_number[current_sub_voxel_index];//获取当前子哈希桶中已占用的点数
			insert_index = sub_voxel_first_index_PL[current_sub_voxel_index] + sub_voxel_occupied_number_temp;//计算插入索引：子哈希桶的起始索引 + 已占用的点数
			sub_voxel_occupied_number[current_sub_voxel_index] = sub_voxel_occupied_number_temp + 1;//更新当前子哈希桶中已占用的点数
		}
		//if without sub_voxel
		else
		{
			int voxel_occupied_number_temp = voxel_occupied_number[current_voxel_hash];//获取当前哈希桶中已占用的点数
			insert_index = (voxel_first_index_PL[current_voxel_hash] + voxel_occupied_number_temp);//计算插入索引：哈希桶的起始索引 + 已占用的点数
			voxel_occupied_number[current_voxel_hash] = voxel_occupied_number_temp + 1;//更新
		}

#ifdef ESSENTIAL_INFO
		if (insert_index >= KNN_reference_data_set_size)// 检查插入索引是否超出范围
		    std::cout << "error! insert_index > KNN_reference_data_set_size" << std::endl;
#endif
		ordered_DSVS[i] = insert_index;//索引数组，存储了重新排序后的位置信息。

        ordered_ref[ordered_DSVS[i]] = KNN_reference_set[i];//根据索引值，重排序，KNN_reference_set[i]：这是原始点云数据中的第 i 个点。

        original_dataset_index[ordered_DSVS[i]] = i;//更新原始数据集索引
//本质：将 KNN_reference_set[i] 赋值到 ordered_ref 的指定位置 ordered_DSVS[i]，从而实现数据的重新排序。
	}
}

/*  就像把一堆书按照书架编号（哈希桶）整理，每个书架的书都按顺序排列，并且每个书架的起始位置都记录在 query_set_first_index 中。
    original_dataset_index：
    KNN_reference_set_size：
    data_set_hash：
    voxel_occupied_number：
    query_set_first_index：
    ordered_DSVS：
    My_PointXYZI* ordered_ref：
    My_PointXYZI* KNN_reference_set：
*/
void reorder_data_set(int* original_dataset_index, int KNN_reference_set_size, int* data_set_hash, int* voxel_occupied_number, int* query_set_first_index, int* ordered_DSVS, My_PointXYZI* ordered_ref, My_PointXYZI* KNN_reference_set)
{
    for (int i = 0; i < KNN_reference_set_size; i++)
	{
        int current_voxel_hash = data_set_hash[i];
		int insert_index;

        int voxel_occupied_number_temp = voxel_occupied_number[current_voxel_hash];
//query_set_first_index 的作用：为每个哈希桶分配起始索引，确保重新排序后的数据集中每个哈希桶的点是连续存储的。
		insert_index = (query_set_first_index[current_voxel_hash] + voxel_occupied_number_temp);
		voxel_occupied_number[current_voxel_hash] = voxel_occupied_number_temp + 1;

#ifdef ESSENTIAL_INFO
		if (insert_index >= KNN_reference_data_set_size)
			std::cout << "error! insert_index > KNN_reference_data_set_size" << std::endl;
#endif
		ordered_DSVS[i] = insert_index;

        ordered_ref[ordered_DSVS[i]] = KNN_reference_set[i];

        original_dataset_index[ordered_DSVS[i]] = i;
    }
}

/*  根据reorder_data_set 生成一个有序的查询点集 ordered_query。同时，它还记录了原始查询点集中每个点的索引 original_dataset_index 和重新排序后的索引 ordered_DSVS
    original_dataset_index：用于存储重新排序后每个点在原始查询点集中的索引
    KNN_reference_set_size：查询点集的大小。
    data_set_hash：哈希值
    voxel_occupied_number：哈希桶中已经存储的点数。
    query_set_first_index：重新排序后的查询点的起始索引。
    ordered_DSVS：存储重新排序后每个点的索引。
    My_PointXYZI* ordered_query：重新排序后的查询点集。
    My_PointXYZI* KNN_reference_set：原始查询点集。
*/
void reorder_query(int* original_dataset_index, int KNN_reference_set_size, int* data_set_hash, int* voxel_occupied_number, int* query_set_first_index, int* ordered_DSVS, My_PointXYZI* ordered_query, My_PointXYZI* KNN_reference_set)
{
    for (int i = 0; i < KNN_reference_set_size; i++)
	{
        int current_voxel_hash = data_set_hash[i];
		int insert_index;

        int voxel_occupied_number_temp = voxel_occupied_number[current_voxel_hash];//当前哈希桶中已经存储的点数。
		insert_index = (query_set_first_index[current_voxel_hash] + voxel_occupied_number_temp);//当前点在重新排序后的查询点的插入位置，等于哈希桶的起始索引 query_set_first_index[current_voxel_hash] 加上当前哈希桶中已存储的点数。
		voxel_occupied_number[current_voxel_hash] = voxel_occupied_number_temp + 1;

#ifdef ESSENTIAL_INFO
		if (insert_index >= KNN_reference_data_set_size)
			std::cout << "error! insert_index > KNN_reference_data_set_size" << std::endl;
#endif
		ordered_DSVS[i] = insert_index;

        ordered_query[ordered_DSVS[i]] = KNN_reference_set[i];

        original_dataset_index[ordered_DSVS[i]] = i;
    }
}
/*  读取查询点：从输入流 reference_input 中读取查询点，并将其写入输出流 reference_input1
    读取查询点索引：从输入流 query_index1 中读取查询点的索引，并将其写入输出流 query_index2
    计算查询点的空间索引：根据查询点的索引 query_index_temp，计算其在三维空间中的 x、y、z 索引。
    计算局部网格的中心坐标：根据查询点的空间索引，计算其所属体素的中心坐标，并将结果写入输出流local_grid_center_x y z
    reference_input：输入流，包含查询点集。
    reference_input1：输出流，用于存储读取的查询点。
    query_index1：输入流，包含查询点的索引。
    query_index2：输出流，用于存储读取的查询点索引。
    local_grid_center_x y z:输出流，用于存储每个查询点所属体素的中心坐标。
    query_set_size：查询点集的大小。
*/
void initial(hls::stream<My_PointXYZI>& reference_input, hls::stream<My_PointXYZI>& reference_input1, hls::stream<int>& query_index1, hls::stream<int>& query_index2, hls::stream<float>& local_grid_center_x, hls::stream<float>& local_grid_center_y, hls::stream<float>& local_grid_center_z, int query_set_size)
{
    for (int i = 0; i < query_set_size; i++)
    {
#pragma HLS LOOP_TRIPCOUNT max=k_query_set_size_max min=k_query_set_size_min

    	My_PointXYZI temp = reference_input.read();
    	reference_input1.write(temp);//从输入流 reference_input 中读取一个查询点，并将其写入输出流 reference_input1。

    	int query_index_temp = query_index1.read();	// read query_i_copy
	    query_index2.write(query_index_temp);//从输入流 query_index1 中读取查询点的索引，并将其写入输出流 query_index2。
//变量表示在 x、y、z 三个方向上，空间被划分成多少个体素（voxel）。
        int local_x_split_size = voxel_split_array_size_PL_x_array_size;
        int local_y_split_size = voxel_split_array_size_PL_y_array_size;
        int local_z_split_size = voxel_split_array_size_PL_z_array_size;
//查询点xyz索引
        int query_x_index = int(query_index_temp / local_y_split_size / local_z_split_size);
        int query_y_index = int((query_index_temp - query_x_index * local_y_split_size * local_z_split_size) / local_z_split_size);
        int query_z_index = query_index_temp - query_x_index * local_y_split_size * local_z_split_size - query_y_index * local_z_split_size;
//通过查询点的索引和体素的大小，计算其所属体素的中心坐标。
        float temp_x = (query_x_index + 0.5);//偏移量，例如索引是 0，则中心位置是 0.5
        float temp_y = (query_y_index + 0.5);
        float temp_z = (query_z_index + 0.5);
        float local_grid_center_x_temp = data_set_max_min_PL_xmin + temp_x * voxel_split_unit_PL;//data_set_max_min_PL_xmin y z：空间在 x、y、z 方向上的最小值（起点）。
        float local_grid_center_y_temp = data_set_max_min_PL_ymin + temp_y * voxel_split_unit_PL;//voxel_split_unit_PL：每个体素大小（边长）。
        float local_grid_center_z_temp = data_set_max_min_PL_zmin + temp_z * voxel_split_unit_PL;//通过公式 中心坐标 = 起点 + (索引 + 0.5) * 网格大小，可以计算出局部网格的中心坐标。
//写入
        local_grid_center_x.write(local_grid_center_x_temp);
        local_grid_center_y.write(local_grid_center_y_temp);
        local_grid_center_z.write(local_grid_center_z_temp);
    }
}
/*  遍历查询点集，为每个查询点搜索其附近的候选体素。
    根据体素的标志位（sub_voxel_flag）判断体素是否有效，并将有效的候选体素索引写入输出流。
    控制候选体素的数量，确保不超过预设的最大值（k_near_voxel_size）。
    reference_input：输入流，包含查询点集。
    reference_input1：输出流，存储读取的查询点。
    query_index2：输入流，包含查询点的索引
    query_index3：输出流，存储读取的查询点索引。
    valid_near_voxels：输出流，存储有效的候选体素索引。
    voxel_flag：输出流，存储候选体素的标志位（0 或 1）。
    local_grid_center_x y z：输入流，体素的中心坐标。
    sub_voxel_flag_index_PL：数组，存储每个体素的标志位。
    query_set_size：
*/
void search_near_cells(hls::stream<My_PointXYZI>& reference_input, hls::stream<My_PointXYZI>& reference_input1, hls::stream<int>& query_index2, hls::stream<int>& query_index3, hls::stream<int>& valid_near_voxels, hls::stream<int>& voxel_flag, hls::stream<float>& local_grid_center_x, hls::stream<float>& local_grid_center_y, hls::stream<float>& local_grid_center_z, int* sub_voxel_flag_index_PL, int query_set_size)
{
    for (int i = 0; i < query_set_size; i++)
	{
#pragma HLS LOOP_TRIPCOUNT max=k_query_set_size_max min=k_query_set_size_min

    	My_PointXYZI temp = reference_input.read();//遍历查询点集，读取每个查询点的坐标和索引。
    	reference_input1.write(temp);//将读取的查询点写入输出流 reference_input1。
//读取查询点所属局部网格的中心坐标。
    	float local_grid_center_x_temp = local_grid_center_x.read();
        float local_grid_center_y_temp = local_grid_center_y.read();
        float local_grid_center_z_temp = local_grid_center_z.read();

        int query_index_temp = query_index2.read();	// read query_i_copy
	    query_index3.write(query_index_temp);//读取查询点的索引，并将其写入输出流 query_index3。

        int valid_voxel_count_num = 0;//记录当前有效的候选体素数量。
        int to_be_added_index[27];//存储查询点附近的候选体素索引。
//loc用于确定查询点相对于局部网格中心的位置。 还是不太懂 ？？？ loc 用于记录查询点的位置类型。
        int loc = 0; // back 3 7 front 1 5  
                     //      2 6       0 4
        //search candidate voxels.
//根据查询点的坐标与局部网格中心坐标的关系，确定查询点的位置（如 x y z、-x y z 等）。根据查询点的位置，计算其附近的候选体素索引，并存储到 to_be_added_index 数组中。
//在局部区域内，只有 8 个体素与查询点直接相邻，其他体素与查询点的相关性较低，可以被忽略。
        if (temp.x >= local_grid_center_x_temp && temp.y >= local_grid_center_y_temp && temp.z >= local_grid_center_z_temp) // x y z
        {
            to_be_added_index[0] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[1] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[2] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[3] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[4] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[5] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[6] = 1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[7] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 0;
            loc = 7;
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
            to_be_added_index[7] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 0;
            loc = 3;
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
            to_be_added_index[7] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 0;
            loc = 5;
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
            to_be_added_index[7] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 0;
            loc = 6;
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
            to_be_added_index[7] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 0;
            loc = 1;
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
            to_be_added_index[7] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 0;
            loc = 4;
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
            to_be_added_index[7] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 0;
            loc = 2;
        }
        else if (temp.x < local_grid_center_x_temp && temp.y < local_grid_center_y_temp && temp.z < local_grid_center_z_temp) // -x -y -z
        {
            to_be_added_index[0] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[1] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[2] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[3] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[4] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 1;
            to_be_added_index[5] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + -1 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[6] = -1 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 0;
            to_be_added_index[7] = 0 * voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size + 0 * voxel_split_array_size_PL_z_array_size + 0;
            loc = 0;
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
//to_be_added_index_table[27] 是一个偏移量表，用于计算查询点周围 27 个体素的索引。它的作用是提供一种高效的方式，通过查询点的索引和偏移量，快速定位其周围的候选体素。
        int to_be_added_index_table[27] = {0, voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size, -voxel_split_array_size_PL_y_array_size * voxel_split_array_size_PL_z_array_size,
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
        
        int loc_table[27*8] = {0,1,2,3,4,5,6,7, 0,1,2,3,0,1,2,3, 4,5,6,7,4,5,6,7,
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
        
        int candidate_region_index = query_index_temp + to_be_added_index_table[0];//根据查询点的索引和 to_be_added_index_table 中的偏移量，计算候选体素的索引。

        int sub_voxel_flag = sub_voxel_flag_index_PL[candidate_region_index];

        if (sub_voxel_flag == k_sub_voxel_number_max)//如果体素标志位 sub_voxel_flag 等于 k_sub_voxel_number_max，表示该体素是有效的候选体素。
        {

            if (valid_voxel_count_num < k_near_voxel_size - 1)
            {
                valid_near_voxels.write(candidate_region_index);
                voxel_flag.write(1);
                valid_voxel_count_num++;
            }
            for (int ii = 1; ii < 27; ii++)
            {
                candidate_region_index = query_index_temp + to_be_added_index_table[ii];

                    sub_voxel_flag = sub_voxel_flag_index_PL[candidate_region_index];
                    if (sub_voxel_flag == k_sub_voxel_number_max)
                    {

                        if (valid_voxel_count_num < k_near_voxel_size - 1)
                        {
                            valid_near_voxels.write(candidate_region_index);
                            voxel_flag.write(1);
                            valid_voxel_count_num++;
                        }

                    }
                    else/////////////////////
                    {
                        int candidate_search_sub_voxel_index = sub_voxel_flag + loc_table[8*ii+loc];
                        if (valid_voxel_count_num < k_near_voxel_size - 1)
                        {
                            valid_near_voxels.write(candidate_search_sub_voxel_index);
                            voxel_flag.write(0);
                            valid_voxel_count_num++;
                        }                
                    }
            }
        }
        else//否则，根据 loc_table 计算子体素的索引。
        {
            int candidate_search_sub_voxel_index = sub_voxel_flag + loc_table[loc];
            if (valid_voxel_count_num < k_near_voxel_size - 1)
            {
                for (int paral_ins_index = 0; paral_ins_index < k_sub_voxel_size; paral_ins_index++)
                {
                    valid_near_voxels.write(sub_voxel_flag + paral_ins_index);
                    voxel_flag.write(0);
                }
                valid_voxel_count_num++;
            }
            for (int ii = 1; ii < 27; ii++)
            {
                candidate_region_index = query_index_temp + to_be_added_index_table[ii];

                    sub_voxel_flag = sub_voxel_flag_index_PL[candidate_region_index];

                    if (sub_voxel_flag == k_sub_voxel_number_max)
                    {
                        if (valid_voxel_count_num < k_near_voxel_size - 1)
                        {
                            valid_near_voxels.write(candidate_region_index);
                            voxel_flag.write(1);
                            valid_voxel_count_num++;
                        }

                    }
                    else
                    {
                        int candidate_search_sub_voxel_index = sub_voxel_flag + loc_table[8*ii+loc];
                        if (valid_voxel_count_num < k_near_voxel_size - 1)
                        {
                            valid_near_voxels.write(candidate_search_sub_voxel_index);
                            voxel_flag.write(0);
                            valid_voxel_count_num++;
                        }                
                    }
            }
        }

        valid_near_voxels.write(k_reference_set_size + 1);
        voxel_flag.write(1);

    }
}

void search_candidate_neighbors(hls::stream<My_PointXYZI>& reference_input, hls::stream<My_PointXYZI>& reference_input1, hls::stream<int>& query_index3, hls::stream<int>& query_index4, hls::stream<int>& valid_near_voxels, hls::stream<int>& voxel_flag, hls::stream<int> candidate_neighbors[k_transform_neighbor_num], int* voxel_first_index_PL, int* sub_voxel_first_index_PL, int query_set_size)
{
    for (int i = 0; i < query_set_size; i++)
	{
#pragma HLS LOOP_TRIPCOUNT max=k_query_set_size_max min=k_query_set_size_min

    	My_PointXYZI temp = reference_input.read();
    	reference_input1.write(temp);

        int query_index_temp = query_index3.read();	// read query_i_copy
	    query_index4.write(query_index_temp);

        int candidate_neighbors_count_local = 0;

	    int over_threshold_num = 0;
	    int over_threshold_sub_voxel[k_over_thre_sub_voxel_num];
        
        loop_valid_voxels:
        for(int region_i = 0; region_i < 34; region_i++)
        {
            int current_search_hash_local;
            int voxel_flag_local;

            current_search_hash_local = valid_near_voxels.read();
            voxel_flag_local = voxel_flag.read();

            if (current_search_hash_local != k_reference_set_size + 1)
            {
                    //if near voxel
                    if (voxel_flag_local == 1)
                    {
                        int current_search_hash = current_search_hash_local;
                        if (current_search_hash >= 0 && current_search_hash < total_calculated_voxel_size)
                        {

                            int hash_start_index = voxel_first_index_PL[current_search_hash];
                            int current_voxel_size = voxel_first_index_PL[current_search_hash + 1] - hash_start_index; //size of current voxel.

    #ifdef ESSENTIAL_INFO
        if (current_voxel_size >= k_transform_neighbor_num)
            std::cout << "error !  current_voxel_size >= k_transform_neighbor_num! " << current_voxel_size << " >= " << k_transform_neighbor_num << std::endl;
    #endif

                            if (current_voxel_size > 0)
                            {
                                for (int temp_index = 0; temp_index < k_transform_neighbor_num; temp_index++)
                                {
                                    if (temp_index < current_voxel_size)
                                    {
                                        int candidate_neighbor_index = hash_start_index + temp_index;
                                        candidate_neighbors[temp_index].write(candidate_neighbor_index);
                                    }
                                    else
                                    {
                                        candidate_neighbors[temp_index].write(k_reference_set_size);
                                    }
                                }
                                candidate_neighbors_count_local = candidate_neighbors_count_local + 1;
                            }
                        }
                    }
    #ifdef SUB_VOXEL_SPLIT
                    else
                    {
                        int current_search_sub_hash = current_search_hash_local;
                        if (current_search_sub_hash < k_sub_voxel_number_max-1)	//remove sub_voxel out of range
                        {
                            int sub_hash_start_index = sub_voxel_first_index_PL[current_search_sub_hash];
                            int current_sub_voxel_size;

                            if(sub_voxel_first_index_PL[current_search_sub_hash + 1] > sub_hash_start_index)
                                current_sub_voxel_size = sub_voxel_first_index_PL[current_search_sub_hash + 1] - sub_hash_start_index; //size of current voxel.
                            else
                                current_sub_voxel_size = 0;

                            if (candidate_neighbors_count_local < k_select_loop_num - 1 && current_sub_voxel_size > 0)
                            {
                                if (current_sub_voxel_size >= k_transform_neighbor_num)
                                {
                                    if(over_threshold_num < k_over_thre_sub_voxel_num)
                                    {
                                        over_threshold_sub_voxel[over_threshold_num] = current_search_sub_hash;
                                        over_threshold_num = over_threshold_num + 1;
                                    }
                                }
                                for (int temp_index = 0; temp_index < k_transform_neighbor_num; temp_index++)
                                {
                                    if (temp_index < current_sub_voxel_size)
                                    {
                                        int candidate_neighbor_index = sub_hash_start_index + temp_index;
                                        candidate_neighbors[temp_index].write(candidate_neighbor_index);
                                    }
                                    else
                                    {
                                        candidate_neighbors[temp_index].write(k_reference_set_size);
                                    }
                                }

                                candidate_neighbors_count_local = candidate_neighbors_count_local + 1;
                            }
                        }
                    }
    #endif
            }	//if !empty()
            else
            {
                break;
            }
        }	// for valid_voxels

        //for over threshold sub-voxel
        for(int over_sub_index = 0; over_sub_index < over_threshold_num; over_sub_index++)
        {
            int current_search_sub_hash = over_threshold_sub_voxel[over_sub_index];
            int sub_hash_start_index = sub_voxel_first_index_PL[current_search_sub_hash] + k_transform_neighbor_num;
            int current_sub_voxel_size = sub_voxel_first_index_PL[current_search_sub_hash + 1] - sub_hash_start_index; //size of current voxel.

            if (candidate_neighbors_count_local < k_select_loop_num - 1 && current_sub_voxel_size > 0)
            {
                for (int temp_index = 0; temp_index < k_transform_neighbor_num; temp_index++)
                {
                    if (temp_index < current_sub_voxel_size)
                    {
                        int candidate_neighbor_index = sub_hash_start_index + temp_index;
                        candidate_neighbors[temp_index].write(candidate_neighbor_index);
                    }
                    else
                    {
                        candidate_neighbors[temp_index].write(k_reference_set_size);
                    }
                }
            }

            candidate_neighbors_count_local = candidate_neighbors_count_local + 1;

        }

        for (int temp_index = 0; temp_index < k_transform_neighbor_num; temp_index++)
        {
            if (temp_index == k_transform_neighbor_num - 1)
                candidate_neighbors[temp_index].write(0);
            else
                candidate_neighbors[temp_index].write(k_reference_set_size + 1);
        }
        candidate_neighbors_count_local = candidate_neighbors_count_local + 1; 
    }
}



void get_point(hls::stream<My_PointXYZI>& reference_input, hls::stream<int>& query_index4, hls::stream<int>& query_index5, hls::stream<int> candidate_neighbors[k_transform_neighbor_num], hls::stream<float> candidate_distance[k_transform_neighbor_num], hls::stream<int> original_dataset_index_local[k_transform_neighbor_num], int* voxel_first_index_PL, My_PointXYZI* data_set, int* original_dataset_index, int query_set_size, int& dataset_buffer_max_index_PL, int& dataset_buffer_min_index_PL, int data_set_size)
{
//#pragma HLS ARRAY_PARTITION dim=1 factor=60 type=cyclic variable=original_data_index_buffer
//#pragma HLS ARRAY_PARTITION dim=1 factor=60 type=cyclic variable=data_set_buffer
    int c0513 = 0;
    for (int i = 0; i < query_set_size; i++)
    {
#pragma HLS LOOP_TRIPCOUNT max=k_query_set_size_max min=k_query_set_size_min

    	My_PointXYZI temp = reference_input.read();
    	//reference_input1.write(temp);

    	int query_index_temp = query_index4.read();	// read query_i_copy
	    query_index5.write(query_index_temp);

        get_point_label0:for (int count_i = 0; count_i < 68; count_i++)
        {
            int candidate_neighbors_local1[k_transform_neighbor_num];

            for (int cache_temp_index = 0; cache_temp_index < k_transform_neighbor_num; cache_temp_index++)
            {

                candidate_neighbors_local1[cache_temp_index] = candidate_neighbors[cache_temp_index].read();
            }

            if (candidate_neighbors_local1[0] == k_reference_set_size + 1)
            {
                for (int temp_index = 0; temp_index < k_transform_neighbor_num; temp_index++)
                {
                    candidate_distance[temp_index].write(k_reference_set_size + 1);
                    original_dataset_index_local[temp_index].write(k_reference_set_size + 1);
                }
                break;
            }
            else
            {

                    for (int temp_index = 0; temp_index < k_transform_neighbor_num; temp_index++)
                    {


     int current_neighbor_index;
                        current_neighbor_index = candidate_neighbors_local1[temp_index];

                        if (current_neighbor_index > 0 && current_neighbor_index < data_set_size)
                        {
                            My_PointXYZI current_data_point = data_set[current_neighbor_index];
                            original_dataset_index_local[temp_index].write(original_dataset_index[current_neighbor_index]);
                            candidate_distance[temp_index].write(cal_dist(temp, current_data_point));
                            c0513 = c0513 + 1;
                        }
                        else
                        {
                            candidate_distance[temp_index].write(100000);
                            original_dataset_index_local[temp_index].write(100000);
                        }
                    }
            }
        }
    }
}



void select_knn(int* query_result, float* nearest_distance, hls::stream<float> candidate_distance_s[k_transform_neighbor_num], hls::stream<int> original_dataset_index[k_transform_neighbor_num], int query_set_size, hls::stream<int>& query_index5)
{

    for (int i = 0; i < query_set_size; i++)
    {
#pragma HLS LOOP_TRIPCOUNT max=k_query_set_size_max min=k_query_set_size_min
        int nearest_index_PL_local[k_nearest_number_max];
        float nearest_distance_PL_local[k_nearest_number_max];

        int original_dataset_index_local[k_transform_neighbor_num];
        float candidate_distance[k_transform_neighbor_num];
        int cmp_array[k_nearest_number_max + k_transform_neighbor_num];

        int query_index_temp = query_index5.read();	// read query_i_copy

    loop_reset_result_array:
        for (int j = 0; j < k_nearest_number_max; j++)
        {

            nearest_index_PL_local[j] = 0;
            nearest_distance_PL_local[j] = 100000;
        }

        int read_pointer = 0;

    loop_dataflow_cal_knn:
        for (int count_i = 0; count_i < 68; count_i++)
        {

            for (int cache_temp_index = 0; cache_temp_index < k_transform_neighbor_num; cache_temp_index++)
            {

                candidate_distance[cache_temp_index] = candidate_distance_s[cache_temp_index].read();

                original_dataset_index_local[cache_temp_index] = original_dataset_index[cache_temp_index].read();
                
            }

            if (candidate_distance[0] == k_reference_set_size + 1)
            {
                break;
            }

           
                    // cmp_array of nearest_neighbor_PL
                    for (int parallel_j = 0; parallel_j < k_nearest_number_max; parallel_j++)
                    {

                        cmp_array[parallel_j] = parallel_j;
                    }

                    for (int parallel_i = 0; parallel_i < k_transform_neighbor_num; parallel_i++)
                    {

                        cmp_array[parallel_i + k_nearest_number_max ] = -1;

                        for (int parallel_j = 0; parallel_j < k_transform_neighbor_num; parallel_j++)
                        {
                            if (candidate_distance[parallel_i] > candidate_distance[parallel_j] || ( candidate_distance[parallel_i] == candidate_distance[parallel_j] && parallel_i >= parallel_j ))
                                cmp_array[parallel_i + k_nearest_number_max ] ++;
                        }
                        for (int parallel_j = 0; parallel_j < k_nearest_number_max; parallel_j++)
                        {
                            if (candidate_distance[parallel_i] >= nearest_distance_PL_local[parallel_j])
                                cmp_array[parallel_i + k_nearest_number_max ] ++;
                            else cmp_array[parallel_j] ++;
                        }
                    }

                    //move the element in nearest_distance_PL_local
                    for (int parallel_i = k_nearest_number_max - 1; parallel_i >= 0; parallel_i--)
                    {


                        if (cmp_array[parallel_i] < k_nearest_number_max && cmp_array[parallel_i] != parallel_i)
                        {
                            nearest_distance_PL_local[cmp_array[parallel_i]] = nearest_distance_PL_local[parallel_i];
                            nearest_index_PL_local[cmp_array[parallel_i]] = nearest_index_PL_local[parallel_i];
                        }
                    }

                    //move the element in candidate_distance
                    for (int parallel_i = 0; parallel_i < k_transform_neighbor_num; parallel_i++)
                    {

                        if (cmp_array[parallel_i + k_nearest_number_max ] < k_nearest_number_max)
                        {
                            nearest_distance_PL_local[cmp_array[parallel_i + k_nearest_number_max ]] = candidate_distance[parallel_i];
                            nearest_index_PL_local[cmp_array[parallel_i + k_nearest_number_max ]] = original_dataset_index_local[parallel_i];
                        }
                    }

        } //while()


        {
        loop_output:
            for (int j = 0; j < k_nearest_number_max; j++)
            {
                query_result[i] = nearest_index_PL_local[j];
                nearest_distance[i] = nearest_distance_PL_local[j];
            }
        }
    }
}



void DSVS_build(int* original_dataset_index, My_PointXYZI* KNN_reference_set, int KNN_reference_set_size, type_flag_hw sharp_flag, int* ordered_DSVS, My_PointXYZI* ordered_query, int* voxel_first_index_PL, int* sub_voxel_flag_index_PL, int* sub_voxel_first_index_PL, bool reorder_query_set, My_PointXYZI_HW16* ordered_ref16_x, My_PointXYZI_HW16* ordered_ref16_y, My_PointXYZI_HW16* ordered_ref16_z, inthw16* original_dataset_index16, indexint* index16, indexint* subindex16, int& packs)
{
    int voxel_occupied_number[k_voxels_number_max];		//the first index of each grid when merge them in ordered hash array.
	int count_voxel_size[k_voxels_number_max];		//count the number of data in each grid  todo
	int data_set_hash[KNN_reference_set_size];			//hash_value <= k_voxels_number_max, can be optimized..
	int data_set_sub_hash[KNN_reference_set_size];			//hash_value <= k_voxels_number_max, can be optimized..
	int query_set_first_index[k_voxels_number_max];  

    for (int i = 0; i < k_voxels_number_max; i++)
	{
		voxel_occupied_number[i] = 0;
		count_voxel_size[i] = 0;
		query_set_first_index[i] = 0;
	}

	calculate_hash(KNN_reference_set, data_set_hash, KNN_reference_set_size);
	
    calculate_subhash(KNN_reference_set, data_set_hash, data_set_sub_hash, KNN_reference_set_size);

    count_hash(data_set_hash, count_voxel_size, KNN_reference_set_size);

    if (!reorder_query_set)
        cal_hash_first_index(voxel_first_index_PL, count_voxel_size, total_calculated_voxel_size);
    else
        cal_hash_first_index(query_set_first_index, count_voxel_size, total_calculated_voxel_size);

#ifdef SUB_VOXEL_SPLIT
    
	// sub_voxel
	int sub_voxel_occupied_number[k_sub_voxel_number_max];
	int sub_voxel_size[k_sub_voxel_number_max];	//count the number of data in each split cell

	if (!reorder_query_set)	
    {
		int sub_voxel_first_index_sentry = 0;
		int bigger_voxel_number = 0;		

        subdivide_data_set(voxel_first_index_PL, sub_voxel_flag_index_PL, sub_voxel_first_index_sentry, bigger_voxel_number);

	loop_reset_split_arrays:
		for (int i = 0; i < sub_voxel_first_index_sentry; i++)
		{
			sub_voxel_size[i] = 0;
			sub_voxel_first_index_PL[i] = 0;
			sub_voxel_occupied_number[i] = 0;
		}

        calculate_split_data_set_hash(KNN_reference_set_size, data_set_hash, data_set_sub_hash, sub_voxel_flag_index_PL, sub_voxel_size);

        cal_sub_voxel_first_index(sub_voxel_flag_index_PL, sub_voxel_first_index_PL, voxel_first_index_PL, sub_voxel_first_index_sentry);

        cal_sub_voxel_first_index2(sub_voxel_first_index_PL, sub_voxel_size, sub_voxel_first_index_sentry);

        DEBUG_INFO(sub_voxel_first_index_sentry);
        DEBUG_INFO(bigger_voxel_number);
	}

    if (!reorder_query_set)
    {
        reorder_data_set_reference(original_dataset_index, KNN_reference_set_size, data_set_hash, sub_voxel_flag_index_PL, data_set_sub_hash, voxel_occupied_number, sub_voxel_occupied_number, voxel_first_index_PL, sub_voxel_first_index_PL, ordered_DSVS, ordered_query, KNN_reference_set);
        
        int jj = 0;
        int num1 = 0;


        
        for (int i = 0; i < total_calculated_voxel_size; i++)
        {
            int c = 0;
            if (sub_voxel_flag_index_PL[i] == k_sub_voxel_number_max)
            {
                if (count_voxel_size[i] > 0)
                {
                    
                    if (count_voxel_size[i] >= 1)
                    {
                        ordered_ref16_x[jj].p1 = ordered_query[voxel_first_index_PL[i]].x;
                        ordered_ref16_y[jj].p1 = ordered_query[voxel_first_index_PL[i]].y;
                        ordered_ref16_z[jj].p1 = ordered_query[voxel_first_index_PL[i]].z;
                        original_dataset_index16[jj].p1 = original_dataset_index[voxel_first_index_PL[i]];
                        num1 = num1 + 1;
                    }
                    else
                    {
                        ordered_ref16_x[jj].p1 = 0;
                        ordered_ref16_y[jj].p1 = 0;
                        ordered_ref16_z[jj].p1 = 0;
                        original_dataset_index16[jj].p1 = k_reference_set_size;
                    }
                    if (count_voxel_size[i] >= 2)
                    {
                        ordered_ref16_x[jj].p2 = ordered_query[voxel_first_index_PL[i]+1].x;
                        ordered_ref16_y[jj].p2 = ordered_query[voxel_first_index_PL[i]+1].y;
                        ordered_ref16_z[jj].p2 = ordered_query[voxel_first_index_PL[i]+1].z;
                        original_dataset_index16[jj].p2 = original_dataset_index[voxel_first_index_PL[i]+1];
                        num1 = num1 + 1;
                    }
                    else
                    {
                        ordered_ref16_x[jj].p2 = 0;
                        ordered_ref16_y[jj].p2 = 0;
                        ordered_ref16_z[jj].p2 = 0;
                        original_dataset_index16[jj].p2 = k_reference_set_size;
                    }
                    if (count_voxel_size[i] >= 3)
                    {
                        ordered_ref16_x[jj].p3 = ordered_query[voxel_first_index_PL[i]+2].x;
                        ordered_ref16_y[jj].p3 = ordered_query[voxel_first_index_PL[i]+2].y;
                        ordered_ref16_z[jj].p3 = ordered_query[voxel_first_index_PL[i]+2].z;
                        original_dataset_index16[jj].p3 = original_dataset_index[voxel_first_index_PL[i]+2];
                        num1 = num1 + 1;
                    }
                    else
                    {
                        ordered_ref16_x[jj].p3 = 0;
                        ordered_ref16_y[jj].p3 = 0;
                        ordered_ref16_z[jj].p3 = 0;
                        original_dataset_index16[jj].p3 = k_reference_set_size;
                    }
                    if (count_voxel_size[i] >= 4)
                    {
                        ordered_ref16_x[jj].p4 = ordered_query[voxel_first_index_PL[i]+3].x;
                        ordered_ref16_y[jj].p4 = ordered_query[voxel_first_index_PL[i]+3].y;
                        ordered_ref16_z[jj].p4 = ordered_query[voxel_first_index_PL[i]+3].z;
                        original_dataset_index16[jj].p4 = original_dataset_index[voxel_first_index_PL[i]+3];
                        num1 = num1 + 1;
                    }
                    else
                    {
                        ordered_ref16_x[jj].p4 = 0;
                        ordered_ref16_y[jj].p4 = 0;
                        ordered_ref16_z[jj].p4 = 0;
                        original_dataset_index16[jj].p4 = k_reference_set_size;
                    }

                    if (count_voxel_size[i] >= 5)
                    {
                        ordered_ref16_x[jj].p5 = ordered_query[voxel_first_index_PL[i]+4].x;
                        ordered_ref16_y[jj].p5 = ordered_query[voxel_first_index_PL[i]+4].y;
                        ordered_ref16_z[jj].p5 = ordered_query[voxel_first_index_PL[i]+4].z;
                        original_dataset_index16[jj].p5 = original_dataset_index[voxel_first_index_PL[i]+4];
                        num1 = num1 + 1;
                    }
                    else
                    {
                        ordered_ref16_x[jj].p5 = 0;
                        ordered_ref16_y[jj].p5 = 0;
                        ordered_ref16_z[jj].p5 = 0;
                        original_dataset_index16[jj].p5 = k_reference_set_size;
                    }
                    if (count_voxel_size[i] >= 6)
                    {
                        ordered_ref16_x[jj].p6 = ordered_query[voxel_first_index_PL[i]+5].x;
                        ordered_ref16_y[jj].p6 = ordered_query[voxel_first_index_PL[i]+5].y;
                        ordered_ref16_z[jj].p6 = ordered_query[voxel_first_index_PL[i]+5].z;
                        original_dataset_index16[jj].p6 = original_dataset_index[voxel_first_index_PL[i]+5];
                        num1 = num1 + 1;  
                    }
                    else
                    {
                        ordered_ref16_x[jj].p6 = 0;
                        ordered_ref16_y[jj].p6 = 0;
                        ordered_ref16_z[jj].p6 = 0;
                        original_dataset_index16[jj].p6 = k_reference_set_size;
                    }
                    if (count_voxel_size[i] >= 7)
                    {
                        ordered_ref16_x[jj].p7 = ordered_query[voxel_first_index_PL[i]+6].x;
                        ordered_ref16_y[jj].p7 = ordered_query[voxel_first_index_PL[i]+6].y;
                        ordered_ref16_z[jj].p7 = ordered_query[voxel_first_index_PL[i]+6].z;
                        original_dataset_index16[jj].p7 = original_dataset_index[voxel_first_index_PL[i]+6];
                        num1 = num1 + 1; 
                    }
                    else
                    {
                        ordered_ref16_x[jj].p7 = 0;
                        ordered_ref16_y[jj].p7 = 0;
                        ordered_ref16_z[jj].p7 = 0;
                        original_dataset_index16[jj].p7 = k_reference_set_size;
                    }
                    if (count_voxel_size[i] >= 8)
                    {
                        ordered_ref16_x[jj].p8 = ordered_query[voxel_first_index_PL[i]+7].x;
                        ordered_ref16_y[jj].p8 = ordered_query[voxel_first_index_PL[i]+7].y;
                        ordered_ref16_z[jj].p8 = ordered_query[voxel_first_index_PL[i]+7].z;
                        original_dataset_index16[jj].p8 = original_dataset_index[voxel_first_index_PL[i]+7];
                        num1 = num1 + 1;
                    }
                    else
                    {
                        ordered_ref16_x[jj].p8 = 0;
                        ordered_ref16_y[jj].p8 = 0;
                        ordered_ref16_z[jj].p8 = 0;
                        original_dataset_index16[jj].p8 = k_reference_set_size;
                    }

                    if (count_voxel_size[i] >= 9)
                    {
                        ordered_ref16_x[jj].p9 = ordered_query[voxel_first_index_PL[i]+8].x;
                        ordered_ref16_y[jj].p9 = ordered_query[voxel_first_index_PL[i]+8].y;
                        ordered_ref16_z[jj].p9 = ordered_query[voxel_first_index_PL[i]+8].z;
                        original_dataset_index16[jj].p9 = original_dataset_index[voxel_first_index_PL[i]+8];
                        num1 = num1 + 1; 
                    }
                    else
                    {
                        ordered_ref16_x[jj].p9 = 0;
                        ordered_ref16_y[jj].p9 = 0;
                        ordered_ref16_z[jj].p9 = 0;
                        original_dataset_index16[jj].p9 = k_reference_set_size;
                    }
                    if (count_voxel_size[i] >= 10)
                    {
                        ordered_ref16_x[jj].p10 = ordered_query[voxel_first_index_PL[i]+9].x;
                        ordered_ref16_y[jj].p10 = ordered_query[voxel_first_index_PL[i]+9].y;
                        ordered_ref16_z[jj].p10 = ordered_query[voxel_first_index_PL[i]+9].z;
                        original_dataset_index16[jj].p10 = original_dataset_index[voxel_first_index_PL[i]+9];
                        num1 = num1 + 1;
                    }
                    else
                    {
                        ordered_ref16_x[jj].p10 = 0;
                        ordered_ref16_y[jj].p10 = 0;
                        ordered_ref16_z[jj].p10 = 0;
                        original_dataset_index16[jj].p10 = k_reference_set_size;
                    }
                    if (count_voxel_size[i] >= 11)
                    {
                        ordered_ref16_x[jj].p11 = ordered_query[voxel_first_index_PL[i]+10].x;
                        ordered_ref16_y[jj].p11 = ordered_query[voxel_first_index_PL[i]+10].y;
                        ordered_ref16_z[jj].p11 = ordered_query[voxel_first_index_PL[i]+10].z;
                        original_dataset_index16[jj].p11 = original_dataset_index[voxel_first_index_PL[i]+10];
                        num1 = num1 + 1;
                    }
                    else
                    {
                        ordered_ref16_x[jj].p11 = 0;
                        ordered_ref16_y[jj].p11 = 0;
                        ordered_ref16_z[jj].p11 = 0;
                        original_dataset_index16[jj].p11 = k_reference_set_size;
                    }
                    if (count_voxel_size[i] >= 12)
                    {
                        ordered_ref16_x[jj].p12 = ordered_query[voxel_first_index_PL[i]+11].x;
                        ordered_ref16_y[jj].p12 = ordered_query[voxel_first_index_PL[i]+11].y;
                        ordered_ref16_z[jj].p12 = ordered_query[voxel_first_index_PL[i]+11].z;
                        original_dataset_index16[jj].p12 = original_dataset_index[voxel_first_index_PL[i]+11];
                        num1 = num1 + 1; 
                    }
                    else
                    {
                        ordered_ref16_x[jj].p12 = 0;
                        ordered_ref16_y[jj].p12 = 0;
                        ordered_ref16_z[jj].p12 = 0;
                        original_dataset_index16[jj].p12 = k_reference_set_size;
                    }

                    if (count_voxel_size[i] >= 13)
                    {
                        ordered_ref16_x[jj].p13 = ordered_query[voxel_first_index_PL[i]+12].x;
                        ordered_ref16_y[jj].p13 = ordered_query[voxel_first_index_PL[i]+12].y;
                        ordered_ref16_z[jj].p13 = ordered_query[voxel_first_index_PL[i]+12].z;
                        original_dataset_index16[jj].p13 = original_dataset_index[voxel_first_index_PL[i]+12];
                        num1 = num1 + 1;
                    }
                    else
                    {
                        ordered_ref16_x[jj].p13 = 0;
                        ordered_ref16_y[jj].p13 = 0;
                        ordered_ref16_z[jj].p13 = 0;
                        original_dataset_index16[jj].p13 = k_reference_set_size;
                    }
                    if (count_voxel_size[i] >= 14)
                    {
                        ordered_ref16_x[jj].p14 = ordered_query[voxel_first_index_PL[i]+13].x;
                        ordered_ref16_y[jj].p14 = ordered_query[voxel_first_index_PL[i]+13].y;
                        ordered_ref16_z[jj].p14 = ordered_query[voxel_first_index_PL[i]+13].z;
                        original_dataset_index16[jj].p14 = original_dataset_index[voxel_first_index_PL[i]+13];
                        num1 = num1 + 1;
                    }
                    else
                    {
                        ordered_ref16_x[jj].p14 = 0;
                        ordered_ref16_y[jj].p14 = 0;
                        ordered_ref16_z[jj].p14 = 0;
                        original_dataset_index16[jj].p14 = k_reference_set_size;
                    }
                    if (count_voxel_size[i] >= 15)
                    {
                        ordered_ref16_x[jj].p15 = ordered_query[voxel_first_index_PL[i]+14].x;
                        ordered_ref16_y[jj].p15 = ordered_query[voxel_first_index_PL[i]+14].y;
                        ordered_ref16_z[jj].p15 = ordered_query[voxel_first_index_PL[i]+14].z;
                        original_dataset_index16[jj].p15 = original_dataset_index[voxel_first_index_PL[i]+14];
                        num1 = num1 + 1;
                    }
                    else
                    {
                        ordered_ref16_x[jj].p15 = 0;
                        ordered_ref16_y[jj].p15 = 0;
                        ordered_ref16_z[jj].p15 = 0;
                        original_dataset_index16[jj].p15 = k_reference_set_size;
                    }
                    if (count_voxel_size[i] >= 16)
                    {
                        ordered_ref16_x[jj].p16 = ordered_query[voxel_first_index_PL[i]+15].x;
                        ordered_ref16_y[jj].p16 = ordered_query[voxel_first_index_PL[i]+15].y;
                        ordered_ref16_z[jj].p16 = ordered_query[voxel_first_index_PL[i]+15].z;
                        original_dataset_index16[jj].p16 = original_dataset_index[voxel_first_index_PL[i]+15];
                        num1 = num1 + 1;
                    }
                    else
                    {
                        ordered_ref16_x[jj].p16 = 0;
                        ordered_ref16_y[jj].p16 = 0;
                        ordered_ref16_z[jj].p16 = 0;
                        original_dataset_index16[jj].p16 = k_reference_set_size;
                    }

                    jj = jj + 1;
                    c = 1;

                }

                if (count_voxel_size[i] > 16)
                {
                    
                    if (count_voxel_size[i] >= 1+16)
                    {
                        ordered_ref16_x[jj].p1 = ordered_query[voxel_first_index_PL[i]+16].x;
                        ordered_ref16_y[jj].p1 = ordered_query[voxel_first_index_PL[i]+16].y;
                        ordered_ref16_z[jj].p1 = ordered_query[voxel_first_index_PL[i]+16].z;
                        original_dataset_index16[jj].p1 = original_dataset_index[voxel_first_index_PL[i]+16];
                        num1 = num1 + 1;
                    }
                    else
                    {
                        ordered_ref16_x[jj].p1 = 0;
                        ordered_ref16_y[jj].p1 = 0;
                        ordered_ref16_z[jj].p1 = 0;
                        original_dataset_index16[jj].p1 = k_reference_set_size;
                    }
                    if (count_voxel_size[i] >= 2+16)
                    {
                        ordered_ref16_x[jj].p2 = ordered_query[voxel_first_index_PL[i]+1+16].x;
                        ordered_ref16_y[jj].p2 = ordered_query[voxel_first_index_PL[i]+1+16].y;
                        ordered_ref16_z[jj].p2 = ordered_query[voxel_first_index_PL[i]+1+16].z;
                        original_dataset_index16[jj].p2 = original_dataset_index[voxel_first_index_PL[i]+1+16];
                        num1 = num1 + 1;
                    }
                    else
                    {
                        ordered_ref16_x[jj].p2 = 0;
                        ordered_ref16_y[jj].p2 = 0;
                        ordered_ref16_z[jj].p2 = 0;
                        original_dataset_index16[jj].p2 = k_reference_set_size;
                    }
                    if (count_voxel_size[i] >= 3+16)
                    {
                        ordered_ref16_x[jj].p3 = ordered_query[voxel_first_index_PL[i]+2+16].x;
                        ordered_ref16_y[jj].p3 = ordered_query[voxel_first_index_PL[i]+2+16].y;
                        ordered_ref16_z[jj].p3 = ordered_query[voxel_first_index_PL[i]+2+16].z;
                        original_dataset_index16[jj].p3 = original_dataset_index[voxel_first_index_PL[i]+2+16];
                        num1 = num1 + 1;
                    }
                    else
                    {
                        ordered_ref16_x[jj].p3 = 0;
                        ordered_ref16_y[jj].p3 = 0;
                        ordered_ref16_z[jj].p3 = 0;
                        original_dataset_index16[jj].p3 = k_reference_set_size;
                    }
                    if (count_voxel_size[i] >= 4+16)
                    {
                        ordered_ref16_x[jj].p4 = ordered_query[voxel_first_index_PL[i]+3+16].x;
                        ordered_ref16_y[jj].p4 = ordered_query[voxel_first_index_PL[i]+3+16].y;
                        ordered_ref16_z[jj].p4 = ordered_query[voxel_first_index_PL[i]+3+16].z;
                        original_dataset_index16[jj].p4 = original_dataset_index[voxel_first_index_PL[i]+3+16];
                        num1 = num1 + 1;
                    }
                    else
                    {
                        ordered_ref16_x[jj].p4 = 0;
                        ordered_ref16_y[jj].p4 = 0;
                        ordered_ref16_z[jj].p4 = 0;
                        original_dataset_index16[jj].p4 = k_reference_set_size;
                    }

                    if (count_voxel_size[i] >= 5+16)
                    {
                        ordered_ref16_x[jj].p5 = ordered_query[voxel_first_index_PL[i]+4+16].x;
                        ordered_ref16_y[jj].p5 = ordered_query[voxel_first_index_PL[i]+4+16].y;
                        ordered_ref16_z[jj].p5 = ordered_query[voxel_first_index_PL[i]+4+16].z;
                        original_dataset_index16[jj].p5 = original_dataset_index[voxel_first_index_PL[i]+4+16];
                        num1 = num1 + 1;
                    }
                    else
                    {
                        ordered_ref16_x[jj].p5 = 0;
                        ordered_ref16_y[jj].p5 = 0;
                        ordered_ref16_z[jj].p5 = 0;
                        original_dataset_index16[jj].p5 = k_reference_set_size;
                    }
                    if (count_voxel_size[i] >= 6+16)
                    {
                        ordered_ref16_x[jj].p6 = ordered_query[voxel_first_index_PL[i]+5+16].x;
                        ordered_ref16_y[jj].p6 = ordered_query[voxel_first_index_PL[i]+5+16].y;
                        ordered_ref16_z[jj].p6 = ordered_query[voxel_first_index_PL[i]+5+16].z;
                        original_dataset_index16[jj].p6 = original_dataset_index[voxel_first_index_PL[i]+5+16];
                        num1 = num1 + 1; 
                    }
                    else
                    {
                        ordered_ref16_x[jj].p6 = 0;
                        ordered_ref16_y[jj].p6 = 0;
                        ordered_ref16_z[jj].p6 = 0;
                        original_dataset_index16[jj].p6 = k_reference_set_size;
                    }
                    if (count_voxel_size[i] >= 7+16)
                    {
                        ordered_ref16_x[jj].p7 = ordered_query[voxel_first_index_PL[i]+6+16].x;
                        ordered_ref16_y[jj].p7 = ordered_query[voxel_first_index_PL[i]+6+16].y;
                        ordered_ref16_z[jj].p7 = ordered_query[voxel_first_index_PL[i]+6+16].z;
                        original_dataset_index16[jj].p7 = original_dataset_index[voxel_first_index_PL[i]+6+16];
                        num1 = num1 + 1;
                    }
                    else
                    {
                        ordered_ref16_x[jj].p7 = 0;
                        ordered_ref16_y[jj].p7 = 0;
                        ordered_ref16_z[jj].p7 = 0;
                        original_dataset_index16[jj].p7 = k_reference_set_size;
                    }
                    if (count_voxel_size[i] >= 8+16)
                    {
                        ordered_ref16_x[jj].p8 = ordered_query[voxel_first_index_PL[i]+7+16].x;
                        ordered_ref16_y[jj].p8 = ordered_query[voxel_first_index_PL[i]+7+16].y;
                        ordered_ref16_z[jj].p8 = ordered_query[voxel_first_index_PL[i]+7+16].z;
                        original_dataset_index16[jj].p8 = original_dataset_index[voxel_first_index_PL[i]+7+16];
                        num1 = num1 + 1;
                    }
                    else
                    {
                        ordered_ref16_x[jj].p8 = 0;
                        ordered_ref16_y[jj].p8 = 0;
                        ordered_ref16_z[jj].p8 = 0;
                        original_dataset_index16[jj].p8 = k_reference_set_size;
                    }

                    if (count_voxel_size[i] >= 9+16)
                    {
                        ordered_ref16_x[jj].p9 = ordered_query[voxel_first_index_PL[i]+8+16].x;
                        ordered_ref16_y[jj].p9 = ordered_query[voxel_first_index_PL[i]+8+16].y;
                        ordered_ref16_z[jj].p9 = ordered_query[voxel_first_index_PL[i]+8+16].z;
                        original_dataset_index16[jj].p9 = original_dataset_index[voxel_first_index_PL[i]+8+16];
                        num1 = num1 + 1; 
                    }
                    else
                    {
                        ordered_ref16_x[jj].p9 = 0;
                        ordered_ref16_y[jj].p9 = 0;
                        ordered_ref16_z[jj].p9 = 0;
                        original_dataset_index16[jj].p9 = k_reference_set_size;
                    }
                    if (count_voxel_size[i] >= 10+16)
                    {
                        ordered_ref16_x[jj].p10 = ordered_query[voxel_first_index_PL[i]+9+16].x;
                        ordered_ref16_y[jj].p10 = ordered_query[voxel_first_index_PL[i]+9+16].y;
                        ordered_ref16_z[jj].p10 = ordered_query[voxel_first_index_PL[i]+9+16].z;
                        original_dataset_index16[jj].p10 = original_dataset_index[voxel_first_index_PL[i]+9+16];
                        num1 = num1 + 1;
                    }
                    else
                    {
                        ordered_ref16_x[jj].p10 = 0;
                        ordered_ref16_y[jj].p10 = 0;
                        ordered_ref16_z[jj].p10 = 0;
                        original_dataset_index16[jj].p10 = k_reference_set_size;
                    }
                    if (count_voxel_size[i] >= 11+16)
                    {
                        ordered_ref16_x[jj].p11 = ordered_query[voxel_first_index_PL[i]+10+16].x;
                        ordered_ref16_y[jj].p11 = ordered_query[voxel_first_index_PL[i]+10+16].y;
                        ordered_ref16_z[jj].p11 = ordered_query[voxel_first_index_PL[i]+10+16].z;
                        original_dataset_index16[jj].p11 = original_dataset_index[voxel_first_index_PL[i]+10+16];
                        num1 = num1 + 1;
                    }
                    else
                    {
                        ordered_ref16_x[jj].p11 = 0;
                        ordered_ref16_y[jj].p11 = 0;
                        ordered_ref16_z[jj].p11 = 0;
                        original_dataset_index16[jj].p11 = k_reference_set_size;
                    }
                    if (count_voxel_size[i] >= 12+16)
                    {
                        ordered_ref16_x[jj].p12 = ordered_query[voxel_first_index_PL[i]+11+16].x;
                        ordered_ref16_y[jj].p12 = ordered_query[voxel_first_index_PL[i]+11+16].y;
                        ordered_ref16_z[jj].p12 = ordered_query[voxel_first_index_PL[i]+11+16].z;
                        original_dataset_index16[jj].p12 = original_dataset_index[voxel_first_index_PL[i]+11+16];
                        num1 = num1 + 1;  
                    }
                    else
                    {
                        ordered_ref16_x[jj].p12 = 0;
                        ordered_ref16_y[jj].p12 = 0;
                        ordered_ref16_z[jj].p12 = 0;
                        original_dataset_index16[jj].p12 = k_reference_set_size;
                    }

                    if (count_voxel_size[i] >= 13+16)
                    {
                        ordered_ref16_x[jj].p13 = ordered_query[voxel_first_index_PL[i]+12+16].x;
                        ordered_ref16_y[jj].p13 = ordered_query[voxel_first_index_PL[i]+12+16].y;
                        ordered_ref16_z[jj].p13 = ordered_query[voxel_first_index_PL[i]+12+16].z;
                        original_dataset_index16[jj].p13 = original_dataset_index[voxel_first_index_PL[i]+12+16];
                        num1 = num1 + 1;
                    }
                    else
                    {
                        ordered_ref16_x[jj].p13 = 0;
                        ordered_ref16_y[jj].p13 = 0;
                        ordered_ref16_z[jj].p13 = 0;
                        original_dataset_index16[jj].p13 = k_reference_set_size;
                    }
                    if (count_voxel_size[i] >= 14+16)
                    {
                        ordered_ref16_x[jj].p14 = ordered_query[voxel_first_index_PL[i]+13+16].x;
                        ordered_ref16_y[jj].p14 = ordered_query[voxel_first_index_PL[i]+13+16].y;
                        ordered_ref16_z[jj].p14 = ordered_query[voxel_first_index_PL[i]+13+16].z;
                        original_dataset_index16[jj].p14 = original_dataset_index[voxel_first_index_PL[i]+13+16];
                        num1 = num1 + 1;
                    }
                    else
                    {
                        ordered_ref16_x[jj].p14 = 0;
                        ordered_ref16_y[jj].p14 = 0;
                        ordered_ref16_z[jj].p14 = 0;
                        original_dataset_index16[jj].p14 = k_reference_set_size;
                    }
                    if (count_voxel_size[i] >= 15+16)
                    {
                        ordered_ref16_x[jj].p15 = ordered_query[voxel_first_index_PL[i]+14+16].x;
                        ordered_ref16_y[jj].p15 = ordered_query[voxel_first_index_PL[i]+14+16].y;
                        ordered_ref16_z[jj].p15 = ordered_query[voxel_first_index_PL[i]+14+16].z;
                        original_dataset_index16[jj].p15 = original_dataset_index[voxel_first_index_PL[i]+14+16];
                        num1 = num1 + 1;
                    }
                    else
                    {
                        ordered_ref16_x[jj].p15 = 0;
                        ordered_ref16_y[jj].p15 = 0;
                        ordered_ref16_z[jj].p15 = 0;
                        original_dataset_index16[jj].p15 = k_reference_set_size;
                    }
                    if (count_voxel_size[i] >= 16+16)
                    {
                        ordered_ref16_x[jj].p16 = ordered_query[voxel_first_index_PL[i]+15+16].x;
                        ordered_ref16_y[jj].p16 = ordered_query[voxel_first_index_PL[i]+15+16].y;
                        ordered_ref16_z[jj].p16 = ordered_query[voxel_first_index_PL[i]+15+16].z;
                        original_dataset_index16[jj].p16 = original_dataset_index[voxel_first_index_PL[i]+15+16];
                        num1 = num1 + 1; 
                    }
                    else
                    {
                        ordered_ref16_x[jj].p16 = 0;
                        ordered_ref16_y[jj].p16 = 0;
                        ordered_ref16_z[jj].p16 = 0;
                        original_dataset_index16[jj].p16 = k_reference_set_size;
                    }

                    jj = jj + 1;
                    c = 2;

                }
                
                if (count_voxel_size[i] > 32)
                {
                    
                    if (count_voxel_size[i] >= 1+16+16)
                    {
                        ordered_ref16_x[jj].p1 = ordered_query[voxel_first_index_PL[i]+16+16].x;
                        ordered_ref16_y[jj].p1 = ordered_query[voxel_first_index_PL[i]+16+16].y;
                        ordered_ref16_z[jj].p1 = ordered_query[voxel_first_index_PL[i]+16+16].z;
                        original_dataset_index16[jj].p1 = original_dataset_index[voxel_first_index_PL[i]+16+16];
                        num1 = num1 + 1;
                    }
                    else
                    {
                        ordered_ref16_x[jj].p1 = 0;
                        ordered_ref16_y[jj].p1 = 0;
                        ordered_ref16_z[jj].p1 = 0;
                        original_dataset_index16[jj].p1 = k_reference_set_size;
                    }
                    if (count_voxel_size[i] >= 2+16+16)
                    {
                        ordered_ref16_x[jj].p2 = ordered_query[voxel_first_index_PL[i]+1+16+16].x;
                        ordered_ref16_y[jj].p2 = ordered_query[voxel_first_index_PL[i]+1+16+16].y;
                        ordered_ref16_z[jj].p2 = ordered_query[voxel_first_index_PL[i]+1+16+16].z;
                        original_dataset_index16[jj].p2 = original_dataset_index[voxel_first_index_PL[i]+1+16+16];
                        num1 = num1 + 1;
                    }
                    else
                    {
                        ordered_ref16_x[jj].p2 = 0;
                        ordered_ref16_y[jj].p2 = 0;
                        ordered_ref16_z[jj].p2 = 0;
                        original_dataset_index16[jj].p2 = k_reference_set_size;
                    }
                    if (count_voxel_size[i] >= 3+16+16)
                    {
                        ordered_ref16_x[jj].p3 = ordered_query[voxel_first_index_PL[i]+2+16+16].x;
                        ordered_ref16_y[jj].p3 = ordered_query[voxel_first_index_PL[i]+2+16+16].y;
                        ordered_ref16_z[jj].p3 = ordered_query[voxel_first_index_PL[i]+2+16+16].z;
                        original_dataset_index16[jj].p3 = original_dataset_index[voxel_first_index_PL[i]+2+16+16];
                        num1 = num1 + 1;
                    }
                    else
                    {
                        ordered_ref16_x[jj].p3 = 0;
                        ordered_ref16_y[jj].p3 = 0;
                        ordered_ref16_z[jj].p3 = 0;
                        original_dataset_index16[jj].p3 = k_reference_set_size;
                    }
                    if (count_voxel_size[i] >= 4+16+16)
                    {
                        ordered_ref16_x[jj].p4 = ordered_query[voxel_first_index_PL[i]+3+16+16].x;
                        ordered_ref16_y[jj].p4 = ordered_query[voxel_first_index_PL[i]+3+16+16].y;
                        ordered_ref16_z[jj].p4 = ordered_query[voxel_first_index_PL[i]+3+16+16].z;
                        original_dataset_index16[jj].p4 = original_dataset_index[voxel_first_index_PL[i]+3+16+16];
                        num1 = num1 + 1;
                    }
                    else
                    {
                        ordered_ref16_x[jj].p4 = 0;
                        ordered_ref16_y[jj].p4 = 0;
                        ordered_ref16_z[jj].p4 = 0;
                        original_dataset_index16[jj].p4 = k_reference_set_size;
                    }

                    if (count_voxel_size[i] >= 5+16+16)
                    {
                        ordered_ref16_x[jj].p5 = ordered_query[voxel_first_index_PL[i]+4+16+16].x;
                        ordered_ref16_y[jj].p5 = ordered_query[voxel_first_index_PL[i]+4+16+16].y;
                        ordered_ref16_z[jj].p5 = ordered_query[voxel_first_index_PL[i]+4+16+16].z;
                        original_dataset_index16[jj].p5 = original_dataset_index[voxel_first_index_PL[i]+4+16+16];
                        num1 = num1 + 1;
                    }
                    else
                    {
                        ordered_ref16_x[jj].p5 = 0;
                        ordered_ref16_y[jj].p5 = 0;
                        ordered_ref16_z[jj].p5 = 0;
                        original_dataset_index16[jj].p5 = k_reference_set_size;
                    }
                    if (count_voxel_size[i] >= 6+16+16)
                    {
                        ordered_ref16_x[jj].p6 = ordered_query[voxel_first_index_PL[i]+5+16+16].x;
                        ordered_ref16_y[jj].p6 = ordered_query[voxel_first_index_PL[i]+5+16+16].y;
                        ordered_ref16_z[jj].p6 = ordered_query[voxel_first_index_PL[i]+5+16+16].z;
                        original_dataset_index16[jj].p6 = original_dataset_index[voxel_first_index_PL[i]+5+16+16];
                        num1 = num1 + 1; 
                    }
                    else
                    {
                        ordered_ref16_x[jj].p6 = 0;
                        ordered_ref16_y[jj].p6 = 0;
                        ordered_ref16_z[jj].p6 = 0;
                        original_dataset_index16[jj].p6 = k_reference_set_size;
                    }
                    if (count_voxel_size[i] >= 7+16+16)
                    {
                        ordered_ref16_x[jj].p7 = ordered_query[voxel_first_index_PL[i]+6+16+16].x;
                        ordered_ref16_y[jj].p7 = ordered_query[voxel_first_index_PL[i]+6+16+16].y;
                        ordered_ref16_z[jj].p7 = ordered_query[voxel_first_index_PL[i]+6+16+16].z;
                        original_dataset_index16[jj].p7 = original_dataset_index[voxel_first_index_PL[i]+6+16+16];
                        num1 = num1 + 1;
                    }
                    else
                    {
                        ordered_ref16_x[jj].p7 = 0;
                        ordered_ref16_y[jj].p7 = 0;
                        ordered_ref16_z[jj].p7 = 0;
                        original_dataset_index16[jj].p7 = k_reference_set_size;
                    }
                    if (count_voxel_size[i] >= 8+16+16)
                    {
                        ordered_ref16_x[jj].p8 = ordered_query[voxel_first_index_PL[i]+7+16+16].x;
                        ordered_ref16_y[jj].p8 = ordered_query[voxel_first_index_PL[i]+7+16+16].y;
                        ordered_ref16_z[jj].p8 = ordered_query[voxel_first_index_PL[i]+7+16+16].z;
                        original_dataset_index16[jj].p8 = original_dataset_index[voxel_first_index_PL[i]+7+16+16];
                        num1 = num1 + 1;
                    }
                    else
                    {
                        ordered_ref16_x[jj].p8 = 0;
                        ordered_ref16_y[jj].p8 = 0;
                        ordered_ref16_z[jj].p8 = 0;
                        original_dataset_index16[jj].p8 = k_reference_set_size;
                    }

                    if (count_voxel_size[i] >= 9+16+16)
                    {
                        ordered_ref16_x[jj].p9 = ordered_query[voxel_first_index_PL[i]+8+16+16].x;
                        ordered_ref16_y[jj].p9 = ordered_query[voxel_first_index_PL[i]+8+16+16].y;
                        ordered_ref16_z[jj].p9 = ordered_query[voxel_first_index_PL[i]+8+16+16].z;
                        original_dataset_index16[jj].p9 = original_dataset_index[voxel_first_index_PL[i]+8+16+16];
                        num1 = num1 + 1;
                    }
                    else
                    {
                        ordered_ref16_x[jj].p9 = 0;
                        ordered_ref16_y[jj].p9 = 0;
                        ordered_ref16_z[jj].p9 = 0;
                        original_dataset_index16[jj].p9 = k_reference_set_size;
                    }
                    if (count_voxel_size[i] >= 10+16+16)
                    {
                        ordered_ref16_x[jj].p10 = ordered_query[voxel_first_index_PL[i]+9+16+16].x;
                        ordered_ref16_y[jj].p10 = ordered_query[voxel_first_index_PL[i]+9+16+16].y;
                        ordered_ref16_z[jj].p10 = ordered_query[voxel_first_index_PL[i]+9+16+16].z;
                        original_dataset_index16[jj].p10 = original_dataset_index[voxel_first_index_PL[i]+9+16+16];num1 = num1 + 1; 
                    }
                    else
                    {
                        ordered_ref16_x[jj].p10 = 0;
                        ordered_ref16_y[jj].p10 = 0;
                        ordered_ref16_z[jj].p10 = 0;
                        original_dataset_index16[jj].p10 = k_reference_set_size;
                    }
                    if (count_voxel_size[i] >= 11+16+16)
                    {
                        ordered_ref16_x[jj].p11 = ordered_query[voxel_first_index_PL[i]+10+16+16].x;
                        ordered_ref16_y[jj].p11 = ordered_query[voxel_first_index_PL[i]+10+16+16].y;
                        ordered_ref16_z[jj].p11 = ordered_query[voxel_first_index_PL[i]+10+16+16].z;
                        original_dataset_index16[jj].p11 = original_dataset_index[voxel_first_index_PL[i]+10+16+16];
                        num1 = num1 + 1;
                    }
                    else
                    {
                        ordered_ref16_x[jj].p11 = 0;
                        ordered_ref16_y[jj].p11 = 0;
                        ordered_ref16_z[jj].p11 = 0;
                        original_dataset_index16[jj].p11 = k_reference_set_size;
                    }
                    if (count_voxel_size[i] >= 12+16+16)
                    {
                        ordered_ref16_x[jj].p12 = ordered_query[voxel_first_index_PL[i]+11+16+16].x;
                        ordered_ref16_y[jj].p12 = ordered_query[voxel_first_index_PL[i]+11+16+16].y;
                        ordered_ref16_z[jj].p12 = ordered_query[voxel_first_index_PL[i]+11+16+16].z;
                        original_dataset_index16[jj].p12 = original_dataset_index[voxel_first_index_PL[i]+11+16+16];
                        num1 = num1 + 1; 
                    }
                    else
                    {
                        ordered_ref16_x[jj].p12 = 0;
                        ordered_ref16_y[jj].p12 = 0;
                        ordered_ref16_z[jj].p12 = 0;
                        original_dataset_index16[jj].p12 = k_reference_set_size;
                    }

                    if (count_voxel_size[i] >= 13+16+16)
                    {
                        ordered_ref16_x[jj].p13 = ordered_query[voxel_first_index_PL[i]+12+16+16].x;
                        ordered_ref16_y[jj].p13 = ordered_query[voxel_first_index_PL[i]+12+16+16].y;
                        ordered_ref16_z[jj].p13 = ordered_query[voxel_first_index_PL[i]+12+16+16].z;
                        original_dataset_index16[jj].p13 = original_dataset_index[voxel_first_index_PL[i]+12+16+16];
                        num1 = num1 + 1;
                    }
                    else
                    {
                        ordered_ref16_x[jj].p13 = 0;
                        ordered_ref16_y[jj].p13 = 0;
                        ordered_ref16_z[jj].p13 = 0;
                        original_dataset_index16[jj].p13 = k_reference_set_size;
                    }
                    if (count_voxel_size[i] >= 14+16+16)
                    {
                        ordered_ref16_x[jj].p14 = ordered_query[voxel_first_index_PL[i]+13+16+16].x;
                        ordered_ref16_y[jj].p14 = ordered_query[voxel_first_index_PL[i]+13+16+16].y;
                        ordered_ref16_z[jj].p14 = ordered_query[voxel_first_index_PL[i]+13+16+16].z;
                        original_dataset_index16[jj].p14 = original_dataset_index[voxel_first_index_PL[i]+13+16+16];
                        num1 = num1 + 1;
                    }
                    else
                    {
                        ordered_ref16_x[jj].p14 = 0;
                        ordered_ref16_y[jj].p14 = 0;
                        ordered_ref16_z[jj].p14 = 0;
                        original_dataset_index16[jj].p14 = k_reference_set_size;
                    }
                    if (count_voxel_size[i] >= 15+16+16)
                    {
                        ordered_ref16_x[jj].p15 = ordered_query[voxel_first_index_PL[i]+14+16+16].x;
                        ordered_ref16_y[jj].p15 = ordered_query[voxel_first_index_PL[i]+14+16+16].y;
                        ordered_ref16_z[jj].p15 = ordered_query[voxel_first_index_PL[i]+14+16+16].z;
                        original_dataset_index16[jj].p15 = original_dataset_index[voxel_first_index_PL[i]+14+16+16];
                        num1 = num1 + 1;
                    }
                    else
                    {
                        ordered_ref16_x[jj].p15 = 0;
                        ordered_ref16_y[jj].p15 = 0;
                        ordered_ref16_z[jj].p15 = 0;
                        original_dataset_index16[jj].p15 = k_reference_set_size;
                    }
                    if (count_voxel_size[i] >= 16+16+16)
                    {
                        ordered_ref16_x[jj].p16 = ordered_query[voxel_first_index_PL[i]+15+16+16].x;
                        ordered_ref16_y[jj].p16 = ordered_query[voxel_first_index_PL[i]+15+16+16].y;
                        ordered_ref16_z[jj].p16 = ordered_query[voxel_first_index_PL[i]+15+16+16].z;
                        original_dataset_index16[jj].p16 = original_dataset_index[voxel_first_index_PL[i]+15+16+16];
                        num1 = num1 + 1;
                    }
                    else
                    {
                        ordered_ref16_x[jj].p16 = 0;
                        ordered_ref16_y[jj].p16 = 0;
                        ordered_ref16_z[jj].p16 = 0;
                        original_dataset_index16[jj].p16 = k_reference_set_size;
                    }

                    jj = jj + 1;
                    c = 3;
                }
                if (c == 1)
                {
                    index16[i] = jj-1;
                    index16[i](31,29) = 1;
                }
                else if (c == 2)
                {
                    index16[i] = jj-2;
                    index16[i](31,29) = 2;
                }
                else if (c == 3)
                {
                    index16[i] = jj-3;
                    index16[i](31,29) = 3;
                }
                else
                {
                    index16[i] = jj;
                }
            }



            else
            {
                for (int ij = 0; ij < k_sub_voxel_size; ij++)
                {
                    int c1 = 0;
                    int temp = sub_voxel_flag_index_PL[i] + ij;
                    if (sub_voxel_size[temp] > 0)
                    {
                        
                        if (sub_voxel_size[temp] >= 1)
                        {
                            ordered_ref16_x[jj].p1 = ordered_query[sub_voxel_first_index_PL[temp]].x;
                            ordered_ref16_y[jj].p1 = ordered_query[sub_voxel_first_index_PL[temp]].y;
                            ordered_ref16_z[jj].p1 = ordered_query[sub_voxel_first_index_PL[temp]].z;
                            original_dataset_index16[jj].p1 = original_dataset_index[sub_voxel_first_index_PL[temp]];
                            num1 = num1 + 1;
                        }
                        else
                        {
                            ordered_ref16_x[jj].p1 = 0;
                            ordered_ref16_y[jj].p1 = 0;
                            ordered_ref16_z[jj].p1 = 0;
                            original_dataset_index16[jj].p1 = k_reference_set_size;
                        }
                        if (sub_voxel_size[temp] >= 2)
                        {
                            ordered_ref16_x[jj].p2 = ordered_query[sub_voxel_first_index_PL[temp]+1].x;
                            ordered_ref16_y[jj].p2 = ordered_query[sub_voxel_first_index_PL[temp]+1].y;
                            ordered_ref16_z[jj].p2 = ordered_query[sub_voxel_first_index_PL[temp]+1].z;
                            original_dataset_index16[jj].p2 = original_dataset_index[sub_voxel_first_index_PL[temp]+1];
                            num1 = num1 + 1;
                        }
                        else
                        {
                            ordered_ref16_x[jj].p2 = 0;
                            ordered_ref16_y[jj].p2 = 0;
                            ordered_ref16_z[jj].p2 = 0;
                            original_dataset_index16[jj].p2 = k_reference_set_size;
                        }
                        if (sub_voxel_size[temp] >= 3)
                        {
                            ordered_ref16_x[jj].p3 = ordered_query[sub_voxel_first_index_PL[temp]+2].x;
                            ordered_ref16_y[jj].p3 = ordered_query[sub_voxel_first_index_PL[temp]+2].y;
                            ordered_ref16_z[jj].p3 = ordered_query[sub_voxel_first_index_PL[temp]+2].z;
                            original_dataset_index16[jj].p3 = original_dataset_index[sub_voxel_first_index_PL[temp]+2];
                            num1 = num1 + 1;
                        }
                        else
                        {
                            ordered_ref16_x[jj].p3 = 0;
                            ordered_ref16_y[jj].p3 = 0;
                            ordered_ref16_z[jj].p3 = 0;
                            original_dataset_index16[jj].p3 = k_reference_set_size;
                        }
                        if (sub_voxel_size[temp] >= 4)
                        {
                            ordered_ref16_x[jj].p4 = ordered_query[sub_voxel_first_index_PL[temp]+3].x;
                            ordered_ref16_y[jj].p4 = ordered_query[sub_voxel_first_index_PL[temp]+3].y;
                            ordered_ref16_z[jj].p4 = ordered_query[sub_voxel_first_index_PL[temp]+3].z;
                            original_dataset_index16[jj].p4 = original_dataset_index[sub_voxel_first_index_PL[temp]+3];
                            num1 = num1 + 1;
                        }
                        else
                        {
                            ordered_ref16_x[jj].p4 = 0;
                            ordered_ref16_y[jj].p4 = 0;
                            ordered_ref16_z[jj].p4 = 0;
                            original_dataset_index16[jj].p4 = k_reference_set_size;
                        }

                        if (sub_voxel_size[temp] >= 5)
                        {
                            ordered_ref16_x[jj].p5 = ordered_query[sub_voxel_first_index_PL[temp]+4].x;
                            ordered_ref16_y[jj].p5 = ordered_query[sub_voxel_first_index_PL[temp]+4].y;
                            ordered_ref16_z[jj].p5 = ordered_query[sub_voxel_first_index_PL[temp]+4].z;
                            original_dataset_index16[jj].p5 = original_dataset_index[sub_voxel_first_index_PL[temp]+4];
                            num1 = num1 + 1;
                        }
                        else
                        {
                            ordered_ref16_x[jj].p5 = 0;
                            ordered_ref16_y[jj].p5 = 0;
                            ordered_ref16_z[jj].p5 = 0;
                            original_dataset_index16[jj].p5 = k_reference_set_size;
                        }
                        if (sub_voxel_size[temp] >= 6)
                        {
                            ordered_ref16_x[jj].p6 = ordered_query[sub_voxel_first_index_PL[temp]+5].x;
                            ordered_ref16_y[jj].p6 = ordered_query[sub_voxel_first_index_PL[temp]+5].y;
                            ordered_ref16_z[jj].p6 = ordered_query[sub_voxel_first_index_PL[temp]+5].z;
                            original_dataset_index16[jj].p6 = original_dataset_index[sub_voxel_first_index_PL[temp]+5];
                            num1 = num1 + 1;   
                        }
                        else
                        {
                            ordered_ref16_x[jj].p6 = 0;
                            ordered_ref16_y[jj].p6 = 0;
                            ordered_ref16_z[jj].p6 = 0;
                            original_dataset_index16[jj].p6 = k_reference_set_size;
                        }
                        if (sub_voxel_size[temp] >= 7)
                        {
                            ordered_ref16_x[jj].p7 = ordered_query[sub_voxel_first_index_PL[temp]+6].x;
                            ordered_ref16_y[jj].p7 = ordered_query[sub_voxel_first_index_PL[temp]+6].y;
                            ordered_ref16_z[jj].p7 = ordered_query[sub_voxel_first_index_PL[temp]+6].z;
                            original_dataset_index16[jj].p7 = original_dataset_index[sub_voxel_first_index_PL[temp]+6];
                            num1 = num1 + 1; 
                        }
                        else
                        {
                            ordered_ref16_x[jj].p7 = 0;
                            ordered_ref16_y[jj].p7 = 0;
                            ordered_ref16_z[jj].p7 = 0;
                            original_dataset_index16[jj].p7 = k_reference_set_size;
                        }
                        if (sub_voxel_size[temp] >= 8)
                        {
                            ordered_ref16_x[jj].p8 = ordered_query[sub_voxel_first_index_PL[temp]+7].x;
                            ordered_ref16_y[jj].p8 = ordered_query[sub_voxel_first_index_PL[temp]+7].y;
                            ordered_ref16_z[jj].p8 = ordered_query[sub_voxel_first_index_PL[temp]+7].z;
                            original_dataset_index16[jj].p8 = original_dataset_index[sub_voxel_first_index_PL[temp]+7];
                            num1 = num1 + 1; 
                        }
                        else
                        {
                            ordered_ref16_x[jj].p8 = 0;
                            ordered_ref16_y[jj].p8 = 0;
                            ordered_ref16_z[jj].p8 = 0;
                            original_dataset_index16[jj].p8 = k_reference_set_size;
                        }

                        if (sub_voxel_size[temp] >= 9)
                        {
                            ordered_ref16_x[jj].p9 = ordered_query[sub_voxel_first_index_PL[temp]+8].x;
                            ordered_ref16_y[jj].p9 = ordered_query[sub_voxel_first_index_PL[temp]+8].y;
                            ordered_ref16_z[jj].p9 = ordered_query[sub_voxel_first_index_PL[temp]+8].z;
                            original_dataset_index16[jj].p9 = original_dataset_index[sub_voxel_first_index_PL[temp]+8];
                            num1 = num1 + 1; 
                        }
                        else
                        {
                            ordered_ref16_x[jj].p9 = 0;
                            ordered_ref16_y[jj].p9 = 0;
                            ordered_ref16_z[jj].p9 = 0;
                            original_dataset_index16[jj].p9 = k_reference_set_size;
                        }
                        if (sub_voxel_size[temp] >= 10)
                        {
                            ordered_ref16_x[jj].p10 = ordered_query[sub_voxel_first_index_PL[temp]+9].x;
                            ordered_ref16_y[jj].p10 = ordered_query[sub_voxel_first_index_PL[temp]+9].y;
                            ordered_ref16_z[jj].p10 = ordered_query[sub_voxel_first_index_PL[temp]+9].z;
                            original_dataset_index16[jj].p10 = original_dataset_index[sub_voxel_first_index_PL[temp]+9];
                            num1 = num1 + 1;  
                        }
                        else
                        {
                            ordered_ref16_x[jj].p10 = 0;
                            ordered_ref16_y[jj].p10 = 0;
                            ordered_ref16_z[jj].p10 = 0;
                            original_dataset_index16[jj].p10 = k_reference_set_size;
                        }
                        if (sub_voxel_size[temp] >= 11)
                        {
                            ordered_ref16_x[jj].p11 = ordered_query[sub_voxel_first_index_PL[temp]+10].x;
                            ordered_ref16_y[jj].p11 = ordered_query[sub_voxel_first_index_PL[temp]+10].y;
                            ordered_ref16_z[jj].p11 = ordered_query[sub_voxel_first_index_PL[temp]+10].z;
                            original_dataset_index16[jj].p11 = original_dataset_index[sub_voxel_first_index_PL[temp]+10];
                            num1 = num1 + 1;
                        }
                        else
                        {
                            ordered_ref16_x[jj].p11 = 0;
                            ordered_ref16_y[jj].p11 = 0;
                            ordered_ref16_z[jj].p11 = 0;
                            original_dataset_index16[jj].p11 = k_reference_set_size;
                        }
                        if (sub_voxel_size[temp] >= 12)
                        {
                            ordered_ref16_x[jj].p12 = ordered_query[sub_voxel_first_index_PL[temp]+11].x;
                            ordered_ref16_y[jj].p12 = ordered_query[sub_voxel_first_index_PL[temp]+11].y;
                            ordered_ref16_z[jj].p12 = ordered_query[sub_voxel_first_index_PL[temp]+11].z;
                            original_dataset_index16[jj].p12 = original_dataset_index[sub_voxel_first_index_PL[temp]+11];
                            num1 = num1 + 1; 
                        }
                        else
                        {
                            ordered_ref16_x[jj].p12 = 0;
                            ordered_ref16_y[jj].p12 = 0;
                            ordered_ref16_z[jj].p12 = 0;
                            original_dataset_index16[jj].p12 = k_reference_set_size;
                        }

                        if (sub_voxel_size[temp] >= 13)
                        {
                            ordered_ref16_x[jj].p13 = ordered_query[sub_voxel_first_index_PL[temp]+12].x;
                            ordered_ref16_y[jj].p13 = ordered_query[sub_voxel_first_index_PL[temp]+12].y;
                            ordered_ref16_z[jj].p13 = ordered_query[sub_voxel_first_index_PL[temp]+12].z;
                            original_dataset_index16[jj].p13 = original_dataset_index[sub_voxel_first_index_PL[temp]+12];
                            num1 = num1 + 1;
                        }
                        else
                        {
                            ordered_ref16_x[jj].p13 = 0;
                            ordered_ref16_y[jj].p13 = 0;
                            ordered_ref16_z[jj].p13 = 0;
                            original_dataset_index16[jj].p13 = k_reference_set_size;
                        }
                        if (sub_voxel_size[temp] >= 14)
                        {
                            ordered_ref16_x[jj].p14 = ordered_query[sub_voxel_first_index_PL[temp]+13].x;
                            ordered_ref16_y[jj].p14 = ordered_query[sub_voxel_first_index_PL[temp]+13].y;
                            ordered_ref16_z[jj].p14 = ordered_query[sub_voxel_first_index_PL[temp]+13].z;
                            original_dataset_index16[jj].p14 = original_dataset_index[sub_voxel_first_index_PL[temp]+13];
                            num1 = num1 + 1;
                        }
                        else
                        {
                            ordered_ref16_x[jj].p14 = 0;
                            ordered_ref16_y[jj].p14 = 0;
                            ordered_ref16_z[jj].p14 = 0;
                            original_dataset_index16[jj].p14 = k_reference_set_size;
                        }
                        if (sub_voxel_size[temp] >= 15)
                        {
                            ordered_ref16_x[jj].p15 = ordered_query[sub_voxel_first_index_PL[temp]+14].x;
                            ordered_ref16_y[jj].p15 = ordered_query[sub_voxel_first_index_PL[temp]+14].y;
                            ordered_ref16_z[jj].p15 = ordered_query[sub_voxel_first_index_PL[temp]+14].z;
                            original_dataset_index16[jj].p15 = original_dataset_index[sub_voxel_first_index_PL[temp]+14];
                            num1 = num1 + 1;
                        }
                        else
                        {
                            ordered_ref16_x[jj].p15 = 0;
                            ordered_ref16_y[jj].p15 = 0;
                            ordered_ref16_z[jj].p15 = 0;
                            original_dataset_index16[jj].p15 = k_reference_set_size;
                        }
                        if (sub_voxel_size[temp] >= 16)
                        {
                            ordered_ref16_x[jj].p16 = ordered_query[sub_voxel_first_index_PL[temp]+15].x;
                            ordered_ref16_y[jj].p16 = ordered_query[sub_voxel_first_index_PL[temp]+15].y;
                            ordered_ref16_z[jj].p16 = ordered_query[sub_voxel_first_index_PL[temp]+15].z;
                            original_dataset_index16[jj].p16 = original_dataset_index[sub_voxel_first_index_PL[temp]+15];
                            num1 = num1 + 1;
                        }
                        else
                        {
                            ordered_ref16_x[jj].p16 = 0;
                            ordered_ref16_y[jj].p16 = 0;
                            ordered_ref16_z[jj].p16 = 0;
                            original_dataset_index16[jj].p16 = k_reference_set_size;
                        }
                        
                        jj = jj + 1;
                        c1 = 1;

                    }

                    if (sub_voxel_size[temp] > 16)
                    {
                        
                        if (sub_voxel_size[temp] >= 1+16)
                        {
                            ordered_ref16_x[jj].p1 = ordered_query[sub_voxel_first_index_PL[temp]+16].x;
                            ordered_ref16_y[jj].p1 = ordered_query[sub_voxel_first_index_PL[temp]+16].y;
                            ordered_ref16_z[jj].p1 = ordered_query[sub_voxel_first_index_PL[temp]+16].z;
                            original_dataset_index16[jj].p1 = original_dataset_index[sub_voxel_first_index_PL[temp]+16];
                            num1 = num1 + 1;
                        }
                        else
                        {
                            ordered_ref16_x[jj].p1 = 0;
                            ordered_ref16_y[jj].p1 = 0;
                            ordered_ref16_z[jj].p1 = 0;
                            original_dataset_index16[jj].p1 = k_reference_set_size;
                        }
                        if (sub_voxel_size[temp] >= 2+16)
                        {
                            ordered_ref16_x[jj].p2 = ordered_query[sub_voxel_first_index_PL[temp]+1+16].x;
                            ordered_ref16_y[jj].p2 = ordered_query[sub_voxel_first_index_PL[temp]+1+16].y;
                            ordered_ref16_z[jj].p2 = ordered_query[sub_voxel_first_index_PL[temp]+1+16].z;
                            original_dataset_index16[jj].p2 = original_dataset_index[sub_voxel_first_index_PL[temp]+1+16];
                            num1 = num1 + 1;
                        }
                        else
                        {
                            ordered_ref16_x[jj].p2 = 0;
                            ordered_ref16_y[jj].p2 = 0;
                            ordered_ref16_z[jj].p2 = 0;
                            original_dataset_index16[jj].p2 = k_reference_set_size;
                        }
                        if (sub_voxel_size[temp] >= 3+16)
                        {
                            ordered_ref16_x[jj].p3 = ordered_query[sub_voxel_first_index_PL[temp]+2+16].x;
                            ordered_ref16_y[jj].p3 = ordered_query[sub_voxel_first_index_PL[temp]+2+16].y;
                            ordered_ref16_z[jj].p3 = ordered_query[sub_voxel_first_index_PL[temp]+2+16].z;
                            original_dataset_index16[jj].p3 = original_dataset_index[sub_voxel_first_index_PL[temp]+2+16];
                            num1 = num1 + 1; 
                        }
                        else
                        {
                            ordered_ref16_x[jj].p3 = 0;
                            ordered_ref16_y[jj].p3 = 0;
                            ordered_ref16_z[jj].p3 = 0;
                            original_dataset_index16[jj].p3 = k_reference_set_size;
                        }
                        if (sub_voxel_size[temp] >= 4+16)
                        {
                            ordered_ref16_x[jj].p4 = ordered_query[sub_voxel_first_index_PL[temp]+3+16].x;
                            ordered_ref16_y[jj].p4 = ordered_query[sub_voxel_first_index_PL[temp]+3+16].y;
                            ordered_ref16_z[jj].p4 = ordered_query[sub_voxel_first_index_PL[temp]+3+16].z;
                            original_dataset_index16[jj].p4 = original_dataset_index[sub_voxel_first_index_PL[temp]+3+16];
                            num1 = num1 + 1;
                        }
                        else
                        {
                            ordered_ref16_x[jj].p4 = 0;
                            ordered_ref16_y[jj].p4 = 0;
                            ordered_ref16_z[jj].p4 = 0;
                            original_dataset_index16[jj].p4 = k_reference_set_size;
                        }

                        if (sub_voxel_size[temp] >= 5+16)
                        {
                            ordered_ref16_x[jj].p5 = ordered_query[sub_voxel_first_index_PL[temp]+4+16].x;
                            ordered_ref16_y[jj].p5 = ordered_query[sub_voxel_first_index_PL[temp]+4+16].y;
                            ordered_ref16_z[jj].p5 = ordered_query[sub_voxel_first_index_PL[temp]+4+16].z;
                            original_dataset_index16[jj].p5 = original_dataset_index[sub_voxel_first_index_PL[temp]+4+16];
                            num1 = num1 + 1;
                        }
                        else
                        {
                            ordered_ref16_x[jj].p5 = 0;
                            ordered_ref16_y[jj].p5 = 0;
                            ordered_ref16_z[jj].p5 = 0;
                            original_dataset_index16[jj].p5 = k_reference_set_size;
                        }
                        if (sub_voxel_size[temp] >= 6+16)
                        {
                            ordered_ref16_x[jj].p6 = ordered_query[sub_voxel_first_index_PL[temp]+5+16].x;
                            ordered_ref16_y[jj].p6 = ordered_query[sub_voxel_first_index_PL[temp]+5+16].y;
                            ordered_ref16_z[jj].p6 = ordered_query[sub_voxel_first_index_PL[temp]+5+16].z;
                            original_dataset_index16[jj].p6 = original_dataset_index[sub_voxel_first_index_PL[temp]+5+16];
                            num1 = num1 + 1;  
                        }
                        else
                        {
                            ordered_ref16_x[jj].p6 = 0;
                            ordered_ref16_y[jj].p6 = 0;
                            ordered_ref16_z[jj].p6 = 0;
                            original_dataset_index16[jj].p6 = k_reference_set_size;
                        }
                        if (sub_voxel_size[temp] >= 7+16)
                        {
                            ordered_ref16_x[jj].p7 = ordered_query[sub_voxel_first_index_PL[temp]+6+16].x;
                            ordered_ref16_y[jj].p7 = ordered_query[sub_voxel_first_index_PL[temp]+6+16].y;
                            ordered_ref16_z[jj].p7 = ordered_query[sub_voxel_first_index_PL[temp]+6+16].z;
                            original_dataset_index16[jj].p7 = original_dataset_index[sub_voxel_first_index_PL[temp]+6+16];
                            num1 = num1 + 1; 
                        }
                        else
                        {
                            ordered_ref16_x[jj].p7 = 0;
                            ordered_ref16_y[jj].p7 = 0;
                            ordered_ref16_z[jj].p7 = 0;
                            original_dataset_index16[jj].p7 = k_reference_set_size;
                        }
                        if (sub_voxel_size[temp] >= 8+16)
                        {
                            ordered_ref16_x[jj].p8 = ordered_query[sub_voxel_first_index_PL[temp]+7+16].x;
                            ordered_ref16_y[jj].p8 = ordered_query[sub_voxel_first_index_PL[temp]+7+16].y;
                            ordered_ref16_z[jj].p8 = ordered_query[sub_voxel_first_index_PL[temp]+7+16].z;
                            original_dataset_index16[jj].p8 = original_dataset_index[sub_voxel_first_index_PL[temp]+7+16];
                            num1 = num1 + 1;
                        }
                        else
                        {
                            ordered_ref16_x[jj].p8 = 0;
                            ordered_ref16_y[jj].p8 = 0;
                            ordered_ref16_z[jj].p8 = 0;
                            original_dataset_index16[jj].p8 = k_reference_set_size;
                        }

                        if (sub_voxel_size[temp] >= 9+16)
                        {
                            ordered_ref16_x[jj].p9 = ordered_query[sub_voxel_first_index_PL[temp]+8+16].x;
                            ordered_ref16_y[jj].p9 = ordered_query[sub_voxel_first_index_PL[temp]+8+16].y;
                            ordered_ref16_z[jj].p9 = ordered_query[sub_voxel_first_index_PL[temp]+8+16].z;
                            original_dataset_index16[jj].p9 = original_dataset_index[sub_voxel_first_index_PL[temp]+8+16];
                            num1 = num1 + 1; 
                        }
                        else
                        {
                            ordered_ref16_x[jj].p9 = 0;
                            ordered_ref16_y[jj].p9 = 0;
                            ordered_ref16_z[jj].p9 = 0;
                            original_dataset_index16[jj].p9 = k_reference_set_size;
                        }
                        if (sub_voxel_size[temp] >= 10+16)
                        {
                            ordered_ref16_x[jj].p10 = ordered_query[sub_voxel_first_index_PL[temp]+9+16].x;
                            ordered_ref16_y[jj].p10 = ordered_query[sub_voxel_first_index_PL[temp]+9+16].y;
                            ordered_ref16_z[jj].p10 = ordered_query[sub_voxel_first_index_PL[temp]+9+16].z;
                            original_dataset_index16[jj].p10 = original_dataset_index[sub_voxel_first_index_PL[temp]+9+16];
                            num1 = num1 + 1;  
                        }
                        else
                        {
                            ordered_ref16_x[jj].p10 = 0;
                            ordered_ref16_y[jj].p10 = 0;
                            ordered_ref16_z[jj].p10 = 0;
                            original_dataset_index16[jj].p10 = k_reference_set_size;
                        }
                        if (sub_voxel_size[temp] >= 11+16)
                        {
                            ordered_ref16_x[jj].p11 = ordered_query[sub_voxel_first_index_PL[temp]+10+16].x;
                            ordered_ref16_y[jj].p11 = ordered_query[sub_voxel_first_index_PL[temp]+10+16].y;
                            ordered_ref16_z[jj].p11 = ordered_query[sub_voxel_first_index_PL[temp]+10+16].z;
                            original_dataset_index16[jj].p11 = original_dataset_index[sub_voxel_first_index_PL[temp]+10+16];
                            num1 = num1 + 1;
                        }
                        else
                        {
                            ordered_ref16_x[jj].p11 = 0;
                            ordered_ref16_y[jj].p11 = 0;
                            ordered_ref16_z[jj].p11 = 0;
                            original_dataset_index16[jj].p11 = k_reference_set_size;
                        }
                        if (sub_voxel_size[temp] >= 12+16)
                        {
                            ordered_ref16_x[jj].p12 = ordered_query[sub_voxel_first_index_PL[temp]+11+16].x;
                            ordered_ref16_y[jj].p12 = ordered_query[sub_voxel_first_index_PL[temp]+11+16].y;
                            ordered_ref16_z[jj].p12 = ordered_query[sub_voxel_first_index_PL[temp]+11+16].z;
                            original_dataset_index16[jj].p12 = original_dataset_index[sub_voxel_first_index_PL[temp]+11+16];
                            num1 = num1 + 1; 
                        }
                        else
                        {
                            ordered_ref16_x[jj].p12 = 0;
                            ordered_ref16_y[jj].p12 = 0;
                            ordered_ref16_z[jj].p12 = 0;
                            original_dataset_index16[jj].p12 = k_reference_set_size;
                        }

                        if (sub_voxel_size[temp] >= 13+16)
                        {
                            ordered_ref16_x[jj].p13 = ordered_query[sub_voxel_first_index_PL[temp]+12+16].x;
                            ordered_ref16_y[jj].p13 = ordered_query[sub_voxel_first_index_PL[temp]+12+16].y;
                            ordered_ref16_z[jj].p13 = ordered_query[sub_voxel_first_index_PL[temp]+12+16].z;
                            original_dataset_index16[jj].p13 = original_dataset_index[sub_voxel_first_index_PL[temp]+12+16];
                            num1 = num1 + 1;
                        }
                        else
                        {
                            ordered_ref16_x[jj].p13 = 0;
                            ordered_ref16_y[jj].p13 = 0;
                            ordered_ref16_z[jj].p13 = 0;
                            original_dataset_index16[jj].p13 = k_reference_set_size;
                        }
                        if (sub_voxel_size[temp] >= 14+16)
                        {
                            ordered_ref16_x[jj].p14 = ordered_query[sub_voxel_first_index_PL[temp]+13+16].x;
                            ordered_ref16_y[jj].p14 = ordered_query[sub_voxel_first_index_PL[temp]+13+16].y;
                            ordered_ref16_z[jj].p14 = ordered_query[sub_voxel_first_index_PL[temp]+13+16].z;
                            original_dataset_index16[jj].p14 = original_dataset_index[sub_voxel_first_index_PL[temp]+13+16];
                            num1 = num1 + 1;
                        }
                        else
                        {
                            ordered_ref16_x[jj].p14 = 0;
                            ordered_ref16_y[jj].p14 = 0;
                            ordered_ref16_z[jj].p14 = 0;
                            original_dataset_index16[jj].p14 = k_reference_set_size;
                        }
                        if (sub_voxel_size[temp] >= 15+16)
                        {
                            ordered_ref16_x[jj].p15 = ordered_query[sub_voxel_first_index_PL[temp]+14+16].x;
                            ordered_ref16_y[jj].p15 = ordered_query[sub_voxel_first_index_PL[temp]+14+16].y;
                            ordered_ref16_z[jj].p15 = ordered_query[sub_voxel_first_index_PL[temp]+14+16].z;
                            original_dataset_index16[jj].p15 = original_dataset_index[sub_voxel_first_index_PL[temp]+14+16];
                            num1 = num1 + 1;
                        }
                        else
                        {
                            ordered_ref16_x[jj].p15 = 0;
                            ordered_ref16_y[jj].p15 = 0;
                            ordered_ref16_z[jj].p15 = 0;
                            original_dataset_index16[jj].p15 = k_reference_set_size;
                        }
                        if (sub_voxel_size[temp] >= 16+16)
                        {
                            ordered_ref16_x[jj].p16 = ordered_query[sub_voxel_first_index_PL[temp]+15+16].x;
                            ordered_ref16_y[jj].p16 = ordered_query[sub_voxel_first_index_PL[temp]+15+16].y;
                            ordered_ref16_z[jj].p16 = ordered_query[sub_voxel_first_index_PL[temp]+15+16].z;
                            original_dataset_index16[jj].p16 = original_dataset_index[sub_voxel_first_index_PL[temp]+15+16];
                            num1 = num1 + 1;
                        }
                        else
                        {
                            ordered_ref16_x[jj].p16 = 0;
                            ordered_ref16_y[jj].p16 = 0;
                            ordered_ref16_z[jj].p16 = 0;
                            original_dataset_index16[jj].p16 = k_reference_set_size;
                        }

                        jj = jj + 1;
                        c1 = 2;

                    }

                    if (sub_voxel_size[temp] > 32)
                    {

                        if (sub_voxel_size[temp] >= 1+16+16)
                        {
                            ordered_ref16_x[jj].p1 = ordered_query[sub_voxel_first_index_PL[temp]+16+16].x;
                            ordered_ref16_y[jj].p1 = ordered_query[sub_voxel_first_index_PL[temp]+16+16].y;
                            ordered_ref16_z[jj].p1 = ordered_query[sub_voxel_first_index_PL[temp]+16+16].z;
                            original_dataset_index16[jj].p1 = original_dataset_index[sub_voxel_first_index_PL[temp]+16+16];
                            num1 = num1 + 1;
                        }
                        else
                        {
                            ordered_ref16_x[jj].p1 = 0;
                            ordered_ref16_y[jj].p1 = 0;
                            ordered_ref16_z[jj].p1 = 0;
                            original_dataset_index16[jj].p1 = k_reference_set_size;
                        }
                        if (sub_voxel_size[temp] >= 2+16+16)
                        {
                            ordered_ref16_x[jj].p2 = ordered_query[sub_voxel_first_index_PL[temp]+1+16+16].x;
                            ordered_ref16_y[jj].p2 = ordered_query[sub_voxel_first_index_PL[temp]+1+16+16].y;
                            ordered_ref16_z[jj].p2 = ordered_query[sub_voxel_first_index_PL[temp]+1+16+16].z;
                            original_dataset_index16[jj].p2 = original_dataset_index[sub_voxel_first_index_PL[temp]+1+16+16];
                            num1 = num1 + 1;
                        }
                        else
                        {
                            ordered_ref16_x[jj].p2 = 0;
                            ordered_ref16_y[jj].p2 = 0;
                            ordered_ref16_z[jj].p2 = 0;
                            original_dataset_index16[jj].p2 = k_reference_set_size;
                        }
                        if (sub_voxel_size[temp] >= 3+16+16)
                        {
                            ordered_ref16_x[jj].p3 = ordered_query[sub_voxel_first_index_PL[temp]+2+16+16].x;
                            ordered_ref16_y[jj].p3 = ordered_query[sub_voxel_first_index_PL[temp]+2+16+16].y;
                            ordered_ref16_z[jj].p3 = ordered_query[sub_voxel_first_index_PL[temp]+2+16+16].z;
                            original_dataset_index16[jj].p3 = original_dataset_index[sub_voxel_first_index_PL[temp]+2+16+16];
                            num1 = num1 + 1;
                        }
                        else
                        {
                            ordered_ref16_x[jj].p3 = 0;
                            ordered_ref16_y[jj].p3 = 0;
                            ordered_ref16_z[jj].p3 = 0;
                            original_dataset_index16[jj].p3 = k_reference_set_size;
                        }
                        if (sub_voxel_size[temp] >= 4+16+16)
                        {
                            ordered_ref16_x[jj].p4 = ordered_query[sub_voxel_first_index_PL[temp]+3+16+16].x;
                            ordered_ref16_y[jj].p4 = ordered_query[sub_voxel_first_index_PL[temp]+3+16+16].y;
                            ordered_ref16_z[jj].p4 = ordered_query[sub_voxel_first_index_PL[temp]+3+16+16].z;
                            original_dataset_index16[jj].p4 = original_dataset_index[sub_voxel_first_index_PL[temp]+3+16+16];
                            num1 = num1 + 1;
                        }
                        else
                        {
                            ordered_ref16_x[jj].p4 = 0;
                            ordered_ref16_y[jj].p4 = 0;
                            ordered_ref16_z[jj].p4 = 0;
                            original_dataset_index16[jj].p4 = k_reference_set_size;
                        }

                        if (sub_voxel_size[temp] >= 5+16+16)
                        {
                            ordered_ref16_x[jj].p5 = ordered_query[sub_voxel_first_index_PL[temp]+4+16+16].x;
                            ordered_ref16_y[jj].p5 = ordered_query[sub_voxel_first_index_PL[temp]+4+16+16].y;
                            ordered_ref16_z[jj].p5 = ordered_query[sub_voxel_first_index_PL[temp]+4+16+16].z;
                            original_dataset_index16[jj].p5 = original_dataset_index[sub_voxel_first_index_PL[temp]+4+16+16];
                            num1 = num1 + 1;
                        }
                        else
                        {
                            ordered_ref16_x[jj].p5 = 0;
                            ordered_ref16_y[jj].p5 = 0;
                            ordered_ref16_z[jj].p5 = 0;
                            original_dataset_index16[jj].p5 = k_reference_set_size;
                        }
                        if (sub_voxel_size[temp] >= 6+16+16)
                        {
                            ordered_ref16_x[jj].p6 = ordered_query[sub_voxel_first_index_PL[temp]+5+16+16].x;
                            ordered_ref16_y[jj].p6 = ordered_query[sub_voxel_first_index_PL[temp]+5+16+16].y;
                            ordered_ref16_z[jj].p6 = ordered_query[sub_voxel_first_index_PL[temp]+5+16+16].z;
                            original_dataset_index16[jj].p6 = original_dataset_index[sub_voxel_first_index_PL[temp]+5+16+16];
                            num1 = num1 + 1;  
                        }
                        else
                        {
                            ordered_ref16_x[jj].p6 = 0;
                            ordered_ref16_y[jj].p6 = 0;
                            ordered_ref16_z[jj].p6 = 0;
                            original_dataset_index16[jj].p6 = k_reference_set_size;
                        }
                        if (sub_voxel_size[temp] >= 7+16+16)
                        {
                            ordered_ref16_x[jj].p7 = ordered_query[sub_voxel_first_index_PL[temp]+6+16+16].x;
                            ordered_ref16_y[jj].p7 = ordered_query[sub_voxel_first_index_PL[temp]+6+16+16].y;
                            ordered_ref16_z[jj].p7 = ordered_query[sub_voxel_first_index_PL[temp]+6+16+16].z;
                            original_dataset_index16[jj].p7 = original_dataset_index[sub_voxel_first_index_PL[temp]+6+16+16];
                            num1 = num1 + 1;
                        }
                        else
                        {
                            ordered_ref16_x[jj].p7 = 0;
                            ordered_ref16_y[jj].p7 = 0;
                            ordered_ref16_z[jj].p7 = 0;
                            original_dataset_index16[jj].p7 = k_reference_set_size;
                        }
                        if (sub_voxel_size[temp] >= 8+16+16)
                        {
                            ordered_ref16_x[jj].p8 = ordered_query[sub_voxel_first_index_PL[temp]+7+16+16].x;
                            ordered_ref16_y[jj].p8 = ordered_query[sub_voxel_first_index_PL[temp]+7+16+16].y;
                            ordered_ref16_z[jj].p8 = ordered_query[sub_voxel_first_index_PL[temp]+7+16+16].z;
                            original_dataset_index16[jj].p8 = original_dataset_index[sub_voxel_first_index_PL[temp]+7+16+16];
                            num1 = num1 + 1;
                        }
                        else
                        {
                            ordered_ref16_x[jj].p8 = 0;
                            ordered_ref16_y[jj].p8 = 0;
                            ordered_ref16_z[jj].p8 = 0;
                            original_dataset_index16[jj].p8 = k_reference_set_size;
                        }

                        if (sub_voxel_size[temp] >= 9+16+16)
                        {
                            ordered_ref16_x[jj].p9 = ordered_query[sub_voxel_first_index_PL[temp]+8+16+16].x;
                            ordered_ref16_y[jj].p9 = ordered_query[sub_voxel_first_index_PL[temp]+8+16+16].y;
                            ordered_ref16_z[jj].p9 = ordered_query[sub_voxel_first_index_PL[temp]+8+16+16].z;
                            original_dataset_index16[jj].p9 = original_dataset_index[sub_voxel_first_index_PL[temp]+8+16+16];
                            num1 = num1 + 1; 
                        }
                        else
                        {
                            ordered_ref16_x[jj].p9 = 0;
                            ordered_ref16_y[jj].p9 = 0;
                            ordered_ref16_z[jj].p9 = 0;
                            original_dataset_index16[jj].p9 = k_reference_set_size;
                        }
                        if (sub_voxel_size[temp] >= 10+16+16)
                        {
                            ordered_ref16_x[jj].p10 = ordered_query[sub_voxel_first_index_PL[temp]+9+16+16].x;
                            ordered_ref16_y[jj].p10 = ordered_query[sub_voxel_first_index_PL[temp]+9+16+16].y;
                            ordered_ref16_z[jj].p10 = ordered_query[sub_voxel_first_index_PL[temp]+9+16+16].z;
                            original_dataset_index16[jj].p10 = original_dataset_index[sub_voxel_first_index_PL[temp]+9+16+16];
                            num1 = num1 + 1;  
                        }
                        else
                        {
                            ordered_ref16_x[jj].p10 = 0;
                            ordered_ref16_y[jj].p10 = 0;
                            ordered_ref16_z[jj].p10 = 0;
                            original_dataset_index16[jj].p10 = k_reference_set_size;
                        }
                        if (sub_voxel_size[temp] >= 11+16+16)
                        {
                            ordered_ref16_x[jj].p11 = ordered_query[sub_voxel_first_index_PL[temp]+10+16+16].x;
                            ordered_ref16_y[jj].p11 = ordered_query[sub_voxel_first_index_PL[temp]+10+16+16].y;
                            ordered_ref16_z[jj].p11 = ordered_query[sub_voxel_first_index_PL[temp]+10+16+16].z;
                            original_dataset_index16[jj].p11 = original_dataset_index[sub_voxel_first_index_PL[temp]+10+16+16];
                            num1 = num1 + 1;
                        }
                        else
                        {
                            ordered_ref16_x[jj].p11 = 0;
                            ordered_ref16_y[jj].p11 = 0;
                            ordered_ref16_z[jj].p11 = 0;
                            original_dataset_index16[jj].p11 = k_reference_set_size;
                        }
                        if (sub_voxel_size[temp] >= 12+16+16)
                        {
                            ordered_ref16_x[jj].p12 = ordered_query[sub_voxel_first_index_PL[temp]+11+16+16].x;
                            ordered_ref16_y[jj].p12 = ordered_query[sub_voxel_first_index_PL[temp]+11+16+16].y;
                            ordered_ref16_z[jj].p12 = ordered_query[sub_voxel_first_index_PL[temp]+11+16+16].z;
                            original_dataset_index16[jj].p12 = original_dataset_index[sub_voxel_first_index_PL[temp]+11+16+16];
                            num1 = num1 + 1;  
                        }
                        else
                        {
                            ordered_ref16_x[jj].p12 = 0;
                            ordered_ref16_y[jj].p12 = 0;
                            ordered_ref16_z[jj].p12 = 0;
                            original_dataset_index16[jj].p12 = k_reference_set_size;
                        }

                        if (sub_voxel_size[temp] >= 13+16+16)
                        {
                            ordered_ref16_x[jj].p13 = ordered_query[sub_voxel_first_index_PL[temp]+12+16+16].x;
                            ordered_ref16_y[jj].p13 = ordered_query[sub_voxel_first_index_PL[temp]+12+16+16].y;
                            ordered_ref16_z[jj].p13 = ordered_query[sub_voxel_first_index_PL[temp]+12+16+16].z;
                            original_dataset_index16[jj].p13 = original_dataset_index[sub_voxel_first_index_PL[temp]+12+16+16];
                            num1 = num1 + 1;
                        }
                        else
                        {
                            ordered_ref16_x[jj].p13 = 0;
                            ordered_ref16_y[jj].p13 = 0;
                            ordered_ref16_z[jj].p13 = 0;
                            original_dataset_index16[jj].p13 = k_reference_set_size;
                        }
                        if (sub_voxel_size[temp] >= 14+16+16)
                        {
                            ordered_ref16_x[jj].p14 = ordered_query[sub_voxel_first_index_PL[temp]+13+16+16].x;
                            ordered_ref16_y[jj].p14 = ordered_query[sub_voxel_first_index_PL[temp]+13+16+16].y;
                            ordered_ref16_z[jj].p14 = ordered_query[sub_voxel_first_index_PL[temp]+13+16+16].z;
                            original_dataset_index16[jj].p14 = original_dataset_index[sub_voxel_first_index_PL[temp]+13+16+16];
                            num1 = num1 + 1;
                        }
                        else
                        {
                            ordered_ref16_x[jj].p14 = 0;
                            ordered_ref16_y[jj].p14 = 0;
                            ordered_ref16_z[jj].p14 = 0;
                            original_dataset_index16[jj].p14 = k_reference_set_size;
                        }
                        if (sub_voxel_size[temp] >= 15+16+16)
                        {
                            ordered_ref16_x[jj].p15 = ordered_query[sub_voxel_first_index_PL[temp]+14+16+16].x;
                            ordered_ref16_y[jj].p15 = ordered_query[sub_voxel_first_index_PL[temp]+14+16+16].y;
                            ordered_ref16_z[jj].p15 = ordered_query[sub_voxel_first_index_PL[temp]+14+16+16].z;
                            original_dataset_index16[jj].p15 = original_dataset_index[sub_voxel_first_index_PL[temp]+14+16+16];
                            num1 = num1 + 1;
                        }
                        else
                        {
                            ordered_ref16_x[jj].p15 = 0;
                            ordered_ref16_y[jj].p15 = 0;
                            ordered_ref16_z[jj].p15 = 0;
                            original_dataset_index16[jj].p15 = k_reference_set_size;
                        }
                        if (sub_voxel_size[temp] >= 16+16+16)
                        {
                            ordered_ref16_x[jj].p16 = ordered_query[sub_voxel_first_index_PL[temp]+15+16+16].x;
                            ordered_ref16_y[jj].p16 = ordered_query[sub_voxel_first_index_PL[temp]+15+16+16].y;
                            ordered_ref16_z[jj].p16 = ordered_query[sub_voxel_first_index_PL[temp]+15+16+16].z;
                            original_dataset_index16[jj].p16 = original_dataset_index[sub_voxel_first_index_PL[temp]+15+16+16];
                            num1 = num1 + 1;
                        }
                        else
                        {
                            ordered_ref16_x[jj].p16 = 0;
                            ordered_ref16_y[jj].p16 = 0;
                            ordered_ref16_z[jj].p16 = 0;
                            original_dataset_index16[jj].p16 = k_reference_set_size;
                        }

                        jj = jj + 1;
                        c1 = 3;
                    }

                    if (c1 == 1)
                    {
                        subindex16[temp] = jj-1;
                        subindex16[temp](31,29) = 1;
                    }
                    else if (c1 == 2)
                    {
                        subindex16[temp] = jj-2;
                        subindex16[temp](31,29) = 2;
                    }
                    else if (c1 == 3)
                    {
                        subindex16[temp] = jj-3;
                        subindex16[temp](31,29) = 3;
                    }
                    else
                    {
                        subindex16[temp] = jj;
                    }
                }
            }
        }

        DEBUG_INFO(jj);
        DEBUG_INFO(num1);
        packs = jj;
    }
    else
    {
        reorder_query(original_dataset_index, KNN_reference_set_size, data_set_hash, voxel_occupied_number, query_set_first_index, ordered_DSVS, ordered_query, KNN_reference_set);

    }

#else

    if (!reorder_query_set)
        reorder_data_set(original_dataset_index, KNN_reference_set_size, data_set_hash, voxel_occupied_number, voxel_first_index_PL, ordered_DSVS, ordered_query, KNN_reference_set);
    else
        reorder_query(original_dataset_index, KNN_reference_set_size, data_set_hash, voxel_occupied_number, query_set_first_index, ordered_DSVS, ordered_query, KNN_reference_set);

#endif

}


