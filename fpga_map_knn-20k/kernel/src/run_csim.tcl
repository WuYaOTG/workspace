exec rm -f "time_record_hls_run.txt"
puts "successful!!!  the start time is [clock format [clock seconds]]"
exec echo "successful!!! runall start time at [clock format [clock seconds]]" >> "time_record_hls_run.txt"

set Project     hls_csim.prj
set Solution    solution_all
set Device      "xczu7ev-ffvc1156-2-e"
set Flow        "vivado"
set Clock       10

open_project $Project -reset   

set_top DSVS_search_hw

add_files knn_odom.cpp -cflags -I.
add_files knn_odom_hw.cpp -cflags -I.
add_files knn_odom.h -cflags -I
add_files -tb main.cpp -cflags -I.
add_files -tb timer.h -cflags -I.
# add_files -tb fpga_host_class.cpp -cflags -I.
# add_files -tb fpga_host_class.h -cflags -I.
# add_files -tb /home/sunhao/LOAM_FPGA/HLS_validation/knn_odom_hls/knn_odom/test_data/sharp_000001.txt -cflags -I.
# add_files -tb /home/sunhao/LOAM_FPGA/HLS_validation/knn_odom_hls/knn_odom/test_data/less_sharp_000000.txt -cflags -I.

# default vivado flow
open_solution $Solution -reset   
set_part $Device
create_clock -period $Clock -name default
config_interface -m_axi_alignment_byte_size 64 -m_axi_latency 0 -m_axi_max_widen_bitwidth 512 -m_axi_offset slave
config_rtl -register_reset_num 3

csim_design

exit