############################################################
## This file is generated automatically by Vitis HLS.
## Please DO NOT edit it.
## Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
############################################################
open_project 0617icecs
set_top DSVS_search_hw
add_files knn_odom.cpp
add_files knn_odom.h
add_files knn_odom_hw.cpp
add_files -tb main.cpp
add_files -tb timer.h
open_solution "solution1" -flow_target vitis
set_part {xczu7ev-ffvc1156-2-e}
create_clock -period 5 -name default
#source "./0617icecs/solution1/directives.tcl"
csim_design
csynth_design
cosim_design
export_design -format ip_catalog
