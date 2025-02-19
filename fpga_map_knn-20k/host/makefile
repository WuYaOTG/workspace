# 2022.09.27 sunhao on RICL AUROS Server
############################## Help Section ##############################
.PHONY: help

help::
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make clean
	$(ECHO) "      Command to remove the generated files."
	$(ECHO) ""
	$(ECHO) "  make all
	$(ECHO) "      Command to generate the execute files."
	$(ECHO) ""

# ####################### Setting file directory #######################################
MK_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
CUR_DIR := $(patsubst %/,%,$(dir $(MK_PATH)))
BUILD_DIR := $(CUR_DIR)/build
BUILD_REPORT_DIR := $(CUR_DIR)/reports/build
TEMP_DIR := $(BUILD_DIR)/temp
TEMP_REPORT_DIR := $(BUILD_DIR)/reports/temp
SRC_DIR := $(CUR_DIR)/../kernel/src/

# ####################### Setting objects and source files #############################
HOST_EXE := $(BUILD_DIR)/host.exe

HOST_OBJECTS += $(BUILD_DIR)/knn_odom.o
HOST_OBJECTS += $(BUILD_DIR)/main.o
HOST_OBJECTS += $(BUILD_DIR)/knn_odom_hw.o
HOST_OBJECTS += $(BUILD_DIR)/fpga_host_class.o
HOST_OBJECTS += $(BUILD_DIR)/xcl2.o
HOST_OBJECTS += $(BUILD_DIR)/power_monitor_104_api.o
HOST_SRCS += $(SRC_DIR)/knn_odom.cpp
HOST_SRCS += $(SRC_DIR)/main.cpp
HOST_SRCS += $(SRC_DIR)/knn_odom_hw.cpp
HOST_SRCS += $(SRC_DIR)/fpga_host_class.cpp
HOST_SRCS += $(SRC_DIR)/xcl2.cpp
HOST_SRCS += $(SRC_DIR)/power_monitor_104_api.cpp

# ####################### Setting compile environment ##################################
HOST_CXX := $(XILINX_VITIS)/gnu/aarch64/lin/aarch64-linux/bin/aarch64-linux-gnu-g++
HOST_C ?= $(XILINX_VITIS)/gnu/aarch64/lin/aarch64-linux/bin/aarch64-linux-gnu-gcc
TARGET ?= hw
# VITIS_PLATFORM ?= xilinx_zcu102_base_202210_1
VITIS_PLATFORM ?= xilinx_zcu104_base_202220_1

# ####################### Setting compile and link flags ##################################
CXXFLAGS += -I$(SRC_DIR)/
CXXFLAGS += -I$(XILINX_XRT)/include 
CXXFLAGS += -I$(XILINX_HLS)/include 
CXXFLAGS += -I$(XILINX_VIVADO)/include 
CXXFLAGS += -I/home/sunhao/MyLibrary/my_include/
CXXFLAGS += -I/home/sunhao/MyLibrary/aarch64_zcu102_PCL/dependancy_include/
CXXFLAGS += -I/home/sunhao/MyLibrary/aarch64_zcu102_PCL/dependancy_include/eigen3/
CXXFLAGS += -std=c++11 -O2 -g -Wall -c -Wno-unknown-pragmas -Wno-unused-label -fmessage-length=0
CXXFLAGS += -DVITIS_PLATFORM=$(VITIS_PLATFORM) 
CXXFLAGS += -D__USE_XOPEN2K8  
CXXFLAGS += --sysroot=/opt/petalinux/2022.2/sysroots/cortexa72-cortexa53-xilinx-linux/
CXXFLAGS += -I/opt/petalinux/2022.2/sysroots/cortexa72-cortexa53-xilinx-linux/usr/include/xrt/
CXXFLAGS += -I/opt/petalinux/2022.2/sysroots/cortexa72-cortexa53-xilinx-linux/usr/include/
CXXFLAGS += -D_USE_OPENCL_
CXXFLAGS += -DUSE_PCL_ON
CXXFLAGS += -DEST_POWER

LDFLAGS += -L/home/sunhao/MyLibrary/aarch64_zcu102_PCL/dependancy_a
LDFLAGS += -Wl,-Bstatic 
LDFLAGS += -lpcl_segmentation -lpcl_sample_consensus -lpcl_tracking -lpcl_registration -lpcl_io -lpcl_ml -lpcl_io_ply -lpcl_keypoints -lpcl_kdtree -lpcl_surface -lpcl_features -lpcl_stereo -lpcl_filters -lpcl_octree -lpcl_recognition -lpcl_search -lpcl_common -lboost_stacktrace_basic -lboost_timer -lboost_container -lboost_regex -lboost_signals -lboost_iostreams -lboost_math_c99l -lboost_exception -lboost_context -lboost_wserialization -lboost_chrono -lboost_test_exec_monitor -lboost_filesystem -lboost_atomic -lboost_math_tr1l -lboost_locale -lboost_log_setup -lboost_coroutine -lboost_type_erasure -lboost_wave -lboost_math_tr1f -lboost_unit_test_framework -lboost_stacktrace_noop -lboost_graph -lboost_math_c99f -lboost_math_c99 -lboost_math_tr1 -lboost_fiber -lboost_log -lboost_random -lboost_thread -lboost_prg_exec_monitor -lboost_date_time -lboost_program_options -lboost_contract -lboost_serialization -lboost_system -lflann_cpp_s -lflann_s -lpcap
LDFLAGS += -Wl,-Bdynamic
LDFLAGS += -L$(XILINX_XRT)/lib -Wno-unused-label -Wno-narrowing -DVERBOSE
LDFLAGS += -lxilinxopencl -lpthread -lrt -ldl -lcrypt -lstdc++ 
LDFLAGS += -L/opt/petalinux/2022.2/sysroots/cortexa72-cortexa53-xilinx-linux/usr/lib/ 
LDFLAGS += --sysroot=/opt/petalinux/2022.2/sysroots/cortexa72-cortexa53-xilinx-linux/
LDFLAGS += -fopenmp --static # 静态链接

############################## Setting Makefile Targets ##############################
.PHONY: check_xrt
check_xrt:
ifeq (,$(wildcard $(XILINX_XRT)/lib/libxilinxopencl.so))
	@echo "Cannot locate XRT installation. Please set XILINX_XRT variable." && false
endif

.PHONY: all clean
all:  $(HOST_EXE)

clean:
	@rm -rf $(BUILD_DIR)
	@rm *.log

# ################ Setting Rules for host ###############
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	-@mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -o "$@" "$<"

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	-@mkdir -p $(@D)
	$(HOST_C) $(CXXFLAGS) -o "$@" "$<"

$(HOST_EXE): $(HOST_OBJECTS)
	@echo "Compiling Host: $(HOST_CXX) -o "$@" $(+) $(LDFLAGS)"
	$(HOST_CXX) -o "$@" $(+) $(LDFLAGS)
	@echo "GENERATE $(HOST_EXE) Done!"
