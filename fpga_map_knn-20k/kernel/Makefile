############################## Help Section ##############################
.PHONY: help

help::
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make clean
	$(ECHO) "      Command to remove the generated files."
	$(ECHO) ""
	$(ECHO) "  make all
	$(ECHO) "      Command to generate hardware accelerator(.xclbin)."
	$(ECHO) ""

# ####################### Setting file directory #######################################
MK_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
CUR_DIR := $(patsubst %/,%,$(dir $(MK_PATH)))
BUILD_DIR := $(CUR_DIR)/vitis_build.prj
BUILD_REPORT_DIR := $(BUILD_DIR)/reports/build
TEMP_DIR := $(BUILD_DIR)/temp
TEMP_REPORT_DIR := $(BUILD_DIR)/reports/temp
LOG_DIR=$(BUILD_DIR)/logs

# ####################### Setting objects and source files #############################
BUILD_XCLBIN := $(BUILD_DIR)/kernel.xclbin
BUILD_XOBJECT += $(TEMP_DIR)/knn_odom_hw.xo
BUILD_SOURCE += ./src/knn_odom_hw.cpp

# ####################### Setting compile environment ##################################
VPP ?= v++
TARGET ?= hw
# PLATFORM ?= xilinx_zcu102_base_202210_1
PLATFORM := xilinx_zcu104_base_202220_1
KERNEL_NAME = DSVS_search_hw

# ####################### Setting compile and link flags ##################################
VPP_FLAGS += -I./src
VPP_FLAGS += -t $(TARGET)
VPP_FLAGS += --platform $(PLATFORM)

# Kernel linker flags
VPP_FLAGS += --config ./krnl_odom.cfg

############################## Setting Makefile Targets ##############################
all: $(BUILD_XCLBIN)
	@cp $(BUILD_XCLBIN) $(CUR_DIR)

# ################ Setting Rules for Binary Containers (Building Kernels) ###############
#  ###### Linking Kernel ######
$(BUILD_XCLBIN): $(BUILD_XOBJECT)
	@echo  "Linking Kernel: $(VPP) -l $(VPP_FLAGS) $(BUILD_XOBJECT) -o $@ $^"
	@mkdir -p $(BUILD_DIR)
	$(VPP) -l $(VPP_FLAGS) $(BUILD_XOBJECT) $(KRNL_LINK_CFG) --temp_dir $(TEMP_DIR) --report_dir $(BUILD_REPORT_DIR)/$(KERNEL_NAME) -o $@ $^

#  ###### Compile Kernel ######
$(BUILD_XOBJECT): $(BUILD_SOURCE)
	@echo  "Compiling Kernel: $(VPP) -c $(VPP_FLAGS) $(BUILD_SOURCE) -k $(KERNEL_NAME) -o $@ $^"
	@mkdir -p $(TEMP_DIR)
	$(VPP) -c $(VPP_FLAGS) $(BUILD_SOURCE) -k $(KERNEL_NAME) --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o $@ $^

clean:
	@rm -rf *.json .run .Xil .ipcache *.jou *.log $(TEMP_DIR) $(TEMP_REPORT_DIR) $(BUILD_DIR) $(BUILD_REPORT_DIR)