############################## Help Section ##############################
.PHONY: help

help::
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make clean
	$(ECHO) "      Command to remove the generated files."
	$(ECHO) ""
	$(ECHO) "  make all
	$(ECHO) "      Command to package host execute file and hardware accelerator."
	$(ECHO) ""

# ####################### Setting file directory #######################################
MK_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
CUR_DIR := $(patsubst %/,%,$(dir $(MK_PATH)))
BUILD_DIR := $(CUR_DIR)/build
KERNEL_PATH := $(CUR_DIR)/../kernel/vitis_build.prj/
HOST_PATH := $(CUR_DIR)/../host/build

# ####################### Setting compile environment ##################################
XILINX_VITIS ?= /tools/Xilinx/Vitis/2022.2
VPP ?= ${XILINX_VITIS}/bin/v++
TARGET ?= hw
# VITIS_PLATFORM = xilinx_zcu102_base_202210_1
VITIS_PLATFORM ?= xilinx_zcu104_base_202220_1
TARGET_XCLBIN = $(BUILD_DIR)/kernel.xclbin

# ####################### Setting package flags ##################################
PACKAGE_FLAGS += --package.sd_file $(HOST_PATH)/*.exe
PACKAGE_FLAGS += --package.out_dir=$(CUR_DIR)/package.out
PACKAGE_FLAGS += -t $(TARGET) --config package.cfg
PACKAGE_FLAGS += $(KERNEL_PATH)/*.xclbin -o $(TARGET_XCLBIN)

############################## Setting Makefile Targets ##############################
all: package

package:
	@echo Running Package Step for the System Project.
	@echo $(VPP) --package --config package.cfg  $(KERNEL_PATH)/*.xclbin -o $(TARGET_XCLBIN) $(PACKAGE_FLAGS)
	$(VPP) --package $(PACKAGE_FLAGS)

clean:
	rm -rf $(BUILD_DIR)
	rm -rf ./package.out
	rm -rf ./.Xil
	rm -rf ./_x
	rm ./*.log

.PHONY: all package clean


