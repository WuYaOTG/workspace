# set the compiling environment
source /tools/Xilinx/Vitis/2022.2/settings64.sh
unset LD_LIBRARY_PATH
export PLATFORM_REPO_PATHS=/tools/Xilinx/Vitis/2022.2/base_platforms/
export ROOTFS=/opt/xilinx/sysroot/xilinx-zynqmp-common-v2022.2/
export SYSROOT=/opt/petalinux/2022.2/sysroots/cortexa72-cortexa53-xilinx-linux/
source /opt/petalinux/2022.2/environment-setup-cortexa72-cortexa53-xilinx-linux


time1=$(date)
echo "build host start time at $time1"
make all > build.log 2>&1
time1=$(date)
echo "Finish end time at $time1"

tail -n 2 build.log