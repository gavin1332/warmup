PADDLE_LIB=~/workroot/paddle_api
rm ./demo_trainer
rm ./startup_program
rm ./main_program
rm ./CMakeCache.txt
rm ./cmake_install.cmake
rm -rf ./CMakeFiles
rm -rf ./breakpoint

mkdir breakpoint
# PADDLE_LIB is the same with FLUID_INSTALL_DIR when building the lib
cmake .. -DPADDLE_LIB=$PADDLE_LIB \
         -DWITH_MKLDNN=OFF \
         -DWITH_MKL=OFF
make

# copy startup_program and main_program to this dir
cp ../startup_program .
cp ../main_program .

# run demo cpp trainer
./demo_trainer
