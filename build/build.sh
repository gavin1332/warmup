PADDLE_LIB=~/workroot/paddle_api
rm -f ./startup_program
rm -f ./main_program
rm -rf ./breakpoint

make clean

mkdir -p breakpoint
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
