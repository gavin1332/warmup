# warmup
Newbie collaboration test
### Step 1. build paddle lib

```

# WITH_MKL=ON|OFF
# WITH_MKLDNN=ON|OFF

PADDLE_LIB=/paddle/lib/dir
cmake .. -DFLUID_INSTALL_DIR=$PADDLE_LIB \
         -DCMAKE_BUILD_TYPE=Release \
         -DWITH_GPU=OFF \
         -DWITH_STYLE_CHECK=OFF \
         -DWITH_MKL=OFF \
         -DWITH_MKLDNN=OFF
make -j8
make -j8 fluid_lib_dist
```

### step 2. generate program desc
```
# please install paddle before run this scripe
pip install --upgrade paddlepaddle-*.whl
python mnist_network.py
```

This will generate two program desc files:
  - startup_program: used to init all parameters
  - main_program: main logic of the network

### step 3. build mnist_train and run it.


```
# Make a build dir at the same dir of this README.md document.
# The demo dir can be put anywhere.
mkdir build
cd build
sh build.sh
```
