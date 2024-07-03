#!/bin/bash

SRC_DIR=$HOME/packages
INSTALL_DIR=$HOME/local

rm -rf $INSTALL_DIR $SRC_DIR
mkdir -p $INSTALL_DIR $SRC_DIR

# Install CMake
cd $SRC_DIR;
git clone https://gitlab.kitware.com/cmake/cmake.git;
cd cmake;
git fetch --all --tags --prune;
git checkout tags/v3.28.1;
./bootstrap --prefix=$INSTALL_DIR/cmake -- -DCMAKE_USE_OPENSSL=OFF;
make -j;
make install -j;

# Install OpenBLAS
cd $SRC_DIR;
git clone https://github.com/OpenMathLib/OpenBLAS.git;
cd OpenBLAS;
git fetch --all --tags --prune;
git checkout tags/v0.3.26;
make -j;
make install -j PREFIX=$INSTALL_DIR/openblas;

# Install FxT
cd $SRC_DIR;
wget http://download.savannah.nongnu.org/releases/fkt/fxt-0.3.14.tar.gz
tar xf fxt-0.3.14.tar.gz;
mkdir -p fxt-0.3.14/build;
cd fxt-0.3.14/build;
../configure --prefix=$INSTALL_DIR/fxt;
make -j;
make install -j;

# Install New Madeleine
#cd $SRC_DIR
#git clone https://gitlab.inria.fr/pm2/pm2.git
#cd pm2
#git checkout 7e3edd5002885d4a114c5e551106f28ae9630e01
#cd scripts
#./pm2-build-packages ./madmpi.conf --prefix=$INSTALL_DIR/nmad

# Install StarPU
cd $SRC_DIR;
git clone --recurse-submodules https://gitlab.inria.fr/starpu/starpu.git;
cd starpu;
git fetch --all --tags --prune;
git checkout tags/starpu-1.4.1;
./autogen.sh;
mkdir build;
cd build;
../configure --prefix=$INSTALL_DIR/starpu --disable-opencl --disable-build-doc --disable-build-examples --disable-build-test;
make -j;
make install -j;
make clean;
../configure --prefix=$INSTALL_DIR/starpu-fxt --disable-opencl --disable-build-doc --disable-build-examples --disable-build-test --with-fxt=$INSTALL_DIR/fxt;
make -j;
make install -j;
