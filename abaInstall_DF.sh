#!/usr/bin/env bash
rm -rf build/

# cmake -B build \
#   -DCMAKE_INSTALL_PREFIX=/data/home/df_iopcas_jghan/software/pure-abacus/abacus-develop \
#   -DCMAKE_CXX_COMPILER=icpx \
#   -DMPI_CXX_COMPILER=mpiicpc \
#   -DELPA_DIR=/data/home/df_iopcas_jghan/software/elpa-2021.11.001 \
#   -DLibxc_DIR=/data/home/df_iopcas_jghan/software/libxc \
#   -DLIBRI_DIR=/data/home/df_iopcas_jghan/software/LibRI \
#   -DLIBCOMM_DIR=/data/home/df_iopcas_jghan/software/LibComm \
#   -DCEREAL_INCLUDE_DIR=/data/home/df_iopcas_jghan/software/cereal-1.3.2/include &&

cmake -B build \
  -DCMAKE_INSTALL_PREFIX=/data/home/df_iopcas_jghan/software/pure-abacus/abacus-develop \
  -DCMAKE_CXX_COMPILER=icpx \
  -DMPI_CXX_COMPILER=mpiicpc \
  -DELPA_DIR=/data/home/df_iopcas_jghan/software/elpa-2021.11.001 \
  -DLibxc_DIR=/data/home/df_iopcas_jghan/software/libxc \
  -DLIBRI_DIR=/data/home/df_iopcas_jghan/software/LibRI \
  -DLIBCOMM_DIR=/data/home/df_iopcas_jghan/software/LibComm \
  -DCEREAL_INCLUDE_DIR=/data/home/df_iopcas_jghan/software/cereal-1.3.2/include \
  -DTorch_DIR=/data/home/df_iopcas_jghan/software/libtorch/share/cmake/Torch &&

cmake --build build -j 8 &&
cmake --install build
