rm -r build

# cmake -B build -DCMAKE_INSTALL_PREFIX=/home/jghan/software/rdmft-abacus -DCMAKE_CXX_COMPILER=icpx -DMPI_CXX_COMPILER=mpiicpc -DELPA_DIR=/home/jghan/resource/elpa-2021.11/ -DLibxc_DIR=/home/jghan/resource/libxc/ -DLIBRI_DIR=/home/jghan/resource/LibRI -DLIBCOMM_DIR=/home/jghan/resource/LibComm -DCEREAL_INCLUDE_DIR=/home/jghan/resource/cereal/cereal-1.3.2/include
cmake -B build -DCMAKE_INSTALL_PREFIX=/home/jghan/software/rdmft-abacus -DCMAKE_CXX_COMPILER=icpx -DMPI_CXX_COMPILER=mpiicpc -DELPA_DIR=/home/jghan/resource/elpa-2021.11-old/ -DLibxc_DIR=/home/jghan/resource/libxc/ -DLIBRI_DIR=/home/jghan/resource/LibRI-abacus-v3.9.0.2 -DLIBCOMM_DIR=/home/jghan/resource/LibComm -DCEREAL_INCLUDE_DIR=/home/jghan/resource/cereal/cereal-1.3.2/include -DTorch_DIR=/home/jghan/software/libtorch/share/cmake/Torch

# cmake --build build -j 24 2>job.err
cmake --build build -j 8

cmake --install build
