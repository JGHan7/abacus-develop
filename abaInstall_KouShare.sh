rm -rf build

#cmake -B build -DCMAKE_INSTALL_PREFIX=/public1/home/t6s000394/jghan/software/abacus-develop/rdmft-abacus/ -DCMAKE_CXX_COMPILER=icpx -DELPA_DIR=/public1/home/t6s000394/jghan/software/elpa-2021.11/ -DCEREAL_INCLUDE_DIR=/public1/home/t6s000394/jghan/software/cereal/cereal-1.3.2/include
#cmake -B build -DCMAKE_INSTALL_PREFIX=/public1/home/t6s000394/jghan/software/abacus-develop/rdmft-abacus/ -DCMAKE_CXX_COMPILER=icpx -DMPI_CXX_COMPILER=mpiicpc -DELPA_DIR=/public1/home/t6s000394/jghan/software/elpa-2021.11/ -DLibxc_DIR=/public1/home/t6s000394/jghan/software/libxc/ -DLIBRI_DIR=/public1/home/t6s000394/jghan/software/LibRI -DLIBCOMM_DIR=/public1/home/t6s000394/jghan/software/LibComm -DCEREAL_INCLUDE_DIR=/public1/home/t6s000394/jghan/software/cereal/cereal-1.3.2/include -DDEBUG_INFO=ON
#cmake -B build -DCMAKE_INSTALL_PREFIX=/public1/home/t6s000394/jghan/software/abacus-develop/rdmft-abacus/ -DCMAKE_CXX_COMPILER=icpx -DMPI_CXX_COMPILER=mpiicpc -DELPA_DIR=/public1/home/t6s000394/jghan/software/elpa-2021.11/ -DLibxc_DIR=/public1/home/t6s000394/jghan/software/libxc/ -DLIBRI_DIR=/public1/home/t6s000394/jghan/software/LibRI -DLIBCOMM_DIR=/public1/home/t6s000394/jghan/software/LibComm -DCEREAL_INCLUDE_DIR=/public1/home/t6s000394/jghan/software/cereal/cereal-1.3.2/include

# cmake -B build -DCMAKE_INSTALL_PREFIX=/mnt/sg001/home/ks_iopcas_jghan/software/rdmft-abacus -DCMAKE_CXX_COMPILER=icpx -DMPI_CXX_COMPILER=mpiicpc -DELPA_DIR=/mnt/sg001/home/ks_iopcas_jghan/software/elpa-2021.11.001 -DLibxc_DIR=/mnt/sg001/home/ks_iopcas_jghan/software/libxc -DLIBRI_DIR=/mnt/sg001/home/ks_iopcas_jghan/software/LibRI -DLIBCOMM_DIR=/mnt/sg001/home/ks_iopcas_jghan/software/LibComm -DCEREAL_INCLUDE_DIR=/mnt/sg001/home/ks_iopcas_jghan/software/cereal-1.3.2/include -DTorch_DIR=/mnt/sg001/home/ks_iopcas_jghan/software/libtorch/share/cmake/Torch -DCMAKE_BUILD_TYPE=Debug

cmake -B build -DCMAKE_INSTALL_PREFIX=/mnt/sg001/home/ks_iopcas_jghan/software/rdmft-abacus -DCMAKE_CXX_COMPILER=icpx -DMPI_CXX_COMPILER=mpiicpc -DELPA_DIR=/mnt/sg001/home/ks_iopcas_jghan/software/elpa-2021.11.001 -DLibxc_DIR=/mnt/sg001/home/ks_iopcas_jghan/software/libxc -DLIBRI_DIR=/mnt/sg001/home/ks_iopcas_jghan/software/LibRI -DLIBCOMM_DIR=/mnt/sg001/home/ks_iopcas_jghan/software/LibComm -DCEREAL_INCLUDE_DIR=/mnt/sg001/home/ks_iopcas_jghan/software/cereal-1.3.2/include -DTorch_DIR=/mnt/sg001/home/ks_iopcas_jghan/software/libtorch/share/cmake/Torch

# cmake -B build -DCMAKE_INSTALL_PREFIX=/mnt/sg001/home/ks_iopcas_jghan/software/rdmft-abacus -DCMAKE_CXX_COMPILER=icpx -DMPI_CXX_COMPILER=mpiicpc -DELPA_DIR=/mnt/sg001/home/ks_iopcas_jghan/software/elpa-2021.11.001 -DLibxc_DIR=/mnt/sg001/home/ks_iopcas_jghan/software/libxc -DLIBRI_DIR=/mnt/sg001/home/ks_iopcas_jghan/software/LibRI -DLIBCOMM_DIR=/mnt/sg001/home/ks_iopcas_jghan/software/LibComm -DCEREAL_INCLUDE_DIR=/mnt/sg001/home/ks_iopcas_jghan/software/cereal-1.3.2/include

#cmake --build build -j 42 2>job.err
cmake --build build -j 32

cmake --install build
