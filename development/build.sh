echo "tensorrt playground"
cd tensorrt
mkdir build
cd build
cmake ..
make -j

echo "benchmark"
cd ../../benchmark
mkdir build
cd build
cmake ..
make -j