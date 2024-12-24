#include "SuperPoint.h"

#include <iostream>

int main() {
    SuperPointConfig config;
    config.dlaCore = 0;
    config.onnxFilePath = "SuperPoint.onnx";
    config.engineFilePath = "SuperPoint.engine";
    config.inputTensorNames = {"l_data_"};

    SuperPoint spoint(config);
    int buildStatus = spoint.build();
    if (!buildStatus)
        std::cout << "Successfully built engine!" << std::endl;
    else
        std::cout << "Failed to build engine... error code " << buildStatus << std::endl;
}