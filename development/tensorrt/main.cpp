#include "SuperPoint.h"

#include <iostream>

int main() {
    SuperPointConfig config;
    config.dlaCore = 0;
    config.onnxFilePath = "SuperPoint.onnx";
    config.engineFilePath = "SuperPoint.engine";
    config.inputTensorNames = {"l_data_"};
    config.outputTensorNames = {"where_2", "div"};
    config.confThreshold = 0.0005;
    config.removeBorders = 0;
    config.maxKeypoints = 1024;
    config.useDlaCore = false;

    SuperPoint spoint(config);
    int buildStatus = spoint.build();
    if (!buildStatus)
        std::cout << "Successfully built or loaded engine!" << std::endl;
    else
        std::cout << "Failed to build engine... error code " << buildStatus << std::endl;

    cv::Mat image0 = cv::imread("00000.jpg");

    // cv::Mat image1 = cv::imread("000001.jpg");

    Eigen::Matrix<double, 259, Eigen::Dynamic> feature_points0;

    if (!spoint.infer(image0, feature_points0)) {
        std::cout << "Something FATAL happened while running inference!" << std::endl;
        return 1;
    }

    spoint.visualize("out", image0);

    return 0;
}