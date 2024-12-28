#include "SuperPoint.h"
#include "LightGlue.h"

#include <iostream>

int main() {
    SuperPointConfig config;
    config.dlaCore = 0;
    config.onnxFilePath = "SuperPoint.onnx";
    config.engineFilePath = "SuperPoint.engine";
    config.inputTensorNames = {"input"};
    config.outputTensorNames = {"keypoints", "scores", "descriptors"};
    config.confThreshold = 0.0005;
    config.removeBorders = 0;
    config.maxKeypoints = 1024;
    config.useDlaCore = false;
    config.logSeverity = tensorrt_log::Logger::Severity::kVERBOSE;
    config.memoryPoolLimit = 512_MiB;

    SuperPoint spoint(config);
    int buildStatus = spoint.build();
    if (!buildStatus)
        std::cout << "Successfully built or loaded SuperPoint engine!" << std::endl;
    else
        std::cout << "Failed to build SuperPoint engine... error code " << buildStatus << std::endl;

    cv::Mat image0 = cv::imread("00000.jpg");

    // cv::Mat image1 = cv::imread("000001.jpg");

    Eigen::Matrix<double, 259, Eigen::Dynamic> feature_points0;

    if (!spoint.infer(image0, feature_points0)) {
        std::cout << "Something FATAL happened while running inference!" << std::endl;
        return 1;
    }

    spoint.visualize(image0, feature_points0, "out");

    return 0;

    LightGlueConfig lgConfig;
    lgConfig.onnxFilePath = "LightGlue.onnx";
    lgConfig.inputTensorNames = {"l_keypoints_", "l_descriptors_"};
    lgConfig.outputTensorNames = {"cat_18", "index_19"};
    lgConfig.logSeverity = tensorrt_log::Logger::Severity::kVERBOSE;

    LightGlue lightGlue(lgConfig);
    buildStatus = lightGlue.build();
    if (!buildStatus)
        std::cout << "Successfully built or loaded LightGlue engine!" << std::endl;
    else
        std::cout << "Failed to build LightGlue engine... error code " << buildStatus << std::endl;

    return 0;
}