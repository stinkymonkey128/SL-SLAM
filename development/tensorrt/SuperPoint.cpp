#include "SuperPoint.h"

using namespace tensorrt_log;

SuperPoint::SuperPoint(SuperPointConfig config) :
    config_(config),
    engine_(nullptr)
{
    setReportableSeverity(config_.logSeverity);
}

int SuperPoint::build() {
    if (deserializeEngine())
        return 0;

    std::unique_ptr<nvinfer1::IBuilder> builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder)
        return 1;

    const uint32_t explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    std::unique_ptr<nvinfer1::INetworkDefinition> network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
        return 2;

    std::unique_ptr<nvinfer1::IBuilderConfig> config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
        return 3;

    std::unique_ptr<nvonnxparser::IParser> parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
    if (!parser)
        return 4;

    auto profile = builder->createOptimizationProfile();
    if (!profile)
        return 5;

    profile->setDimensions(config_.inputTensorNames[0].c_str(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, 1, 100, 100));
    profile->setDimensions(config_.inputTensorNames[0].c_str(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, 1, 500, 500));
    profile->setDimensions(config_.inputTensorNames[0].c_str(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(1, 1, 1500, 1500));
    
    config->addOptimizationProfile(profile);

    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed)
        return 6;

    auto profileStream = tensorrt_common::makeCudaStream();
    if (!profileStream)
        return 7;

    config->setProfileStream(*profileStream);

    std::unique_ptr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
        return 8;

    std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(gLogger.getTRTLogger())};
    if (!runtime)
        return 9;

    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()));
    if (!engine_)
        return 10;
    
    saveEngine();
    return 0;
}

bool SuperPoint::infer(const cv::Mat& img, Eigen::Matrix<double, 259, Eigen::Dynamic>& features) {
    if (!context_) {
        context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
        if (!context_)
            return false;
    }

    CHECK_RETURN_W_MSG(context_->setInputShape(config_.inputTensorNames[0].c_str(), nvinfer1::Dims4(1, 1, img.rows, img.cols)), false, "Invalid binding dimensions");

    tensorrt_buffer::BufferManager buffers(engine_, 0, context_.get());

    for (int32_t i = 0, e = engine_->getNbIOTensors(); i < e; i++) {
        auto const name = engine_->getIOTensorName(i);
        context_->setTensorAddress(name, buffers.getDeviceBuffer(name));    
    }

    if (!context_->allInputDimensionsSpecified())
        return false;

    ASSERT(config_.inputTensorNames.size() == 1);

    if (!processInput(buffers, img))
        return false;

    buffers.copyInputToDevice();
    
    bool status = context_->executeV2(buffers.getDeviceBindings().data());
    if (!status)
        return false;
    
    buffers.copyOutputToHost();

    if (!processOutput(buffers, features))
        return false;

    return true;
}

bool SuperPoint::deserializeEngine() {
    std::ifstream file(config_.engineFilePath.c_str(), std::ios::binary);
    if (file.is_open()) {
        file.seekg(0, std::ifstream::end);
        size_t size = file.tellg();
        file.seekg(0, std::ifstream::beg);
        char *model_stream = new char[size];
        file.read(model_stream, size);
        file.close();
        nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
        if (runtime == nullptr) return false;
        engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(model_stream, size));
        if (engine_ == nullptr) return false;
        return true;
    }

    return false;
}

bool SuperPoint::constructNetwork(
    std::unique_ptr<nvinfer1::IBuilder>& builder,
    std::unique_ptr<nvinfer1::INetworkDefinition>& network,
    std::unique_ptr<nvinfer1::IBuilderConfig>& config,
    std::unique_ptr<nvonnxparser::IParser>& parser
) const {
    auto parsed = parser->parseFromFile(config_.onnxFilePath.c_str(), static_cast<int>(gLogger.getReportableSeverity()));
    if (!parsed)
        return false;

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, config_.memoryPoolLimit);
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    if (config_.useDlaCore)
        tensorrt_common::enableDLA(builder.get(), config.get(), config_.dlaCore);

    return true;
}

void SuperPoint::saveEngine() {
    if (engine_) {
        nvinfer1::IHostMemory* data = engine_->serialize();
        std::ofstream file(config_.engineFilePath, std::ios::binary);

        if (!file)
            return;

        file.write(reinterpret_cast<const char*>(data->data()), data->size());
    }
}

bool SuperPoint::processInput(const tensorrt_buffer::BufferManager& buffers, const cv::Mat& img) {
    float* imageInput = static_cast<float*>(buffers.getHostBuffer(config_.inputTensorNames[0]));

    cv::Mat gray;
    if (img.channels() == 3)
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else
        gray = img;

    for (int row = 0; row < gray.rows; row++) {
        for (int col = 0; col < gray.cols; col++) {
            imageInput[row * img.cols + col] =  float(gray.at<unsigned char>(row, col)) / 255.f;
        }
    }

    return true;
}

bool SuperPoint::processOutput(const tensorrt_buffer::BufferManager& buffers, Eigen::Matrix<double, 259, Eigen::Dynamic>& features) {
    float* keypoints = static_cast<float*>(buffers.getHostBuffer(config_.outputTensorNames[0]));
    float* scores = static_cast<float*>(buffers.getHostBuffer(config_.outputTensorNames[1]));
    float* descriptors = static_cast<float*>(buffers.getHostBuffer(config_.outputTensorNames[2]));
    
    nvinfer1::Dims kpsDims = context_->getTensorShape(config_.outputTensorNames[0].c_str());
    int numKeypoints = kpsDims.d[1]; // 99% sure this is not a dynamic axes but graph view shows it is

    features.resize(259, numKeypoints);

    for (int i = 0; i < numKeypoints; i++) {
        features(0, i) = static_cast<double>(scores[i]);
        features(1, i) = static_cast<double>(keypoints[i*2]);
        features(2, i) = static_cast<double>(keypoints[i*2 + 1]);

        for (int d = 0; d < 256; d++)
            features(3 + d, i) = static_cast<double>(descriptors[i*256 + d]);
    }

    return true;
}

void SuperPoint::visualize(
    const cv::Mat& img, 
    const Eigen::Matrix<double, 259, Eigen::Dynamic>& features, 
    const std::string& outImgPath
) {
    cv::Mat display;

    if (img.channels() == 1)
        cv::cvtColor(img, display, cv::COLOR_GRAY2BGR);
    else
        display = img.clone();

    const int numKeypoints = features.cols();

    for (int i = 0; i < numKeypoints; ++i) {
        cv::circle(
            display,
            cv::Point(static_cast<int>(features(1, i)), static_cast<int>(features(2, i))),
            1,
            cv::Scalar(0, 255, 0),
            -1
        );
    }

    cv::imwrite(outImgPath, display);
}