#include "LightGlue.h"

using namespace tensorrt_log;

LightGlue::LightGlue(LightGlueConfig config) :
    config_(config),
    engine_(nullptr)
{
    setReportableSeverity(config_.logSeverity);
}

int LightGlue::build() {
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

    profile->setDimensions(config_.outputTensorNames[0].c_str(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2(1, 3));
    profile->setDimensions(config_.outputTensorNames[0].c_str(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2(512, 3));
    profile->setDimensions(config_.outputTensorNames[0].c_str(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2(1024, 3));
    
    profile->setDimensions(config_.outputTensorNames[1].c_str(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2(1, 1));
    profile->setDimensions(config_.outputTensorNames[1].c_str(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2(1, 512));
    profile->setDimensions(config_.outputTensorNames[1].c_str(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2(1, 1024));
    
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

bool LightGlue::deserializeEngine() {
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

bool LightGlue::constructNetwork(
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

void LightGlue::saveEngine() {
    if (engine_) {
        nvinfer1::IHostMemory* data = engine_->serialize();
        std::ofstream file(config_.engineFilePath, std::ios::binary);

        if (!file)
            return;

        file.write(reinterpret_cast<const char*>(data->data()), data->size());
    }
}

bool LightGlue::infer(
    const std::vector<Eigen::Matrix<double, 259, Eigen::Dynamic>>& features,
    std::vector<Eigen::VectorXi>& matches,
    std::vector<Eigen::VectorXd>& scores,
    int imgHeight,
    int imgWidth
) {
    if (!context_) {
        context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
        if (!context_)
            return false;
    }

    tensorrt_buffer::BufferManager buffers(engine_, 0, context_.get());

    for (int32_t i = 0, e = engine_->getNbIOTensors(); i < e; i++) {
        auto const name = engine_->getIOTensorName(i);
        context_->setTensorAddress(name, buffers.getDeviceBuffer(name));    
    }

    if (!context_->allInputDimensionsSpecified())
        return false;

    ASSERT(config_.inputTensorNames.size() == 2);

    if (!processInput(buffers, features, imgHeight, imgWidth))
        return false;

    buffers.copyInputToDevice();

    bool status = context_->executeV2(buffers.getDeviceBindings().data());
    if (!status)
        return false;

    buffers.copyOutputToHost();

    if (!processOutput(buffers, matches, scores))
        return false;

    return true;
}

bool LightGlue::processInput(
    const tensorrt_buffer::BufferManager& buffers, 
    const std::vector<Eigen::Matrix<double, 259, Eigen::Dynamic>>& features,
    int imgHeight,
    int imgWidth
) {
    float* keypoints = static_cast<float*>(buffers.getHostBuffer(config_.inputTensorNames[0]));
    float* descriptors = static_cast<float*>(buffers.getHostBuffer(config_.inputTensorNames[1]));
    // keypoints [2, 1024, 2]
    // descriptors [2, 1024, 256]

    for (int i = 0; i < 2; i++) {
        for (int point = 0; point < MAX_KEYPOINTS; point++) {
            double x = features[i](1, point);
            double y = features[i](2, point);

            float norm_x = 2.f * static_cast<float>(x) / static_cast<float>(imgWidth) - 1.f;
            float norm_y = 2.f * static_cast<float>(y) / static_cast<float>(imgHeight) - 1.f;

            keypoints[i * MAX_KEYPOINTS + point * 2] = norm_x;
            keypoints[i * MAX_KEYPOINTS + point * 2 + 1] = norm_y;

            for (int d = 0; d < 256; d++)
                descriptors[i * MAX_KEYPOINTS * 256 + point * 256 + d] = features[i](3 + d, point);
        }
    }

    return true;
}

bool LightGlue::processOutput(
    const tensorrt_buffer::BufferManager& buffers, 
    std::vector<Eigen::VectorXi>& matches,
    std::vector<Eigen::VectorXd>& scores
) {
    float* outMatches = static_cast<float*>(buffers.getHostBuffer(config_.outputTensorNames[0]));
    float* outScores = static_cast<float*>(buffers.getHostBuffer(config_.outputTensorNames[1]));

    nvinfer1::Dims matchDims = context_->getTensorShape(config_.outputTensorNames[0].c_str());
    int numMatches = matchDims.d[0];

    std::cout << "NUMBER OF MATCHES " << numMatches << std::endl;

    return true;
}