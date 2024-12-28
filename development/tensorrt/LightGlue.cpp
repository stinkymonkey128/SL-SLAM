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
    /*
    auto profile = builder->createOptimizationProfile();
    if (!profile)
        return 5;

    // keypoints (2, n, 2) aka 2 images, n keypoints, 2d space
    profile->setDimensions(config_.inputTensorNames[0].c_str(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims3(2, 1, 2));
    profile->setDimensions(config_.inputTensorNames[0].c_str(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims3(2, 512, 2));
    profile->setDimensions(config_.inputTensorNames[0].c_str(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims3(2, 1024, 2));
    
    // descriptors (2, n, ddim) aka 2 images, n keypoints, descripor dimension aka vector embeddings of points in our case dim is 256
    profile->setDimensions(config_.inputTensorNames[1].c_str(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims3(2, 1, 256));
    profile->setDimensions(config_.inputTensorNames[1].c_str(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims3(2, 512, 256));
    profile->setDimensions(config_.inputTensorNames[1].c_str(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims3(2, 1024, 256)); 

    config->addOptimizationProfile(profile);*/

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
    std::vector<Eigen::VectorXi>& indices,
    std::vector<Eigen::VectorXd>& scores
) {
    if (!context_) {
        context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
        if (!context_)
            return false;
    }

    return true;
}