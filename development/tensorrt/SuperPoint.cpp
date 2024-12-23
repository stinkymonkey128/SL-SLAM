#include "SuperPoint.h"

using namespace tensorrt_log;

SuperPoint::SuperPoint(SuperPointConfig config) :
    config_(config),
    engine_(nullptr)
{
    setReportableSeverity(Logger::Severity::kINTERNAL_ERROR);
}

bool SuperPoint::build() {
    if (deserializeEngine())
        return true;

    std::unique_ptr<nvinfer1::IBuilder> builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder)
        return false;

    const uint32_t explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    std::unique_ptr<nvinfer1::INetworkDefinition> network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
        return false;

    std::unique_ptr<nvinfer1::IBuilderConfig> config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
        return false;

    std::unique_ptr<nvonnxparser::IParser> parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
    if (!parser)
        return false;

    auto profile = builder->createOptimizationProfile();
    if (!profile)
        return false;

    profile->setDimensions(config_.inputTensorNames[0].c_str(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, 1, 100, 100));
    profile->setDimensions(config_.inputTensorNames[0].c_str(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, 1, 500, 500));
    profile->setDimensions(config_.inputTensorNames[0].c_str(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(1, 1, 1500, 1500));
    config->addOptimizationProfile(profile);

    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed)
        return false;

    cudaStream_t profileStream;
    cudaStreamCreate(&profileStream);
    if (!profileStream)
        return false;

    config->setProfileStream(profileStream);

    std::unique_ptr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
        return false;

    std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(gLogger.getTRTLogger())};
    if (!runtime)
        return false;

    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()));
    if (!engine_)
        return false;
    
    saveEngine();
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

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 512_MiB);
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    config->setDLACore(config_.dlaCore);

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