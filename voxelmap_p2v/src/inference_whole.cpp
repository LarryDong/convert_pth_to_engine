#include <iostream>
#include <chrono>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <unordered_map>
#include <algorithm>
#include <iomanip>

using namespace nvinfer1;

// TensorRT Logger
class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
} gLogger;

// 读取新的CSV文件格式
void readCSVData(const std::string& filename, float* p, float* points) {
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("无法打开CSV文件: " + filename);
    }

    std::string line;
    
    // 读取第一行作为p
    if (std::getline(file, line)) {
        std::replace(line.begin(), line.end(), ',', ' ');
        std::stringstream ss(line);
        ss >> p[0] >> p[1] >> p[2];
    }

    // 读取后续50行作为points
    int row = 0;
    while (std::getline(file, line) && row < 50) {
        std::replace(line.begin(), line.end(), ',', ' ');
        std::stringstream ss(line);
        ss >> points[row*3] >> points[row*3+1] >> points[row*3+2];
        row++;
    }

    if (row < 50) {
        std::cerr << "警告: CSV文件只包含 " << row << " 个点，不足50个" << std::endl;
    }
}

// 加载TensorRT引擎
ICudaEngine* loadEngine(const std::string& engine_file, IRuntime*& runtime) {
    std::ifstream file(engine_file, std::ios::binary);
    if (!file.good()) {
        throw std::runtime_error("无法打开TensorRT engine文件: " + engine_file);
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);

    runtime = createInferRuntime(gLogger);
    return runtime->deserializeCudaEngine(buffer.data(), size);
}

// 打印数组数据
void printArray(const std::string& name, const float* data, int size, int max_print = 5) {
    std::cout << name << ": [";
    for (int i = 0; i < std::min(size, max_print); ++i) {
        std::cout << data[i];
        if (i < size - 1) std::cout << ", ";
    }
    if (size > max_print) std::cout << "...";
    std::cout << "]" << std::endl;
}

// 专门打印points数据的函数 (50x3格式)
void printPointsMatrix(const float* points, const std::string& title = "Points数据") {
    std::cout << "\n" << title << " (50x3):\n";
    std::cout << std::fixed << std::setprecision(6); // 固定6位小数
    
    for (int i = 0; i < 50; ++i) {
        std::cout << "  [" << std::setw(10) << points[i*3] << ", " 
                          << std::setw(10) << points[i*3 + 1] << ", " 
                          << std::setw(10) << points[i*3 + 2] << "]\n";
    }
}

// 执行TensorRT推理
void runInference(IExecutionContext* context, 
                 const float* input_points, 
                 const float* input_p,
                 float* output_features,
                 float* output_weight,
                 float* output_p2v) {
    const int batch_size = 1;
    const int input1_size = 1 * 50 * 3;
    const int input2_size = 1 * 3;
    const int output1_size = 1 * 1027;
    const int output2_size = 1 * 3;
    const int output3_size = 1 * 1;

    void* buffers[5];
    cudaMalloc(&buffers[0], input1_size * sizeof(float));
    cudaMalloc(&buffers[1], input2_size * sizeof(float));
    cudaMalloc(&buffers[2], output1_size * sizeof(float));
    cudaMalloc(&buffers[3], output2_size * sizeof(float));
    cudaMalloc(&buffers[4], output3_size * sizeof(float));

    cudaMemcpy(buffers[0], input_points, input1_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(buffers[1], input_p, input2_size * sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto start = std::chrono::high_resolution_clock::now();
    context->enqueueV2(buffers, stream, nullptr);
    cudaStreamSynchronize(stream);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> duration = end - start;
    std::cout << "推理时间: " << duration.count() << " 毫秒" << std::endl;

    cudaMemcpy(output_features, buffers[2], output1_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_weight, buffers[4], output3_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_p2v, buffers[3], output2_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaStreamDestroy(stream);
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    cudaFree(buffers[2]);
    cudaFree(buffers[3]);
    cudaFree(buffers[4]);
}

int main(int argc, char** argv) {
    try {
        // 初始化TensorRT
        IRuntime* runtime = nullptr;
        ICudaEngine* engine = loadEngine("/home/server/xec/voxelmap_p2v/best_voxel.engine", runtime);
        if (!engine) throw std::runtime_error("加载TensorRT引擎失败");
        IExecutionContext* context = engine->createExecutionContext();

        // 准备输入输出缓冲区
        float input_points[1 * 50 * 3];
        float input_p[1 * 3];
        float output_features[1 * 1027];
        float output_weight[1 * 1];
        float output_p2v[1 * 3];

        readCSVData("/home/server/xec/voxelmap_p2v/data_new.csv", input_p, input_points);
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> voxel_points = Eigen::Map<Eigen::Matrix<float, 50, 3, Eigen::RowMajor>>(input_points);
        Eigen::Vector3f p = Eigen::Map<Eigen::Vector3f>(input_p);

        std::memcpy(input_points, voxel_points.data(), sizeof(float) * 50 * 3);
        std::memcpy(input_p, p.data(), sizeof(float) * 3);

        // 打印输入
        printArray("输入P", input_p, 3);
        // printPointsMatrix(input_points, "体素点示例");
        // 执行推理
        runInference(context, input_points, input_p, output_features, output_weight, output_p2v);

        // 打印输出 (只打印weight和p2v)
        std::cout << "输出Weight: " << output_weight[0] << std::endl;
        printArray("输出P2V", output_p2v, 3);

        context->destroy();
        engine->destroy();
        runtime->destroy();
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}