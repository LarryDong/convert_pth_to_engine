#include <torch/script.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

using namespace std;


const string model_file = "/home/larry/codeGit/implict_voxel/src/feat_voxel_map/checkpoint/best_voxel.pt";
const string data_file = "/home/larry/featVoxelMap_ws/data/data_50.csv";


int main() {
    // 1. 设置设备（CPU/CUDA）
    torch::Device device(torch::kCPU);
    torch::jit::script::Module module;
    module = torch::jit::load(model_file, device);
    module.eval();
    cout << "Model loaded." << endl;

    // read csv file and convert
    std::ifstream file(data_file);
    std::string line;
    std::vector<std::vector<float>> data;

    // Read all lines
    while (std::getline(file, line)) {
        std::vector<float> row;
        std::stringstream ss(line);
        std::string value;
        
        while (std::getline(ss, value, ',')) {
            row.push_back(std::stof(value));
        }
        data.push_back(row);
    }

    cout << "Test data loaded." << endl;

    // Separate p (first line) and points (remaining 50 lines)
    std::vector<float> p = data[0];
    std::vector<std::vector<float>> points(data.begin() + 1, data.end());

    // Convert to tensors
    torch::Tensor p_tensor = torch::tensor(p, torch::dtype(torch::kFloat32)).unsqueeze(0);

    // 更安全的points处理方式
    std::vector<float> flattened_points;
    for (const auto& row : points) {
        flattened_points.insert(flattened_points.end(), row.begin(), row.end());
    }
    torch::Tensor points_tensor = torch::tensor(flattened_points, torch::dtype(torch::kFloat32)).reshape({1, static_cast<long>(points.size()), static_cast<long>(points[0].size())});

    // Test data
    cout << "Loaded P: " << p_tensor << endl;
    cout << "points_tensor: " << points_tensor << endl;


    // Test output:
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(points_tensor);
    inputs.push_back(p_tensor);

    // 修改后的前向传播部分
    try {
        // 先不假设输出是元组，直接获取输出
        auto output = module.forward(inputs);
        assert(output.isTuple());
            
        auto tuple_ptr = output.toTuple();
        const auto& elements = tuple_ptr->elements();

        // Output 1: P2V
        torch::Tensor tensor1 = elements[0].toTensor().contiguous().to(torch::kCPU);
        std::vector<float> vec1(tensor1.data_ptr<float>(), tensor1.data_ptr<float>() + tensor1.numel());

        cout <<"--> P2V: " << endl;
        for(auto x:vec1)
            cout << x << ", ";
        cout << endl;

        // Output 2: weight.
        torch::Tensor tensor2 = elements[1].toTensor();
        float scalar_value = tensor2.item<float>();
        cout <<"--> Weight: " << scalar_value << endl;

        // Output 3: global feature:
        torch::Tensor tensor3 = elements[2].toTensor().contiguous().to(torch::kCPU);
        std::vector<float> vec3(tensor3.data_ptr<float>(),tensor3.data_ptr<float>() + tensor3.numel());

    } catch (const c10::Error& e) {
        std::cerr << "Error during inference: " << e.what() << "\n";
        return -1;
    }

    return 0;
}
