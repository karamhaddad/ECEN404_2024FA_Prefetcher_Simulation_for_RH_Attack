#include "cache.h"
#include "tensorflow/c/c_api.h"
#include <vector>
#include <bitset>
#include <memory>
#include <ostream>
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/graph/graph.h"
#include <iostream>

/*  Howdy, in this file, I have created my prefetcher. There are many like it, but this one is mine.
    This is an LSTM prefetcher that takes in the previous 4 block deltas, and predicts the next one,
    which then get converted into the next address, and sent to the beyond to be useful.
*/

// Make sure tensorflow version is compatable with your model
using namespace tensorflow;

static std::vector<std::vector<float>> previous_inputs_vector(3); // Buffer for the previous 3 input vectors

// This is now unused, but it was a variable we used to keep track of the cycel for predictions
unsigned int Cycle = 0;          // Global variable for Cycle 

tensorflow::SessionOptions session_options_ = tensorflow::SessionOptions();  // Configuration Options for TF session
tensorflow::RunOptions run_options_ = tensorflow::RunOptions();              // Run-time options for TF session 
tensorflow::SavedModelBundle model_ = tensorflow::SavedModelBundle();        // Object where saved model is stored
std::unique_ptr<Session> session;   
const std::string export_dir = "/mnt/md0/jupyter/students/nathanielbush/block_offset_v1/model";  // Directory for model

/// Tensor building function for 4 timesteps
// This function creates the input tensor that is being fed intot he model

tensorflow::Tensor build_input_tensor(const std::vector<std::vector<float>>& prev_inputs, const std::vector<float>& curr_input) 
{
    // Create a tensor with shape 
    // (1 sample, 4 timesteps, 7 features)
    tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, 4, 7}));
    auto input_tensor_map = input_tensor.tensor<float, 3>();

    // Check 4 timesteps and 7 features each
    for (int timestep = 0; timestep < 3; ++timestep) {

        for (int i = 0; i < 7; ++i) {
            
            input_tensor_map(0, timestep, i) = prev_inputs[timestep][i]; // Previous inputs at timesteps 0, 1, 2
        
        }
    }

    // fill tensor with the current input
    for (int i = 0; i < 7; ++i) {
        input_tensor_map(0, 3, i) = curr_input[i]; // Current input at timestep 3
    }

    return input_tensor;
}


/* This functino takes in the integer form of the block delta, and converts it into 
    vector form, this makes it easier to break everything up into small chunks
*/
std::vector<float> int_to_binary_vector(uint64_t value, int bit_size) 
{

    std::vector<float> binary_vector(bit_size, 0);

    for (int i = 0; i < bit_size; ++i) {

        binary_vector[bit_size - i - 1] = (value >> i) & 1;

    }
    return binary_vector;
}

/* This function takes in the input from the current address, and extracts the block number
    The block number was chosen as it contains the most value information, and isn't too long.
*/
std::vector<float> process_input1(uint64_t addr) 
{
    // Get the block offset and block number
    uint64_t block_offset = addr % 4096; // Get the bottom 12 bits
    uint64_t block_number = block_offset >> 6; // Block number (bits 6-11)

    // Sanity check make sure it is 7 bits
    std::vector<float> input_vector(7, 0);
    for (int i = 0; i < 7; ++i) {
        input_vector[6 - i] = (block_number >> i) & 1; // Extract 7 bits
    }

    return input_vector; // Return 7-bit vector representing block number
}

/* This function converts the block delta into two's compliment, checking that no page
    boundaries were crossed. If a page boundary is crossed, then it returns null.
    These output bits basically get added onto the current page address to get the whole
    next prediction address.
*/
uint64_t process_output(uint64_t current_addr, const std::vector<float>& output_bits)
{
    // Initialize block delta
    int64_t block_delta = 0;

    // Convert to integer (easier for math than using bitwise notation)

    for (size_t i = 0; i < output_bits.size(); ++i) {

        block_delta |= static_cast<int64_t>(output_bits[i]) << (output_bits.size() - 1 - i);

    }

    // INitialize predicted_address
    uint64_t predicted_address;

    // Check if in twos compliment
    int sign_bit = static_cast<int>(output_bits[0]);  // The first bit is the sign bit

    // If in twos compliment and negative, subtract the delta
    if (sign_bit == 1) {

        // Convert two's complement to positive magnitude
        block_delta = block_delta & 0b0111111;  // Remove the sign bit
        block_delta = ~block_delta + 1;  // Invert bits and add 1 (two's complement to magnitude)
        predicted_address = current_addr - (block_delta << 6);  // Subtract the magnitude from address

    } else {

        // If positive, add to address
        predicted_address = current_addr + (block_delta << 6);
    }

    // Get the original page number (bits 64-52)
    uint64_t current_page_number = (current_addr >> 52) & 0x1FFF;  // Check bits 64-52

    // Extract the new page number after adding/subtracting the delta
    uint64_t new_page_number = (predicted_address >> 52) & 0x1FFF;

    // Check if the predicted address stays within the same page
    if (new_page_number != current_page_number) {
        return 0;  // Do not issue prefetch if it crosses a page boundary
    }

    return predicted_address;  // Return the valid prefetch address
}



// Convert the oiutput tensoor into vector form for processing
std::vector<float> tensor_to_vector(const Tensor& output_tensor) 
{
    auto flat = output_tensor.flat<float>();
    return std::vector<float>(flat.data(), flat.data() + flat.size());
}

// initialzie the prefetcher
void CACHE::prefetcher_initialize() {
    // Attempts to load the model from the directory ({"serve"} sets to run only)
    auto status = tensorflow::LoadSavedModel(session_options_, 
                                                    run_options_, 
                                                    export_dir, 
                                                    {"serve"}, 
                                                    &model_);

    if (!status.ok()) {
        std::cerr << "Failed to load saved model: " << status.ToString() << std::endl;
        return;
    }
}

void CACHE::prefetcher_cycle_operate() {}

void CACHE::prefetcher_final_stats() {}

uint32_t CACHE::prefetcher_cache_fill(uint64_t addr, uint32_t set, uint32_t way, uint8_t prefetch, uint64_t evicted_addr, uint32_t metadata_in)
{
  return metadata_in;
}

/* This function is what runs every time the cache operates, it is the 'meat' of the prefetcher, 
    and all the other functions should be called somewhere in here. 
    The basic algorith goes:
    1) create the input tensor from the current and previous addresses
    2) run the model with said input tensor
    3) convert the output tensor into a useable form
    4) create the next predicted address with the output tensor
    5) prefetch
    
*/
uint32_t CACHE::prefetcher_cache_operate(uint64_t addr, uint64_t ip, uint8_t cache_hit, bool useful_prefetch, uint8_t type, uint32_t metadata_in) 
{
    // Prepare inputs for the tensor

    // Creates a current input vector 
    std::vector<float> current_input_vector = process_input1(addr);

    // Check if previous inputs are empty (initialize with zeros if necessary)
    // Should only run for the first 3 prefetches
    if (previous_inputs_vector[0].empty()) {
        
        for (int i = 0; i < 3; ++i) {
            previous_inputs_vector[i].resize(7, 0.0);
        }
    }

    // Create input tensor
    tensorflow::Tensor input_tensor = build_input_tensor(previous_inputs_vector, current_input_vector);

    // Update the previous input vectors for the next run
    previous_inputs_vector.erase(previous_inputs_vector.begin()); // Remove the oldest timestep
    previous_inputs_vector.push_back(current_input_vector);       // Add the current input as the newest timestep

    // Run the tensorflow model

    // Prepare the input for the TF session using the correct node names
    std::vector<std::pair<string, tensorflow::Tensor>> inputs = {{"serving_default_lstm_input", input_tensor}};
    
    // Create an output tensor for the session output
    std::vector<Tensor> outputs;
    tensorflow::Status run_status = model_.session->Run(inputs, {"StatefulPartitionedCall"}, {}, &outputs);

    if (!run_status.ok()) {
        std::cerr << "Failed to run TensorFlow session: " << run_status.ToString() << "\n";
        return metadata_in;
    }

    // Convert the output tensor to a binary vector and process it
    std::vector<float> output_vector = tensor_to_vector(outputs[0]);
    std::vector<float> binary_output(output_vector.size());
    std::transform(output_vector.begin(), output_vector.end(), binary_output.begin(), [](float value) {
        return value > 0.5 ? 1 : 0;
    });

    // Process the predicted block offset and check if it's valid for prefetching
    uint64_t predicted_address = process_output(addr, binary_output);
    
    if (predicted_address != 0) {
        prefetch_line(predicted_address, true, metadata_in);  // Issue a prefetch only if valid
    }

    return metadata_in;
}
