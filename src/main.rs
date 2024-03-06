use ndarray::{Array4};
use crate::onnx_runtime::onnxruntime::{parse_input_tensor};
use crate::operations::{Input, Output};
use crate::operations::soft_max::SoftMax;

mod onnx_proto3;
mod node;
mod graph;
mod onnx_runtime;
mod operations;


fn main() {

    //Testing Mnist-7
    let mut dep_graph = onnx_runtime::onnxruntime::get_computational_graph("src/mnist-7/model.onnx".to_string());
    let arr = parse_input_tensor("src/mnist-7/test_data_set_0/input_0.pb".to_string()).unwrap();
    let out = dep_graph.run(arr).unwrap();
    println!("Result from graph: ");
    let graph_result = match out {
        Output::TensorD(array) => array,
        _ => panic!("Errore output")
    };
    println!("{}", graph_result.clone());

    println!("Test_Data_set/Output_data: ");
    let arr = parse_input_tensor("src/mnist-7/test_data_set_0/output_0.pb".to_string()).unwrap();
    let result = match arr {
        Input::TensorD(array) => array,
        _ => panic!("Errore output")
    };
    println!("{}", result.clone());

    //Testing Modified Mnist-7
    println!("\nTesting add node");
    let operation = SoftMax::new();
    dep_graph.add_node("Softmax_213".to_string(), Box::new(operation), &["Plus214".to_string()]);
    let arr = parse_input_tensor("src/mnist-7/test_data_set_0/input_0.pb".to_string()).unwrap();
    let out = dep_graph.run(arr).unwrap();
    let result = out.into_raw_vec().unwrap();
    result.into_iter().for_each(|val| print!("{} ", val));
    println!();

    //Testing Removing Node Mnist-7
    println!("\nTesting remove previous created node");
    dep_graph.remove_node("Softmax_213".to_string()).unwrap();
    let arr = parse_input_tensor("src/mnist-7/test_data_set_0/input_0.pb".to_string()).unwrap();
    let out = dep_graph.run(arr).unwrap();
    let result = out.into_raw_vec().unwrap();
    result.into_iter().for_each(|val| print!("{} ", val));
    println!();

    //Testing Removing Node in the middle Mnist-7
    println!("\nTesting Add - Modify sequence");
    let operation = SoftMax::new();
    dep_graph.add_node("Softmax_213".to_string(), Box::new(operation), &["Times212".to_string()]);
    dep_graph.modify_node_dep("Plus214".to_string(), Some("Times212".to_string()), Some("Softmax_213".to_string())).unwrap();
    let arr = parse_input_tensor("src/mnist-7/test_data_set_0/input_0.pb".to_string()).unwrap();
    let out = dep_graph.run(arr).unwrap();
    let result = out.into_raw_vec().unwrap();
    result.into_iter().for_each(|val| print!("{} ", val));
    println!();

    //Testing Random Input on the same Network
    println!("\nUsing random input on the same network");
    let tmp_array: Vec<f32> = Array4::from_elem((1,1,28,28), 0.7).into_raw_vec();
    let net_input = Input::from_raw_vec(tmp_array, &[1, 1, 28, 28]).unwrap();
    let out = dep_graph.run(net_input).unwrap();
    let result = out.into_raw_vec().unwrap();
    result.into_iter().for_each(|val| print!("{} ", val));
    println!();

    //Testing Googlenet
    let mut dep_graph = onnx_runtime::onnxruntime::get_computational_graph("src/googlenet/model.onnx".to_string());
    let arr = parse_input_tensor("src/googlenet/test_data_set_0/input_0.pb".to_string()).unwrap();
    let out = dep_graph.run(arr).unwrap();
    println!("Result from graph: ");
    let graph_result = match out {
        Output::TensorD(array) => array,
        _ => panic!("Errore output")
    };
    println!("{}", graph_result.clone());


    println!("Test_Data_set/Output_data: ");
    let arr = parse_input_tensor("src/googlenet/test_data_set_0/output_0.pb".to_string()).unwrap();
    let result = match arr {
        Input::TensorD(array) => array,
        _ => panic!("Errore output")
    };
    println!("{}", result.clone());
}

