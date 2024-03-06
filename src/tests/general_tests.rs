
#[cfg(test)]
pub mod general_tests {

use std::cmp::max;
use std::collections::HashSet;
use std::fs::File;
use std::io::Read;
use ndarray::{arr1, Array, Array3, Dim, Dimension, Ix4, Shape};
use ndarray::{Array1, Array2, Array4, ArrayD, Ix2, IxDyn};
use protobuf::Message;
use crate::operations::add::Add;
use crate::operations::averagepool::AveragePool;
use crate::operations::gemm::Gemm;
use crate::operations::local_response_normalization::LRN;
use crate::operations::maxpool::MaxPool;
use crate::onnx_proto3::{ModelProto, NodeProto};
use crate::onnx_runtime;
use crate::operations::{Compute, Input, Output};
use crate::operations::conv::{Conv};
use crate::operations::reshape::Reshape;

    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }

    #[test]
    fn test_max_pool_stride1(){
        let mut max_pool_node = MaxPool::new(Some(Shape::from(Dim([7, 7]))),
                                             Some(arr1(&[0,0,0,0])), Some(arr1(&[1,1])));
        let mut prova = Array4::from_elem((1, 3, 9, 9), 0.0);
        let mut comparison = Array4::from_elem((1, 3, 3, 3), 0.0);
        for i in 0..3 {
            prova[[0, i, 0, 0]] = 1.0;
            prova[[0, i, 0, 8]] = 1.0;
            prova[[0, i, 8, 0]] = 1.0;
            prova[[0, i, 8, 8]] = 1.0;
            comparison[[0, i, 0, 0]] = 1.0;
            comparison[[0, i, 0, 2]] = 1.0;
            comparison[[0, i, 2, 0]] = 1.0;
            comparison[[0, i, 2, 2]] = 1.0;
        }
        let input_d = Input::TensorD(prova.into_shape(IxDyn(&[1, 3, 9, 9])).unwrap());
        let result = match max_pool_node.compute(input_d) {
            Output::TensorD(arr) => arr.into_dimensionality::<Ix4>().unwrap(),
            _ => panic!("Wrong result")
        };
        assert_eq!(result, comparison);
    }

    #[test]
    fn test_max_pool_stride2(){
        let mut kernel_vec:[usize; 2] = [0; 2];
        let mut input: Vec<i64> = vec![7, 7];
        let input = input.into_iter().map(|val| val as usize).collect::<Vec<usize>>();
        kernel_vec.copy_from_slice(&input);
        let mut max_pool_node = MaxPool::new(Some(Shape::from(Dim(kernel_vec) )),
                                             Some(arr1(&[0,0,0,0])), Some(arr1(&[2,2])));
        let mut prova = Array4::from_elem((1, 3, 9, 9), 0.0);
        let mut comparison = Array4::from_elem((1, 3, 2, 2), 0.0);
        for i in 0..3 {
            prova[[0, i, 0, 0]] = 1.0;
            prova[[0, i, 0, 8]] = 2.0;
            prova[[0, i, 8, 0]] = 1.5;
            prova[[0, i, 8, 8]] = 3.0;
            prova[[0, i, 4, 4]] = 0.5;
            comparison[[0, i, 0, 0]] = 1.0;
            comparison[[0, i, 0, 1]] = 2.0;
            comparison[[0, i, 1, 0]] = 1.5;
            comparison[[0, i, 1, 1]] = 3.0;
        }
        let input_d = Input::TensorD(prova.into_shape(IxDyn(&[1, 3, 9, 9])).unwrap());
        let result = match max_pool_node.compute(input_d) {
            Output::TensorD(arr) => arr.into_dimensionality::<Ix4>().unwrap(),
            _ => panic!("Wrong result")
        };
        assert_eq!(result, comparison);
    }

    #[test]
    fn test_average_pool_stride2(){
        let mut avg_pool_node = AveragePool::new(Some(Shape::from(Dim([7, 7]))),
                                                 Some(arr1(&[0,0,0,0])), Some(arr1(&[2,2])));
        let mut prova = Array4::from_elem((1, 3, 9, 9), 1.0);
        let mut comparison = Array4::from_elem((1, 3, 2, 2), 1.0);
        let input_d = Input::TensorD(prova.into_shape(IxDyn(&[1, 3, 9, 9])).unwrap());
        let result = match avg_pool_node.compute(input_d) {
            Output::TensorD(arr) => arr.into_dimensionality::<Ix4>().unwrap(),
            _ => panic!("Wrong result")
        };
        assert_eq!(result, comparison);
    }

    #[test]
    fn test_lrn(){
        // Create two test Array4 instances (representing images)
        let vec1 = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
            19.0, 20.0, 21.0, 22.0, 23.0, 24.0
        ];
        let vec2 = vec![0.99795485,  1.9948157,   2.9902866,   3.9840744,   4.9758863,   5.965434,
                        6.95243,     7.936592,    8.917418,    9.894319,   10.86688,    11.834699,
                        12.797375,   13.754522,   14.705759,   15.650717,   16.844427,   17.81153,
                        18.774221,   19.732246,   20.685343,   21.633266,   22.57577,    23.512613
        ];
        let test_data_1: Array4<f32> = Array4::from_shape_vec(
            Shape::from(Dim([1, 3, 2, 4])), vec1).unwrap();

        let test_data_2: Array4<f32> = Array4::from_shape_vec(
            Shape::from(Dim([1, 3, 2, 4])), vec2).unwrap();

        let mut lrn_node = LRN::new(0.0001, 0.75, 1.0, 3);

        let input_d = Input::TensorD(test_data_1.into_shape(IxDyn(&[1, 3, 2, 4])).unwrap());
        let result = match lrn_node.compute(input_d) {
            Output::TensorD(arr) => arr.into_dimensionality::<Ix4>().unwrap(),
            _ => panic!("Wrong result")
        };
        assert_eq!(result, test_data_2);
    }

    #[test]
    fn test_max_pool_from_python_results(){
        let vec1 = vec![
            1.0, 2.0, 3.0, 4.0, 5.0,
            6.0, 7.0, 8.0, 9.0, 10.0,
            11.0, 12.0, 13.0, 14.0, 15.0,
            16.0, 17.0, 18.0, 19.0, 20.0,
            21.0, 22.0, 23.0, 24.0, 25.0
        ];
        let vec2 = vec![
            13.0, 14.0, 15.0, 15.0, 15.0,
            18.0, 19.0, 20.0, 20.0, 20.0,
            23.0, 24.0, 25.0, 25.0, 25.0,
            23.0, 24.0, 25.0, 25.0, 25.0,
            23.0, 24.0, 25.0, 25.0, 25.0
        ];
        let test_data_1: Array4<f32> = Array4::from_shape_vec(
            Shape::from(Dim([1, 1, 5, 5])), vec1).unwrap();

        let test_data_2: Array4<f32> = Array4::from_shape_vec(
            Shape::from(Dim([1, 1, 5, 5])), vec2).unwrap();

        let mut max_pool_node = MaxPool::new(Some(Shape::from(Dim([5, 5]) )),
                                             Some(arr1(&[2,2,2,2])), Some(arr1(&[1,1])));

        let input_d = Input::TensorD(test_data_1.into_shape(IxDyn(&[1, 1, 5, 5])).unwrap());
        let result = match max_pool_node.compute(input_d) {
            Output::TensorD(arr) => arr.into_dimensionality::<Ix4>().unwrap(),
            _ => panic!("Wrong result")
        };
        assert_eq!(result, test_data_2);
    }

    #[test]
    fn test_max_pool_parsing(){
        let mut input_onnx = File::open("src/tests/gender_googlenet.onnx").unwrap();
        //Onnx file into byte array
        let mut byte_array = Vec::<u8>::new();
        input_onnx.read_to_end(&mut byte_array).unwrap();
        //Parsing del byte array nella struttura onnx_proto3.rs
        let model: ModelProto = match Message::parse_from_bytes(&byte_array) {
            Ok(model) => model,
            Err(err) => {
                eprintln!("Failed to parse the ONNX googlenet: {}", err);
                return;
            }
        };
        let graph = model.get_graph();
        //Estrazione dei nodi dal protoGrafo
        let nodes = graph.get_node();
        let mut max_nodes: Vec<MaxPool> = Vec::new();
        let mut count = 0;

        for node in nodes.iter(){
            if node.op_type == "MaxPool"{
                count+=1;
                max_nodes.push(MaxPool::parse_from_proto_node(node.attribute.as_slice()));
            }
        }
        for node in max_nodes.iter(){
            print!("Kernel Shape: ");
            node.kernel_shape.raw_dim().slice().iter().for_each(|el| print!("{}", el));
            println!();
            print!("Pads: ");
            node.pads.iter().for_each(|el| print!("{}", *el));
            println!();
            print!("Strides: ");
            node.strides.iter().for_each(|el| print!("{}", *el));
            println!();
            println!();
        }
        assert_eq!(max_nodes.len(), count);
    }

#[test]
fn test_conv_parsing(){
    let mut input_onnx = File::open("src/tests/gender_googlenet.onnx").unwrap();
    //Onnx file into byte array
    let mut byte_array = Vec::<u8>::new();
    input_onnx.read_to_end(&mut byte_array).unwrap();
    //Parsing del byte array nella struttura onnx_proto3.rs
    let model: ModelProto = match Message::parse_from_bytes(&byte_array) {
        Ok(model) => model,
        Err(err) => {
            eprintln!("Failed to parse the ONNX googlenet: {}", err);
            return;
        }
    };
    let graph = model.get_graph();
    //Estrazione dei nodi dal protoGrafo
    let nodes = graph.get_node();
    let mut max_nodes: Vec<Conv> = Vec::new();
    let mut count = 0;

    for node in nodes.iter(){
        if node.op_type == "Conv"{
            count+=1;
            max_nodes.push(Conv::parse_from_proto_node(node.attribute.as_slice()));
        }
    }
    for node in max_nodes.iter(){
        print!("autopad: ");
        print!("{}", node.autopad);
        println!();
        print!("strides: ");
        node.strides.iter().for_each(|el| print!("{}", *el));
        println!();
        print!("pads: ");
        node.pads.iter().for_each(|el| print!("{}", *el));
        println!();
        print!("dilations: ");
        node.dilations.iter().for_each(|el| print!("{}", *el));
        println!();
        print!("group: ");
        print!("{}", node.group);
        println!();
        print!("kernel_shape: ");
        node.kernel_shape.raw_dim().slice().iter().for_each(|el| print!("{}", el));
        println!();
        println!();
    }
    assert_eq!(max_nodes.len(), count);
}

    #[test]
    fn test_lrn_parsing(){
        let mut input_onnx = File::open("src/tests/gender_googlenet.onnx").unwrap();
        //Onnx file into byte array
        let mut byte_array = Vec::<u8>::new();
        input_onnx.read_to_end(&mut byte_array).unwrap();
        //Parsing del byte array nella struttura onnx_proto3.rs
        let model: ModelProto = match Message::parse_from_bytes(&byte_array) {
            Ok(model) => model,
            Err(err) => {
                eprintln!("Failed to parse the ONNX googlenet: {}", err);
                return;
            }
        };
        let graph = model.get_graph();
        //Estrazione dei nodi dal protoGrafo
        let nodes = graph.get_node();
        let mut lrn_nodes: Vec<LRN> = Vec::new();
        let mut count = 0;

        for node in nodes.iter(){
            if node.op_type == "LRN"{
                count+=1;
                lrn_nodes.push(LRN::parse_from_proto_node(node.attribute.as_slice()));
            }
        }
        for node in lrn_nodes.iter(){
            println!("{} - {} - {} - {}", node.alpha, node.beta, node.bias, node.size);
        }
        assert_eq!(lrn_nodes.len(), count);
    }

    #[test]
    fn test_gemm_parsing(){
        let mut input_onnx = File::open("src/tests/gender_googlenet.onnx").unwrap();
        //Onnx file into byte array
        let mut byte_array = Vec::<u8>::new();
        input_onnx.read_to_end(&mut byte_array).unwrap();
        //Parsing del byte array nella struttura onnx_proto3.rs
        let model: ModelProto = match Message::parse_from_bytes(&byte_array) {
            Ok(model) => model,
            Err(err) => {
                eprintln!("Failed to parse the ONNX googlenet: {}", err);
                return;
            }
        };
        let graph = model.get_graph();
        //Estrazione dei nodi dal protoGrafo
        let nodes = graph.get_node();
        let mut gemm_nodes: Vec<Gemm> = Vec::new();
        let mut count = 0;

        for node in nodes.iter(){
            if node.op_type == "Gemm"{
                count+=1;
                gemm_nodes.push(Gemm::parse_from_proto_node(node.attribute.as_slice()));
            }
        }
        for node in gemm_nodes.iter(){
            println!("{} - {} - {} - {}", node.alpha, node.beta, node.trans_a, node.trans_b);
        }
        assert_eq!(gemm_nodes.len(), count);
    }

    #[test]
    fn test_add(){
        let vec1 = vec![
            1.0, 2.0, 3.0, 4.0, 5.0,
            6.0, 7.0, 8.0, 9.0, 10.0,
            11.0, 12.0, 13.0, 14.0, 15.0,
            16.0
        ];
        let vec2 = vec![
            13.0, 14.0, 15.0, 15.0
        ];

        //Retrieved from python NumPy
        let vec3 = vec![14.0, 15.0, 16.0, 17.0, 19.0,
                        20.0, 21.0, 22.0, 24.0, 25.0, 26.0, 27.0,
                        28.0, 29.0, 30.0, 31.0];

        let to_compare = Array4::from_shape_vec(
            Shape::from(Dim([1, 4, 2, 2])), vec3).unwrap();

        let test_data_1: Array4<f32> = Array4::from_shape_vec(
            Shape::from(Dim([1, 4, 2, 2])), vec1).unwrap();

        let test_data_2: Array3<f32> = Array3::from_shape_vec(
            Shape::from(Dim([4, 1, 1])), vec2).unwrap();

        let mut add_node = Add::new();

        let input_1 = test_data_1.into_shape(IxDyn(&[1, 4, 2, 2])).unwrap();
        let input_2 = test_data_2.into_shape(IxDyn(&[4, 1, 1])).unwrap();
        let input_d = Input::Tensor4List(Vec::from([input_1, input_2]));
        let result = match add_node.compute(input_d) {
            Output::TensorD(arr) => arr.into_dimensionality::<Ix4>().unwrap(),
            _ => panic!("Wrong result")
        };
        assert_eq!(result, to_compare);
    }

#[test]
fn test_conv_with_padding_pads(){
    let mut kernel_vec:[usize; 2] = [0; 2];
    let mut input: Vec<i64> = vec![3, 3];
    let input = input.into_iter().map(|val| val as usize).collect::<Vec<usize>>();
    kernel_vec.copy_from_slice(&input);
    let mut conv_node = Conv::new(None, Some(arr1(&[1, 1])), Some(1), Some(Shape::from(Dim(kernel_vec))), Some(arr1(&[1, 1, 1, 1])), Some(arr1(&[1, 1])));

    // Define the input values
    let x_values = vec![
        0.0, 1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0, 9.0,
        10.0, 11.0, 12.0, 13.0, 14.0,
        15.0, 16.0, 17.0, 18.0, 19.0,
        20.0, 21.0, 22.0, 23.0, 24.0,
    ];

    // Create a 1x1x5x5 array
    let x = Array::from_shape_vec((1, 1, 5, 5), x_values).unwrap();
    let mut weight = Array4::from_elem((1, 1, 3, 3), 1.0);

    // Define the expected output
    let y_with_padding_values = vec![
        12.0, 21.0, 27.0, 33.0, 24.0, 33.0, 54.0, 63.0, 72.0, 51.0, 63.0, 99.0, 108.0, 117.0, 81.0, 93.0, 144.0, 153.0,
        162.0, 111.0, 72.0, 111.0, 117.0, 123.0, 84.0,
    ];

    // Create a 1x1x5x5 array
    let y_with_padding = Array::from_shape_vec((1, 1, 5, 5), y_with_padding_values).unwrap();
    let mut vec_of_arrayd: Vec<ArrayD<f32>> = Vec::new();
    vec_of_arrayd.push(x.into_dyn());
    vec_of_arrayd.push(weight.into_dyn());
    let mut input_d = Input::Tensor4List(vec_of_arrayd);
    let result = match conv_node.compute(input_d) {
        Output::TensorD(arr) => arr.into_dimensionality::<Ix4>().unwrap(),
        _ => panic!("Wrong result")
    };
    println!("{}", result);
    assert_eq!(result, y_with_padding);
}

#[test]
fn test_conv_with_padding_pads_strides(){
    let mut kernel_vec:[usize; 2] = [0; 2];
    let mut input: Vec<i64> = vec![3, 3];
    let input = input.into_iter().map(|val| val as usize).collect::<Vec<usize>>();
    kernel_vec.copy_from_slice(&input);
    let mut conv_node = Conv::new(None, Some(arr1(&[1, 1])), Some(1), Some(Shape::from(Dim(kernel_vec))), Some(arr1(&[1, 1, 1, 1])), Some(arr1(&[2, 2])));

    // Define the input values
    let x_values = vec![
        0.0, 1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0, 9.0,
        10.0, 11.0, 12.0, 13.0, 14.0,
        15.0, 16.0, 17.0, 18.0, 19.0,
        20.0, 21.0, 22.0, 23.0, 24.0,
        25.0, 26.0 ,27.0, 28.0, 29.0,
        30.0, 31.0, 32.0, 33.0, 34.0
    ];

    // Create a 1x1x5x5 array
    let x = Array::from_shape_vec((1, 1, 7, 5), x_values).unwrap();
    let mut weight = Array4::from_elem((1, 1, 3, 3), 1.0);

    // Define the expected output
    let y_with_padding_values = vec![
        12.0, 27.0, 24.0, 63.0, 108.0, 81.0, 123.0, 198.0, 141.0, 112.0, 177.0, 124.0
    ];

    // Create a 1x1x5x5 array
    let y_with_padding = Array::from_shape_vec((1, 1, 4, 3), y_with_padding_values).unwrap();
    let mut vec_of_arrayd: Vec<ArrayD<f32>> = Vec::new();
    vec_of_arrayd.push(x.into_dyn());
    vec_of_arrayd.push(weight.into_dyn());
    let mut input_d = Input::Tensor4List(vec_of_arrayd);
    let result = match conv_node.compute(input_d) {
        Output::TensorD(arr) => arr.into_dimensionality::<Ix4>().unwrap(),
        _ => panic!("Wrong result")
    };
    println!("{}", result);
    assert_eq!(result, y_with_padding);
}

#[test]
fn test_conv_with_padding_pads_along_1dimension_strides(){
    let mut kernel_vec:[usize; 2] = [0; 2];
    let mut input: Vec<i64> = vec![3, 3];
    let input = input.into_iter().map(|val| val as usize).collect::<Vec<usize>>();
    kernel_vec.copy_from_slice(&input);
    let mut conv_node = Conv::new(None, Some(arr1(&[1, 1])), Some(1), Some(Shape::from(Dim(kernel_vec))), Some(arr1(&[1, 0, 1, 0])), Some(arr1(&[2, 2])));

    // Define the input values
    let x_values = vec![
        0.0, 1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0, 9.0,
        10.0, 11.0, 12.0, 13.0, 14.0,
        15.0, 16.0, 17.0, 18.0, 19.0,
        20.0, 21.0, 22.0, 23.0, 24.0,
        25.0, 26.0 ,27.0, 28.0, 29.0,
        30.0, 31.0, 32.0, 33.0, 34.0
    ];

    // Create a 1x1x5x5 array
    let x = Array::from_shape_vec((1, 1, 7, 5), x_values).unwrap();
    let mut weight = Array4::from_elem((1, 1, 3, 3), 1.0);

    // Define the expected output
    let y_with_padding_values = vec![
        21.0, 33.0,
        99.0, 117.0,
        189.0, 207.0,
        171.0, 183.0
    ];

    // Create a 1x1x4x2 array
    let y_with_padding = Array::from_shape_vec((1, 1, 4, 2), y_with_padding_values).unwrap();
    let mut vec_of_arrayd: Vec<ArrayD<f32>> = Vec::new();
    vec_of_arrayd.push(x.into_dyn());
    vec_of_arrayd.push(weight.into_dyn());
    let mut input_d = Input::Tensor4List(vec_of_arrayd);
    let result = match conv_node.compute(input_d) {
        Output::TensorD(arr) => arr.into_dimensionality::<Ix4>().unwrap(),
        _ => panic!("Wrong result")
    };
    println!("{}", result);
    assert_eq!(result, y_with_padding);
}

#[test]
fn test_conv_without_padding_pads_strides(){
    let mut kernel_vec:[usize; 2] = [0; 2];
    let mut input: Vec<i64> = vec![3, 3];
    let input = input.into_iter().map(|val| val as usize).collect::<Vec<usize>>();
    kernel_vec.copy_from_slice(&input);
    let mut conv_node = Conv::new(None, Some(arr1(&[1, 1])), Some(1), Some(Shape::from(Dim(kernel_vec))), Some(arr1(&[0, 0, 0, 0])), Some(arr1(&[2, 2])));

    // Define the input values
    let x_values = vec![
        0.0, 1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0, 9.0,
        10.0, 11.0, 12.0, 13.0, 14.0,
        15.0, 16.0, 17.0, 18.0, 19.0,
        20.0, 21.0, 22.0, 23.0, 24.0,
        25.0, 26.0 ,27.0, 28.0, 29.0,
        30.0, 31.0, 32.0, 33.0, 34.0
    ];

    // Create a 1x1x5x5 array
    let x = Array::from_shape_vec((1, 1, 7, 5), x_values).unwrap();
    let mut weight = Array4::from_elem((1, 1, 3, 3), 1.0);

    // Define the expected output
    let y_with_padding_values = vec![
        54.0, 72.0,
        144.0, 162.0,
        234.0, 252.0
    ];

    // Create a 1x1x5x5 array
    let y_with_padding = Array::from_shape_vec((1, 1, 3, 2), y_with_padding_values).unwrap();
    let mut vec_of_arrayd: Vec<ArrayD<f32>> = Vec::new();
    vec_of_arrayd.push(x.into_dyn());
    vec_of_arrayd.push(weight.into_dyn());
    let mut input_d = Input::Tensor4List(vec_of_arrayd);
    let result = match conv_node.compute(input_d) {
        Output::TensorD(arr) => arr.into_dimensionality::<Ix4>().unwrap(),
        _ => panic!("Wrong result")
    };
    println!("{}", result);
    assert_eq!(result, y_with_padding);
}


#[test]
fn test_conv_without_padding_pads_bias(){
    let mut kernel_vec:[usize; 2] = [0; 2];
    let mut input: Vec<i64> = vec![3, 3];
    let input = input.into_iter().map(|val| val as usize).collect::<Vec<usize>>();
    kernel_vec.copy_from_slice(&input);
    let mut conv_node = Conv::new(None, Some(arr1(&[1, 1])), Some(1), Some(Shape::from(Dim(kernel_vec))), None, Some(arr1(&[1, 1])));

    // Define the input values
    let x_values = vec![
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0,
        20.0, 21.0, 22.0, 23.0, 24.0,
    ];

    // Create a 1x1x5x5 array
    let x = Array::from_shape_vec((1, 1, 5, 5), x_values).unwrap();
    let mut weight = Array4::from_elem((1, 1, 3, 3), 1.0);

    let mut bias = Array1::from_elem((1), 3.0);

    // Define the expected output
    let y_values = vec![
        57.0, 66.0, 75.0, 102.0, 111.0, 120.0, 147.0, 156.0, 165.0,
    ];

    // Create a 1x1x5x5 array
    let y = Array::from_shape_vec((1, 1, 3, 3), y_values).unwrap();
    //let input_d = Input::TensorD(x.into_shape(IxDyn(&[1, 3, 9, 9])).unwrap());
    //let weight_d = Input::TensorD(weight.into_shape(IxDyn(&[1 ,1, 3, 3])).unwrap());
    let mut vec_of_arrayd: Vec<ArrayD<f32>> = Vec::new();
    vec_of_arrayd.push(x.into_dyn());
    vec_of_arrayd.push(weight.into_dyn());
    vec_of_arrayd.push(bias.into_dyn());
    let mut input_d = Input::Tensor4List(vec_of_arrayd);
    let result = match conv_node.compute(input_d) {
        Output::TensorD(arr) => arr.into_dimensionality::<Ix4>().unwrap(),
        _ => panic!("Wrong result")
    };
    println!("{}", result);
    assert_eq!(result, y);
}

    #[test]
    fn test_create_mapping(){
        let model = onnx_runtime::onnxruntime::parse_onnx("src/tests/mnist-7.onnx".to_string()).unwrap();
        let graph = model.get_graph();
        let tot_out = graph.get_node()
            .iter().map(|n| n.get_output().len()).reduce(|v1, v2| v1 + v2).unwrap();
        let mapping = onnx_runtime::onnxruntime::get_in_out_mapping(graph);
        let mapping_len = mapping.len();
        for (key, value) in mapping.into_iter(){
            println!("key = {}, value = {}", key.clone(), value.clone())
        }
        assert_eq!(tot_out, mapping_len);
    }
}