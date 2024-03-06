//Interface struct for

pub mod onnxruntime {
    use std::collections::HashMap;
    use std::fs::File;
    use std::io::Read;
    use ndarray::{ArrayD, IxDyn};
    use protobuf::Message;
    use crate::operations::add::Add;
    use crate::operations::averagepool::AveragePool;
    use crate::operations::concat::Concat;
    use crate::operations::conv::Conv;
    use crate::operations::dropout::Dropout;
    use crate::operations::gemm::Gemm;
    use crate::graph::DepGraph;
    use crate::operations::input::InputNode;
    use crate::operations::local_response_normalization::LRN;
    use crate::operations::matmul::MatMul;
    use crate::operations::maxpool::MaxPool;
    use crate::node::Node;
    use crate::onnx_proto3::{GraphProto, ModelProto, TensorProto};
    use crate::operations::{Compute, Input};
    use crate::operations::relu::Relu;
    use crate::operations::reshape::Reshape;
    use crate::operations::soft_max::SoftMax;
    use crate::operations::start::Start;

    #[derive(Debug)]
    pub enum Error{
        ProtoBufError,
        InputParsingError,
        ShapeError,
        ConversionError
    }

    pub fn parse_onnx(path: String) -> Result<ModelProto, Error>{
        let mut input_onnx = File::open(path.as_str()).unwrap();
        //Onnx file into byte array
        let mut byte_array = Vec::<u8>::new();
        input_onnx.read_to_end(&mut byte_array).unwrap();
        //Parsing del byte array nella struttura onnx_proto3.rs
        let model: ModelProto = match Message::parse_from_bytes(&byte_array) {
            Ok(model) => model,
            Err(err) => {
                eprintln!("Failed to parse the ONNX googlenet: {}", err);
                return Err(Error::ProtoBufError);
            }
        };
        return Ok(model);
    }
    pub fn get_computational_graph(path: String) -> DepGraph{
        let model = parse_onnx(path).unwrap();
        let graph = model.get_graph();
        let mut nodes = get_nodes(graph);
        let mut initializers = parse_initializers(graph);
        nodes.append(&mut initializers);
        let input_name = graph.get_input()[0].name.clone();
        nodes.push(Node::new(input_name.clone(), Box::new(InputNode::new())));
        let node_map = nodes.into_iter().map(|n| (n.id(), n)).collect::<HashMap<String, Node>>();
        DepGraph::new(node_map, input_name)
    }

    pub fn parse_initializers(graph: &GraphProto) -> Vec<Node>{
        let starting_nodes = graph.get_initializer();
        let mut cnt = 0;
        let nodes: Vec<Node> = starting_nodes.into_iter().map(|tensor| {
            cnt += 1;
            let dims: Vec<usize> = tensor.get_dims().iter().map(|val| *val as usize).collect();
            let raw = tensor.get_raw_data();
            let mut data: Vec<f32> = Vec::new();
            match tensor.get_data_type() {
                1 => {
                    if raw.len() != 0 {
                        data = parse_from_raw_data(raw);
                    } else {
                        data = tensor.get_float_data().into_iter().map(|val| *val).collect();
                    }
                },
                7 => {
                    data = tensor.get_int64_data().into_iter().map(|val| *val as f32).collect();
                    if data.len() == 0{
                        data = parse_int64_from_raw_data(tensor.get_raw_data());
                    }},
                _ => ()
            }
            let tensor_d = ArrayD::from_shape_vec(IxDyn(&dims), data).unwrap();
            let tmp_node = Node::new(tensor.name.clone(), Box::new(Start::new(tensor_d)));
            return tmp_node
        }).collect();
        println!("All parsed = {}",starting_nodes.len() == nodes.len());
        return nodes;
    }

    pub fn parse_from_raw_data(raw: &[u8]) -> Vec<f32>{
        return raw.chunks_exact(4) // Split into chunks of 4 bytes (size of f32)
            .map(|chunk| {
                let mut bytes_array = [0; 4];
                bytes_array.copy_from_slice(chunk);
                f32::from_bits(u32::from_le_bytes(bytes_array)) // Convert u8 to f32
            })
            .collect();
    }

    pub fn parse_int64_from_raw_data(raw: &[u8]) -> Vec<f32>{
        return raw.chunks_exact(8) // Split into chunks of 4 bytes (size of f32)
            .map(|chunk| {
                let mut bytes_array = [0; 8];
                bytes_array.copy_from_slice(chunk);
                i64::from_le_bytes(bytes_array) as f32
            })
            .collect();
    }

    #[allow(dead_code)]
    pub fn parse_input_tensor(path: String) -> Result<Input, Error>{
        let mut input_tensor = File::open(path).unwrap();
        //Onnx file into byte array
        let mut byte_array = Vec::<u8>::new();
        input_tensor.read_to_end(&mut byte_array).unwrap();
        //Parsing del byte array nella struttura onnx_proto3.rs
        let input_parsed: TensorProto = match Message::parse_from_bytes(&byte_array) {
            Ok(model) => model,
            Err(err) => {
                eprintln!("Failed to parse the ONNX googlenet: {}", err);
                return Err(Error::InputParsingError);
            }
        };
        let dims = input_parsed.get_dims()
            .into_iter().map(|v| *v as usize).collect::<Vec<usize>>();
        let vec = parse_from_raw_data(input_parsed.get_raw_data());
        return Ok(Input::TensorD( ArrayD::from_shape_vec(IxDyn(&dims), vec).unwrap()));
    }

    pub fn get_in_out_mapping(graph: &GraphProto) -> HashMap<String, String>{
        let nodes = graph.get_node();
        let mut index = 0;
        return nodes.into_iter().flat_map(|x| {
            let mut name = x.name.clone();
            if name.len() == 0{
                name = format!("Node_{}", index);
                index += 1;
            }
            return x.get_output().into_iter().map(|s| (s.clone(), name.clone())).collect::<Vec<(String, String)>>();
        }).collect::<HashMap<String, String>>();
    }

    pub fn get_nodes(graph: &GraphProto) -> Vec<Node>{
        let mut index = 0;
        let alias = get_in_out_mapping(graph);
        return graph.get_node().into_iter().map(|node| {
            let mut id = node.name.clone();
            if id.len() == 0{
                id = format!("Node_{}", index);
                index += 1;
            }
            let res: Box<dyn Compute + Send + Sync> = match node.get_op_type(){
                    "Softmax" => Box::new(SoftMax::parse_from_proto_node()),
                    "Relu" => Box::new(Relu::parse_from_proto_node()),
                    "Concat" => Box::new(Concat::parse_from_proto_node()),
                    "Dropout" => Box::new(Dropout::parse_from_proto_node()),
                    "MaxPool" => Box::new(MaxPool::parse_from_proto_node(node.get_attribute())),
                    "LRN" => Box::new(LRN::parse_from_proto_node(node.get_attribute())),
                    "AveragePool" => Box::new(AveragePool::parse_from_proto_node(node.get_attribute())),
                    "Conv" => Box::new(Conv::parse_from_proto_node(node.get_attribute())),
                    "Reshape" => Box::new(Reshape::parse_from_proto_node()),
                    "Gemm" => Box::new(Gemm::parse_from_proto_node(node.get_attribute())),
                    "MatMul" => Box::new(MatMul::parse_from_proto_node()),
                    "Add" => Box::new(Add::parse_from_proto_node()),
                    _ => panic!("Unknown operation type!")
                };
            let mut new_node = Node::new(id, res);
            for dep in node.get_input(){
                let mut tmp = dep;
                if alias.contains_key(dep) {
                    tmp = alias.get(dep).unwrap();
                }
                new_node.add_dep((*tmp).clone());
            }
            return new_node;
        }).collect::<Vec<Node>>();
    }

}