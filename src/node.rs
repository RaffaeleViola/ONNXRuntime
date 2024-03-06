use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fmt::{Display, Formatter};
use std::sync::{Arc, RwLock};
use ndarray::{ArrayD};
use crate::operations::{Compute, Input, Output};


pub struct Node
{
    pub id: String,
    pub deps: Vec<String>,
    pub operation: Box<dyn Compute + Send + Sync>,
    pub output: Option<Output>
}

pub struct SimpleNode {
    id: String,
    pub deps: HashSet<String>,
}

impl SimpleNode
{
    pub fn new(id: String, deps: HashSet<String>) -> SimpleNode {
        SimpleNode {
            id,
            deps,
        }
    }
    pub fn id(&self) -> String {
        self.id.clone()
    }
    pub fn deps(&self) -> &HashSet<String> {
        &self.deps
    }
}

impl Node
{
    pub fn new(id: String, operation: Box<dyn Compute + Send + Sync>) -> Node {
        Node {
            id,
            deps: Vec::default(),
            operation,
            output: None
        }
    }

    pub fn id(&self) -> String {
        self.id.clone()
    }
    pub fn deps(&self) -> HashSet<String> {

        HashSet::from_iter(self.deps.clone().into_iter())
    }
    pub fn add_dep(&mut self, dep: String) {
        self.deps.push(dep);
    }

    pub fn compute_operation(&mut self, nodes: &HashMap<String, Arc<RwLock<Node>>>) -> () {
        if self.deps.len() == 1{
            let elem = self.deps.iter().next().unwrap().clone();
            let only_dep = nodes.get(&elem).unwrap();
            let input = match only_dep.read().unwrap().output.clone().unwrap() {
                Output::TensorD(array) => Input::TensorD(array),
                _ => panic!("wrong output")
            };
            self.output = Some(self.operation.compute(input));
            /*match self.output.clone().unwrap(){
                Output::TensorD(arr) => (),
                _ => ()
            }*/

        }else if self.deps.len() > 1 {
            let mut inputs = Vec::<ArrayD<f32>>::new();
            self.deps.iter().for_each(|dep| {
                let elem = nodes.get(dep).unwrap();
                let input = match elem.read().unwrap().output.clone().unwrap() {
                    Output::TensorD(array) => array,
                    _ => panic!("wrong output")
                };
                inputs.push(input);
            });
            self.output = Some(self.operation.compute(Input::Tensor4List(inputs)));
            /*match self.output.clone().unwrap(){
                Output::TensorD(arr) => (),
                _ => ()
            }*/
        }else{
            self.output = Some(self.operation.compute(Input::Empty));
        }
    }
}

impl Display for Node {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let id = self.id();
        let inputs = match self.deps.len() {
            d if d > 0 => self.deps.iter()
                .map(|v| (*v).clone()).reduce(|d1, d2| d1 + " --- " + d2.as_str()).unwrap(),
            _ => "None -> Ready Node".to_string()
        };
        let op_type = self.operation.op_type();
        write!(f, "id: {}\nop_type: {}\ninputs: {}\n", id, op_type, inputs)
    }
}
