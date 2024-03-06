use crossbeam::channel::{unbounded};
use std::collections::{HashMap, HashSet};
use std::{error, mem, thread};
use std::fmt;
use std::fmt::{Debug, Display, Formatter};
use std::sync::{Arc, RwLock};
use crate::node::{Node, SimpleNode};
use crate::operations::{Compute, Input, Output};

pub enum Error {
    NodeNotfound
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Node Not Found")
    }
}

impl Debug for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Node Not Found")
    }
}

impl error::Error for Error {}

pub type InnerDependencyMap = HashMap<String, HashSet<String>>;
pub type DependencyMap = Arc<RwLock<InnerDependencyMap>>;


pub struct DepGraph
{
    pub ready_nodes: Vec<String>,
    pub deps: DependencyMap,
    pub rdeps: DependencyMap,
    pub nodes: HashMap<String, Arc<RwLock<Node>>>,
    pub original_nodes: Vec<SimpleNode>,
    pub input_name: String
}

impl DepGraph
{
    pub fn new(nodes: HashMap<String, Node>, input_name: String) -> Self {
        let simple_nodes = nodes.values()
            .map(|node| SimpleNode::new(node.id().clone(), node.deps().clone()))
            .collect::<Vec<SimpleNode>>();
        let (deps, rdeps, ready_nodes) = DepGraph::parse_nodes(&simple_nodes);
        let nodes_safe = nodes
            .into_iter()
            .map( |(key, val)| (key, Arc::new(RwLock::new(val)))).collect();
        DepGraph {
            ready_nodes,
            deps,
            rdeps,
            nodes: nodes_safe,
            original_nodes: simple_nodes,
            input_name
        }
    }

    fn parse_nodes(nodes: &Vec<SimpleNode>) -> (DependencyMap, DependencyMap, Vec<String>) {

        let mut deps = InnerDependencyMap::default();
        let mut rdeps = InnerDependencyMap::default();
        let mut ready_nodes = Vec::<String>::default();

        for node in nodes {
            deps.insert(node.id().clone(), node.deps().clone());

            if node.deps().is_empty() {
                ready_nodes.push(node.id().clone());
            }

            for node_dep in node.deps() {
                if !rdeps.contains_key(node_dep) {
                    let mut dep_rdeps = HashSet::new();
                    dep_rdeps.insert(node.id().clone());
                    rdeps.insert(node_dep.clone(), dep_rdeps.clone());
                } else {
                    let dep_rdeps = rdeps.get_mut(node_dep).unwrap();
                    dep_rdeps.insert(node.id().clone());
                }
            }
        }

        (
            Arc::new(RwLock::new(deps)),
            Arc::new(RwLock::new(rdeps)),
            ready_nodes,
        )
    }

    pub fn run(&mut self, input: Input) -> Option<Output> {
        let input_array = match input{
            Input::TensorD(vec) => vec,
            _ => panic!("Wrong Input")
        };
        let (deps, rdeps, ready_nodes) = DepGraph::parse_nodes(&self.original_nodes);
        self.deps = deps;
        self.rdeps = rdeps;
        self.ready_nodes = ready_nodes;
        //Main thread works as dispatcher
        let mut threads = Vec::new();
        let (tx_input, rx_input): (crossbeam::channel::Sender<String>, crossbeam::channel::Receiver<String>) = unbounded();
        let (tx_output, rx_output): (crossbeam::channel::Sender<String>, crossbeam::channel::Receiver<String>)= unbounded();
        let safe_map = Arc::new(self.nodes.clone());
        for i in 0..4 {
            let rx = rx_input.clone();
            let tx = tx_output.clone();
            let node_map = safe_map.clone();

            let input_name = self.input_name.clone();
            let cloned_input = input_array.clone();

            let thread = thread::spawn(move | | {
                    while let Ok(node) = rx.recv() {
                        println!("Thread_{} computing node: {}", i, node.clone());
                        let node_to_process = node_map.get(&node).unwrap();
                        if node != input_name {
                            node_to_process.write().unwrap().compute_operation(&node_map);
                        }else{
                            let out = node_to_process.write().unwrap().operation.compute(Input::TensorD(cloned_input.clone()));
                            node_to_process.write().unwrap().output = Some(out);
                        }
                        tx.send(node).unwrap();
                    }
            });
            threads.push(thread);
        }
        //Start Injection
        while let Some(node_to_process) = self.ready_nodes.pop(){
            tx_input.send(node_to_process).unwrap();
        }
        //Running
        let mut last_node = None;
        while let Ok(node_to_add) = rx_output.recv() {
            self.ready_nodes.append(&mut remove_node_id(node_to_add.clone(), &self.deps, &self.rdeps).unwrap());
            //No more nodes leave
            if self.deps.read().unwrap().is_empty() {
                last_node = Some(node_to_add);
                break;
            }
            while let Some(node_to_process) = self.ready_nodes.pop(){
                tx_input.send(node_to_process).unwrap();
            }
        }

        mem::drop(tx_input);
        for handle in threads {handle.join().unwrap();}
        if let Some(inference) = last_node {
            return self.nodes.get(&inference).unwrap().read().unwrap().output.clone();
        }else{
            return None;
        }
    }

    #[allow(dead_code)]
    pub fn add_node(&mut self, name: String, operation: Box<dyn Compute + Send + Sync>, deps: &[String]){
        let mut new_node = Node::new(name.clone(), operation);
        deps.iter().for_each(|x | new_node.add_dep(x.clone()));
        self.nodes.insert(name.clone(), Arc::new(RwLock::new(new_node)));
        self.original_nodes.push(SimpleNode::new(name.clone(), deps.into_iter()
            .map(|n| n.clone()).collect::<HashSet<String>>()));
    }

    #[allow(dead_code)]
    pub fn remove_node(&mut self, name: String) -> Result<(), Error>{
        return match self.nodes.remove(&name){
            Some(_x) => {
                let mut found = 0;
                for (ind, node) in self.original_nodes.iter().enumerate(){
                    if node.id() == name {
                        found = ind;
                        break
                    }
                }
                self.original_nodes.remove(found);
                Ok(())
            },
            None => Err(Error::NodeNotfound)
        }
    }

    #[allow(dead_code)]
    pub fn modify_node_dep(&mut self, name: String, to_remove: Option<String>, to_add: Option<String>) -> Result<(), Error>{
        let node = match self.nodes.get_mut(&mut name.clone()){
            Some(val) => val,
            None => return Err(Error::NodeNotfound)
        };
        if to_add.is_some(){
            node.write().unwrap().deps.push(to_add.clone().unwrap().clone());
            let mut found = 0;
            for (ind, node) in self.original_nodes.iter().enumerate(){
                if node.id() == name{
                    found = ind;
                    break
                }
            }
            let tmp = to_add.clone().unwrap().clone();
            self.original_nodes[found].deps.insert(tmp);
        }
        if to_remove.is_some(){
            let to_compare = to_remove.unwrap().clone();
            node.write().unwrap().deps.retain(|v| *v != to_compare);
            let mut found = 0;
            for (ind, node) in self.original_nodes.iter().enumerate(){
                if node.id() == name{
                    found = ind;
                    break
                }
            }
            self.original_nodes[found].deps.remove(&to_compare);
        }
        Ok(())
    }

}

pub fn remove_node_id(
    id: String,
    deps: &DependencyMap,
    rdeps: &DependencyMap,
) -> Result<Vec<String>, Error>
{
    let rdep_ids = {
        match rdeps.read().unwrap().get(&id) {
            Some(node) => node.clone(),
            // If no node depends on a node, it will not appear
            // in rdeps.
            None => Default::default(),
        }
    };

    let mut deps = deps.write().unwrap();
    let next_nodes = rdep_ids
        .iter()
        .filter_map(|rdep_id| {
            let rdep = match deps.get_mut(&*rdep_id) {
                Some(rdep) => rdep,
                None => return None,
            };

            rdep.remove(&id);

            if rdep.is_empty() {
                Some(rdep_id.clone())
            } else {
                None
            }
        })
        .collect();

    // Remove the current node from the list of dependencies.
    deps.remove(&id);

    Ok(next_nodes)
}
