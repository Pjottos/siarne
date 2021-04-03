use rand::prelude::*;

use std::iter;

pub struct Network<const MAX_CONNECTIONS: usize> {
    accumulators: Vec<ActionPotential>,

    // parameters
    outputs: Vec<[NeuronId; MAX_CONNECTIONS]>,
    effects: Vec<[Effect; MAX_CONNECTIONS]>,
    action_potentials: Vec<ActionPotential>,
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Eq, Ord, Default)]
struct NeuronId(u32);

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Eq, Ord, Default)]
struct ActionPotential(i32);

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Eq, Ord, Default)]
struct Effect(i8);

impl<const MAX_CONNECTIONS: usize> Network<MAX_CONNECTIONS> {
    pub fn new(neuron_count: u32) -> Self {
        assert!(neuron_count > 0);

        let mut rng = thread_rng();

        let outputs = iter::repeat_with(|| {
            let mut tmp = [Default::default(); MAX_CONNECTIONS];
            for i in 0..MAX_CONNECTIONS {
                tmp[i] = NeuronId(rng.gen_range(0..neuron_count));
            }
            tmp
        })
            .take(neuron_count as usize)
            .collect();
            
        let effects = iter::repeat_with(|| {
            let mut tmp = [Default::default(); MAX_CONNECTIONS];
            for i in 0..MAX_CONNECTIONS {
                tmp[i] = Effect(rng.gen());
            }
            tmp
        })
            .take(neuron_count as usize)
            .collect();
        
        let action_potentials = iter::repeat_with(|| ActionPotential(rng.gen()))
            .take(neuron_count as usize)
            .collect();

        Self {
            accumulators: vec![Default::default(); neuron_count as usize],

            outputs,
            effects,
            action_potentials,
        }
    }
}
