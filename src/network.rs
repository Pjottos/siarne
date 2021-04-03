use rand::prelude::*;

use std::iter;

pub struct Network {
    accumulators: Vec<ActionPotential>,
    connection_range: usize,

    // parameters
    action_potentials: Vec<ActionPotential>,
    effects: Vec<Effect>,
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Eq, Ord, Default)]
struct ActionPotential(i32);

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Eq, Ord, Default)]
struct Effect(i8);

impl Network {
    pub fn new(neuron_count: usize, connection_range: usize) -> Self {
        assert!(neuron_count > 0);
        assert!(connection_range > 0);
        let effect_count = neuron_count.checked_mul(connection_range)
            .expect("neuron_count or connection_range too big");

        let mut rng = thread_rng();

        let action_potentials = iter::repeat_with(|| ActionPotential(rng.gen()))
            .take(neuron_count)
            .collect();

        let effects = iter::repeat_with(|| Effect(rng.gen()))
            .take(effect_count)
            .collect();
        
        Self {
            accumulators: vec![Default::default(); neuron_count as usize],
            connection_range,
            
            action_potentials,
            effects,
        }
    }
}
