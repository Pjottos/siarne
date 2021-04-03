use rand::prelude::*;

use std::{iter, num::NonZeroUsize};

const ACCUMULATOR_BUF_COUNT: usize = 2;

pub struct Network {
    accumulators: [Vec<ActionPotential>; ACCUMULATOR_BUF_COUNT],
    current_accumulator_buf: usize,
    connection_range: usize,

    // parameters
    action_potentials: Vec<ActionPotential>,
    effects: Vec<Effect>,
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Eq, Ord, Default)]
pub struct ActionPotential(pub i32);

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Eq, Ord, Default)]
pub struct Effect(pub i8);

impl Network {
    pub fn new(neuron_count: NonZeroUsize, connection_range: NonZeroUsize) -> Self {
        let effect_count = neuron_count.get().checked_mul(connection_range.get())
            .expect("neuron_count or connection_range too big");

        let mut rng = thread_rng();

        let action_potentials = iter::repeat_with(|| ActionPotential(rng.gen()))
            .take(neuron_count.get())
            .collect();

        let effects = iter::repeat_with(|| Effect(rng.gen()))
            .take(effect_count)
            .collect();

        let accumulator_buf = vec![ActionPotential(0); neuron_count.get()];
        
        Self {
            accumulators: [accumulator_buf.clone(), accumulator_buf],
            current_accumulator_buf: 0,
            connection_range: connection_range.get(),
            
            action_potentials,
            effects,
        }
    }

    pub fn with_params(action_potentials: Vec<ActionPotential>, effects: Vec<Effect>) -> Self {
        let neuron_count = action_potentials.len();
        assert!(neuron_count > 0);
        let connection_range = effects.len() / neuron_count;
        assert!(connection_range > 0);

        let accumulator_buf = vec![ActionPotential(0); neuron_count];
        
        Self {
            accumulators: [accumulator_buf.clone(), accumulator_buf],
            current_accumulator_buf: 0,
            connection_range,
            
            action_potentials,
            effects,
        }
    }

    pub fn action_potentials(&self) -> &[ActionPotential] {
        &self.action_potentials
    }

    pub fn action_potentials_mut(&mut self) -> &mut [ActionPotential] {
        &mut self.action_potentials
    }
    
    pub fn effects(&self) -> &[Effect] {
        &self.effects
    }

    pub fn effects_mut(&mut self) -> &mut [Effect] {
        &mut self.effects
    }

    pub fn last_accumulator_buf(&self) -> &[ActionPotential] {
        let index = (self.current_accumulator_buf + ACCUMULATOR_BUF_COUNT - 1) % ACCUMULATOR_BUF_COUNT;
        &self.accumulators[index]
    }

    pub fn last_accumulator_buf_mut(&mut self) -> &mut [ActionPotential] {
        let index = (self.current_accumulator_buf + ACCUMULATOR_BUF_COUNT - 1) % ACCUMULATOR_BUF_COUNT;
        &mut self.accumulators[index]
    }

    fn advance_accumulator_buf(&mut self) -> &mut [ActionPotential] {
        let i = (self.current_accumulator_buf + 1) % ACCUMULATOR_BUF_COUNT;

        let count = self.accumulators[i].len();
        self.accumulators[i].clear();
        self.accumulators[i].resize(count, ActionPotential(0));

        self.current_accumulator_buf = i;
        &mut self.accumulators[i]
    }

    pub fn tick(&mut self) {
        todo!();
    }
}
