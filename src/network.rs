//! Code related to creating and executing [Network]s

use rand::prelude::*;

use std::iter;

const ACCUMULATOR_BUF_COUNT: usize = 2;

/// A structure containing a collection of interconnected neurons.
pub struct Network {
    accumulators: [Option<Vec<ActionPotential>>; ACCUMULATOR_BUF_COUNT],
    current_cum_buf: usize,
    connection_count: usize,

    // parameters
    action_potentials: Vec<ActionPotential>,
    effects: Vec<Effect>,
}

/// The action potential of a neuron, i.e the minimum sum of its
/// inputs for it to fire.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Eq, Ord, Default)]
pub struct ActionPotential(pub i32);

/// The effect of a connection is the value added to the input of a neuron
/// when the neuron at the other end of the connection fires. Connections
/// are one-directional.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Eq, Ord, Default)]
pub struct Effect(pub i8);

impl Network {
    /// Create a [Network] with randomly initialized parameters.  
    /// `connection_count` is the amount of inputs per neuron.
    /// # Panics
    /// When `neuron_count` or `connection_count` is 0.  
    /// When `connection_count` > `neuron_count`.   
    /// When the result of `neuron_count * connection_count` does not fit in a [usize].  
    pub fn new(neuron_count: usize, connection_count: usize) -> Self {
        assert!(neuron_count > 0);
        assert!(connection_count > 0);
        assert!(connection_count <= neuron_count);
        let effect_count = neuron_count.checked_mul(connection_count)
            .expect("neuron_count or connection_count too big");

        let mut rng = thread_rng();

        let action_potentials = iter::repeat_with(|| ActionPotential(rng.gen()))
            .take(neuron_count)
            .collect();

        let effects = iter::repeat_with(|| Effect(rng.gen()))
            .take(effect_count)
            .collect();

        let accumulator_buf = vec![ActionPotential(0); neuron_count];
        
        Self {
            accumulators: [Some(accumulator_buf.clone()), Some(accumulator_buf)],
            current_cum_buf: 0,
            connection_count: connection_count,
            
            action_potentials,
            effects,
        }
    }

    /// Create a [Network] with the specified parameters.
    /// # Panics
    /// When `action_potentials` or `effects` is empty.  
    /// When there are more action potentials than effects.  
    /// When the amount of effects per action potential is greater than the amount of action potentials.  
    pub fn with_params(action_potentials: Vec<ActionPotential>, effects: Vec<Effect>) -> Self {
        let neuron_count = action_potentials.len();
        assert!(neuron_count > 0);
        let connection_count = effects.len() / neuron_count;
        assert!(connection_count > 0);
        assert!(connection_count <= neuron_count);

        let accumulator_buf = vec![ActionPotential(0); neuron_count];
        
        Self {
            accumulators: [Some(accumulator_buf.clone()), Some(accumulator_buf)],
            current_cum_buf: 0,
            connection_count,
            
            action_potentials,
            effects,
        }
    }

    /// Returns a slice of the action potentials of the neurons.
    pub fn action_potentials(&self) -> &[ActionPotential] {
        &self.action_potentials
    }

    /// Returns a mutable slice of the action potentials of the neurons.
    pub fn action_potentials_mut(&mut self) -> &mut [ActionPotential] {
        &mut self.action_potentials
    }
    
    /// Returns a slice of the effects of the connections between neurons.  
    /// This is a matrix with dimensions `connection_count` x `neuron_count` where the center of 
    /// each row (rounded to 0) is the connection of the neuron to itself. The other connections
    /// are to adjacent neurons where the neurons are arranged in a circle.
    pub fn effects(&self) -> &[Effect] {
        &self.effects
    }

    /// Returns a mutable slice of the effects of the connections between neurons.  
    /// See [Network::effects] for more information.
    pub fn effects_mut(&mut self) -> &mut [Effect] {
        &mut self.effects
    }

    /// Returns a slice of the inputs of the neurons on the last executed tick.  
    /// These values will be used in the next tick to determine which neurons should fire.
    /// The values can also be used to extract output from the network by reading the values
    /// at arbitrary indices so long as the same indices are used each time. 
    /// # Examples
    /// ```
    /// # use ans::Network;
    /// let mut net = Network::new(16, 2);
    /// 
    /// // introduce information from outside the network
    /// net.last_accumulator_buf_mut()[0].0 += 123; 
    ///
    /// net.tick();
    ///
    /// // keep in mind it might take multiple ticks for the
    /// // input introduced at one neuron to propagate to the output
    /// // depending on the connection_count and distance between the neurons
    /// println!("Output: {}", net.last_accumulator_buf()[1].0);
    /// ```
    pub fn last_accumulator_buf(&self) -> &[ActionPotential] {
        let index = (self.current_cum_buf + ACCUMULATOR_BUF_COUNT - 1) % ACCUMULATOR_BUF_COUNT;
        self.accumulators[index].as_ref().unwrap()
    }

    /// Returns a mutable slice of the inputs of the neurons on the last executed tick.  
    /// For more information see [Network::last_accumulator_buf].
    pub fn last_accumulator_buf_mut(&mut self) -> &mut [ActionPotential] {
        let index = (self.current_cum_buf + ACCUMULATOR_BUF_COUNT - 1) % ACCUMULATOR_BUF_COUNT;
        self.accumulators[index].as_mut().unwrap()
    }

    /// Execute a tick on the network, updating [Network::last_accumulator_buf]
    #[inline]
    pub fn tick(&mut self) {
        let mut cum = self.accumulators[self.current_cum_buf].take().unwrap();

        for (neuron, &value) in self.last_accumulator_buf().iter().enumerate() {
            // safety: length of parameter vecs must not change after construction of network
            unsafe {
                if value >= *self.action_potentials.get_unchecked(neuron) {
                    self.apply_effects(neuron, &mut cum);
                }
            }
        }

        self.accumulators[self.current_cum_buf] = Some(cum);
        self.advance_cum_buf();
    }

    #[inline]
    unsafe fn apply_effects(&self, neuron: usize, cum: &mut [ActionPotential]) {
        // think of the neurons as being arranged in a circle.
        // for any given neuron, we apply effects to neurons within 
        // a slice of this circle with a size specified by self.connection_count.
        // the neuron is at the center of this slice and the slice cannot overlap.

        // amount of neurons before and after current neuron (on the circle).
        let extent_back = self.connection_count / 2;
        let extent_front = if extent_back == 0 {
            0
        } else if self.connection_count % 2 != 0 {
            extent_back
        } else {
            extent_back - 1
        };

        let (neuron_start, effect_start, count) = if neuron < extent_back {
            // the amount of neurons appearing before index 0 on the circle
            // is guaranteed to be at least 1
            let back_count = extent_back - neuron;
            let back_start = self.action_potentials.len() - back_count;

            for i in 0..back_count {
                // in this loop effect_start is 0 because it happens first.
                // in the next loop it will continue where this loop left off.
                self.apply_single_effect(
                    back_start + i,
                    neuron,
                    0 + i,
                    cum,
                );
            }

            // handle rest of effects in next loop
            (0, back_count, self.connection_count - back_count)
        } else if (neuron + extent_front) >= self.action_potentials.len() {
            let front_count = (neuron + extent_front) - (self.action_potentials.len() - 1);
            // each row in self.effects is self.connection_count long.
            // so the last n rows start at index self.connection_count - n.
            // n = front_count in this case
            let front_effect_start = self.connection_count - front_count;

            for i in 0..front_count {
                self.apply_single_effect(
                    0 + i,
                    neuron,
                    front_effect_start + i,
                    cum,
                );
            }

            // neuron >= extent_back so we can go to the start without wrapping around
            let neuron_start = neuron - extent_back;
            (neuron_start, 0, self.connection_count - front_count)
        } else {
            // whole slice fits within the array bounds
            let neuron_start = neuron - extent_back;
            (neuron_start, 0, self.connection_count)
        };

        for i in 0..count {
            self.apply_single_effect(
                neuron_start + i,
                neuron,
                effect_start + i,
                cum,
            );
        }
    }

    #[inline]
    unsafe fn apply_single_effect(
        &self, 
        dst_neuron: usize, 
        src_neuron: usize, 
        local_effect: usize, 
        cum: &mut [ActionPotential]
    ) {
        let effect = *self.effects.get_unchecked((src_neuron * self.connection_count) + local_effect);
        cum.get_unchecked_mut(dst_neuron).0 += effect.0 as i32;
    }
    
    #[inline]
    fn advance_cum_buf(&mut self) {
        let i = (self.current_cum_buf + 1) % ACCUMULATOR_BUF_COUNT;

        let mut cum = self.accumulators[i].take().unwrap();
        let count = cum.len();
        cum.clear();
        cum.resize(count, ActionPotential(0));
        self.accumulators[i] = Some(cum);

        self.current_cum_buf = i;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn catch_zero_neuron_count() {
        let _ = Network::new(0, 1);
    }
    
    #[test]
    #[should_panic]
    fn catch_zero_connection_count() {
        let _ = Network::new(1, 0);
    }
    
    #[test]
    #[should_panic]
    fn catch_invalid_connection_count() {
        let _ = Network::new(1, 2);
    }

    #[test]
    #[should_panic]
    fn catch_effect_count_overflow() {
        let max = usize::MAX;
        let _ = Network::new(max, 2);
    }

    #[test]
    #[should_panic]
    fn catch_zero_action_potentials() {
        let _ = Network::with_params(vec![], vec![Effect(0)]);
    }
    
    #[test]
    #[should_panic]
    fn catch_zero_effects() {
        let _ = Network::with_params(vec![ActionPotential(0)], vec![]);
    }
    
    #[test]
    #[should_panic]
    fn catch_too_little_effects() {
        let _ = Network::with_params(vec![ActionPotential(0); 2], vec![Effect(0)]);
    }
    
    #[test]
    #[should_panic]
    fn catch_too_many_effects() {
        let _ = Network::with_params(vec![ActionPotential(0); 2], vec![Effect(0); 6]);
    }

    #[test]
    fn test_tick() {
        let action_potentials = vec![
            ActionPotential(0),
            ActionPotential(16),
            ActionPotential(-16),
        ];
        let effects = vec![
            Effect(-8), Effect(1), Effect(2),
            Effect(-17), Effect(127), Effect(0),
            Effect(127), Effect(0), Effect(0),
        ];

        let mut net = Network::with_params(action_potentials, effects);
        net.tick();

        let expected = [
            ActionPotential(1),
            ActionPotential(129),
            ActionPotential(-8),
        ];

        let result = net.last_accumulator_buf();

        assert_eq!(
            result,
            &expected,
            "accumulator buf had unexpected contents",
        );
    }
}
