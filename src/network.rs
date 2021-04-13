//! Code related to creating and executing [Network]s

use rand::prelude::*;

use std::{iter, ops::Range};

const ACCUMULATOR_BUF_COUNT: usize = 2;

/// A structure containing a collection of interconnected neurons.
pub struct Network {
    accumulators: [Option<Box<[NeuronValue]>>; ACCUMULATOR_BUF_COUNT],
    current_cum_buf: usize,
    connection_count: usize,

    // parameters
    tresholds: Box<[NeuronValue]>,
    effects: Box<[Effect]>,
}

/// A value related to the input of a neuron.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Eq, Ord, Default)]
pub struct NeuronValue(pub i32);

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

        let tresholds = iter::repeat_with(|| NeuronValue(rng.gen()))
            .take(neuron_count)
            .collect::<Vec<_>>()
            .into();

        let effects = iter::repeat_with(|| Effect(rng.gen()))
            .take(effect_count)
            .collect::<Vec<_>>()
            .into();

        let accumulator_buf: Box<[NeuronValue]> = vec![NeuronValue(0); neuron_count].into();
        
        Self {
            accumulators: [Some(accumulator_buf.clone()), Some(accumulator_buf)],
            current_cum_buf: 0,
            connection_count: connection_count,
            
            tresholds,
            effects,
        }
    }

    /// Create a [Network] with the specified parameters.
    /// # Panics
    /// When `tresholds` or `effects` is empty.  
    /// When there are more tresholds than effects.  
    /// When the amount of effects per treshold is greater than the amount of tresholds.  
    pub fn with_params(tresholds: Box<[NeuronValue]>, effects: Box<[Effect]>) -> Self {
        let neuron_count = tresholds.len();
        assert!(neuron_count > 0);
        let connection_count = effects.len() / neuron_count;
        assert!(connection_count > 0);
        assert!(connection_count <= neuron_count);

        let accumulator_buf: Box<[NeuronValue]> = vec![NeuronValue(0); neuron_count].into();
        
        Self {
            accumulators: [Some(accumulator_buf.clone()), Some(accumulator_buf)],
            current_cum_buf: 0,
            connection_count,
            
            tresholds,
            effects,
        }
    }

    /// Returns a slice of the fire tresholds of the neurons.
    #[inline]
    pub fn tresholds(&self) -> &[NeuronValue] {
        &self.tresholds
    }

    /// Returns a mutable slice of the fire tresholds of the neurons.
    #[inline]
    pub fn tresholds_mut(&mut self) -> &mut [NeuronValue] {
        &mut self.tresholds
    }
    
    /// Returns a slice of the effects of the connections between neurons.  
    /// This is a matrix with dimensions `connection_count` x `neuron_count`
    /// where each row is the outputs to nearby neurons. At column `connection_count` / 2
    /// the connection from the neuron to itself is stored, the other columns store 
    /// connections to neurons before and after the neuron. You can think of neurons
    /// being arranged in a circle.
    /// # Examples
    /// ```
    /// # use ans::Network;
    /// let net = Network::new(3, 3);
    /// 
    /// // print connection effects from neuron to neuron
    /// println!("0 -> 2: {:?}", net.effects()[(0 * 3) + 0]);
    /// println!("0 -> 0: {:?}", net.effects()[(0 * 3) + 1]);
    /// println!("0 -> 1: {:?}", net.effects()[(0 * 3) + 2]);
    ///
    /// println!("1 -> 0: {:?}", net.effects()[(1 * 3) + 0]);
    /// println!("1 -> 1: {:?}", net.effects()[(1 * 3) + 1]);
    /// println!("1 -> 2: {:?}", net.effects()[(1 * 3) + 2]);
    ///
    /// println!("2 -> 1: {:?}", net.effects()[(2 * 3) + 0]);
    /// println!("2 -> 2: {:?}", net.effects()[(2 * 3) + 1]);
    /// println!("2 -> 0: {:?}", net.effects()[(2 * 3) + 2]);
    /// ```
    #[inline]
    pub fn effects(&self) -> &[Effect] {
        &self.effects
    }

    /// Returns a mutable slice of the effects of the connections between neurons.  
    /// See [Network::effects] for more information about the layout of this slice.
    #[inline]
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
    #[inline]
    pub fn last_accumulator_buf(&self) -> &[NeuronValue] {
        let index = (self.current_cum_buf + ACCUMULATOR_BUF_COUNT - 1) % ACCUMULATOR_BUF_COUNT;
        self.accumulators[index].as_ref().unwrap()
    }

    /// Returns a mutable slice of the inputs of the neurons on the last executed tick.  
    /// For more information see [Network::last_accumulator_buf].
    #[inline]
    pub fn last_accumulator_buf_mut(&mut self) -> &mut [NeuronValue] {
        let index = (self.current_cum_buf + ACCUMULATOR_BUF_COUNT - 1) % ACCUMULATOR_BUF_COUNT;
        self.accumulators[index].as_mut().unwrap()
    }

    /// Execute a tick on the network, updating [Network::last_accumulator_buf]
    pub fn tick(&mut self) {
        let mut cum = self.accumulators[self.current_cum_buf].take().unwrap();
        let inputs = self.last_accumulator_buf();
        let neuron_count = self.tresholds.len();

        // think of the neurons as being arranged in a circle.
        // for any given neuron, we observe a slice of this circle with
        // a size specified by self.connection_count where the current
        // neuron is at the center of this slice. for each neuron in the slice, we add  
        // the effect of it on the current neuron if the input is above the treshold.

        // amount of neurons before and after any neuron (on the circle).
        let extent_back = self.connection_count / 2;
        let extent_front = if extent_back == 0 {
            0
        } else if self.connection_count % 2 != 0 {
            extent_back
        } else {
            extent_back - 1
        };

        for src in 0..extent_back {
            // safety: it is assumed parameter and accumulator slices do not change size
            // after construction of the network. As that might invalidate self.connection_count
            unsafe {
                let input = inputs.get_unchecked(src);
                let treshold = self.tresholds.get_unchecked(src);

                if input >= treshold {
                    let wrapping_range = neuron_count - extent_back + src..neuron_count;
                    self.apply_effects(
                        &mut cum,
                        src,
                        0..self.connection_count - wrapping_range.len(),
                        wrapping_range.len(),
                    );
                    self.apply_effects(
                        &mut cum,
                        src,
                        wrapping_range,
                        0,
                    );
                }
            }
        }

        for src in extent_back..neuron_count - extent_front {
            unsafe {
                let input = inputs.get_unchecked(src);
                let treshold = self.tresholds.get_unchecked(src);

                if input >= treshold {
                    self.apply_effects(
                        &mut cum,
                        src,
                        src - extent_back..src + extent_front + 1,
                        0,
                    );
                }
            }
        }

        for src in neuron_count - extent_front..neuron_count {
            unsafe {
                let input = inputs.get_unchecked(src);
                let treshold = self.tresholds.get_unchecked(src);

                if input >= treshold {
                    let non_wrapping_range = src - extent_back..neuron_count;
                    self.apply_effects(
                        &mut cum,
                        src,
                        0..self.connection_count - non_wrapping_range.len(),
                        non_wrapping_range.len(),
                    );
                    
                    self.apply_effects(
                        &mut cum,
                        src,
                        non_wrapping_range,
                        0,
                    );
                }
            }
        }

        self.accumulators[self.current_cum_buf] = Some(cum);
        self.advance_cum_buf();
    }

    #[inline]
    unsafe fn apply_effects(
        &self,
        cum: &mut [NeuronValue],
        src: usize,
        dst_range: Range<usize>,
        offset: usize,
    ) {
        let base = (src * self.connection_count) + offset;
        for (i, dst) in dst_range.enumerate() {
            let effect = self.effects.get_unchecked(base + i);
            cum.get_unchecked_mut(dst).0 += effect.0 as i32;
        }
    }

    #[inline]
    fn advance_cum_buf(&mut self) {
        let i = (self.current_cum_buf + 1) % ACCUMULATOR_BUF_COUNT;

        self.accumulators[i]
            .as_mut()
            .unwrap() 
            .fill(NeuronValue(0));

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
        let _ = Network::with_params(vec![].into(), vec![Effect(0)].into());
    }
    
    #[test]
    #[should_panic]
    fn catch_zero_effects() {
        let _ = Network::with_params(vec![NeuronValue(0)].into(), vec![].into());
    }
    
    #[test]
    #[should_panic]
    fn catch_too_little_effects() {
        let _ = Network::with_params(vec![NeuronValue(0); 2].into(), vec![Effect(0)].into());
    }
    
    #[test]
    #[should_panic]
    fn catch_too_many_effects() {
        let _ = Network::with_params(vec![NeuronValue(0); 2].into(), vec![Effect(0); 6].into());
    }

    #[test]
    fn test_tick() {
        let action_potentials = vec![
            NeuronValue(0),
            NeuronValue(16),
            NeuronValue(-16),
        ].into();

        let effects = vec![
            Effect(-8), Effect(1), Effect(2),
            Effect(-17), Effect(127), Effect(0),
            Effect(127), Effect(0), Effect(0),
        ].into();

        let mut net = Network::with_params(action_potentials, effects);

        net.tick();

        assert_eq!(
            net.last_accumulator_buf(),
            &[
                NeuronValue(1),
                NeuronValue(129),
                NeuronValue(-8),
            ],
        );

        net.tick();

        assert_eq!(
            net.last_accumulator_buf(),
            &[
                NeuronValue(-16),
                NeuronValue(256),
                NeuronValue(-8),
            ],
        );
    }
}
