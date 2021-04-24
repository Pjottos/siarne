//! Code related to creating and executing [Network]s

use rand::prelude::*;

use std::{iter, ops::Range};

const ACCUMULATOR_BUF_COUNT: usize = 2;

#[derive(Debug)]
pub enum Error {
    ZeroNeurons,
    ZeroConnections,
    TooManyConnections,
    EffectCountOverflow,
    InvalidNeuronIndex,
}

/// A structure containing a collection of interconnected neurons.
pub struct Network {
    accumulators: [Option<Box<[NeuronValue]>>; ACCUMULATOR_BUF_COUNT],
    current_cum_buf: usize,
    connection_count: usize,

    // parameters
    tresholds: Box<[NeuronValue]>,
    effects: Box<[Effect]>,
    input_neurons: Box<[usize]>,
    output_neurons: Box<[usize]>,
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
    /// # Errors
    /// When `neuron_count` is 0, [Error::ZeroNeurons].
    /// When `connection_count` is 0, [Error::ZeroConnections].
    /// When `connection_count` > `neuron_count`, [Error::TooManyConnections].   
    /// When the result of `neuron_count * connection_count` does not fit in a [usize], [Error::EffectCountOverflow].  
    pub fn new(neuron_count: usize, connection_count: usize, input_count: usize, output_count: usize) -> Result<Self, Error> {
        if neuron_count == 0 {
            return Err(Error::ZeroNeurons);
        }
        if connection_count == 0 {
            return Err(Error::ZeroConnections);
        }
        if connection_count > neuron_count {
            return Err(Error::TooManyConnections);
        }

        let effect_count = neuron_count
            .checked_mul(connection_count)
            .ok_or(Error::EffectCountOverflow)?;
        
        let mut rng = thread_rng();

        let tresholds = iter::repeat_with(|| NeuronValue(rng.gen()))
            .take(neuron_count)
            .collect();

        let effects = iter::repeat_with(|| Effect(rng.gen()))
            .take(effect_count)
            .collect();
        
        let neuron_dist = rand::distributions::Uniform::from(0..neuron_count);

        let input_neurons = iter::repeat_with(|| neuron_dist.sample(&mut rng))
            .take(input_count)
            .collect();

        let output_neurons = iter::repeat_with(|| neuron_dist.sample(&mut rng))
            .take(output_count)
            .collect();
            
        let accumulator_buf: Box<[NeuronValue]> = vec![NeuronValue(0); neuron_count].into();
        
        Ok(Self {
            accumulators: [Some(accumulator_buf.clone()), Some(accumulator_buf)],
            current_cum_buf: 0,
            connection_count,
            
            tresholds,
            effects,
            input_neurons,
            output_neurons,
        })
    }

    /// Create a [Network] with the specified parameters.
    /// # Errors
    /// When `tresholds` is empty, [Error::ZeroNeurons].    
    /// When the amount of effects per neuron is less than 1, [Error::ZeroConnections].    
    /// When there are more connections per neuron than neurons, [Error::TooManyConnections].  
    /// When `input_neurons` contains an out of bounds index, [Error::InvalidNeuronIndex].  
    pub fn with_params(
        tresholds: Box<[NeuronValue]>,
        effects: Box<[Effect]>,
        input_neurons: Box<[usize]>,
        output_neurons: Box<[usize]>,
    ) -> Result<Self, Error> {
        let neuron_count = tresholds.len();
        if neuron_count == 0 {
            return Err(Error::ZeroNeurons);
        }

        let connection_count = effects.len() / neuron_count;
        if connection_count == 0 {
            return Err(Error::ZeroConnections);
        }

        if connection_count > neuron_count {
            return Err(Error::TooManyConnections);
        }

        if !(
            input_neurons.iter().copied().all(|i| i < neuron_count) &&
            output_neurons.iter().copied().all(|i| i < neuron_count)
        ) {
            return Err(Error::InvalidNeuronIndex);
        }

        let accumulator_buf: Box<[NeuronValue]> = vec![NeuronValue(0); neuron_count].into();
        
        Ok(Self {
            accumulators: [Some(accumulator_buf.clone()), Some(accumulator_buf)],
            current_cum_buf: 0,
            connection_count,
            
            tresholds,
            effects,
            input_neurons,
            output_neurons,
        })
    }

    /// Returns a slice of the activation tresholds of the neurons.
    #[inline]
    pub fn tresholds(&self) -> &[NeuronValue] {
        &self.tresholds
    }

    /// Returns a mutable slice of the activation tresholds of the neurons.
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
    /// let net = Network::new(3, 3, 0, 0).unwrap();
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

    /// Applies the specified inputs to the neurons designated as input neurons, in order.  
    /// # Panics
    /// When `inputs.len()` is not equal to the input neuron count.  
    pub fn apply_inputs(&mut self, inputs: &[NeuronValue]) {
        assert_eq!(self.input_neurons.len(), inputs.len());

        let cum = self.accumulators[self.last_accumulator_buf_index()]
            .as_mut()
            .unwrap();

        self.input_neurons
            .iter()
            .copied()
            .zip(inputs.iter().copied())
            .for_each(|(neuron, input)| {
                unsafe {
                    cum.get_unchecked_mut(neuron).0 += input.0;
                }
            });
    }

    /// Read values of the designated output neurons into the specified buffer, in order.
    /// # Panics
    /// When `outputs.len()` is not equal to the output neuron count.  
    pub fn read_outputs(&self, outputs: &mut [NeuronValue]) {
        assert_eq!(self.output_neurons.len(), outputs.len());

        let cum = self.last_accumulator_buf();

        self.output_neurons
            .iter()
            .copied()
            .zip(outputs.iter_mut())
            .for_each(|(neuron, output)| {
                unsafe {
                    *output = *cum.get_unchecked(neuron);
                }
            });
    }

    /// Execute a tick on the network, evaluating each neuron and applying effects to other neurons if it fires.  
    /// Only the result of the last tick is considered, i.e the effects are applied on a zeroed buffer,
    /// but whether to apply an effect or not is determined by looking at the buffer from the last tick.
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
            // after construction of the network. Unless self.connection_count is updated as well.
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
    fn last_accumulator_buf(&self) -> &[NeuronValue] {
        self.accumulators[self.last_accumulator_buf_index()].as_ref().unwrap()
    }

    #[inline]
    fn last_accumulator_buf_index(&self) -> usize {
        (self.current_cum_buf + ACCUMULATOR_BUF_COUNT - 1) % ACCUMULATOR_BUF_COUNT
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
    fn input_validation() {
        match Network::new(0, 1, 0, 0) {
            Err(Error::ZeroNeurons) => (),
            _ => panic!(),
        }

        match Network::new(1, 0, 0, 0) {
            Err(Error::ZeroConnections) => (),
            _ => panic!(),
        }

        match Network::new(1, 2, 0, 0) {
            Err(Error::TooManyConnections) => (),
            _ => panic!(),
        }

        match Network::new(usize::MAX, 2, 0, 0) {
            Err(Error::EffectCountOverflow) => (),
            _ => panic!(),
        }

        match Network::with_params(vec![].into(), vec![Effect(0)].into(), vec![].into(), vec![].into()) {
            Err(Error::ZeroNeurons) => (),
            _ => panic!(),
        }

        match Network::with_params(vec![NeuronValue(0); 2].into(), vec![Effect(0)].into(), vec![].into(), vec![].into()) {
            Err(Error::ZeroConnections) => (),
            _ => panic!(),
        }

        match Network::with_params(vec![NeuronValue(0); 2].into(), vec![Effect(0); 6].into(), vec![].into(), vec![].into()) {
            Err(Error::TooManyConnections) => (),
            _ => panic!(),
        }

        match Network::with_params(vec![NeuronValue(0); 2].into(), vec![Effect(0); 4].into(), vec![2usize].into(), vec![].into()) {
            Err(Error::InvalidNeuronIndex) => (),
            _ => panic!(),
        }

        match Network::with_params(vec![NeuronValue(0); 2].into(), vec![Effect(0); 4].into(), vec![].into(), vec![2usize].into()) {
            Err(Error::InvalidNeuronIndex) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn tick_io() {
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

        let input_neurons = vec![
            2,
            0,
        ].into();
        
        let output_neurons = vec![
            2,
            0,
            1,
        ].into();

        let mut net = Network::with_params(
            action_potentials,
            effects,
            input_neurons,
            output_neurons,
        ).unwrap();

        // first evaluate 2 ticks for expected output
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

        // then check if inputs and outputs are handled correctly
        net.apply_inputs(
            &[
                NeuronValue(-9),
                NeuronValue(0),
            ]
        );

        net.tick();

        assert_eq!(
            net.last_accumulator_buf(),
            &[
                NeuronValue(-17),
                NeuronValue(127),
                NeuronValue(0),
            ],
        ); 

        let mut outputs = [NeuronValue(0); 3];

        net.read_outputs(&mut outputs);

        assert_eq!(
            &outputs,
            &[
                NeuronValue(0),
                NeuronValue(-17),
                NeuronValue(127),
            ],
        );
    }
}
