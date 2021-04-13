use crate::{Network, network::{NeuronValue, Effect}};

use rand::{prelude::*, distributions};
use rand_chacha::ChaCha8Rng;

/// Parameters for a noise pass, see [build_network_from_noise].
#[derive(Debug, Clone, Copy)]
pub struct NoisePassParams {
    pub seed: u64,
    pub power: u8,
}

/// Apply noise to the parameters of a [Network].
/// This process is deterministic.  
/// `power` is a value related to the magnitude of the noise.
/// The higher this value, the more the network parameters will change on average.
/// The following table gives an estimate of the probabilities of certain offsets on the parameters  
/// ```text
/// | power   | 0    | 1, -1 | 2, -2 | 3, -3 |  
/// |---------|------|-------|-------|-------|
/// | 0       | 0.50 | 0.13  | 0.04  | 0.02  |  
/// | 1       | 0.33 | 0.13  | 0.06  | 0.03  |  
/// | 2       | 0.25 | 0.12  | 0.06  | 0.04  |  
/// | 3       | 0.20 | 0.11  | 0.06  | 0.04  |  
/// ```
pub fn apply_parameter_noise(
    net: &mut Network, 
    seed: u64,
    power: u8,
) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let dist = distributions::Uniform::from(u64::MIN..=u64::MAX);
    let p = power as u64;

    let mut offset = move || -> i64 {
        let r = dist.sample(&mut rng);
        // for a large part of the domain, this will produce values close to 0
        // also, it is unlikely to skip values at least at reasonable powers
        let mut unsigned = u64::MAX / (r / (1 + p));
        // this makes sure powers >= 1 still have 0 as possible output
        unsigned -= p;
        let sign = -1 + (2 * (r % 2)) as i64;
        
        (unsigned / 2) as i64 * sign
    };
    
    for effect in net.effects_mut().iter_mut() {
        let noise = offset()
            .clamp(i8::MIN as i64, i8::MAX as i64) as i8;
        // saturating add because a small offset should never cause a huge difference in
        // the parameter value
        effect.0 = effect.0.saturating_add(noise);
    }

    for treshold in net.tresholds_mut().iter_mut() {
        let noise = offset()
            .clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        treshold.0 = treshold.0.saturating_add(noise);
    }
    
    // let tmp: Vec<_> = std::iter::repeat_with(|| offset())
    //     .take(655360)
    //     .collect();
    
    // println!("min: {:?}, max: {:?}", tmp.iter().min(), tmp.iter().max());
    // println!("ratio {:.6}", tmp.iter().filter(|&&t| t < 0).count() as f64 / tmp.iter().filter(|&&t| t > 0).count() as f64);
    // for i in 0..5 {
    //     let positive = tmp.iter()
    //         .filter(|&&t| t == i)
    //         .count();
    //     
    //     println!(" {}: {} {:.6}", i, positive, positive as f64 / tmp.len() as f64);
    //     if i != 0 {
    //         let negative = tmp.iter()
    //             .filter(|&&t| t == -i)
    //             .count();
    //     
    //         println!("{}: {} {:.6}", -i, negative, negative as f64 / tmp.len() as f64);
    //     }
    // }
}

/// Constructs a [Network] by generating initial parameters with `seed`,
/// then applying the specified `passes` of noise.  
/// See [apply_parameter_noise] for more information.
/// # Panics
/// See [Network::new]. 
pub fn build_network_from_noise<Is>(
    neuron_count: usize,
    connection_count: usize,
    seed: u64,
    passes: Is,
) -> Network
where
    Is: Iterator<Item = NoisePassParams>
{
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let tresholds = std::iter::repeat_with(|| NeuronValue(rng.gen()))
        .take(neuron_count)
        .collect();
    
    let effects = std::iter::repeat_with(|| Effect(rng.gen()))
        .take(neuron_count.checked_mul(connection_count).unwrap())
        .collect();
    
    let mut net = Network::with_params(tresholds, effects);

    for pass in passes {
        apply_parameter_noise(&mut net, pass.seed, pass.power);
    }

    net
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_determinism() {
        let passes = (0..=u8::MAX)
            .map(|i| NoisePassParams { seed: i as u64 + 1234, power: u8::MAX - i });

        let net = build_network_from_noise(16, 2, 1234, passes);

        assert_eq!(
            net.tresholds(),
            &[
                NeuronValue(-1278279962),
                NeuronValue(1657864061),
                NeuronValue(239123806),
                NeuronValue(-15785932),
                NeuronValue(-455062199),
                NeuronValue(-1731366824),
                NeuronValue(597245901),
                NeuronValue(1358662888),
                NeuronValue(555452750),
                NeuronValue(646707917),
                NeuronValue(-344060829),
                NeuronValue(1485825241),
                NeuronValue(-1644047160),
                NeuronValue(-1839883124),
                NeuronValue(-1904695363),
                NeuronValue(702228411),
            ],
        );

        assert_eq!(
            net.effects(),
            &[
                Effect(117), Effect(-128),
                Effect(-102), Effect(124),
                Effect(-103), Effect(-64),
                Effect(-32), Effect(65),
                Effect(-71), Effect(73),
                Effect(-82), Effect(-128),
                Effect(127), Effect(-87),
                Effect(23), Effect(-74),
                Effect(-97), Effect(-91),
                Effect(-69), Effect(-63),
                Effect(89), Effect(80),
                Effect(-79), Effect(-123),
                Effect(-102), Effect(-3),
                Effect(126), Effect(-107),
                Effect(127), Effect(121),
                Effect(95), Effect(-128),
            ],
        );
    }
}
