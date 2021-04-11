use crate::Network;

use rand::{prelude::*, distributions};
use rand_chacha::ChaCha8Rng;

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

    for action_potential in net.action_potentials_mut().iter_mut() {
        let noise = offset()
            .clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        action_potential.0 = action_potential.0.saturating_add(noise);
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

