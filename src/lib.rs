#[macro_use]
extern crate lazy_static;

use std::sync::Mutex;

lazy_static! {
  static ref GLOBAL_RNG: Mutex<XorShift128Plus> = Mutex::new(XorShift128Plus::new_seeded([4711, 17]));
}

pub type StdRng = XorShift64Star;

pub trait Rng {
  fn gen_u32(&mut self) -> u32 {
    self.gen_u64() as u32
  }
  fn gen_u64(&mut self) -> u64 {
    let x = self.gen_u32() as u64;
    let y = self.gen_u32() as u64;
    x << 32 | y
  }
}


pub trait Rand : Rng {
  fn limit(&mut self, max: u32) -> u32;
  fn chance(&mut self, num: u32, denom: u32) -> bool;
}

impl<T: Rng> Rand for T {
  fn limit(&mut self, max: u32) -> u32 {
    if max < ((std::u32::MAX as f32 * 0.00001) as u32) {
      self.gen_u32() % max
    } else {
      (self.gen_u64() % (max as u64)) as u32
    }
  }

  fn chance(&mut self, num: u32, denom: u32) -> bool {
    self.limit(denom) < num
  }
}

pub trait SeedFrom<T> {
  fn seed_from(T) -> Self;
}

impl<T> SeedFrom<T> for T {
  fn seed_from(x: T) -> Self { x }
}
impl SeedFrom<i32> for u64 {
  fn seed_from(x: i32) -> Self { x as u64 }
}
impl SeedFrom<u32> for u64 {
  fn seed_from(x: u32) -> Self { x as u64 }
}
impl SeedFrom<i64> for u64 {
  fn seed_from(x: i64) -> Self { x as u64 }
}
impl<T> SeedFrom<T> for [u64; 2] where u64: SeedFrom<T> {
  fn seed_from(x: T) -> Self { let y = u64::seed_from(x); [y, y] }
}


pub struct XorShift128Plus {
  state: [u64; 2],
}

impl XorShift128Plus {
  pub fn new_seeded<T>(seed: T) -> Self where [u64; 2]: SeedFrom<T> {
    let mut seed = <[u64; 2]>::seed_from(seed);
    if seed[0] == 0 &&
       seed[1] == 0 {
      seed[0] = 32766;
    }
    XorShift128Plus { state: seed }
  }
  pub fn new() -> Self {
    let mut glob = GLOBAL_RNG.lock().unwrap();
    XorShift128Plus::new_seeded([glob.gen_u64(), glob.gen_u64()])
  }
}

impl Rng for XorShift128Plus {
  fn gen_u64(&mut self) -> u64 {
    let mut x = self.state[0];
    let y = self.state[1];
    self.state[0] = y;
    x ^= x << 23;
    self.state[1] = x ^ y ^ (x >> 17) ^ (y >> 26);
    return self.state[1].wrapping_add(y);
  }
}


pub struct XorShift64Star {
  state: u64,
}

impl XorShift64Star {
  pub fn new_seeded<T>(seed: T) -> Self where u64: SeedFrom<T> {
    let mut seed = u64::seed_from(seed);
    if seed == 0 {
      seed = 32766;
    }
    XorShift64Star { state: seed }
  }
  pub fn new() -> Self {
    XorShift64Star::new_seeded(GLOBAL_RNG.lock().unwrap().gen_u64())
  }
}

impl Rng for XorShift64Star {
  fn gen_u64(&mut self) -> u64 {
    let mut x = self.state;
    x ^= x << 12;
    x ^= x >> 25;
    x ^= x << 27;
    self.state = x;
    x.wrapping_mul(0x2545F4914F6CDD1Du64)
  }
}


#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn seeded_equal() {
    for seed in 0..100 {
      let mut a = StdRng::new_seeded(seed * 245);
      let mut b = StdRng::new_seeded(seed * 245);
      assert!(is_equal(&mut a, &mut b, 1000));
    }
  }

  #[test]
  fn seeded_different() {
    for seed in 0..100 {
      let mut a = StdRng::new_seeded(seed * 245);
      let mut b = StdRng::new_seeded(seed * 245 + 1);
      assert!(!is_equal(&mut a, &mut b, 5000));
      let mut c = StdRng::new_seeded(seed);
      let mut d = StdRng::new_seeded(seed * 2 + 1);
      assert!(!is_equal(&mut c, &mut d, 5000));
    }
  }

  #[test]
  fn unseeded_different() {
    for _ in 0..100 {
      let mut a = StdRng::new();
      let mut b = StdRng::new();
      assert!(!is_equal(&mut a, &mut b, 5000));
    }
  }

  #[test]
  fn consumed_different() {
    for seed in 0..100 {
      let mut a = StdRng::new_seeded(seed * 253);
      let mut b = StdRng::new_seeded(seed * 253);
      b.gen_u32();
      assert!(!is_equal(&mut a, &mut b, 5000));
    }
  }

  fn is_equal(a: &mut Rng, b: &mut Rng, num: u32) -> bool {
    for _ in 0..num {
      if a.gen_u32() != b.gen_u32() {
        return false;
      }
    }
    true
  }

  #[test]
  fn test_u64() {
    let mut a = StdRng::new_seeded(4711);
    let mut b = StdRng::new_seeded(4711);
    let mut bits = 0u64;
    for _ in 0..5000 {
      let x = a.gen_u64();
      assert_eq!(x, b.gen_u64());
      bits |= x;
    }
    assert_eq!(bits, 0xffff_ffff_ffff_ffff);
  }

  #[test]
  fn test_limit() {
    let mut a = StdRng::new();
    let mut counts = [0u32; 47];
    for _ in 0..4700 {
      counts[a.limit(47) as usize] += 1;
    }
    for count in counts.iter() {
      assert!(50 <= *count && *count <= 150);
    }
  }

  #[test]
  fn test_chance() {
    let mut a = StdRng::new();
    let mut count = 0;
    for _ in 0..5432 {
      if a.chance(321, 5432) {
        count += 1;
      }
    }
    assert!(260 <= count && count <= 380);
  }

  #[test]
  fn test_api() {
    fn my_func(r: &mut Rand) {
      r.gen_u32();
      r.gen_u64();
    }

    let mut a = StdRng::new();
    my_func(&mut a);
  }

  #[test]
  fn test_seed_from() {
    XorShift128Plus::new_seeded([1u64, 2u64]);
    XorShift128Plus::new_seeded(-1i64);
    XorShift128Plus::new_seeded(-1i32);
    XorShift128Plus::new_seeded(1u64);
    XorShift128Plus::new_seeded(1u32);
    XorShift64Star::new_seeded(-1i64);
    XorShift64Star::new_seeded(-1i32);
    XorShift64Star::new_seeded(1u64);
    XorShift64Star::new_seeded(1u32);
  }
}

