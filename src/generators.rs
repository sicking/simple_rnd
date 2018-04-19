use super::Rng;
use std::sync::Mutex;

lazy_static! {
  static ref GLOBAL_RNG: Mutex<XorShift128Plus> = Mutex::new(XorShift128Plus::new_seeded([4711, 17]));
}

pub type StdRng = XorShift64Star;

pub trait SeedFrom<T> {
  fn seed_from(T) -> Self;
}

impl<T> SeedFrom<T> for T {
  fn seed_from(x: T) -> Self { x }
}
impl<T> SeedFrom<T> for [u64; 2] where u64: SeedFrom<T> {
  fn seed_from(x: T) -> Self { let y = u64::seed_from(x); [y, y] }
}
macro_rules! impl_seed_from {
  ($from_ty: ty, $to_ty: ty) => (
    impl SeedFrom<$from_ty> for $to_ty {
      fn seed_from(x: $from_ty) -> Self { x as $to_ty }
    }
  )
}
impl_seed_from!(i8, u64);
impl_seed_from!(u8, u64);
impl_seed_from!(i16, u64);
impl_seed_from!(u16, u64);
impl_seed_from!(i32, u64);
impl_seed_from!(u32, u64);
impl_seed_from!(i64, u64);
//impl_seed_from!(u64, u64);
impl_seed_from!(isize, u64);
impl_seed_from!(usize, u64);


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
    self.state[1].wrapping_add(y)
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

  fn is_equal<T: Rng>(a: &mut T, b: &mut T, num: u32) -> bool {
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
  fn test_seed_from() {
    XorShift128Plus::new_seeded([1u64, 2u64]);
    XorShift128Plus::new_seeded(-1i64);
    XorShift128Plus::new_seeded(-1i32);
    XorShift128Plus::new_seeded(-1i16);
    XorShift128Plus::new_seeded(-1i8);
    XorShift128Plus::new_seeded(1u64);
    XorShift128Plus::new_seeded(1u32);
    XorShift128Plus::new_seeded(1u16);
    XorShift128Plus::new_seeded(1u8);
    XorShift64Star::new_seeded(-1i64);
    XorShift64Star::new_seeded(-1i32);
    XorShift64Star::new_seeded(-1i16);
    XorShift64Star::new_seeded(-1i8);
    XorShift64Star::new_seeded(1u64);
    XorShift64Star::new_seeded(1u32);
    XorShift64Star::new_seeded(1u16);
    XorShift64Star::new_seeded(1u8);
  }
}
