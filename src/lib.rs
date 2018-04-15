#[macro_use]
extern crate lazy_static;

use std::sync::Mutex;
use std::collections::HashSet;

lazy_static! {
  static ref GLOBAL_RNG: Mutex<XorShift128Plus> = Mutex::new(XorShift128Plus::new_seeded([4711, 17]));
}

pub type StdRng = XorShift64Star;

pub trait Rangeable : Sized {
  fn rng_below<R: Rng + ?Sized>(rng: &mut R, limit: Self) -> Self;
  fn rng_range<R: Rng + ?Sized>(rng: &mut R, start: Self, end: Self) -> Self;
}

pub trait Rng {
  fn gen_u32(&mut self) -> u32 {
    self.gen_u64() as u32
  }
  fn gen_u64(&mut self) -> u64 {
    let x = self.gen_u32() as u64;
    let y = self.gen_u32() as u64;
    x << 32 | y
  }

  fn below<T>(&mut self, limit: T) -> T where T: Rangeable {
    T::rng_below(self, limit)
  }

  fn range<T>(&mut self, start: T, end: T) -> T where T: Rangeable {
    T::rng_range(self, start, end)
  }

  fn chance(&mut self, num: u32, denom: u32) -> bool {
    self.below(denom) < num
  }

  fn zeroone(&mut self) -> f32 {
    let frac = self.gen_u32() & ((1 << 24) - 1);
    const EXP: u32 = 127u32 << 23;
    unsafe { std::mem::transmute::<u32, f32>(frac | EXP) - 1.0 }
  }

  fn shuffle<U>(&mut self, values: &mut [U]) {
    for i in (1..values.len()).rev() {
      values.swap(i, self.below(i + 1));
    }
  }

  fn permutation(&mut self, size: usize) -> Vec<usize> {
    let mut res = (0usize..size).collect::<Vec<usize>>();
    self.shuffle(&mut res);
    res
  }

  fn choose<'a, T>(&mut self, values: &'a [T]) -> Option<&'a T> {
    if values.is_empty() {
      None
    } else {
      Some(&values[self.below(values.len())])
    }
  }

  fn choose_mut<'a, T>(&mut self, values: &'a mut [T]) -> Option<&'a mut T> {
    if values.is_empty() {
      None
    } else {
      let len = values.len();
      Some(&mut values[self.below(len)])
    }
  }

  fn sample<'a, T>(&'a mut self, values: &'a [T], sample_size: usize) -> SampleIter<'a, Self, T> where Self: Sized {
    SampleIter::new(self, &values, sample_size)
  }
}

enum SampleIterData {
  Picked(HashSet<usize>),
  Remaining(Vec<usize>),
}
pub struct SampleIter<'a, R, T> where R: 'a + Rng, T: 'a {
  rng: &'a mut R,
  values: &'a [T],
  n: usize,
  data: SampleIterData,
}

impl<'a, R, T> SampleIter<'a, R, T> where R: 'a + Rng, T: 'a {
  fn new(rng: &'a mut R, values: &'a [T], sample_size: usize) -> SampleIter<'a, R, T> {
    let set_size = values.len();
    if sample_size > set_size {
      panic!("Sample set smaller than requested sample size");
    }

    // For small values of 'sample_size', it's better to use a hash set to track
    // which entries we've already selected and re-select if we end up
    // selecting the same entry again.
    // If 'sample_size' is closer to values.len(), it's better to track which entries
    // we've not yet selected and pick from this set.
    // Where the line goes between which approach is faster needs to be tuned.
    // The below is a very rough guess.
    if sample_size * 4 < set_size {
      SampleIter { rng, values, n: sample_size, data: SampleIterData::Picked(HashSet::new()) }
    } else {
      SampleIter { rng, values, n: sample_size, data: SampleIterData::Remaining((0..set_size).collect()) }
    }
  }
}

impl<'a, R, T> Iterator for SampleIter<'a, R, T> where R: 'a + Rng, T: 'a {
  type Item = &'a T;

  fn next(&mut self) -> Option<&'a T> {
    if self.n == 0 {
      return None;
    }
    self.n -= 1;

    match self.data {
      SampleIterData::Picked(ref mut picked) => {
        let mut pos = self.rng.below(self.values.len());
        while !picked.insert(pos) {
          pos = self.rng.below(self.values.len());
        }
        // Could use get_unchecked on index reference for performance.
        Some(&self.values[pos])
      },
      SampleIterData::Remaining(ref mut remaining) => {
        let pos = self.rng.below(self.n + 1);
        // Could use get_unchecked on all index references for performance.
        let res_pos = remaining[pos];
        remaining[pos] = remaining[self.n];
        Some(&self.values[res_pos])
      },
    }
  }

  fn size_hint(&self) -> (usize, Option<usize>) {
    (self.n, Some(self.n))
  }
}

impl<'a, R, T> ExactSizeIterator for SampleIter<'a, R, T> where R: 'a + Rng, T: 'a {}

macro_rules! impl_rangable {
  ($ty: ty, $uty: ty) => (
    #[allow(unused_comparisons)]
    impl Rangeable for $ty {
      fn rng_below<R: Rng + ?Sized>(rng: &mut R, limit: $ty) -> $ty {
        if limit < 0 {
          panic!("Rng.below() called with limit < 0");
        }
        if (limit as u32) < 429_496 { // 32bit max / 10000. I.e. bias is less than 0.01%
          (rng.gen_u32() % (limit as u32)) as $ty
        } else {
          (rng.gen_u64() % (limit as u64)) as $ty
        }
      }

      fn rng_range<R: Rng + ?Sized>(rng: &mut R, start: $ty, end: $ty) -> $ty {
        if start >= end {
          panic!("empty or inverted range");
        }
        let diff = end.wrapping_sub(start) as $uty;
        (rng.below(diff) as $ty).wrapping_add(start)
      }
    }
  )
}
impl_rangable!(u8, u8);
impl_rangable!(i8, u8);
impl_rangable!(u16, u16);
impl_rangable!(i16, u16);
impl_rangable!(u32, u32);
impl_rangable!(i32, u32);
impl_rangable!(u64, u64);
impl_rangable!(i64, u64);
impl_rangable!(usize, usize);
impl_rangable!(isize, usize);

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
  fn test_below() {
    let mut a = StdRng::new();
    let mut counts = [0u32; 47];
    for _ in 0..4700 {
      counts[a.below(47usize)] += 1;
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

  /*
  For now I have not figured out how to make this work due to the type parameters in
  several functions.
  #[test]
  fn test_trait_obj() {
    fn my_func(r: &mut Rng) {
      r.gen_u32();
      r.gen_u64();
      r.below(5);
      r.chance(1, 10);
    }

    let mut a = StdRng::new();
    my_func(&mut a);
  }
  */

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

  struct TestRng {
    x: u32,
    fail64: bool,
  }
  impl TestRng {
    fn new(x: u32) -> Self { TestRng { x, fail64: false } }
    fn new_fail64(x: u32) -> Self { TestRng { x, fail64: true } }
  }
  impl Rng for TestRng {
    fn gen_u32(&mut self) -> u32 {
      self.x
    }
    fn gen_u64(&mut self) -> u64 {
      if self.fail64 {
        panic!("gen_u64 called");
      }
      ((self.x as u64) << 32) | (self.x as u64)
    }
  }

  #[test]
  fn test_zeroone() {
    let mut a = StdRng::new();
    for _ in 0..5000 {
      let v = a.zeroone();
      assert!(0.0 <= v && v < 1.0);
    }
    let mut t = TestRng::new(0);
    assert!(t.zeroone() == 0.0);
    t.x = 0xffff_ffff;
    assert!(t.zeroone() > 0.0);
    assert!(t.zeroone() < 1.0);
    assert!(t.zeroone() > 0.999);
  }

  #[test]
  fn test_shuffle() {
    let mut positions = [0u32; 16];
    let mut a = StdRng::new();
    for _ in 0..10000 {
      let mut arr: [usize; 4] = [0, 1, 2, 3];
      a.shuffle(&mut arr);
      for i in 0..4 {
        positions[i * 4 + arr[i]] += 1;
      }
    }
    for count in positions.iter() {
      assert!(2300 <= *count && *count <= 2700);
    }
  }

  #[test]
  fn test_choose() {
    let mut a = StdRng::new();
    let chars = "abcdefghijklmn".chars().collect::<Vec<char>>();
    let mut chosen = Vec::new();
    chosen.resize(chars.len(), 0i32);
    for _ in 0..10000 {
      let picked = *a.choose(&chars).unwrap();
      chosen[(picked as usize) - ('a' as usize)] += 1;
    }
    for count in chosen.iter() {
      let err = *count - (10000 / (chars.len() as i32));
      assert!(-150 <= err && err <= 150);
    }


    chosen.truncate(0);
    chosen.resize(40, 0i32);

    for _ in 0..10000 {
      *a.choose_mut(&mut chosen).unwrap() += 1;
    }
    for count in chosen.iter() {
      let err = *count - (10000 / (chosen.len() as i32));
      assert!(-150 <= err && err <= 150);
    }
  }

  #[test]
  fn test_below_types() {
    let mut a = StdRng::new();
    assert_eq!(a.below(1i8), 0);
    assert_eq!(a.below(1u8), 0);
    assert_eq!(a.below(1i16), 0);
    assert_eq!(a.below(1u16), 0);
    assert_eq!(a.below(1i32), 0);
    assert_eq!(a.below(1u32), 0);
    assert_eq!(a.below(1i64), 0);
    assert_eq!(a.below(1u64), 0);
    assert_eq!(a.below(1isize), 0);
    assert_eq!(a.below(1usize), 0);
    assert!(std::panic::catch_unwind(|| TestRng::new_fail64(0).below(std::u32::MAX)).is_err());
    let mut t = TestRng::new_fail64(0);
    t.below(std::u16::MAX);
    t.below(std::i16::MAX);
    t.below(std::u8::MAX);
    t.below(std::i8::MAX);
  }

  #[test]
  fn test_range_primitives() {
    let mut a = StdRng::new();
    assert_eq!(a.range(-2i8, -1), -2);
    assert_eq!(a.range(1u8, 2), 1);
    assert_eq!(a.range(-2i16, -1), -2);
    assert_eq!(a.range(1u16, 2), 1);
    assert_eq!(a.range(-2i32, -1), -2);
    assert_eq!(a.range(1u32, 2), 1);
    assert_eq!(a.range(-2i64, -1), -2);
    assert_eq!(a.range(1u64, 2), 1);
    assert_eq!(a.range(-2isize, -1), -2);
    assert_eq!(a.range(1usize, 2), 1);

    for _ in 0..10000 {
      let x = a.range(-120i8, 120);
      assert!(-120 <= x && x < 120);
    }

    let x = a.range(-120i8, 120);
    assert!(-120 <= x && x < 120);
    let x = a.range(-32760i16, 32760);
    assert!(-32760 <= x && x < 32760);
    let x = a.range(std::i32::MIN, std::i32::MAX);
    assert!(std::i32::MIN <= x && x < std::i32::MAX);
    let x = a.range(std::i64::MIN, std::i64::MAX);
    assert!(std::i64::MIN <= x && x < std::i64::MAX);
  }

  #[test]
  fn test_below_panics() {
    assert!(std::panic::catch_unwind(|| StdRng::new().below(0u32)).is_err());
    assert!(std::panic::catch_unwind(|| StdRng::new().below(-100i8)).is_err());
    assert!(std::panic::catch_unwind(|| StdRng::new().range(2, 1)).is_err());
    assert!(std::panic::catch_unwind(|| StdRng::new().range(20u8, 10u8)).is_err());
  }

  #[test]
  fn test_permutation() {
    let mut a = StdRng::new();
    let mut found = Vec::new();
    found.resize(20, false);
    let perm = a.permutation(found.len());
    for pos in perm.iter() {
      assert_eq!(found[*pos], false);
      found[*pos] = true;
    }
    for val in found {
      assert_eq!(val, true);
    }
  }

  #[test]
  fn test_sample() {
    let mut a = StdRng::new();
    let vals = (0u16..1000).rev().collect::<Vec<u16>>();
    for i in 0..100 {
      let selected = a.sample(&vals, i * 10);
      assert_eq!(selected.len(), i * 10);
      let mut found = HashSet::new();
      for val_ref in selected {
        assert!(found.insert(*val_ref));
      }
      assert_eq!(found.len(), i * 10);
    }
    for i in 0..100 {
      let selected: Vec<u16> = a.sample(&vals, i * 10).map(|x| *x).collect();
      assert_eq!(selected.len(), i * 10);
      let mut found = HashSet::new();
      for val_ref in selected {
        assert!(found.insert(val_ref));
      }
    }
  }
}
