#[macro_use]
extern crate lazy_static;

mod generators;
pub use generators::{ SeedFrom, XorShift128Plus, XorShift64Star, StdRng };

#[cfg(feature = "bigint")]
mod rng_bigint;
#[cfg(feature = "bigint")]
use rng_bigint::BigUint;

use std::collections::HashSet;
use std::ops::AddAssign;

pub trait Rangeable : Sized {
  type Output;
  fn rng_below<R: Rng + ?Sized>(rng: &mut R, limit: Self) -> Self::Output;
  fn rng_range<R: Rng + ?Sized>(rng: &mut R, start: Self, end: Self) -> Self::Output;
  fn zero() -> Self::Output;
  fn is_neg(&self) -> bool;
}
pub trait ZeroOneable : Sized {
  fn rng_zeroone<R: Rng + ?Sized>(rng: &mut R) -> Self;
}
pub trait ToWeightedChoice : Sized {
  type Item;
  type Weight;
  fn to_weighted_choice(self) -> (Self::Item, Self::Weight);
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

  fn below<T, O>(&mut self, limit: T) -> O where T: Rangeable<Output=O> {
    T::rng_below(self, limit)
  }

  fn range<T, O>(&mut self, start: T, end: T) -> O where T: Rangeable<Output=O> {
    T::rng_range(self, start, end)
  }

  fn chance(&mut self, num: u32, denom: u32) -> bool {
    self.below(denom) < num
  }

  fn zeroone<T: ZeroOneable>(&mut self) -> T {
    T::rng_zeroone(self)
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

  fn choose_weighted<Iter, T, W, S>(&mut self, values: Iter, total_weight: S) -> Option<T>
    where Iter: Iterator,
          Iter::Item: ToWeightedChoice<Item=T, Weight=W>,
          W: Rangeable,
          S: Rangeable<Output=S> + AddAssign<W> + PartialOrd<S> + PartialEq<S> + std::fmt::Debug + Clone {
            // S:Clone is only required by debug assertions
    if total_weight == S::zero() {
      return None;
    }

    let mut cumulative_weight = S::zero();
    let mut debug_result = None;
    let debug_total_weight = if cfg!(debug_assertions) { Some(total_weight.clone()) } else { None };
    let val = self.below(total_weight);

    for (item, weight) in values.map(|x| x.to_weighted_choice()) {
      if weight.is_neg() {
        panic!("Negative weight");
      }

      cumulative_weight += weight;

      if cumulative_weight > val {
        if !cfg!(debug_assertions) {
          return Some(item);
        }
        if debug_result.is_none() {
          debug_result = Some(item);
        }
      }
    }

    debug_assert_eq!(debug_total_weight.unwrap(), cumulative_weight);
    if cfg!(debug_assertions) && debug_result.is_some() {
      return debug_result;
    }

    panic!("total_weight did not match up with sum of weights");
  }

  fn choose_weighted2<ValIter, WeightIter, T, W, S>(&mut self, values: ValIter, weights: WeightIter) -> Option<T>
    where ValIter: Iterator<Item=T>,
          WeightIter: Iterator<Item=W> + Clone,
          W: std::ops::Add<W, Output=S> + Rangeable,
          S: std::iter::Sum<W> + Rangeable<Output=S> + AddAssign<W> + PartialOrd<S> {

    let total_weight: S = weights.clone().map(|w| { if w.is_neg() { panic!("Negative weight"); } w }).sum();

    if total_weight == S::zero() {
      return None;
    }

    let mut cumulative_weight = S::zero();
    let val = self.below(total_weight);

    for (item, weight) in values.zip(weights) {
      cumulative_weight += weight;

      if cumulative_weight > val {
        return Some(item);
      }
    }

    panic!("clone weight did not match up with sum of weights");
  }

  fn sample<'a, T>(&'a mut self, values: &'a [T], sample_size: usize) -> SampleIter<'a, Self, T> where Self: Sized {
    SampleIter::new(self, &values, sample_size)
  }

  #[cfg(feature = "bigint")]
  fn gen_biguint(&mut self, bits: usize) -> BigUint {
    let top_bits = bits % 32;
    let words = bits / 32;
    let mut data = Vec::with_capacity((words + 31) / 32);
    for _ in 0..words {
      data.push(self.gen_u32());
    }
    if top_bits > 0 {
      data.push(self.gen_u32() >> (32 - top_bits));
    }
    BigUint::new(data)
  }

  #[cfg(test)]
  fn test_gen_u32(&mut self, _limit: u32) -> u32 {
    self.gen_u32()
  }
  #[cfg(test)]
  fn test_gen_u64(&mut self, _limit: u64) -> u64 {
    self.gen_u64()
  }
  #[cfg(all(test, feature = "bigint"))]
  fn test_gen_biguint(&mut self, bits: usize, _limit: &BigUint) -> BigUint {
    self.gen_biguint(bits)
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

#[cfg(not(test))]
macro_rules! call_gen_32 {
  ($rng: ident, $limit: expr) => ($rng.gen_u32())
}
#[cfg(test)]
macro_rules! call_gen_32 {
  ($rng: ident, $limit: expr) => ($rng.test_gen_u32($limit))
}
#[cfg(not(test))]
macro_rules! call_gen_64 {
  ($rng: ident, $limit: expr) => ($rng.gen_u64())
}
#[cfg(test)]
macro_rules! call_gen_64 {
  ($rng: ident, $limit: expr) => ($rng.test_gen_u64($limit))
}

macro_rules! impl_rangable {
  ($ty: ty, $uty: ty) => (
    #[allow(unused_comparisons)]
    impl Rangeable for $ty {
      type Output = $ty;

      fn rng_below<R: Rng + ?Sized>(rng: &mut R, limit: $ty) -> $ty {
        if limit.is_neg() {
          panic!("Rng.below() called with limit < 0");
        }
        if (limit as u32) < (std::u32::MAX / 10000) { // I.e. bias is less than 0.01%
          (call_gen_32!(rng, limit as u32) % (limit as u32)) as $ty
        } else {
          (call_gen_64!(rng, limit as u64) % (limit as u64)) as $ty
        }
      }

      fn rng_range<R: Rng + ?Sized>(rng: &mut R, start: $ty, end: $ty) -> $ty {
        if start >= end {
          panic!("empty or inverted range");
        }
        let diff = end.wrapping_sub(start) as $uty;
        (rng.below(diff) as $ty).wrapping_add(start)
      }

      fn zero() -> Self::Output {
        return 0;
      }

      fn is_neg(&self) -> bool {
        return *self < 0;
      }
    }

    #[allow(unused_comparisons)]
    impl<'a> Rangeable for &'a $ty {
      type Output = $ty;

      fn rng_below<R: Rng + ?Sized>(rng: &mut R, limit: &'a $ty) -> $ty {
        <$ty>::rng_below(rng, *limit)
      }

      fn rng_range<R: Rng + ?Sized>(rng: &mut R, start: &'a $ty, end: &'a $ty) -> $ty {
        <$ty>::rng_range(rng, *start, *end)
      }

      fn zero() -> Self::Output {
        return 0;
      }

      fn is_neg(&self) -> bool {
        return **self < 0;
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

macro_rules! impl_float_rangable {
  ($ty: ty, $significand_bits: expr, $exp_bits: expr, $gen_func: ident) => (
    impl Rangeable for $ty {
      type Output = $ty;

      fn rng_below<R: Rng + ?Sized>(rng: &mut R, limit: $ty) -> $ty {
        Self::rng_range(rng, 0.0, limit)
      }

      fn rng_range<R: Rng + ?Sized>(rng: &mut R, start: $ty, end: $ty) -> $ty {
        let start_exp = (start.to_bits() >> $significand_bits) & ((1 << $exp_bits) - 1);
        let end_exp = (end.to_bits() >> $significand_bits) & ((1 << $exp_bits) - 1);
        if start_exp > ((1 << $exp_bits) - 4) || end_exp > ((1 << $exp_bits) - 4) {
          panic!("Overflow or NaN");
        }

        let scale = end - start;
        let offset = start - scale;
        assert!(scale.is_finite() && offset.is_finite() && start.is_finite() && end.is_finite());
        let exp = ((1 << ($exp_bits - 1)) - 1) << $significand_bits;
        loop {
          let frac = rng.$gen_func() & ((1 << $significand_bits) - 1);
          let res = <$ty>::from_bits(frac | exp) * scale + offset;
          // Check for rounding errors
          if res >= start && res < end {
            return res
          }
        }
      }

      fn zero() -> Self::Output {
        return 0.0
      }

      fn is_neg(&self) -> bool {
        return *self < 0.0;
      }
    }
    impl<'a> Rangeable for &'a $ty {
      type Output = $ty;

      fn rng_below<R: Rng + ?Sized>(rng: &mut R, limit: &'a $ty) -> $ty {
        <$ty>::rng_below(rng, *limit)
      }

      fn rng_range<R: Rng + ?Sized>(rng: &mut R, start: &'a $ty, end: &'a $ty) -> $ty {
        <$ty>::rng_range(rng, *start, *end)
      }

      fn zero() -> Self::Output {
        return 0.0;
      }

      fn is_neg(&self) -> bool {
        return **self < 0.0;
      }
    }
  )
}
impl_float_rangable!(f64, 52, 11, gen_u64);
impl_float_rangable!(f32, 23, 8, gen_u32);

macro_rules! impl_zerooneable {
  ($ty: ty, $significand_bits: expr, $gen_func: ident, $exp_bias: expr) => (
    impl ZeroOneable for $ty {
      fn rng_zeroone<R: Rng + ?Sized>(rng: &mut R) -> $ty {
        let frac = rng.$gen_func() & ((1 << $significand_bits) - 1);
        let exp = $exp_bias << $significand_bits;
        <$ty>::from_bits(frac | exp) - 1.0
      }
    }
  )
}
impl_zerooneable!(f32, 23, gen_u32, 127);
impl_zerooneable!(f64, 52, gen_u64, 1023);

impl<'a, T, W> ToWeightedChoice for &'a(T, W) where W: Copy {
  type Item = &'a T;
  type Weight = W;
  fn to_weighted_choice(self) -> (Self::Item, Self::Weight) { (&self.0, self.1) }
}
impl<'a, T, W> ToWeightedChoice for &'a mut (T, W) where W: Copy {
  type Item = &'a mut T;
  type Weight = W;
  fn to_weighted_choice(self) -> (Self::Item, Self::Weight) { (&mut self.0, self.1) }
}
impl<T, W> ToWeightedChoice for (T, W) where W: Copy {
  type Item = T;
  type Weight = W;
  fn to_weighted_choice(self) -> (Self::Item, Self::Weight) { self }
}


#[cfg(test)]
mod tests {
  use super::*;
  use std::panic::catch_unwind;

  #[test]
  fn test_below() {
    let mut a = StdRng::new();
    let mut counts = [0u32; 47];
    for _ in 0..4700 {
      counts[a.below(47usize)] += 1;
    }
    for count in counts.iter() {
      assert!(60 <= *count && *count <= 140);
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

  struct TestRng {
    x: u32,
    fail64: bool,
    max: bool,
  }
  impl TestRng {
    fn new(x: u32) -> Self { TestRng { x, fail64: false, max: false } }
    fn new_fail64(x: u32) -> Self { TestRng { x, fail64: true, max: false } }
    fn new_max() -> Self { TestRng { x: 0, fail64: false, max: true } }
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
    fn test_gen_u32(&mut self, limit: u32) -> u32 {
      if self.max {
        limit - 1
      } else {
        self.gen_u32()
      }
    }
    fn test_gen_u64(&mut self, limit: u64) -> u64 {
      if self.max {
        limit - 1
      } else {
        self.gen_u64()
      }
    }
  }

  #[test]
  fn test_zeroone() {
    let mut a = StdRng::new();
    for _ in 0..5000 {
      let v = a.zeroone::<f32>();
      assert!(0.0 <= v && v < 1.0);
    }
    let mut t = TestRng::new(0);
    assert!(t.zeroone::<f32>() == 0.0);
    t.x = 0xffff_ffff;
    assert!(t.zeroone::<f32>() > 0.999);
    assert!(t.zeroone::<f32>() < 1.0);

    for _ in 0..5000 {
      let v = a.zeroone::<f64>();
      assert!(0.0 <= v && v < 1.0);
    }
    let mut t = TestRng::new(0);
    assert!(t.zeroone::<f64>() == 0.0);
    t.x = 0xffff_ffff;
    assert!(t.zeroone::<f64>() > 0.999);
    assert!(t.zeroone::<f64>() < 1.0);
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
  fn test_choose_weighted() {
    let mut a = StdRng::new();
    let chars = "abcdefghijklmn".chars().collect::<Vec<char>>();
    let weights = vec![1u32, 2, 3, 0, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7];
    let total_weight : u32 = weights.iter().sum();
    assert_eq!(chars.len(), weights.len());

    // Automatic dereferencing when weights are given as references
    let mut chosen = Vec::new();
    chosen.resize(chars.len(), 0i32);
    for _ in 0..10000 {
      let iter = chars.iter().zip(weights.iter());
      let picked = *a.choose_weighted(iter, total_weight).unwrap();
      chosen[(picked as usize) - ('a' as usize)] += 1;
    }
    for (i, count) in chosen.iter().enumerate() {
      let err = *count - ((weights[i] * 10000 / total_weight) as i32);
      assert!(-150 <= err && err <= 150);
    }

    // Mutable items
    chosen.truncate(0);
    chosen.resize(weights.len(), 0);
    for _ in 0..10000 {
      let iter = chosen.iter_mut().zip(weights.iter().map(|x| *x));
      *a.choose_weighted(iter, total_weight).unwrap() += 1;
    }
    for (i, count) in chosen.iter().enumerate() {
      let err = *count - ((weights[i] * 10000 / total_weight) as i32);
      assert!(-150 <= err && err <= 150);
    }

    // Automatic dereferencing when iterating references of item+weight tuples
    let mut weighted_items = weights.iter().rev().map(|x| (0i32, *x)).collect::<Vec<_>>();
    for _ in 0..10000 {
      *a.choose_weighted(weighted_items.iter_mut(), total_weight).unwrap() += 1;
    }
    for &(count, weight) in weighted_items.iter() {
      let err = count - ((weight * 10000 / total_weight) as i32);
      assert!(-150 <= err && err <= 150);
    }

    // Choose last item
    let mut test_rng = TestRng::new_max();
    for _ in 0..3 {
      assert_eq!(*test_rng.choose_weighted(chars.iter().zip(weights.iter()),
                                           total_weight).unwrap(),
                 'n');
    }

    fn test_adjusted_weight_total(delta: i32) {
      let items = vec![(1, 1), (2, 2), (3, 3)];
      if cfg!(debug_assertions) || delta == 0 {
        StdRng::new().choose_weighted(items.iter(), 6+delta);
      } else {
        loop { StdRng::new().choose_weighted(items.iter(), 6+delta); }
      }
    }

    assert!(catch_unwind(|| test_adjusted_weight_total(0)).is_ok());
    assert!(catch_unwind(|| test_adjusted_weight_total(1)).is_err());
    assert!(catch_unwind(|| test_adjusted_weight_total(1000)).is_err());
    if cfg!(debug_assertions) {
      // The non-debug-assertions code can't detect too small total_weight
      assert!(catch_unwind(|| test_adjusted_weight_total(-1)).is_err());
      assert!(catch_unwind(|| test_adjusted_weight_total(-1000)).is_err());
    }
  }

  #[test]
  fn test_choose_weighted2() {
    let mut a = StdRng::new();
    let chars = "abcdefghijklmn".chars().collect::<Vec<char>>();
    let weights = vec![1u32, 2, 3, 0, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7];
    let total_weight: u32 = weights.iter().sum();
    assert_eq!(chars.len(), weights.len());

    let mut chosen = Vec::new();
    chosen.resize(chars.len(), 0i32);
    for _ in 0..10000 {
      let picked = *a.choose_weighted2(chars.iter(), weights.iter()).unwrap();
      chosen[(picked as usize) - ('a' as usize)] += 1;
    }
    for (i, count) in chosen.iter().enumerate() {
      let err = *count - ((weights[i] * 10000 / total_weight) as i32);
      assert!(-150 <= err && err <= 150);
    }

    // Mutable items
    chosen.truncate(0);
    chosen.resize(weights.len(), 0);
    for _ in 0..10000 {
      *a.choose_weighted2(chosen.iter_mut(), weights.iter()).unwrap() += 1;
    }
    for (i, count) in chosen.iter().enumerate() {
      let err = *count - ((weights[i] * 10000 / total_weight) as i32);
      assert!(-150 <= err && err <= 150);
    }

    // Choose last item
    let mut test_rng = TestRng::new_max();
    for _ in 0..3 {
      assert_eq!(test_rng.choose_weighted2(chars.iter().cloned(), weights.iter()).unwrap(),
                 'n');
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
    assert!(catch_unwind(|| TestRng::new_fail64(0).below(std::u32::MAX)).is_err());
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
    assert_eq!(a.range(&-2i32, &-1), -2);
    assert_eq!(a.range(&1u32, &2), 1);
    assert_eq!(a.range(&-2i64, &-1), -2);
    assert_eq!(a.range(&1u64, &2), 1);

    for _ in 0..10000 {
      let x = a.range(-120i8, 120);
      assert!(-120 <= x && x < 120);
      let x = a.range(&-110i8, &110);
      assert!(-110 <= x && x < 110);
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
    assert!(catch_unwind(|| StdRng::new().below(0u32)).is_err());
    assert!(catch_unwind(|| StdRng::new().below(-100i8)).is_err());
    assert!(catch_unwind(|| StdRng::new().range(2, 1)).is_err());
    assert!(catch_unwind(|| StdRng::new().range(20u8, 10u8)).is_err());
    assert!(catch_unwind(|| StdRng::new().below(&0u32)).is_err());
    assert!(catch_unwind(|| StdRng::new().below(&-100i8)).is_err());
    assert!(catch_unwind(|| StdRng::new().range(&2, &1)).is_err());
    assert!(catch_unwind(|| StdRng::new().range(&20u8, &10u8)).is_err());
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

  #[test]
  fn test_float() {
    let mut a = StdRng::new();
    for val in [0.000001f64, 1000000.0, 47.0, f64::from_bits(1), f64::from_bits(0x7fcf_ffff_ffff_ffff)].iter() {
      for _ in 0..1000 {
        let x = a.below(*val);
        assert!(0.0 <= x && x < *val);
        let x = a.range(-*val, *val);
        assert!(-*val <= x && x < *val);
      }
      let x = a.below(val);
      assert!(0.0 <= x && x < *val);
      let x = a.range(&(-*val), val);
      assert!(-*val <= x && x < *val);
    }

    assert!(catch_unwind(|| StdRng::new().below(std::f64::NAN)).is_err());
    assert!(catch_unwind(|| StdRng::new().below(std::f64::INFINITY)).is_err());

    for val in [0.000001f32, 1000000.0, 47.0, f32::from_bits(1), f32::from_bits(0x7e7f_ffff)].iter() {
      for _ in 0..1000 {
        let x = a.below(*val);
        assert!(0.0 <= x && x < *val);
        let x = a.range(-*val, *val);
        assert!(-*val <= x && x < *val);
      }
      let x = a.below(val);
      assert!(0.0 <= x && x < *val);
      let x = a.range(&(-*val), val);
      assert!(-*val <= x && x < *val);
    }

    assert!(catch_unwind(|| StdRng::new().below(std::f32::NAN)).is_err());
    assert!(catch_unwind(|| StdRng::new().below(std::f32::INFINITY)).is_err());
  }
}
