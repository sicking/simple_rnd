use std;
use super::{ Rng, Rangeable };

impl<'a, Rang: Rangeable + Copy> Rangeable for &'a Rang {
  type Output = Rang::Output;

  fn rng_below<R: Rng + ?Sized>(rng: &mut R, limit: &'a Rang) -> Self::Output {
    Rang::rng_below(rng, *limit)
  }

  fn rng_range<R: Rng + ?Sized>(rng: &mut R, start: &'a Rang, end: &'a Rang) -> Self::Output {
    Rang::rng_range(rng, *start, *end)
  }

  fn zero() -> Self::Output {
    Rang::zero()
  }

  fn is_neg(&self) -> bool {
    (*self).is_neg()
  }
}

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

#[cfg(test)]
mod tests {
  use super::*;
  use super::super::StdRng;
  use std::panic::catch_unwind;

  struct TestRng {
    x: u32,
    fail64: bool,
  }
  impl TestRng {
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
}
