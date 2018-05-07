#[allow(unused_imports)]
use std;
use super::{ Rng, Rangeable, RangeImpl };

impl<'a, T: Rangeable + Copy> Rangeable for &'a T {
  type Output = T::Output;
  type Range = RangeRef<T>;

  fn rng_below<'b, R: Rng + ?Sized>(rng: &'b mut R, limit: &'a T) -> Self::Output {
    T::rng_below(rng, *limit)
  }

  fn rng_range<'b, R: Rng + ?Sized>(rng: &'b mut R, start: &'a T, end: &'a T) -> Self::Output {
    T::rng_range(rng, *start, *end)
  }

  fn zero() -> Self::Output {
    T::zero()
  }

  fn is_neg(&self) -> bool {
    (*self).is_neg()
  }
}

pub struct RangeRef<T> where T: Rangeable + Copy {
  inner: T::Range,
}
impl<'a, T> RangeImpl<&'a T> for RangeRef<T> where T: Rangeable + Copy {
  type Output = T::Output;
  fn new_below(limit: &'a T) -> Self {
    RangeRef { inner: T::Range::new_below(*limit) }
  }
  fn new_range(start: &'a T, end: &'a T) -> Self {
    RangeRef { inner: T::Range::new_range(*start, *end) }
  }
  fn gen<'b, R: Rng + ?Sized>(&self, rng: &'b mut R) -> Self::Output {
    self.inner.gen(rng)
  }
}

pub struct RangeInt<T> {
  offset: T,
  limit: T, // limit should be of unsigned type. Bug we're casting to T for simplicity
  zone: T,
}

macro_rules! impl_rangable {
  ($ty: ty, $uty: ty, $gen_ty: ty, $gen_func: ident, $gen_test_func: ident) => (
    #[allow(unused_comparisons)]
    impl Rangeable for $ty {
      type Output = $ty;
      type Range = RangeInt<$ty>;

      fn zero() -> Self::Output {
        return 0;
      }

      fn is_neg(&self) -> bool {
        return *self < 0;
      }
    }

    impl RangeImpl<$ty> for RangeInt<$ty> {
      type Output = $ty;
      fn new_below(limit: $ty) -> Self {
        Self::new_range(0, limit)
      }
      fn new_range(start: $ty, end: $ty) -> Self {
        if start >= end {
          panic!("Rng.range()/Rng.below() called with empty range");
        }
        let limit = end.wrapping_sub(start) as $uty;
        let zone = (0 as $uty).wrapping_sub(limit) % limit;
        RangeInt { offset: start, limit: limit as $ty, zone: zone as $ty }
      }

      #[cfg(not(test))]
      fn gen<R: Rng + ?Sized>(&self, rng: &mut R) -> Self::Output {
        let zone = (0 as $gen_ty).wrapping_sub(self.zone as $gen_ty);
        let limit_ty = self.limit as $uty as $gen_ty;
        loop {
          let res = rng.$gen_func();
          if res < zone {
            return ((res % limit_ty) as $ty).wrapping_add(self.offset);
          }
        }
      }

      #[cfg(test)]
      fn gen<R: Rng + ?Sized>(&self, rng: &mut R) -> Self::Output {
        let zone = (0 as $gen_ty).wrapping_sub(self.zone as $gen_ty);
        let limit_ty = self.limit as $uty as $gen_ty;
        for _ in 0..10 {
          let res = rng.$gen_test_func(limit_ty);
          if res < zone {
            return ((res % limit_ty) as $ty).wrapping_add(self.offset);
          }
        }
        ((rng.$gen_test_func(limit_ty) % limit_ty) as $ty).wrapping_add(self.offset)
      }
    }


  )
}
impl_rangable!(u8, u8, u32, gen_u32, test_gen_u32);
impl_rangable!(i8, u8, u32, gen_u32, test_gen_u32);
impl_rangable!(u16, u16, u32, gen_u32, test_gen_u32);
impl_rangable!(i16, u16, u32, gen_u32, test_gen_u32);
impl_rangable!(u32, u32, u32, gen_u32, test_gen_u32);
impl_rangable!(i32, u32, u32, gen_u32, test_gen_u32);
impl_rangable!(u64, u64, u64, gen_u64, test_gen_u64);
impl_rangable!(i64, u64, u64, gen_u64, test_gen_u64);
impl_rangable!(usize, usize, u64, gen_u64, test_gen_u64);
impl_rangable!(isize, usize, u64, gen_u64, test_gen_u64);

#[cfg(test)]
mod tests {
  use super::*;
  use super::super::StdRng;

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
    let mut rng = StdRng::new();
    assert_eq!(rng.below(1i8), 0);
    assert_eq!(rng.below(1u8), 0);
    assert_eq!(rng.below(1i16), 0);
    assert_eq!(rng.below(1u16), 0);
    assert_eq!(rng.below(1i32), 0);
    assert_eq!(rng.below(1u32), 0);
    assert_eq!(rng.below(1i64), 0);
    assert_eq!(rng.below(1u64), 0);
    assert_eq!(rng.below(1isize), 0);
    assert_eq!(rng.below(1usize), 0);
    assert_eq!(rng.below(1i8), 0);
    assert_eq!(rng.below(1u8), 0);
    assert_eq!(rng.below(1i16), 0);
    assert_eq!(rng.below(1u16), 0);
    assert_eq!(rng.below(1i32), 0);
    assert_eq!(rng.below(1u32), 0);
    assert_eq!(rng.below(1i64), 0);
    assert_eq!(rng.below(1u64), 0);
    assert_eq!(rng.below(1isize), 0);
    assert_eq!(rng.below(1usize), 0);
    assert_eq!(rng.below(&1u8), 0);
    assert_eq!(rng.below(&&1u16), 0);
    assert_eq!(rng.below(&&&1u64), 0);
    assert_eq!(rng.below(&&&&&&1u64), 0);

    let mut t = TestRng::new_fail64(0);
    t.below(std::u32::MAX);
    t.below(std::i32::MAX);
    t.below(std::u16::MAX);
    t.below(std::i16::MAX);
    t.below(std::u8::MAX);
    t.below(std::i8::MAX);
  }

  #[test]
  fn test_range_primitives() {
    let mut rng = StdRng::new();
    assert_eq!(rng.range(-2i8, -1), -2);
    assert_eq!(rng.range(1u8, 2), 1);
    assert_eq!(rng.range(-2i16, -1), -2);
    assert_eq!(rng.range(1u16, 2), 1);
    assert_eq!(rng.range(-2i32, -1), -2);
    assert_eq!(rng.range(1u32, 2), 1);
    assert_eq!(rng.range(-2i64, -1), -2);
    assert_eq!(rng.range(1u64, 2), 1);
    assert_eq!(rng.range(-2isize, -1), -2);
    assert_eq!(rng.range(1usize, 2), 1);
    assert_eq!(rng.range(&-2i32, &-1), -2);
    assert_eq!(rng.range(&1u32, &2), 1);
    assert_eq!(rng.range(&-2i64, &-1), -2);
    assert_eq!(rng.range(&1u64, &2), 1);
    assert_eq!(rng.range(&&-2i32, &&-1), -2);
    assert_eq!(rng.range(&&1u32, &&2), 1);
    assert_eq!(rng.range(&&&-2i64, &&&-1), -2);
    assert_eq!(rng.range(&&&&&1u64, &&&&&2), 1);

    for _ in 0..10000 {
      let x = rng.range(-120i8, 120);
      assert!(-120 <= x && x < 120);
      let x = rng.range(&-110i8, &110);
      assert!(-110 <= x && x < 110);
    }

    let x = rng.range(-120i8, 120);
    assert!(-120 <= x && x < 120);
    let x = rng.range(-32760i16, 32760);
    assert!(-32760 <= x && x < 32760);
    let x = rng.range(std::i32::MIN, std::i32::MAX);
    assert!(std::i32::MIN <= x && x < std::i32::MAX);
    let x = rng.range(std::i64::MIN, std::i64::MAX);
    assert!(std::i64::MIN <= x && x < std::i64::MAX);
  }
}
