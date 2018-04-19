extern crate num_bigint;
extern crate num_traits;

pub use self::num_bigint::{ BigUint, BigInt, Sign };
use super::{ Rng, Rangeable };
use self::num_traits::Zero;
use self::num_traits::Signed;

impl<'a> Rangeable for &'a BigUint {
  type Output = BigUint;

  fn rng_below<R: Rng + ?Sized>(rng: &mut R, limit: &'a BigUint) -> BigUint {
    if limit.is_zero() {
      panic!("Rng.below() called with limit <= 0");
    }

    // This could be done faster if we accessed the inner words in the BigUint.
    let bits = limit.bits();
    loop {
      let res = rng.gen_biguint(bits);
      if res < *limit {
        return res;
      }
    }
  }

  fn rng_range<R: Rng + ?Sized>(rng: &mut R, start: &'a BigUint, end: &'a BigUint) -> BigUint {
    if start >= end {
      panic!("empty or inverted range");
    }
    let diff = end - start;
    rng.below(&diff) + start
  }

  fn zero() -> Self::Output {
    return BigUint::zero()
  }

  fn is_neg(&self) -> bool {
    return false;
  }
}

impl<'a> Rangeable for &'a BigInt {
  type Output = BigInt;

  fn rng_below<R: Rng + ?Sized>(rng: &mut R, limit: &'a BigInt) -> BigInt {
    if !limit.is_positive() {
      panic!("Rng.below() called with limit <= 0");
    }

    // This could be done faster if we accessed the inner words in the BigInt.
    // Or we could reuse the BigUint::rng_below implementation if we could access the
    // inner BigUint.
    let bits = limit.bits();
    loop {
      let res = BigInt::from_biguint(Sign::Plus, rng.gen_biguint(bits));
      if res < *limit {
        return res;
      }
    }
  }

  fn rng_range<R: Rng + ?Sized>(rng: &mut R, start: &'a BigInt, end: &'a BigInt) -> BigInt {
    if start >= end {
      panic!("empty or inverted range");
    }
    let diff = end - start;
    rng.below(&diff) + start
  }

  fn zero() -> Self::Output {
    return BigInt::zero()
  }

  fn is_neg(&self) -> bool {
    return self.is_negative();
  }
}


#[cfg(test)]
mod tests {
  use super::*;
  use super::super::StdRng;
  use std::panic::catch_unwind;

  #[cfg(feature = "bigint")]
  #[test]
  fn test_bigint() {
    let mut a = StdRng::new();
    for bits in [0usize, 1, 3, 5, 31, 32, 33, 50, 63, 64, 65, 1500].iter().cloned() {
      let mut bits_set = BigUint::from(0u32);
      let mut bits_clear = (BigUint::from(1u32) << bits) - 1u32;
      for _ in 0..20 {
        let gen = a.gen_biguint(bits);
        bits_set |= &gen;
        bits_clear &= &gen;
      }
      assert_eq!(bits_set, (BigUint::from_slice(&[1]) << bits) - 1u32);
      assert!(bits_clear.is_zero());
    }

    let lower = BigUint::from_slice(&[1, 1, 5]);
    let upper = BigUint::from_slice(&[14, 1, 47, 12]);
    for _ in 0..10000 {
      let res = a.range(&lower, &upper);
      assert!(lower <= res && res < upper);
    }

    let lower = BigInt::from_slice(Sign::Plus, &[1, 1, 5]);
    let upper = BigInt::from_slice(Sign::Plus, &[14, 1, 47, 12]);
    for _ in 0..10000 {
      let res = a.range(&lower, &upper);
      assert!(lower <= res && res < upper);
    }

    let lower = BigInt::from_slice(Sign::Minus, &[14, 1, 47, 12]);
    let upper = BigInt::from_slice(Sign::Minus, &[1, 1, 5]);
    for _ in 0..10000 {
      let res = a.range(&lower, &upper);
      assert!(lower <= res && res < upper);
    }

    let lower = BigInt::from_slice(Sign::Minus, &[14, 1, 5, 1]);
    let upper = BigInt::from_slice(Sign::Plus, &[1, 1, 0xefff_1234]);
    let mut has_positive = false;
    let mut has_negative = false;
    for _ in 0..10000 {
      let res = a.range(&lower, &upper);
      assert!(lower <= res && res < upper);
      has_positive = has_positive || res.is_positive();
      has_negative = has_negative || res.is_negative();
    }
    assert!(has_positive);
    assert!(has_negative);

    assert!(catch_unwind(|| StdRng::new().below(&BigUint::from(0u32))).is_err());
    assert!(catch_unwind(|| StdRng::new().below(&BigInt::from(0u32))).is_err());
    assert!(catch_unwind(|| StdRng::new().below(&BigInt::from(-1i32))).is_err());
    assert!(catch_unwind(|| StdRng::new().range(&BigUint::from(5u32), &BigUint::from(5u32))).is_err());
    assert!(catch_unwind(|| StdRng::new().range(&BigUint::from(50u32), &BigUint::from(5u32))).is_err());
    assert!(catch_unwind(|| StdRng::new().range(&BigInt::from(-5i32), &BigInt::from(-5i32))).is_err());
    assert!(catch_unwind(|| StdRng::new().range(&BigInt::from(50u32), &BigInt::from(-5i32))).is_err());
  }
}
