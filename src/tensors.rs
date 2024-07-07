use num_traits::{Float, Num, NumOps, NumAssignOps};
use std::ops;
use std::default::Default;
use std::marker::Copy;
use std::cmp::PartialEq;
// for the unsasfe and fast dot product implementation
use std::intrinsics::{fadd_fast, fmul_fast};

// Scalars to be held as components of the tensor.
pub trait Scalar = Num + NumOps + NumAssignOps + Copy + Default;

// R1Tensor short for: Rank 1 Tensor.
// This is a wrapper struct around a contiguous array.
// The data in an array is restricted with the Scalar trait above.
// The dimension (length of the array) is specified by the constant
// parameter D.
#[derive(Debug, Clone, PartialEq)]
pub struct R1Tensor<S: Scalar, const D: usize> {
    components: [S;D]
}

// Construction of the R1Tensor from an array,
// by moving the array into the R1Tensor struct
impl<S: Scalar, const D: usize> From<[S;D]> for R1Tensor<S,D> {
    fn from(array: [S;D]) -> R1Tensor<S,D> {
        R1Tensor { components: array }
    }
}

impl<S: Scalar, const D: usize> ops::Index<usize> for R1Tensor<S,D> {
    type Output = S;
    fn index(&self, index: usize) -> &S {
        &self.components[index]
    }
}

impl<S: Scalar, const D: usize> ops::IndexMut<usize> for R1Tensor<S,D> {
    fn index_mut(&mut self, index: usize) -> &mut S {
        &mut self.components[index]
    }
}

// Really the only requirement here is that the type is inexable
impl<S: Scalar, const D: usize> ops::Add<R1Tensor<S,D>> for &R1Tensor<S,D>
{
    type Output = R1Tensor<S,D>;

    fn add(self, rhs: R1Tensor<S,D>) -> R1Tensor<S,D> {
        let mut output = self.clone();
        for i in 0..D {
            output[i] += rhs[i]
        }
        output
    }
}

impl<S: Scalar, const D: usize> ops::Add<&R1Tensor<S,D>> for &R1Tensor<S,D> {
    type Output = R1Tensor<S,D>;

    fn add(self, rhs: &R1Tensor<S,D>) -> R1Tensor<S,D> {
        let mut output = self.clone();
        for i in 0..D {
            output[i] += rhs[i]
        }
        output
    }
}

impl<S: Scalar, const D: usize>
    ops::AddAssign<R1Tensor<S,D>> for R1Tensor<S,D> {
        fn add_assign(&mut self, rhs: R1Tensor<S,D>) {
            for i in 0..D {
                self[i] += rhs[i]
            }
        }
    }

impl<S: Scalar, const D: usize>
    ops::AddAssign<&R1Tensor<S,D>> for R1Tensor<S,D> {
        fn add_assign(&mut self, rhs: &R1Tensor<S,D>) {
            for i in 0..D {
                self[i] += rhs[i]
            }
        }
}

impl<S: Scalar, const D: usize> ops::Sub<R1Tensor<S,D>> for &R1Tensor<S,D> {
    type Output = R1Tensor<S,D>;

    fn sub(self, rhs: R1Tensor<S,D>) -> R1Tensor<S,D> {
        let mut output = self.clone();
        for i in 0..D {
            output[i] -= rhs[i]
        }
        output
    }
}

impl<S: Scalar, const D: usize> ops::Sub<&R1Tensor<S,D>> for &R1Tensor<S,D> {
    type Output = R1Tensor<S,D>;

    fn sub(self, rhs: &R1Tensor<S,D>) -> R1Tensor<S,D> {
        let mut output = self.clone();
        for i in 0..D {
            output[i] -= rhs[i]
        }
        output
    }
}

impl<S: Scalar, const D: usize>
    ops::SubAssign<R1Tensor<S,D>> for R1Tensor<S,D> {
        fn sub_assign(&mut self, rhs: R1Tensor<S,D>) {
            for i in 0..D {
                self[i] -= rhs[i]
            }
        }
    }

impl<S: Scalar, const D: usize>
    ops::SubAssign<&R1Tensor<S,D>> for R1Tensor<S,D> {
        fn sub_assign(&mut self, rhs: &R1Tensor<S,D>) {
            for i in 0..D {
                self[i] -= rhs[i]
            }
        }
    }

impl<S: Scalar, const D: usize> ops::Mul<S> for &R1Tensor<S,D> {
    type Output = R1Tensor<S,D>;

    fn mul(self, rhs: S) -> R1Tensor<S,D> {
        let mut output = self.clone();
        for i in 0..D {
            output[i] *= rhs
        }
        output
    }
}

impl<S: Scalar, const D: usize> ops::MulAssign<S> for R1Tensor<S,D> {
    fn mul_assign(&mut self, rhs: S) {
        for i in 0..D {
            self[i] *= rhs
        }
    }
}

impl<S: Scalar, const D: usize> ops::Div<S> for &R1Tensor<S,D> {
    type Output = R1Tensor<S,D>;

    fn div(self, rhs: S) -> R1Tensor<S,D> {
        let mut output = self.clone();
        for i in 0..D {
            output[i] /= rhs
        }
        output
    }
}

impl<S: Scalar, const D: usize> ops::DivAssign<S> for R1Tensor<S,D> {
    fn div_assign(&mut self, rhs: S) {
        for i in 0..D {
            self[i] /= rhs
        }
    }
}

// For tensors with the appropriate base type,
// there are additional functions available.
impl<S: Scalar + Float, const D: usize> R1Tensor<S,D> {
    // This is a safe version for the dot product
    fn dotprod(&self, rhs: &Self) -> S {
        self.components.iter().zip(rhs.components.iter())
            .fold(S::default(),
                  |acc, (&lhs, &rhs)| lhs.mul_add(rhs, acc))
    }

    fn dotprod_unsafe(&self, rhs: &Self) -> S {
        unsafe {
        self.components.iter().zip(rhs.components.iter())
            .fold(S::default(),
                  |acc, (&lhs, &rhs)| fadd_fast(acc, fmul_fast(lhs, rhs)))
        }
    }

    #[inline(always)]
    fn norm2(&self) -> S {
        self.components.iter().
            fold(S::default(), |acc, &s| s.mul_add(s, acc))
    }

    fn norm2_unsafe(&self) -> S {
        unsafe {
        self.components.iter().
            fold(S::default(),
                 |acc, &s| fadd_fast(acc, fmul_fast(s,s)))
        }
    }

    // These are not the proper normalise functions
    // and showcases the need to use num-traits
    fn normalise(&mut self) {
        *self /= self.norm2().sqrt()
    }

    fn normalise_unsafe(&mut self) {
        *self /= self.norm2_unsafe().sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_test() {
        let mut u = R1Tensor::<f32, 3>::from([3.0, 4.0, 5.0]);
        let v = R1Tensor::<f32, 3>::from([1.0, 2.0, 3.0]);
        let w = v.clone();
        // Copy of the LHS values but allowing for its use again
        assert_eq!( &u + v, R1Tensor::from([4.0, 6.0, 8.0]) );

        u += w;
        // u was not modified in the previous calculation
        assert_eq!( u, R1Tensor::from([4.0, 6.0, 8.0]) );
    }

    #[test]
    fn dotprod_test() {
        let u = R1Tensor::<f32, 3>::from([3.0, 4.0, 5.0]);
        let v = u.clone();

        assert_eq!( (&u).dotprod(&v), 50.0 );

        assert_eq!( u.dotprod_unsafe(&v), 50.0 );
    }

    #[test]
    fn norm_test() {
        let mut u = R1Tensor::<f32, 3>::from([3.0, 4.0, 5.0]);

        assert_eq!( u.norm2(), 50.0 );

        assert_eq!( u.norm2_unsafe(), 50.0 );

        u.normalise_unsafe();
        assert_eq!( u,
                    R1Tensor::from(
                        [3.0/50.0.sqrt(), 4.0/50.0.sqrt(), 5.0/50.0.sqrt()]) );
    }
}
