use core::ops::{AddAssign, SubAssign};
use num_traits::{Float, Num, NumAssignOps, NumOps};
use std::cmp::PartialEq;
use std::default::Default;
use std::marker::Copy;
use std::ops;
// for the unsasfe and fast dot product implementation
use std::intrinsics::{fadd_fast, fmul_fast};

pub trait FloatOps = Float;
// Scalars to be held as the base numeric data stored in a tensor.
pub trait Scalar = NumOps + NumAssignOps + GroupOps + Copy;
// Group operation under addition, suitable for dimensionful numbers.
pub trait GroupOps<Rhs = Self, Output = Self> = Default
    + PartialEq
    + AddAssign<Rhs>
    + SubAssign<Rhs>
    + ops::Add<Rhs, Output = Output>
    + ops::Sub<Rhs, Output = Output>
    + Copy;

// R1Tensor short for: Rank 1 Tensor.
// This is a wrapper struct around a contiguous array.
// The data in an array is restricted with the Scalar trait above.
// The dimension (length of the array) is specified by the constant
// parameter D.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct R1Tensor<S: GroupOps, const D: usize> {
    components: [S; D],
}

// Cannot simply derive default at the moment so the manual implementation
// is given.
impl<S: GroupOps, const D: usize> Default for R1Tensor<S, D> {
    fn default() -> Self {
        R1Tensor {
            components: [S::default(); D],
        }
    }
}

impl<S: GroupOps, const D: usize> ops::Deref for R1Tensor<S, D> {
    type Target = [S; D];
    fn deref(&self) -> &Self::Target {
        &self.components
    }
}

impl<S: GroupOps, const D: usize> ops::DerefMut for R1Tensor<S, D> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.components
    }
}

// Useful macro for declaration of a concrete sized tensor,
// using the syntax: dimensions![5,8] for 5 outmost dimensions
// and 8 inner dimensions (a rank 2 tensor).
#[macro_export]
macro_rules! dimensions {
    ($basetype: ty, [$basedim: literal]) => {
        R1Tensor< $basetype, $basedim >
    };
    ($basetype: ty, [$basedim: literal, $($nextdim: literal),+]) => {
        R1Tensor< dimensions!( $basetype, [$($nextdim),+]), $basedim >
    };
}

// Construction of the R1Tensor from an array,
// by moving ownership of the array to the R1Tensor struct
impl<S: Scalar, const D: usize> From<[S; D]> for R1Tensor<S, D> {
    fn from(array: [S; D]) -> R1Tensor<S, D> {
        R1Tensor { components: array }
    }
}

impl<S: GroupOps, const D: usize> ops::Index<usize> for R1Tensor<S, D> {
    type Output = S;
    fn index(&self, index: usize) -> &S {
        &self.components[index]
    }
}

impl<S: GroupOps, const D: usize> ops::IndexMut<usize> for R1Tensor<S, D> {
    fn index_mut(&mut self, index: usize) -> &mut S {
        &mut self.components[index]
    }
}

// Really the only requirement here is that the type implements iterators.
// The requirement for impl on ops::Add for use of the +
// makes it impossible to simply create a trait that generates the values.
impl<S: GroupOps, const D: usize> ops::AddAssign<R1Tensor<S, D>> for R1Tensor<S, D> {
    fn add_assign(&mut self, rhs: R1Tensor<S, D>) {
        for (elem_out, elem_rhs) in self.iter_mut().zip(rhs.iter()) {
            *elem_out += *elem_rhs
        }
    }
}

impl<S: GroupOps, const D: usize> ops::AddAssign<&R1Tensor<S, D>> for R1Tensor<S, D> {
    fn add_assign(&mut self, rhs: &R1Tensor<S, D>) {
        for (elem_out, elem_rhs) in self.iter_mut().zip(rhs.iter()) {
            *elem_out += elem_rhs.clone()
        }
    }
}

impl<S: GroupOps, const D: usize> ops::Add<R1Tensor<S, D>> for R1Tensor<S, D> {
    type Output = R1Tensor<S, D>;

    // Dropping both self and rhs saves memory highwater.
    fn add(self, rhs: R1Tensor<S, D>) -> R1Tensor<S, D> {
        // Instead of cloning the value, move ownership into output
        let mut output = self;
        output += rhs;
        output
    }
}


impl<S: GroupOps, const D: usize> ops::Add<&R1Tensor<S, D>> for R1Tensor<S, D> {
    type Output = R1Tensor<S, D>;

    fn add(self, rhs: &R1Tensor<S, D>) -> R1Tensor<S, D> {
        let mut output = self;
        output += rhs;
        output
    }
}

impl<S: GroupOps, const D: usize> ops::Add<R1Tensor<S, D>> for &R1Tensor<S, D> {
    type Output = R1Tensor<S, D>;

    // Just use the reverse of T + &T definition,
    // it conserves space in the same way.
    fn add(self, rhs: R1Tensor<S, D>) -> R1Tensor<S, D> {
        rhs + self
    }
}

impl<S: GroupOps, const D: usize> ops::SubAssign<R1Tensor<S, D>> for R1Tensor<S, D> {
    fn sub_assign(&mut self, rhs: R1Tensor<S, D>) {
        for (elem_out, elem_rhs) in self.iter_mut().zip(rhs.iter()) {
            *elem_out -= *elem_rhs
        }
    }
}

impl<S: GroupOps, const D: usize> ops::Sub<R1Tensor<S, D>> for R1Tensor<S, D> {
    type Output = R1Tensor<S, D>;

    fn sub(self, rhs: R1Tensor<S, D>) -> R1Tensor<S, D> {
        let mut output = self;
        output -= rhs;
        output
    }
}

// Defines scalar-multiplication for the container.
impl<S: Scalar, const D: usize> ops::MulAssign<S> for R1Tensor<S, D> {
    fn mul_assign(&mut self, rhs: S) {
        for elem in self.iter_mut() {
            *elem *= rhs
        }
    }
}

impl<S: Scalar, const D: usize> ops::Mul<S> for R1Tensor<S, D> {
    type Output = R1Tensor<S, D>;

    fn mul(self, rhs: S) -> R1Tensor<S, D> {
        let mut output = self;
        output *= rhs;
        output
    }
}

impl<S: Scalar, const D: usize> ops::DivAssign<S> for R1Tensor<S, D> {
    fn div_assign(&mut self, rhs: S) {
        for elem in self.iter_mut() {
            *elem /= rhs
        }
    }
}

impl<S: Scalar, const D: usize> ops::Div<S> for R1Tensor<S, D> {
    type Output = R1Tensor<S, D>;

    fn div(self, rhs: S) -> R1Tensor<S, D> {
        let mut output = self;
        output /= rhs;
        output
    }
}


// For tensors with the appropriate base type, i.e. a float,
// there are additional functions available.
impl<S: Scalar + Float, const D: usize> R1Tensor<S, D> {
    // This is a safe version for the dot product
    pub fn dotprod(&self, rhs: &Self) -> S {
        self.components
            .iter()
            .zip(rhs.components.iter())
            .fold(S::default(), |acc, (&lhs, &rhs)| lhs.mul_add(rhs, acc))
    }

    pub fn udotprod(&self, rhs: &Self) -> S {
        unsafe {
            self.components
                .iter()
                .zip(rhs.components.iter())
                .fold(S::default(), |acc, (&lhs, &rhs)| {
                    fadd_fast(acc, fmul_fast(lhs, rhs))
                })
        }
    }

    pub fn norm2(&self) -> S {
        self.components
            .iter()
            .fold(S::default(), |acc, &s| s.mul_add(s, acc))
    }

    pub fn unorm2(&self) -> S {
        unsafe {
            self.components
                .iter()
                .fold(S::default(), |acc, &s| fadd_fast(acc, fmul_fast(s, s)))
        }
    }
    // These are not the proper normalise functions
    // and showcases the need to use num-traits
    pub fn normalise(&mut self) {
        *self /= self.norm2().sqrt();
    }

    pub fn unormalise(&mut self) {
        *self /= self.unorm2().sqrt();
    }

    pub fn new_normalised(&self) -> Self {
        let mut output = self.clone();
        output.normalise();
        output
    }

    pub fn new_unormalised(&self) -> Self {
        let mut output = self.clone();
        output.unormalise();
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_test() {
        let mut u = R1Tensor::<f32, 3>::from([3.0, 4.0, 5.0]);
        let v = R1Tensor::<f32, 3>::from([1.0, 2.0, 3.0]);
        // Copy of the LHS values but allowing for its use again
        assert_eq!(u.clone() + &v, R1Tensor::from([4.0, 6.0, 8.0]));

        u += v;
        // u was not modified in the previous calculation
        assert_eq!(u, R1Tensor::from([4.0, 6.0, 8.0]));
    }

    #[test]
    fn dotprod_test() {
        let u = R1Tensor::<f32, 3>::from([3.0, 4.0, 5.0]);
        let v = u.clone();

        assert_eq!((&u).dotprod(&v), 50.0);

        assert_eq!(u.udotprod(&v), 50.0);
    }

    #[test]
    fn norm_test() {
        let mut u = R1Tensor::<f32, 3>::from([3.0, 4.0, 5.0]);

        assert_eq!(u.norm2(), 50.0);

        assert_eq!(u.unorm2(), 50.0);

        u.normalise();
        assert_eq!(
            u,
            R1Tensor::from([3.0 / 50.0.sqrt(), 4.0 / 50.0.sqrt(), 5.0 / 50.0.sqrt()])
        );
    }

    #[test]
    fn multidim_test() {
        let mut matrix = <dimensions!(f32, [3, 3])>::default();
        matrix[2][2] += 1.0;
        let second = <dimensions!(f32, [3, 3])>::default();

        assert_eq!((matrix + &second)[2][2], 1.0);
        assert_eq!(second[2][1], 0.0);
    }
}
