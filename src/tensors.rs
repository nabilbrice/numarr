use core::ops::{AddAssign, SubAssign, MulAssign, DivAssign};
use num_traits::{Float, Num, NumAssignOps, NumOps};
use num_traits::ops::mul_add::{MulAdd, MulAddAssign};
use std::cmp::PartialEq;
use std::default::Default;
use std::marker::Copy;
use std::ops;
// for the unsasfe and fast dot product implementation
use std::intrinsics::{fadd_fast, fmul_fast};

pub trait FloatOps = Float;
// Group operation under addition, suitable for dimensionful numbers.
pub trait AddGroup<Rhs = Self, Output = Self> = Data
    + AddAssign<Rhs>
    + SubAssign<Rhs>
    + ops::Add<Rhs, Output = Output>
    + ops::Sub<Rhs, Output = Output>;

// Because the MulGroup must take Self as Rhs and Output,
// this needs to be element-wise defined for the NumArray too.
pub trait MulGroup<Rhs = Self, Output = Self> = Data
    + MulAssign<Rhs>
    + DivAssign<Rhs>
    + ops::Mul<Rhs, Output = Output>
    + ops::Div<Rhs, Output = Output>;

pub trait FieldOps<Rhs = Self, Output = Self> =
    AddGroup<Self, Self> + MulGroup<Self, Self>;

pub trait Data = Default + Clone + Copy + PartialEq + PartialOrd;

// Because this implements the interface operators directly,
// the operations cannot be made into a trait.
// If the interface was left to the user, it would be easier
// to make:
// trait ArrayOps
// which implements a default fn for add, ..
// and then the interface can be applied when desired.

// This is a transparent wrapper around a contiguous array.
// The data in an array is restricted with the Scalar trait above.
// The dimension (length of the array) is specified by the constant
// parameter D.
// These derive make NumArray itself implement the trait Data,
// so it can hold itself.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct NumArray<S: Data, const D: usize> {
    components: [S; D],
}

// Cannot simply derive default at the moment so the manual implementation
// is given.
impl<S: Data, const D: usize> Default for NumArray<S, D> {
    fn default() -> Self {
        NumArray {
            components: [S::default(); D],
        }
    }
}

impl<S: Data, const D: usize> ops::Deref for NumArray<S, D> {
    type Target = [S; D];
    fn deref(&self) -> &Self::Target {
        &self.components
    }
}

impl<S: Data, const D: usize> ops::DerefMut for NumArray<S, D> {
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
        NumArray< $basetype, $basedim >
    };
    ($basetype: ty, [$basedim: literal, $($nextdim: literal),+]) => {
        NumArray< dimensions!( $basetype, [$($nextdim),+]), $basedim >
    };
}

// For easy creation of multidimensional numarrays:
// numarray!([1.0,2.0]), numarray!(1.0,2.0),
// numarray!([1.0,2.0],[3.0,4.0])
#[macro_export]
macro_rules! numarray {
    // For rank 1 numarrays only:
    ($($e:literal),+) => {
        NumArray::from([$($e),+])
    };
    // For rank 1 matching:
    ([$($e:literal),+]) => {
        NumArray::from([$($e),+])
    };
    // For higher ranks, recursively call itself:
    ($([$($e:literal),+]),+) => {
        NumArray::from([$(numarray!([$($e),+])),+])
    };
}

// Construction of the NumArray from an array,
// by moving ownership of the array to the NumArray struct
impl<S: Data, const D: usize> From<[S; D]> for NumArray<S, D> {
    fn from(array: [S; D]) -> NumArray<S, D> {
        NumArray { components: array }
    }
}

impl<S: Data, const D: usize> FromIterator<S> for NumArray<S, D> {
    fn from_iter<I: IntoIterator<Item=S>>(iter: I) -> Self {
        let mut output: [S; D];
        // SAFETY: later initialised
        unsafe { output = std::mem::uninitialized() };
        for (elem_out, elem_iter) in output.iter_mut().zip(iter) {
            *elem_out = elem_iter
        }
        NumArray::from(output)
    }
}

impl<S: Data, const D: usize> ops::Index<usize> for NumArray<S, D> {
    type Output = S;
    fn index(&self, index: usize) -> &S {
        &self.components[index]
    }
}

impl<S: Data, const D: usize> ops::IndexMut<usize> for NumArray<S, D> {
    fn index_mut(&mut self, index: usize) -> &mut S {
        &mut self.components[index]
    }
}

// Really the only requirement here is that the type implements iterators.
// The requirement for impl on ops::Add for use of the +
// makes it impossible to simply create a trait that generates the values.
impl<S: AddGroup, const D: usize> ops::AddAssign<NumArray<S, D>> for NumArray<S, D> {
    fn add_assign(&mut self, rhs: NumArray<S, D>) {
        self.iter_mut().zip(rhs.iter())
            .for_each(|(mut out, rhs)| *out += *rhs)
    }
}

impl<S: AddGroup, const D: usize> ops::AddAssign<&NumArray<S, D>> for NumArray<S, D> {
    fn add_assign(&mut self, rhs: &NumArray<S, D>) {
        *self += *rhs
    }
}

impl<S: AddGroup, const D: usize> ops::Add<NumArray<S, D>> for NumArray<S, D> {
    type Output = NumArray<S, D>;

    fn add(self, rhs: NumArray<S, D>) -> NumArray<S, D> {
        // Instead of cloning the value, move ownership into output
        let mut output = self;
        output += rhs;
        output
    }
}

impl<S: AddGroup, const D: usize> ops::Add<&NumArray<S, D>> for NumArray<S, D> {
    type Output = NumArray<S, D>;

    fn add(self, rhs: &NumArray<S, D>) -> NumArray<S, D> {
        let mut output = self;
        output += rhs;
        output
    }
}

impl<S: AddGroup, const D: usize> ops::Add<NumArray<S, D>> for &NumArray<S, D> {
    type Output = NumArray<S, D>;

    // Just use the reverse of T + &T definition,
    // it conserves space in the same way.
    fn add(self, rhs: NumArray<S, D>) -> NumArray<S, D> {
        rhs + self
    }
}

impl<S: AddGroup, const D: usize> ops::Add<&NumArray<S, D>> for &NumArray<S, D> {
    type Output = NumArray<S, D>;

    fn add(self, rhs: &NumArray<S, D>) -> NumArray<S, D> {
        let mut output = self.clone();
        output += rhs;
        output
    }
}

impl<S: AddGroup, const D: usize> ops::SubAssign<NumArray<S, D>> for NumArray<S, D> {
    fn sub_assign(&mut self, rhs: NumArray<S, D>) {
        self.iter_mut().zip(rhs.iter())
            .for_each(|(mut out, rhs)| *out -= *rhs)
    }
}

impl<S: AddGroup, const D: usize> ops::Sub<NumArray<S, D>> for NumArray<S, D> {
    type Output = NumArray<S, D>;

    fn sub(self, rhs: NumArray<S, D>) -> NumArray<S, D> {
        let mut output = self;
        output -= rhs;
        output
    }
}

// Multiplication defined for NumArray to close it as a group.
// This is almost a dot product without the sum at the end.
impl<S: MulGroup, const D: usize> ops::MulAssign<NumArray<S, D>> for NumArray<S, D> {
    fn mul_assign(&mut self, rhs: Self) {
        self.iter_mut().zip(rhs.iter())
            .for_each(|(elem_out, elem_rhs)| *elem_out *= *elem_rhs)
    }
}

impl<S: MulGroup, const D: usize> ops::Mul<NumArray<S, D>> for NumArray<S, D> {
    type Output = NumArray<S, D>;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut output = self;
        output *= rhs;
        output
    }
}

// Defines scalar-multiplication for the container.
impl<S: MulGroup, const D: usize> ops::MulAssign<S> for NumArray<S, D> {
    fn mul_assign(&mut self, rhs: S) {
        self.iter_mut().for_each(|elem| *elem *= rhs)
    }
}

impl<S: MulGroup, const D: usize> ops::Mul<S> for NumArray<S, D> {
    type Output = NumArray<S, D>;

    fn mul(self, rhs: S) -> NumArray<S, D> {
        let mut output = self;
        output *= rhs;
        output
    }
}

impl<S: MulGroup, const D: usize> ops::DivAssign<NumArray<S, D>> for NumArray<S, D> {
    fn div_assign(&mut self, rhs: Self) {
        self.iter_mut().zip(rhs.iter())
            .for_each(|(elem_out, elem_rhs)| *elem_out /= *elem_rhs)
    }
}

impl<S: MulGroup, const D: usize> ops::Div<NumArray<S, D>> for NumArray<S, D> {
    type Output = NumArray<S, D>;

    fn div(self, rhs: Self) -> Self::Output {
        let mut output = self.clone();
        output /= rhs;
        output
    }
}

impl<S: MulGroup, const D: usize> ops::DivAssign<S> for NumArray<S, D> {
    fn div_assign(&mut self, rhs: S) {
        self.iter_mut().for_each(|elem| *elem /= rhs)
    }
}

impl<S: MulGroup, const D: usize> ops::Div<S> for NumArray<S, D> {
    type Output = NumArray<S, D>;

    fn div(self, rhs: S) -> NumArray<S, D> {
        let mut output = self;
        output /= rhs;
        output
    }
}

// Only implemented for the NumArrays holding first-order floats
impl<S: Data + Float, const D: usize> NumArray<S, D> {
    #[inline]
    pub fn udotprod(&self, rhs: &Self) -> S {
        unsafe {
            self.iter()
                .zip(rhs.iter())
                .fold(S::default(), |acc, (&lhs, &rhs)| {
                    fadd_fast(acc, fmul_fast(lhs, rhs))
                })
        }
    }
}

impl<S: Data + MulGroup + AddGroup, const D: usize> NumArray<S, D> {
    // This is a safe version for the dot product
    #[inline]
    pub fn dotprod(&self, rhs: &Self) -> S {
        (*self * *rhs).iter().fold(S::default(), |acc, &elem| acc + elem)
    }
}

// This is defined at least for one up since it requires the dotprod already.
// Unless the dotprod is defined for S.
impl<S: Data + MulGroup + AddGroup, const L: usize, const D: usize> NumArray<NumArray<S,L>, D>
{
    #[inline]
    pub fn matmul(&self, rhs: NumArray<S,L>) -> NumArray<S,L> {
        let mut output = <NumArray<S,L>>::default();
        output.iter_mut().zip(self.iter())
            .for_each(|(elem_out, row)| *elem_out = row.dotprod(&rhs));
        output
    }
}


// For tensors with the appropriate base type, i.e. a float,
// there are additional functions available.
impl<S: Data + FieldOps + Float, const D: usize> NumArray<S, D> {
    #[inline]
    pub fn norm2(&self) -> S {
        self.iter()
            .fold(S::default(), |acc, &s| s.mul_add(s, acc))
    }

    #[inline]
    pub fn unorm2(&self) -> S {
        unsafe {
            self.iter()
                .fold(S::default(), |acc, &s| fadd_fast(acc, fmul_fast(s, s)))
        }
    }
    // These are not the proper normalise functions
    // and showcases the need to use num-traits
    #[inline]
    pub fn normalise(&mut self) {
        *self /= self.norm2().sqrt();
    }

    #[inline]
    pub fn unormalise(&mut self) {
        *self /= self.unorm2().sqrt();
    }

    #[inline]
    pub fn new_normalised(&self) -> Self {
        let mut output = self.clone();
        output.normalise();
        output
    }

    #[inline]
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
        let mut u = NumArray::<f32, 3>::from([3.0, 4.0, 5.0]);
        let v = NumArray::<f32, 3>::from([1.0, 2.0, 3.0]);
        // Copy of the LHS values but allowing for its use again
        assert_eq!(u.clone() + &v, NumArray::from([4.0, 6.0, 8.0]));

        u += v;
        // u was not modified in the previous calculation
        assert_eq!(u, NumArray::from([4.0, 6.0, 8.0]));
    }

    #[test]
    fn dotprod_test() {
        let u = NumArray::<f32, 3>::from([3.0, 4.0, 5.0]);
        let v = u.clone();

        assert_eq!((&u).dotprod(&v), 50.0);

        assert_eq!(u.udotprod(&v), 50.0);
    }

    #[test]
    fn norm_test() {
        let mut u = NumArray::<f32, 3>::from([3.0, 4.0, 5.0]);

        assert_eq!(u.norm2(), 50.0);

        assert_eq!(u.unorm2(), 50.0);

        u.unormalise();
        assert_eq!(
            u,
            NumArray::from([3.0 / 50.0.sqrt(), 4.0 / 50.0.sqrt(), 5.0 / 50.0.sqrt()])
        );
    }

    #[test]
    fn multidim_test() {
        let mut matrix = <dimensions!(f32, [3, 3])>::default();
        matrix[2][2] += 1.0;
        let second = <dimensions!(f32, [3, 3])>::default();

        assert_eq!((matrix + &second)[2][2], 1.0);
        assert_eq!(second[2][1], 0.0);

        // Use the macro for generating a NumArray type:
        let subgrid1 = numarray!(1.0,2.0);
        let subgrid2 = numarray!([3.0,4.0]);
        let grid_composed = NumArray::from([subgrid1, subgrid2]);
        let grid_directly = numarray!([1.0,2.0],[3.0,4.0]);

        assert_eq!(grid_composed, grid_directly);
    }

    #[test]
    fn matmul_test() {
        let matrix = numarray!([1.0, 0.5], [0.0, 2.0]);
        let vector = numarray!([3.0,4.0]);

        let result = matrix.matmul(vector);
        assert_eq!(result, numarray!([5.0,8.0]));
    }
}
