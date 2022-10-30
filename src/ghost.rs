use core::{cell::UnsafeCell, marker::PhantomData};

#[derive(Clone, Copy, Default)]
struct InvariantLifetime<'id>(PhantomData<*mut &'id ()>);

#[derive(Default)]
pub struct GhostToken<'id> {
    _marker: InvariantLifetime<'id>,
}

impl<'id> GhostToken<'id> {
    pub fn new<R>(f: impl for<'new_id> FnOnce(GhostToken<'new_id>) -> R) -> R {
        f(Self::default())
    }
}

#[repr(transparent)]
pub struct GhostCell<'id, T> {
    value: UnsafeCell<T>,
    _marker: InvariantLifetime<'id>,
}

impl<'id, T> GhostCell<'id, T> {
    pub fn new(value: T) -> Self {
        Self {
            value: UnsafeCell::new(value),
            _marker: InvariantLifetime::default(),
        }
    }
    pub fn borrow<'a>(&'a self, _token: &'a GhostToken<'id>) -> &T {
        unsafe { &*self.value.get() }
    }
    pub fn borrow_mut<'a>(&'a self, _token: &'a mut GhostToken<'id>) -> &mut T {
        unsafe { &mut *self.value.get() }
    }
}
