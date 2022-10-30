use crate::AsParameters;

pub struct Parameters<'id>(Vec<Box<dyn AsParameters<'id>>>);
