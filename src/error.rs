use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub enum NNError {
    // Model related errors
    InvalidLayerConfiguration(String),
    LayerShapeMismatch(String),
    EmptyModel,
    
    // Training related errors
    InvalidInputShape(String),
    InvalidOutputShape(String),
    InvalidWeightShape(String),
    InvalidBiasShape(String),
    
    // Optimizer related errors
    OptimizerNotSet,
    InvalidOptimizer(String),
    
    // Loss related errors
    LossNotSet,
    InvalidLoss(String),
    
    // File operations
    ModelLoadError(String),
    ModelSaveError(String),
    
    // Activation related errors
    InvalidActivation(String),
    
    // Computation errors
    ComputationError(String),
    
    // Add these two missing variants
    IoError(std::io::Error),
    SerializationError(Box<bincode::ErrorKind>),  // Used for bincode serialization errors

    ShapeError(ndarray::ShapeError),

    Other(String),
}

impl fmt::Display for NNError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            NNError::InvalidLayerConfiguration(msg) => write!(f, "Invalid layer configuration: {}", msg),
            NNError::LayerShapeMismatch(msg) => write!(f, "Layer shape mismatch: {}", msg),
            NNError::EmptyModel => write!(f, "Model has no layers"),
            NNError::InvalidInputShape(msg) => write!(f, "Invalid input shape: {}", msg),
            NNError::InvalidOutputShape(msg) => write!(f, "Invalid output shape: {}", msg),
            NNError::InvalidWeightShape(msg) => write!(f, "Invalid weight shape: {}", msg),
            NNError::InvalidBiasShape(msg) => write!(f, "Invalid bias shape: {}", msg),
            NNError::OptimizerNotSet => write!(f, "Optimizer not set. Call compile() before training"),
            NNError::InvalidOptimizer(msg) => write!(f, "Invalid optimizer configuration: {}", msg),
            NNError::LossNotSet => write!(f, "Loss function not set. Call compile() before training"),
            NNError::InvalidLoss(msg) => write!(f, "Invalid loss function: {}", msg),
            NNError::ModelLoadError(msg) => write!(f, "Failed to load model: {}", msg),
            NNError::ModelSaveError(msg) => write!(f, "Failed to save model: {}", msg),
            NNError::InvalidActivation(msg) => write!(f, "Invalid activation function: {}", msg),
            NNError::ComputationError(msg) => write!(f, "Computation error: {}", msg),
            NNError::IoError(err) => write!(f, "I/O error: {}", err),
            NNError::SerializationError(err) => write!(f, "Serialization error: {}", err),
            NNError::ShapeError(err) => write!(f, "Shape error: {}", err),
            NNError::Other(err) => write!(f, "Other error: {}", err),
        }
    }
}

impl From<std::io::Error> for NNError {
    fn from(err: std::io::Error) -> NNError {
        NNError::IoError(err)
    }
}

impl From<Box<bincode::ErrorKind>> for NNError {
    fn from(err: Box<bincode::ErrorKind>) -> NNError {
        NNError::SerializationError(err)
    }
}

impl Error for NNError {}

pub type Result<T> = std::result::Result<T, NNError>;