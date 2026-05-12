from .base import DetectorProtocol, detection_layer_from_frame

try:
    from .correlation_detector import DropletDetectorCuPy, CorrelationDropletDetector, detect_frame_to_dataset_layer
except Exception as exc:  # allows package import without cupy
    DropletDetectorCuPy = None
    CorrelationDropletDetector = None
    detect_frame_to_dataset_layer = detection_layer_from_frame

try:
    from .enhanced_detector import EnhancerConfig, EnhancedDetector, enhance_detector, make_enhanced_detector
except Exception:
    EnhancerConfig = None
    EnhancedDetector = None
    enhance_detector = None
    make_enhanced_detector = None

try:
    from .neural_detector import NeuralDetectorConfig, NeuralDropletDetector, make_neural_detector
except Exception:
    NeuralDetectorConfig = None
    NeuralDropletDetector = None
    make_neural_detector = None

__all__ = [
    "DetectorProtocol",
    "detection_layer_from_frame",
    "DropletDetectorCuPy",
    "CorrelationDropletDetector",
    "detect_frame_to_dataset_layer",
    "EnhancerConfig",
    "EnhancedDetector",
    "enhance_detector",
    "make_enhanced_detector",
    "NeuralDetectorConfig",
    "NeuralDropletDetector",
    "make_neural_detector",
]
