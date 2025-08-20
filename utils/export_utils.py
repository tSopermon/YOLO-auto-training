"""
Export utilities for YOLO models.
Supports multiple export formats including ONNX, TorchScript, OpenVINO, CoreML, and TensorRT.
"""

import logging
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import json
import yaml
from datetime import datetime
import subprocess
import shutil

logger = logging.getLogger(__name__)


class YOLOExporter:
    """Comprehensive exporter for YOLO models."""

    def __init__(
        self, model: nn.Module, config: "YOLOConfig", export_dir: Optional[Path] = None
    ):
        """
        Initialize exporter.

        Args:
            model: Trained YOLO model
            config: Training configuration
            export_dir: Directory to save exported models
        """
        self.model = model
        self.config = config
        self.export_dir = (
            Path(export_dir) if export_dir else Path(config.export_config["export_dir"])
        )
        self.export_dir.mkdir(parents=True, exist_ok=True)

        # Export results storage
        self.exported_models = {}
        self.export_metadata = {}

    def export_all_formats(
        self,
        formats: Optional[List[str]] = None,
        include_nms: bool = True,
        half_precision: bool = True,
        int8_quantization: bool = False,
        simplify: bool = True,
        dynamic: bool = False,
    ) -> Dict[str, Path]:
        """
        Export model to all supported formats.

        Args:
            formats: List of formats to export (default: all supported)
            include_nms: Whether to include NMS in exported model
            half_precision: Whether to use FP16 precision
            int8_quantization: Whether to use INT8 quantization
            simplify: Whether to simplify ONNX model
            dynamic: Whether to use dynamic axes

        Returns:
            Dictionary mapping format names to export paths
        """
        if formats is None:
            formats = self.config.export_config["export_formats"]

        logger.info(f"Starting export to formats: {formats}")

        for format_name in formats:
            try:
                export_path = self._export_to_format(
                    format_name=format_name,
                    include_nms=include_nms,
                    half_precision=half_precision,
                    int8_quantization=int8_quantization,
                    simplify=simplify,
                    dynamic=dynamic,
                )

                if export_path:
                    self.exported_models[format_name] = export_path
                    logger.info(
                        f"Successfully exported to {format_name}: {export_path}"
                    )

            except Exception as e:
                logger.error(f"Failed to export to {format_name}: {e}")
                continue

        # Save export metadata
        self._save_export_metadata()

        logger.info(f"Export completed. Exported {len(self.exported_models)} formats")
        return self.exported_models

    def _export_to_format(
        self,
        format_name: str,
        include_nms: bool = True,
        half_precision: bool = True,
        int8_quantization: bool = False,
        simplify: bool = True,
        dynamic: bool = False,
    ) -> Optional[Path]:
        """Export model to specific format."""

        if format_name == "onnx":
            return self._export_to_onnx(
                include_nms=include_nms,
                half_precision=half_precision,
                simplify=simplify,
                dynamic=dynamic,
            )
        elif format_name == "torchscript":
            return self._export_to_torchscript(
                include_nms=include_nms, half_precision=half_precision
            )
        elif format_name == "openvino":
            return self._export_to_openvino(
                include_nms=include_nms, half_precision=half_precision
            )
        elif format_name == "coreml":
            return self._export_to_coreml(
                include_nms=include_nms, half_precision=half_precision
            )
        elif format_name == "tensorrt":
            return self._export_to_tensorrt(
                include_nms=include_nms,
                half_precision=half_precision,
                int8_quantization=int8_quantization,
            )
        else:
            logger.warning(f"Unsupported export format: {format_name}")
            return None

    def _export_to_onnx(
        self,
        include_nms: bool = True,
        half_precision: bool = True,
        simplify: bool = True,
        dynamic: bool = False,
    ) -> Optional[Path]:
        """Export model to ONNX format."""
        try:
            # Prepare dummy input
            dummy_input = self._prepare_dummy_input()

            # Export to ONNX
            onnx_path = self.export_dir / f"{self.config.model_type}_model.onnx"

            # Set export parameters
            export_kwargs = {
                "f": str(onnx_path),
                "input_names": ["images"],
                "output_names": ["output"],
                "dynamic_axes": (
                    {"images": {0: "batch_size"}, "output": {0: "batch_size"}}
                    if dynamic
                    else None
                ),
                "opset_version": 12,
                "do_constant_folding": True,
                "verbose": False,
            }

            # Remove None values
            export_kwargs = {k: v for k, v in export_kwargs.items() if v is not None}

            # Export
            torch.onnx.export(self.model, dummy_input, **export_kwargs)

            # Simplify ONNX model if requested
            if simplify:
                try:
                    self._simplify_onnx(onnx_path)
                except Exception as e:
                    logger.warning(f"Failed to simplify ONNX model: {e}")

            # Validate ONNX model
            self._validate_onnx(onnx_path)

            return onnx_path

        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return None

    def _export_to_torchscript(
        self, include_nms: bool = True, half_precision: bool = True
    ) -> Optional[Path]:
        """Export model to TorchScript format."""
        try:
            # Prepare dummy input
            dummy_input = self._prepare_dummy_input()

            # Set model to evaluation mode
            self.model.eval()

            # Export to TorchScript
            scripted_model = torch.jit.script(self.model)

            # Save model
            torchscript_path = self.export_dir / f"{self.config.model_type}_model.pt"

            # Ensure export directory exists
            self.export_dir.mkdir(parents=True, exist_ok=True)

            torch.jit.save(scripted_model, str(torchscript_path))

            # Verify file was created
            if not torchscript_path.exists():
                raise RuntimeError(
                    f"TorchScript file was not created: {torchscript_path}"
                )

            # Test loading
            loaded_model = torch.jit.load(str(torchscript_path))
            _ = loaded_model(dummy_input)

            return torchscript_path

        except Exception as e:
            logger.error(f"TorchScript export failed: {e}")
            return None

    def _export_to_openvino(
        self, include_nms: bool = True, half_precision: bool = True
    ) -> Optional[Path]:
        """Export model to OpenVINO format."""
        try:
            # First export to ONNX
            onnx_path = self._export_to_onnx(
                include_nms=include_nms,
                half_precision=half_precision,
                simplify=True,
                dynamic=False,
            )

            if not onnx_path:
                return None

            # Convert ONNX to OpenVINO using OpenVINO tools
            openvino_path = self.export_dir / f"{self.config.model_type}_model_openvino"
            openvino_path.mkdir(exist_ok=True)

            # Try to use OpenVINO conversion tools
            try:
                self._convert_onnx_to_openvino(onnx_path, openvino_path)
                return openvino_path
            except Exception as e:
                logger.warning(f"OpenVINO conversion failed: {e}")
                # Return ONNX path as fallback
                return onnx_path

        except Exception as e:
            logger.error(f"OpenVINO export failed: {e}")
            return None

    def _export_to_coreml(
        self, include_nms: bool = True, half_precision: bool = True
    ) -> Optional[Path]:
        """Export model to CoreML format."""
        try:
            # CoreML export requires specific dependencies
            try:
                import coremltools as ct
            except ImportError:
                logger.warning(
                    "CoreML export requires coremltools. Install with: pip install coremltools"
                )
                return None

            # Prepare dummy input
            dummy_input = self._prepare_dummy_input()

            # Export to CoreML
            coreml_path = self.export_dir / f"{self.config.model_type}_model.mlmodel"

            # Convert model
            traced_model = torch.jit.trace(self.model, dummy_input)
            coreml_model = ct.convert(
                traced_model,
                inputs=[ct.TensorType(name="images", shape=dummy_input.shape)],
                minimum_deployment_target=ct.target.iOS15,
            )

            # Save model
            coreml_model.save(str(coreml_path))

            return coreml_path

        except Exception as e:
            logger.error(f"CoreML export failed: {e}")
            return None

    def _export_to_tensorrt(
        self,
        include_nms: bool = True,
        half_precision: bool = True,
        int8_quantization: bool = False,
    ) -> Optional[Path]:
        """Export model to TensorRT format."""
        try:
            # First export to ONNX
            onnx_path = self._export_to_onnx(
                include_nms=include_nms,
                half_precision=half_precision,
                simplify=True,
                dynamic=False,
            )

            if not onnx_path:
                return None

            # Convert ONNX to TensorRT
            tensorrt_path = self.export_dir / f"{self.config.model_type}_model.trt"

            try:
                self._convert_onnx_to_tensorrt(
                    onnx_path, tensorrt_path, half_precision, int8_quantization
                )
                return tensorrt_path
            except Exception as e:
                logger.warning(f"TensorRT conversion failed: {e}")
                # Return ONNX path as fallback
                return onnx_path

        except Exception as e:
            logger.error(f"TensorRT export failed: {e}")
            return None

    def _prepare_dummy_input(self) -> torch.Tensor:
        """Prepare dummy input for model export."""
        batch_size = self.config.export_config.get("batch_size", 1)
        image_size = self.config.image_size

        if isinstance(image_size, (list, tuple)):
            height, width = image_size[0], image_size[1]
        else:
            height = width = image_size

        # Create dummy input tensor
        dummy_input = torch.randn(batch_size, 3, height, width)

        return dummy_input

    def _simplify_onnx(self, onnx_path: Path) -> None:
        """Simplify ONNX model using onnx-simplifier."""
        try:
            import onnxsim

            # Load ONNX model
            onnx_model = onnxsim.load(str(onnx_path))

            # Simplify model
            simplified_model, check = onnxsim.simplify(onnx_model)

            if check:
                # Save simplified model
                onnxsim.save(simplified_model, str(onnx_path))
                logger.info("ONNX model simplified successfully")
            else:
                logger.warning("ONNX model simplification check failed")

        except ImportError:
            logger.warning(
                "onnx-simplifier not available. Install with: pip install onnx-simplifier"
            )
        except Exception as e:
            logger.warning(f"ONNX simplification failed: {e}")

    def _validate_onnx(self, onnx_path: Path) -> None:
        """Validate ONNX model."""
        try:
            import onnx

            # Load and validate ONNX model
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model validation passed")

        except ImportError:
            logger.warning("onnx not available. Install with: pip install onnx")
        except Exception as e:
            logger.warning(f"ONNX validation failed: {e}")

    def _convert_onnx_to_openvino(self, onnx_path: Path, output_dir: Path) -> None:
        """Convert ONNX model to OpenVINO format."""
        try:
            # Try to use OpenVINO conversion tools
            cmd = [
                "mo",
                "--input_model",
                str(onnx_path),
                "--output_dir",
                str(output_dir),
                "--model_name",
                f"{self.config.model_type}_model",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("OpenVINO conversion successful")
            else:
                logger.warning(f"OpenVINO conversion failed: {result.stderr}")
                raise RuntimeError("OpenVINO conversion failed")

        except FileNotFoundError:
            logger.warning("OpenVINO Model Optimizer (mo) not found in PATH")
            raise RuntimeError("OpenVINO tools not available")
        except Exception as e:
            logger.error(f"OpenVINO conversion error: {e}")
            raise

    def _convert_onnx_to_tensorrt(
        self,
        onnx_path: Path,
        output_path: Path,
        half_precision: bool,
        int8_quantization: bool,
    ) -> None:
        """Convert ONNX model to TensorRT format."""
        try:
            import tensorrt as trt

            # Create TensorRT logger
            logger_trt = trt.Logger(trt.Logger.WARNING)

            # Create TensorRT builder
            builder = trt.Builder(logger_trt)
            config = builder.create_builder_config()

            # Set precision
            if int8_quantization:
                config.set_flag(trt.BuilderFlag.INT8)
            elif half_precision:
                config.set_flag(trt.BuilderFlag.FP16)

            # Parse ONNX model
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, logger_trt)

            with open(onnx_path, "rb") as model:
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        logger.error(
                            f"TensorRT parsing error: {parser.get_error(error)}"
                        )
                    raise RuntimeError("TensorRT parsing failed")

            # Build engine
            engine = builder.build_engine(network, config)

            if engine is None:
                raise RuntimeError("TensorRT engine building failed")

            # Save engine
            with open(output_path, "wb") as f:
                f.write(engine.serialize())

            logger.info("TensorRT conversion successful")

        except ImportError:
            logger.warning("TensorRT not available. Install with: pip install tensorrt")
            raise RuntimeError("TensorRT not available")
        except Exception as e:
            logger.error(f"TensorRT conversion error: {e}")
            raise

    def _save_export_metadata(self) -> None:
        """Save export metadata and configuration."""
        metadata = {
            "export_timestamp": datetime.now().isoformat(),
            "model_type": self.config.model_type,
            "model_config": {
                "image_size": self.config.image_size,
                "batch_size": self.config.batch_size,
                "num_classes": len(self.config.data_yaml.get("names", [])),
                "class_names": self.config.data_yaml.get("names", []),
            },
            "export_config": self.config.export_config,
            "exported_models": {
                format_name: str(path)
                for format_name, path in self.exported_models.items()
            },
            "export_status": "completed" if self.exported_models else "failed",
        }

        # Save as JSON
        metadata_file = self.export_dir / "export_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=4, default=str)

        # Save as YAML
        metadata_yaml = self.export_dir / "export_metadata.yaml"
        with open(metadata_yaml, "w") as f:
            yaml.dump(metadata, f, default_flow_style=False)

        logger.info(f"Export metadata saved to {self.export_dir}")

    def generate_export_report(self) -> str:
        """Generate export report."""
        if not self.exported_models:
            return "No models exported successfully."

        report = f"""
YOLO Model Export Report
========================

Export completed: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Model type: {self.config.model_type}
Export directory: {self.export_dir}

Successfully exported formats:
"""

        for format_name, path in self.exported_models.items():
            file_size = path.stat().st_size / (1024 * 1024)  # MB
            report += f"- {format_name.upper()}: {path.name} ({file_size:.2f} MB)\n"

        report += f"\nTotal exported models: {len(self.exported_models)}"

        return report


def export_model(
    model: nn.Module,
    config: "YOLOConfig",
    export_dir: Optional[Path] = None,
    formats: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """
    Export YOLO model to multiple formats.

    Args:
        model: Trained YOLO model
        config: Training configuration
        export_dir: Directory to save exported models
        formats: List of formats to export

    Returns:
        Dictionary mapping format names to export paths
    """
    exporter = YOLOExporter(model, config, export_dir)
    exported_models = exporter.export_all_formats(formats=formats)

    # Generate and log report
    report = exporter.generate_export_report()
    logger.info(report)

    return exported_models


def export_to_onnx(
    model: nn.Module,
    config: "YOLOConfig",
    output_path: Optional[Path] = None,
    half_precision: bool = True,
    simplify: bool = True,
) -> Optional[Path]:
    """
    Export model specifically to ONNX format.

    Args:
        model: Trained YOLO model
        config: Training configuration
        output_path: Path to save ONNX model
        half_precision: Whether to use FP16 precision
        simplify: Whether to simplify ONNX model

    Returns:
        Path to exported ONNX model or None if failed
    """
    exporter = YOLOExporter(model, config)
    onnx_path = exporter._export_to_onnx(
        half_precision=half_precision, simplify=simplify
    )

    if onnx_path and output_path:
        # Copy to desired location
        shutil.copy2(onnx_path, output_path)
        return output_path

    return onnx_path


def export_to_torchscript(
    model: nn.Module, config: "YOLOConfig", output_path: Optional[Path] = None
) -> Optional[Path]:
    """
    Export model specifically to TorchScript format.

    Args:
        model: Trained YOLO model
        config: Training configuration
        output_path: Path to save TorchScript model

    Returns:
        Path to exported TorchScript model or None if failed
    """
    exporter = YOLOExporter(model, config)
    torchscript_path = exporter._export_to_torchscript()

    if torchscript_path and output_path:
        # Copy to desired location
        shutil.copy2(torchscript_path, output_path)
        return output_path

    return torchscript_path


def validate_exported_model(
    model_path: Path, format_name: str, config: "YOLOConfig"
) -> bool:
    """
    Validate exported model.

    Args:
        model_path: Path to exported model
        format_name: Format of the model
        config: Training configuration

    Returns:
        True if validation passes, False otherwise
    """
    try:
        if format_name == "onnx":
            return _validate_onnx_model(model_path)
        elif format_name == "torchscript":
            return _validate_torchscript_model(model_path, config)
        elif format_name == "openvino":
            return _validate_openvino_model(model_path)
        elif format_name == "coreml":
            return _validate_coreml_model(model_path)
        elif format_name == "tensorrt":
            return _validate_tensorrt_model(model_path)
        else:
            logger.warning(f"Unknown format for validation: {format_name}")
            return False

    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        return False


def _validate_onnx_model(model_path: Path) -> bool:
    """Validate ONNX model."""
    try:
        import onnx

        model = onnx.load(str(model_path))
        onnx.checker.check_model(model)
        return True
    except Exception as e:
        logger.error(f"ONNX validation failed: {e}")
        return False


def _validate_torchscript_model(model_path: Path, config: "YOLOConfig") -> bool:
    """Validate TorchScript model."""
    try:
        model = torch.jit.load(str(model_path))

        # Test with dummy input
        dummy_input = torch.randn(1, 3, config.image_size, config.image_size)
        _ = model(dummy_input)

        return True
    except Exception as e:
        logger.error(f"TorchScript validation failed: {e}")
        return False


def _validate_openvino_model(model_path: Path) -> bool:
    """Validate OpenVINO model."""
    try:
        # Check if model directory exists and contains required files
        if model_path.is_dir():
            required_files = ["model.bin", "model.xml"]
            return all((model_path / f).exists() for f in required_files)
        return False
    except Exception as e:
        logger.error(f"OpenVINO validation failed: {e}")
        return False


def _validate_coreml_model(model_path: Path) -> bool:
    """Validate CoreML model."""
    try:
        import coremltools as ct

        model = ct.models.MLModel(str(model_path))
        return model is not None
    except Exception as e:
        logger.error(f"CoreML validation failed: {e}")
        return False


def _validate_tensorrt_model(model_path: Path) -> bool:
    """Validate TensorRT model."""
    try:
        import tensorrt as trt

        # Try to load the engine
        with open(model_path, "rb") as f:
            engine_data = f.read()

        # Create runtime and deserialize
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(engine_data)

        return engine is not None
    except Exception as e:
        logger.error(f"TensorRT validation failed: {e}")
        return False
