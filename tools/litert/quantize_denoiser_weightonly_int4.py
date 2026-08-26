"""Real weight-only int8 quantization of the PointDiT-L/16 per-step denoiser
TFLite model -- weights quantized for storage, explicitly dequantized back to
float before each op (compute stays float). No calibration needed. Per the
API's own docstring: targets model SIZE, not latency (compute remains float),
included here for a complete size/speed/accuracy picture across all 3 real
LiteRT quantization strategies this package supports.
"""
from ai_edge_quantizer import quantizer, qtyping

FLOAT_TFLITE = "tools/litert/models/pointdit_l16_denoiser_step.tflite"
WEIGHTONLY_TFLITE = "tools/litert/models/pointdit_l16_denoiser_step_weightonly_int4.tflite"


def main():
    qt = quantizer.Quantizer(FLOAT_TFLITE)
    qt.add_weight_only_config(
        regex=".*", operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        num_bits=4,
    )
    print("Recipe:", qt.get_quantization_recipe())
    print("need_calibration:", qt.need_calibration)

    calibration_result = qt.calibrate({}) if qt.need_calibration else None
    result = qt.quantize(calibration_result, serialize_to_path=WEIGHTONLY_TFLITE)
    print(f"Wrote {WEIGHTONLY_TFLITE}")

    import os
    fp32_size = os.path.getsize(FLOAT_TFLITE)
    wo_size = os.path.getsize(WEIGHTONLY_TFLITE)
    print(f"fp32 size: {fp32_size / 1e6:.1f} MB, weight-only size: {wo_size / 1e6:.1f} MB "
          f"({fp32_size / wo_size:.2f}x smaller)")


if __name__ == "__main__":
    main()
