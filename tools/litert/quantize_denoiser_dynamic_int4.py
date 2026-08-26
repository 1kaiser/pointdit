"""Real dynamic-range int8 quantization of the PointDiT-L/16 per-step denoiser
TFLite model -- weights quantized ahead of time, activations quantized on-the-fly
at runtime (real int8 compute, unlike weight-only's float-compute fallback). No
calibration data needed (unlike static int8), since activation ranges are
determined per-call, not from a fixed calibration set.
"""
from ai_edge_quantizer import quantizer, qtyping

FLOAT_TFLITE = "tools/litert/models/pointdit_l16_denoiser_step.tflite"
DYNAMIC_TFLITE = "tools/litert/models/pointdit_l16_denoiser_step_dynamic_int4.tflite"


def main():
    qt = quantizer.Quantizer(FLOAT_TFLITE)
    qt.add_dynamic_config(
        regex=".*", operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        num_bits=4,
    )
    print("Recipe:", qt.get_quantization_recipe())
    print("need_calibration:", qt.need_calibration)

    calibration_result = qt.calibrate({}) if qt.need_calibration else None
    result = qt.quantize(calibration_result, serialize_to_path=DYNAMIC_TFLITE)
    print(f"Wrote {DYNAMIC_TFLITE}")

    import os
    fp32_size = os.path.getsize(FLOAT_TFLITE)
    dyn_size = os.path.getsize(DYNAMIC_TFLITE)
    print(f"fp32 size: {fp32_size / 1e6:.1f} MB, dynamic-int8 size: {dyn_size / 1e6:.1f} MB "
          f"({fp32_size / dyn_size:.2f}x smaller)")


if __name__ == "__main__":
    main()
