import torch
from deep_feature_extractor import FeatureExtractor, device

def main():
    # 1) Instantiate model exactly as you do in your pipeline
    model = FeatureExtractor(pretrained=True, finetuned_path=None).to(device)
    model.eval()

    # 2) Dummy input covering your expected range: e.g. batch size 1, 3×224×224
    #    If you plan on dynamic input sizes, we’ll add dynamic axes below.
    dummy = torch.randn(1, 3, 224, 224, device=device)

    # 3) Export
    torch.onnx.export(
        model,
        dummy,
        "feat_extractor.onnx",
        opset_version=17,
        input_names=["input"],
        output_names=["features"],
        dynamic_axes={
            "input": {0: "batch", 2: "H", 3: "W"},
            "features": {0: "batch"},
        },
    )
    print("✅ ONNX model saved to feat_extractor.onnx")

if __name__ == "__main__":
    main()
