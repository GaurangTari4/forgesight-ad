import torch

from tools.export_feature_extractor_onnx import ExtractorOutputs, runtime_availability


def test_deployment_runtime_probe_has_expected_keys():
    availability = runtime_availability()
    assert set(availability) == {"onnx", "onnxruntime", "openvino"}
    assert all(isinstance(value, bool) for value in availability.values())


def test_tuple_wrapper_preserves_declared_output_order():
    class FakeExtractor(torch.nn.Module):
        def forward(self, images):
            return {name: torch.full((1, 1), index) for index, name in enumerate(("layer1", "layer2", "layer3", "layer4"))}

    outputs = ExtractorOutputs(FakeExtractor())(torch.zeros(1, 3, 2, 2))
    assert [int(output.item()) for output in outputs] == [0, 1, 2, 3]
