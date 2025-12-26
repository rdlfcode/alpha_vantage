
import torch
import sys
import os

# Add project root to sys.path
sys.path.append(os.getcwd())

from ml.models import FFTModel, TITANS

def test_fft_model_1d():
    input_tensor = torch.randn(4, 60, 5)
    model = FFTModel(seq_len=60, input_dim=5, pred_len=1, use_all_features=False)
    output = model(input_tensor)
    assert output.shape == (4, 1), f"FFT 1D shape mismatch: {output.shape}"
    print("FFT 1D PASS")

def test_fft_model_2d():
    input_tensor = torch.randn(4, 60, 5)
    model = FFTModel(seq_len=60, input_dim=5, pred_len=1, use_all_features=True)
    output = model(input_tensor)
    assert output.shape == (4, 1), f"FFT 2D shape mismatch: {output.shape}"
    print("FFT 2D PASS")

def test_titans_neural():
    input_tensor = torch.randn(4, 60, 5)
    model = TITANS(input_dim=5, d_model=32, memory_type='neural', memory_size=32)
    output = model(input_tensor)
    assert output.shape == (4, 1), f"Titans Neural shape mismatch: {output.shape}"
    print("Titans Neural PASS")

def test_titans_lstm():
    input_tensor = torch.randn(4, 60, 5)
    model = TITANS(input_dim=5, d_model=32, memory_type='lstm', memory_size=32)
    output = model(input_tensor)
    assert output.shape == (4, 1), f"Titans LSTM shape mismatch: {output.shape}"
    print("Titans LSTM PASS")

def test_titans_transformer():
    input_tensor = torch.randn(4, 60, 5)
    model = TITANS(input_dim=5, d_model=32, memory_type='transformer', memory_size=32)
    output = model(input_tensor)
    assert output.shape == (4, 1), f"Titans Transformer shape mismatch: {output.shape}"
    print("Titans Transformer PASS")

if __name__ == "__main__":
    try:
        test_fft_model_1d()
        test_fft_model_2d()
        test_titans_neural()
        test_titans_lstm()
        test_titans_transformer()
        print("ALL TESTS PASSED")
    except Exception as e:
        print(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
