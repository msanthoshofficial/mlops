import pytest
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import create_model

def test_model_creation():
    model = create_model()
    assert model is not None
    # Check output shape
    # Output shape is (None, 1)
    assert model.output_shape == (None, 1)

def test_data_preprocessing():
    # Placeholder test for data loading logic
    # In a real scenario, we would mock the download and check file paths
    pass
