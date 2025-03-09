import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from pathlib import Path
from label_analysis import TotalSegmenterLabels  # Replace 'your_module' with the actual module name

# Sample DataFrames for mocking
mock_labels_df = pd.DataFrame({
    'structure': ['adrenal_gland_left', 'adrenal_gland_right', 'spinal_cord', 'clavicula_left', 'clavicula_right'],
    'structure_short': ['adrenal_gland', 'adrenal_gland', 'spinal_cord', 'shoulder', 'shoulder'],
    'label': [1, 2, 83, 9, 10],
    'label_short': [1, 1, 2, 3, 3],
    'location_localiser': ['background', 'background', 'background', 'bone', 'bone'],
    'location': ['abdomen', 'abdomen', 'neck,chest,abdomen,pelvis', 'chest', 'chest'],
    'side': ['left', 'right', None, 'left', 'right']
})

mock_meta_df = pd.DataFrame({
    'key': ['meta1', 'meta2'],
    'value': ['value1', 'value2']
})

@pytest.fixture
def totalsegmenterlabels():
    with patch.object(Path, 'exists', return_value=True):
        with patch('pandas.read_excel', side_effect=[mock_labels_df, mock_meta_df]):
            tsl = TotalSegmenterLabels()
            return tsl

def test_labels_all(totalsegmenterlabels):
    tsl = totalsegmenterlabels
    labels = tsl.labels(organ="all")
    assert labels == [1, 2, 83, 9, 10], "Labels for all organs do not match expected values"

def test_labels_specific_organ(totalsegmenterlabels):
    tsl = totalsegmenterlabels
    labels = tsl.labels(organ="adrenal_gland")
    assert labels == [1, 2], "Labels for specific organ do not match expected values"

def test_labels_with_side(totalsegmenterlabels):
    tsl = totalsegmenterlabels
    labels = tsl.labels(organ="adrenal_gland", side="left")
    assert labels == [1], "Labels for specific organ and side do not match expected values"

def test_create_remapping(totalsegmenterlabels):
    tsl = totalsegmenterlabels
    labelsets = [[1, 2], [83]]
    labels_out = [10, 20]
    remapping = tsl.create_remapping(labelsets, labels_out)
    expected_remapping = {1: 10, 2: 10, 83: 20, 9: 0, 10: 0}
    for key, value in expected_remapping.items():
        assert remapping[key] == value, f"Remapping for label {key} does not match expected value {value}"

def test_all_property(totalsegmenterlabels):
    tsl = totalsegmenterlabels
    all_labels = tsl.all
    assert all_labels == [1, 2, 83, 9, 10], "The 'all' property does not match expected values"

def test_label_short_property(totalsegmenterlabels):
    tsl = totalsegmenterlabels
    label_short = tsl.label_short
    assert label_short == [1, 1, 2, 3, 3], "The 'label_short' property does not match expected values"

def test_label_localiser_property(totalsegmenterlabels):
    tsl = totalsegmenterlabels
    label_localiser = tsl.label_localiser
    assert label_localiser == ['background', 'background', 'background', 'bone', 'bone'], "The 'label_localiser' property does not match expected values"

@pytest.mark.parametrize("labelsets, labels_out", [
    ([[1, 2], [83]], [10, 20]),
    ([[9], [10, 83]], [5, 15]),
])
def test_create_remapping_multiple_cases(totalsegmenterlabels, labelsets, labels_out):
    tsl = totalsegmenterlabels
    remapping = tsl.create_remapping(labelsets, labels_out)
    for lset, lout in zip(labelsets, labels_out):
        for l in lset:
            assert remapping[l] == lout, f"Remapping for label {l} does not match expected value {lout}"

# Run with pytest to execute all the unit tests

