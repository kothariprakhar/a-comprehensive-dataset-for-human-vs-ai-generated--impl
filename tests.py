import unittest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import MagicMock, patch

# Assuming the code is saved in a module named 'implementation'
# We import the classes dynamically or copy-paste context if run in a standalone file.
# For this test suite, we assume the classes are available in the local namespace.

# --- Mocks to prevent heavy model downloading during tests ---
class MockResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2048) # Mocking the original fc layer
        self.in_features = 2048

    def forward(self, x):
        # Return a tensor of shape (Batch, 2048) simulating features
        return torch.randn(x.size(0), 2048)

# --- Redefining Config for Test Environment ---
TEST_CONFIG = {
    'batch_size': 2,
    'image_size': 224,
    'num_classes_multiclass': 6
}

# Context Check: Ensure classes exist (Simulating import)
try:
    DeFactifyNet
    MS_COCOAI_Dummy
except NameError:
    raise unittest.SkipTest("Classes DeFactifyNet or MS_COCOAI_Dummy not found in scope.")

class TestDeFactifyNet(unittest.TestCase):
    def setUp(self):
        # Patch torchvision.models.resnet50 to avoid downloading weights
        self.resnet_patcher = patch('torchvision.models.resnet50')
        self.mock_resnet_fn = self.resnet_patcher.start()
        
        # Configure the mock to return a lightweight MockResNet
        self.mock_backbone = MockResNet()
        # We need to simulate the structure expected by DeFactifyNet
        # The code accesses self.backbone.fc.in_features
        self.mock_backbone.fc.in_features = 2048
        self.mock_resnet_fn.return_value = self.mock_backbone

        self.device = 'cpu'
        self.model = DeFactifyNet(frozen_backbone=False).to(self.device)

    def tearDown(self):
        self.resnet_patcher.stop()

    def test_model_initialization(self):
        """Test if the model initializes heads with correct dimensions."""
        # Check if backbone.fc was replaced by Identity
        self.assertIsInstance(self.model.backbone.fc, nn.Identity)
        
        # Check Binary Head dimensions (2048 -> 1)
        self.assertEqual(self.model.binary_head[0].in_features, 2048)
        self.assertEqual(self.model.binary_head[-1].out_features, 1)

        # Check Multiclass Head dimensions (2048 -> 6)
        self.assertEqual(self.model.multiclass_head[0].in_features, 2048)
        self.assertEqual(self.model.multiclass_head[-1].out_features, TEST_CONFIG['num_classes_multiclass'])

    def test_frozen_backbone(self):
        """Test if the frozen_backbone flag correctly disables gradients."""
        model_frozen = DeFactifyNet(frozen_backbone=True)
        
        # Backbone parameters should not require grad
        for param in model_frozen.backbone.parameters():
            self.assertFalse(param.requires_grad)
            
        # Head parameters SHOULD require grad
        for param in model_frozen.binary_head.parameters():
            self.assertTrue(param.requires_grad)

    def test_forward_pass_shape(self):
        """Test if the forward pass returns tensors of correct shapes."""
        batch_size = 4
        # Input: (B, 3, 224, 224)
        dummy_input = torch.randn(batch_size, 3, 224, 224)
        
        # The forward method expects the backbone to output features.
        # Our mock backbone returns (B, 2048).
        b_logits, m_logits = self.model(dummy_input)
        
        self.assertEqual(b_logits.shape, (batch_size, 1))
        self.assertEqual(m_logits.shape, (batch_size, TEST_CONFIG['num_classes_multiclass']))

    def test_gradient_flow(self):
        """Test if gradients propagate back to the heads."""
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, 224, 224)
        
        b_logits, m_logits = self.model(dummy_input)
        
        target_b = torch.ones(batch_size, 1)
        target_m = torch.zeros(batch_size, dtype=torch.long)
        
        loss = nn.BCEWithLogitsLoss()(b_logits, target_b) + nn.CrossEntropyLoss()(m_logits, target_m)
        
        loss.backward()
        
        # Check gradients in the last layer of heads
        self.assertIsNotNone(self.model.binary_head[-1].weight.grad)
        self.assertIsNotNone(self.model.multiclass_head[-1].weight.grad)

class TestDataset(unittest.TestCase):
    def test_dataset_structure(self):
        """Test MS_COCOAI_Dummy outputs correct types and shapes."""
        ds = MS_COCOAI_Dummy(num_samples=10, transform=None)
        self.assertEqual(len(ds), 10)
        
        img, b_lbl, m_lbl = ds[0]
        
        # Check Image Shape (C, H, W) due to permute logic in __getitem__ when transform is None
        self.assertEqual(img.shape, (3, 224, 224))
        self.assertIsInstance(img, torch.Tensor)
        
        # Check Label Types
        self.assertIsInstance(b_lbl, torch.Tensor)
        self.assertEqual(b_lbl.dtype, torch.float32)
        
        self.assertIsInstance(m_lbl, torch.Tensor)
        self.assertEqual(m_lbl.dtype, torch.long)

    def test_dataset_integrity(self):
        """Test that labels match the logic (0 -> Real, >0 -> AI)."""
        ds = MS_COCOAI_Dummy(num_samples=50)
        for i in range(len(ds)):
            _, b_lbl, m_lbl = ds[i]
            if m_lbl.item() == 0:
                self.assertEqual(b_lbl.item(), 0.0, "Source 0 should be Real (0.0)")
            else:
                self.assertEqual(b_lbl.item(), 1.0, "Source >0 should be AI (1.0)")

if __name__ == '__main__':
    unittest.main()