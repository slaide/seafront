"""CNN-based Z estimation tracker."""

from __future__ import annotations

import io
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from data_types import CalibrationData, TrackerResult
from trackers.base import BaseTracker


class CNNTracker(BaseTracker):
    """
    Small CNN for Z estimation. Expected to overfit with small calibration sets.

    Shows training vs test performance to indicate overfitting.
    """

    name = "CNN"

    def __init__(self, calibration: CalibrationData, roi_size: int = 64, verbose: bool = False):
        # Use smaller ROI for CNN (64x64)
        super().__init__(calibration, roi_size=64, verbose=verbose)
        self.model = None
        self.train_loss = 0.0
        self.train_mae = 0.0  # Training set MAE
        self.val_mae = 0.0    # Validation set MAE (for overfitting detection)
        self.n_train = 0
        self.n_val = 0
        self.z_mean = 0.0
        self.z_std = 1.0
        self._train_model()

    def _build_model(self):
        """Build a small CNN model using PyTorch."""
        import torch
        import torch.nn as nn

        class SmallCNN(nn.Module):
            def __init__(self):
                super().__init__()
                # Input: 1x64x64
                self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2)  # -> 16x32x32
                self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # -> 32x16x16
                self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # -> 64x8x8
                self.pool = nn.AdaptiveAvgPool2d(1)  # -> 64x1x1
                self.fc = nn.Linear(64, 1)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.conv1(x))
                x = self.relu(self.conv2(x))
                x = self.relu(self.conv3(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x

        return SmallCNN()

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Resize full image to 64x64 and normalize."""
        # Resize full image to 64x64 for CNN input
        resized = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        # Normalize to [0, 1]
        roi_f = resized.astype(np.float32)
        roi_f = (roi_f - roi_f.min()) / (roi_f.max() - roi_f.min() + 1e-8)
        return roi_f

    def _train_model(self):
        """Train the CNN on calibration data with train/val split for overfitting detection."""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError:
            print("    [CNN] PyTorch not available - install with: uv add torch")
            self.model = None
            return

        # Prepare all data
        X_all = []
        for img in self.calibration.reference_images:
            X_all.append(self._preprocess(img))
        X_all = np.array(X_all)[:, np.newaxis, :, :]  # Add channel dim: (N, 1, 64, 64)
        y_all = self.calibration.z_positions.copy()

        # Split into train/val (80/20) - use every 5th sample for validation
        n_samples = len(y_all)
        val_indices = list(range(2, n_samples, 5))  # Every 5th, starting at index 2
        train_indices = [i for i in range(n_samples) if i not in val_indices]

        X_train = X_all[train_indices]
        y_train = y_all[train_indices]
        X_val = X_all[val_indices]
        y_val = y_all[val_indices]

        self.n_train = len(train_indices)
        self.n_val = len(val_indices)

        # Normalize Z targets (fit on train only)
        self.z_mean = y_train.mean()
        self.z_std = y_train.std() + 1e-8
        y_train_norm = (y_train - self.z_mean) / self.z_std
        y_val_norm = (y_val - self.z_mean) / self.z_std

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train_norm).unsqueeze(1)
        X_val_t = torch.FloatTensor(X_val)

        # Build and train model
        self.model = self._build_model()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)

        # Train for many epochs (will overfit!)
        n_epochs = 500
        self.model.train()
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            outputs = self.model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()

        self.train_loss = loss.item()

        # Calculate train and val MAE to show overfitting
        self.model.eval()
        with torch.no_grad():
            # Training MAE
            train_pred = self.model(X_train_t).numpy().flatten()
            train_pred_um = train_pred * self.z_std + self.z_mean
            self.train_mae = float(np.mean(np.abs(train_pred_um - y_train)))

            # Validation MAE
            if len(val_indices) > 0:
                val_pred = self.model(X_val_t).numpy().flatten()
                val_pred_um = val_pred * self.z_std + self.z_mean
                self.val_mae = float(np.mean(np.abs(val_pred_um - y_val)))
            else:
                self.val_mae = float('nan')

    def estimate_z(self, image: np.ndarray) -> TrackerResult:
        if self.model is None:
            return TrackerResult(0.0, 0.0, 0)

        import torch

        # Preprocess
        roi = self._preprocess(image)
        X = torch.FloatTensor(roi[np.newaxis, np.newaxis, :, :])

        # Predict
        self.model.eval()
        with torch.no_grad():
            pred_norm = self.model(X).item()

        z_estimate = pred_norm * self.z_std + self.z_mean

        return TrackerResult(float(z_estimate), 1.0, 0)

    def get_model_size_bytes(self) -> int:
        """Size = PyTorch model weights + normalization params."""
        if self.model is None:
            return 0
        import torch
        # Save model to buffer to get actual size
        buffer = io.BytesIO()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'z_mean': self.z_mean,
            'z_std': self.z_std,
        }, buffer)
        return buffer.tell()
