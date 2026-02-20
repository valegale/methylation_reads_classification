import torch
import torch.nn as nn
import torch.nn.functional as F

class ThreeBranchCNN(nn.Module):
    """
    3-branch CNN: DNA + each methylation type (6mA, 5mC, 4mC)
    Each branch detects short, medium, long motifs separately.
    """
    def __init__(self, num_classes=2):
        super().__init__()

        def make_branch():
            return nn.ModuleDict({
                'conv_small': nn.Conv1d(5, 16, kernel_size=3, padding='same'),
                'conv_medium': nn.Conv1d(5, 16, kernel_size=6, padding='same'),
                'conv_long': nn.Conv1d(5, 16, kernel_size=12, padding='same'),
                'bn_small': nn.BatchNorm1d(16),
                'bn_medium': nn.BatchNorm1d(16),
                'bn_long': nn.BatchNorm1d(16)
            })

        # three branches
        self.branch6mA = make_branch()
        self.branch5mC = make_branch()
        self.branch4mC = make_branch()

        self.pool_avg = nn.AdaptiveAvgPool1d(1)
        self.pool_max = nn.AdaptiveMaxPool1d(1)

        # each branch: 16*4 features (small_avg, medium_avg+max, long_max) = 64
        # three branches → 64*3=192 features
        self.fc = nn.Sequential(
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward_branch(self, x, branch):
        # small
        small = F.relu(branch['bn_small'](branch['conv_small'](x)))
        s_avg = self.pool_avg(small).squeeze(-1)

        # medium
        medium = F.relu(branch['bn_medium'](branch['conv_medium'](x)))
        m_avg = self.pool_avg(medium).squeeze(-1)
        m_max = self.pool_max(medium).squeeze(-1)

        # long
        long = F.relu(branch['bn_long'](branch['conv_long'](x)))
        l_max = self.pool_max(long).squeeze(-1)

        return torch.cat([s_avg, m_avg, m_max, l_max], dim=1)

    def forward(self, x):
        # x: (batch, 7, seq_len)
        dna = x[:, 0:4, :]  # DNA one-hot
        x_6mA = torch.cat([dna, x[:, 4:5, :]], dim=1)
        x_5mC = torch.cat([dna, x[:, 5:6, :]], dim=1)
        x_4mC = torch.cat([dna, x[:, 6:7, :]], dim=1)

        f6 = self.forward_branch(x_6mA, self.branch6mA)
        f5 = self.forward_branch(x_5mC, self.branch5mC)
        f4 = self.forward_branch(x_4mC, self.branch4mC)

        feat = torch.cat([f6, f5, f4], dim=1)
        return self.fc(feat)