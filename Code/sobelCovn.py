import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class SobelConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SobelConv2d, self).__init__()
        self.conv_x = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv_y = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)

        # Sobel kernel for x direction
        sobel_kernel_x = torch.tensor([[-1, 0, 1],
                                       [-2, 0, 2],
                                       [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)

        # Sobel kernel for y direction
        sobel_kernel_y = torch.tensor([[-1, -2, -1],
                                       [0, 0, 0],
                                       [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

        # Extend the sobel kernel to match the number of input and output channels
        sobel_kernel_x = sobel_kernel_x.repeat(out_channels, in_channels, 1, 1).to(device)
        sobel_kernel_y = sobel_kernel_y.repeat(out_channels, in_channels, 1, 1).to(device)

        # Set the sobel kernels as the weight of the convolution layers
        self.conv_x.weight = nn.Parameter(sobel_kernel_x, requires_grad=False)
        self.conv_y.weight = nn.Parameter(sobel_kernel_y, requires_grad=False)

    def forward(self, x):
        edge_x = self.conv_x(x)
        edge_y = self.conv_y(x)
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        return edge


# Example usage
if __name__ == '__main__':
    model = SobelConv2d(in_channels=3, out_channels=3)
    input_tensor = torch.randn((1, 3, 128, 128))
    output_tensor = model(input_tensor)
    print(output_tensor)
