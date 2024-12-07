def get_padding_to_keep_dim(dilation: int, kernel_size: int):
    return (kernel_size * dilation - dilation) // 2
