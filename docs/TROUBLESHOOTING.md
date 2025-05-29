# Troubleshooting

## Common Issues

### FutureWarning: Passing `image` as torch tensor with value range in [-1,1] is deprecated

If you see this warning:

```
/Users/patrickhartono/miniconda3/envs/SD/lib/python3.10/site-packages/diffusers/image_processor.py:724: FutureWarning: Passing `image` as torch tensor with value range in [-1,1] is deprecated. The expected value range for image tensor is [0,1] when passing as pytorch tensor or numpy Array. You passed `image` with value range [-1.0,1.0]
```

#### Explanation

This warning comes from the diffusers library's `VaeImageProcessor`, which now expects image tensors to be in the [0,1] range, but historically StreamDiffusion used the [-1,1] range.

#### Solution

The repository has been updated to use [0,1] range by default in the `process_image` and `pil2tensor` functions. If you're using these functions directly, make sure to use the default range or explicitly specify `range=(0,1)` parameter.

If you're using custom code that directly processes images, convert any tensors from [-1,1] to [0,1] range before passing them to functions from the diffusers library.

To convert from [-1,1] to [0,1]:
```python
image_tensor = (image_tensor + 1) / 2  # Convert from [-1,1] to [0,1]
```

To convert from [0,1] to [-1,1]:
```python
image_tensor = image_tensor * 2 - 1  # Convert from [0,1] to [-1,1]
```
