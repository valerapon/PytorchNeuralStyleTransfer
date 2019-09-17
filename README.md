# PytorchNeuralStyleTransferModified

Code to run Neural Style Transfer from our paper [Image Style Transfer Using Convolutional Neural Networks](http://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html).

Also includes coarse-to-fine high-resolution from our paper [Controlling Perceptual Factors in Neural Style Transfer](https://arxiv.org/abs/1611.07865).

To run the code you need to get the pytorch VGG19-Model from [Simonyan and Zisserman, 2014](https://arxiv.org/abs/1409.1556) by running: 

`sh download_models.sh`

Everything else is in the IPythonNotebook `NeuralStyleTransfer.ipynb`

Have fun :-)

----  

Standard neural transfer was supplemented by masking. There are two IPython Notebook with different overlay methods: overlay with optimizing function and post-processing two images.  
#### There are image and style:  
<img src="Images/image.jpg" width="304"> <img src="Images/style1.jpg" width="400">   
## Post-processing method:  
#### Mask and result:  
<img src="Images/mask.png" width="304"> <img src="Images/post-proc.png" width="304">    

How to create mask:   

```
def create_mask(size):
    spectrum = np.linspace(0, 1, 2 * size - 1)
    mask = torch.tensor(np.zeros((size, size))[np.newaxis, :, :])
    for i in range(size):
        for j in range(size):
            mask[0, i, j] = spectrum[i + j]
    return mask.float()
```   

Main idea is overlay old image and image created with new style. Overlay is performed using the formula:  

```
new_image_tensor = style_tensor * mask + image_tensor * (1 - mask)
```

## Optimizing function method:
### Mask and resultt:  

<img src="Images/mask1.png" width="304">  <img src="Images/opt-func.png" width="304">   
How to create mask:  

```
def create_mask(size_row, size_col):
    spectrum = np.tile(np.linspace(0, 1, size_col), (size_row ,1))
    mask = torch.tensor(spectrum)
    return mask.float()
```  

This mask create not for image, but for <a href="https://www.codecogs.com/eqnedit.php?latex=F" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F" title="F" /></a> matrix in Gramma matrix formation: <a href="https://www.codecogs.com/eqnedit.php?latex=G&space;=&space;F&space;*&space;F^T" target="_blank"><img src="https://latex.codecogs.com/gif.latex?G&space;=&space;F&space;*&space;F^T" title="G = F * F^T" /></a>:  

```
F = F * create_mask(F.shape[1], F.shape[2])
G = torch.bmm(F, F.transpose(1,2))
```
It's main idea.  
Worth noting that we got "inverted" overlay, because we multiply mask <a href="https://www.codecogs.com/eqnedit.php?latex=M" target="_blank"><img src="https://latex.codecogs.com/gif.latex?M" title="M" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=F" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F" title="F" /></a>, but we will can multiply to <a href="https://www.codecogs.com/eqnedit.php?latex=F^T" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F^T" title="F^T" /></a> and will get a corresponding overlap.
