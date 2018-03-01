import numpy as np
from scipy.ndimage import affine_transform

def reslice(data,affine,zooms,new_zooms,order=1, mode='constant', cval=0):
    new_zooms = np.array(new_zooms, dtype='f8')
    zooms = np.array(zooms, dtype='f8')
    R = new_zooms / zooms
    new_shape = zooms / new_zooms * np.array(data.shape[:3])
    new_shape = tuple(np.round(new_shape).astype('i8'))
    kwargs = {'matrix': R, 'output_shape': new_shape, 'order': order,
              'mode': mode, 'cval': cval}
    if data.ndim == 3:
        data2 = affine_transform(input=data, **kwargs)
    if data.ndim == 4:
        data2 = np.zeros(new_shape+(data.shape[-1],), data.dtype)
        for i in range(data.shape[-1]):
            affine_transform(input=data[..., i], output=data2[..., i],
                                **kwargs)
    Rx = np.eye(4)
    Rx[:3, :3] = np.diag(R)
    affine2 = np.dot(affine, Rx)
    return data2, affine2
