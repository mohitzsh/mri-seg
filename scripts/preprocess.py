"""
    Adopted from https://github.com/Ryo-Ito/brain_segmentation/
"""
import argparse
import json
import os

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter
import SimpleITK as sitk
from scipy.ndimage import affine_transform

subjects = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18']
data_prefix = "IBSR_"

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

def preprocess(inputfile, outputfile, order=0, df=None, input_key=None, output_key=None):
    # import pdb; pdb.set_trace()
    img = nib.load(inputfile)
    data = img.get_data()
    affine = img.affine
    zoom = img.header.get_zooms()[:3]
    data, affine = reslice(data, affine, zoom, (1., 1., 1.), order)
    data = np.squeeze(data)
    data = np.pad(data, [(0, 256 - len_) for len_ in data.shape], "constant")
    if order == 0:
        if df is not None:
            tmp = np.zeros_like(data)
            for target, source in zip(df[output_key], df[input_key]):
                tmp[np.where(data == source)] = target
            data = tmp
        data = np.int32(data)
        assert data.ndim == 3, data.ndim
    else:
        data_sub = data - gaussian_filter(data, sigma=1)
        img = sitk.GetImageFromArray(np.copy(data_sub))
        img = sitk.AdaptiveHistogramEqualization(img)
        data_clahe = sitk.GetArrayFromImage(img)[:, :, :, None]
        data = np.concatenate((data_clahe, data[:, :, :, None]), 3)
        data = (data - np.mean(data, (0, 1, 2))) / np.std(data, (0, 1, 2))
        assert data.ndim == 4, data.ndim
        assert np.allclose(np.mean(data, (0, 1, 2)), 0.), np.mean(data, (0, 1, 2))
        assert np.allclose(np.std(data, (0, 1, 2)), 1.), np.std(data, (0, 1, 2))
        data = np.float32(data)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, outputfile)
    print("{} Done!".format(inputfile))


def main():
    parser = argparse.ArgumentParser(description="preprocess dataset")
    parser.add_argument(
        "--img_directory", "-i", type=str,
        help="directory containing the images")
    parser.add_argument(
        "--label_directory","-l", type=str,
        help="directory containing the labels")
    parser.add_argument(
        "--img_suffix", type=str,default="_ana_strip.nii.gz",
        help="suffix of images")
    parser.add_argument(
        "--label_suffix", type=str,default="_segTRI_ana.nii.gz",
        help="suffix of labels")
    parser.add_argument(
        "--output_directory", "-o", type=str,
        help="directory of preprocessed dataset")
    parser.add_argument(
        "--output_file", "-f", type=str, default="dataset.json",
        help="json file of preprocessed dataset, default=dataset.json")
    parser.add_argument(
        "--label_file", type=str, default=None,
        help="csv file with label translation rule, default=None")
    parser.add_argument(
        "--input_key", type=str, default=None,
        help="specifies column for input of label translation, default=None")
    parser.add_argument(
        "--output_key", type=str, default=None,
        help="specifies column for output of label translation, default=None")
    parser.add_argument(
        "--n_classes", type=int,default=5,
        help="number of classes to classify")
    args = parser.parse_args()
    print(args)
    if args.label_file is None:
        df = None
    else:
        df = pd.read_csv(args.label_file)

    dataset = {"in_channels": 2, "n_classes": args.n_classes}
    dataset_list = []

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)
    for subject in subjects:
        output_folder = os.path.join(args.output_directory, subject)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        filedict = {"subject": subject}

        filename = data_prefix + subject + args.img_suffix
        outputfile = os.path.join(output_folder, filename)
        filedict["image"] = outputfile
        preprocess(
            os.path.join(args.img_directory, filename),
            outputfile,
            order=1)

        filename = data_prefix + subject + args.label_suffix
        outputfile = os.path.join(output_folder, filename)
        filedict["label"] = outputfile
        preprocess(
            os.path.join(args.label_directory, filename),
            outputfile,
            order=0,
            df=df,
            input_key=args.input_key,
            output_key=args.output_key)

        dataset_list.append(filedict)
    dataset["data"] = dataset_list

    with open(os.path.join(args.output_directory,args.output_file), "w") as f:
        json.dump(dataset, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    main()
