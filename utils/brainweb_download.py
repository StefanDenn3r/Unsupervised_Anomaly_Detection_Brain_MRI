from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from nibabel import load, save, Nifti1Image
from tqdm import tqdm


@dataclass
class Item:
    url: str
    path: str
    severity: str


def _download(url: str, data: str, fname: str):
    resp = requests.post(url, data=data, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
            desc=fname,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def _download_and_convert(
        url: str, fname: Path, alias: str, name: str, institution: str,
        email: str, force_update: bool = False
):
    minc_fname = fname.with_suffix('.mnc.gz')
    fname.parent.mkdir(parents=True, exist_ok=True)
    if minc_fname.exists() and not force_update:
        print(
            f"Skipping {str(minc_fname)} download, "
            f"since it already exists."
        )
    else:
        _download(
            url=url,
            data=f'do_download_alias={alias}'
                 f'&format_value=minc'
                 f'&zip_value=gnuzip'
                 f'&who_name={name}'
                 f'&who_institution={institution}'
                 f'&who_email={email}'
                 f'&download_for_real=%5BStart+download%21%5D',
            fname=str(minc_fname)
        )
    nii_fname = fname.with_suffix('.nii.gz')
    if nii_fname.exists() and not force_update:
        print(
            f"Skipping conversion of {str(minc_fname)}, "
            f"since {str(nii_fname)} already exists."
        )
    else:
        try:
            minc = load(minc_fname)
            affine = np.array([[0, 0, 1, 0],
                               [0, 1, 0, 0],
                               [1, 0, 0, 0],
                               [0, 0, 0, 1]])
            out = Nifti1Image(minc.get_fdata(), affine=affine)
            save(out, nii_fname)
            print(f'Successfully converted to {str(nii_fname)}')
        except Exception as e:
            print(e)


def _download_data(base_dir, name, institution, email):
    modality = 'T2'
    slice_thickness = '1mm'
    noise_levels = ['pn0', 'pn1', 'pn3', 'pn5']
    intensity_non_uniformities = ['rf0', 'rf20', 'rf40']
    severities = [
        Item(
            url='https://brainweb.bic.mni.mcgill.ca/cgi/brainweb2',
            path='lesions/severe',
            severity='AI+msles2'
        ),
        Item(
            url='https://brainweb.bic.mni.mcgill.ca/cgi/brainweb1',
            path='normal',
            severity='ICBM+normal'
        )
    ]

    for item in severities:
        out_dir = (base_dir / item.path)
        for noise_level in noise_levels:
            for intensity_non_uniformity in intensity_non_uniformities:
                alias = f"{modality}+{item.severity}+{slice_thickness}+{noise_level}+{intensity_non_uniformity}"
                fname = out_dir / alias.replace("+", "_").lower()
                _download_and_convert(
                    url=item.url,
                    fname=fname,
                    alias=alias,
                    name=name,
                    institution=institution,
                    email=email
                )


def _download_labels(base_dir, name, institution, email):
    items = [
        Item(
            url="https://brainweb.bic.mni.mcgill.ca/cgi/brainweb1",
            path="normal",
            severity="phantom_1.0mm_normal_crisp"
        ),  # normal
        Item(
            url="https://brainweb.bic.mni.mcgill.ca/cgi/brainweb2",
            path="severe_lesions",
            severity="phantom_1.0mm_msles3_crisp"
        )  # severe
    ]

    gt_dir = (base_dir / 'groundtruth')
    for item in items:
        _download_and_convert(
            url=item.url,
            fname=gt_dir / item.path,
            alias=item.severity,
            name=name,
            institution=institution,
            email=email
        )


def download_brainweb_dataset(
        # default path for google colab example
        base_dir: Path = Path('/content/data/Brainweb'),
        name: str = "",
        institution: str = "",
        email: str = "",
):
    _download_data(base_dir, name, institution, email)
    _download_labels(base_dir, name, institution, email)
