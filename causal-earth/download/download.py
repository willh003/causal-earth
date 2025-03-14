
from typing import Sequence, Union, Dict, List
import hashlib
import pickle
import os
import urllib.request
from tqdm import tqdm
import tarfile
from links import DOWNLOAD_LINKS


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def get_sha_of_file(file: str, buf_size: int = 100*1024*1024):
    sha = hashlib.sha256()
    with open(file, 'rb') as f:
        while True:
            data = f.read(buf_size)
            if not data:
                break
            sha.update(data)
    return sha.hexdigest()


class Downloader():
    """Downloader Class for EarthNet2021
    """    

    def __init__(self, data_dir: str, links:Dict[str,List[List[str]]]):
        """Initialize Downloader Class

        Args:
            data_dir (str): The directory where the data shall be saved in, we recommend data/dataset/
            links: dict of {split_type :[ [fname_1, link_1, sha_1], ... [fname_N, link_N, sha_N] ] }
        """        
        self.data_dir = data_dir
        self.__URL__ = links
        os.makedirs(data_dir, exist_ok = True)

    @classmethod
    def get(cls, data_dir: str, splits: Union[str,Sequence[str]], links:Dict[str,List[List[str]]], overwrite: bool = False, delete: bool = True):
        """Download the EarthNet2021 Dataset
        
        Before downloading, ensure that you have enough free disk space. We recommend 1 TB.

        Specify the directory data_dir, where it should be saved. Then choose, which of the splits you want to download.
        All available splits: ["train","iid","ood","extreme","seasonal"]
        You can either give "all" to splits or a List of splits, for example ["train","iid"].

        Args:
            data_dir (str): The directory where the data shall be saved in, we recommend data/dataset/
            splits (Sequence[str]): Either "all" or a subset of ["train","iid","ood","extreme","seasonal"]. This determines the splits that are downloaded.
            links: dict of {split_type :[ [fname_1, link_1, sha_1], ... [fname_N, link_N, sha_N] ] }
            overwrite (bool, optional): If True, overwrites an existing gzipped tarball by downloading it again. Defaults to False.
            delete (bool, optional): If True, deletes the downloaded tarball after unpacking it. Defaults to True.
        """        
        self = cls(data_dir, links)        
        print(f"splits: {splits}")

        if isinstance(splits, str):
            if splits == "all":
                splits = ["train","iid","ood","extreme","seasonal"]
        if isinstance(splits, str):
            splits_set = {splits}
            splits = (splits,)
        elif isinstance(splits, list) and len(splits) == 1:
            splits_set = {splits[0]}
        else:
            splits_set = set(splits)
            
        assert(splits_set.issubset(set(["train","iid","ood","extreme","seasonal"])))

        progress_file = os.path.join(self.data_dir, ".PROGRESS")
        try:
            with open(progress_file, "rb") as fp:
                progress_list = pickle.load(fp)
            print("Resuming Download.")
        except:
            progress_list = []

        for split in splits:
            print(f"Downloading split {split} to {self.data_dir}")
            
            for filename, dl_url, sha in tqdm(self.__URL__[split]):
                if filename in progress_list and not overwrite:
                    print(f"{filename} allready downloaded")
                    continue
                tmp_path = os.path.join(self.data_dir, filename)
                print("Downloading...")
                with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
                    urllib.request.urlretrieve(dl_url, filename = tmp_path, reporthook=t.update_to)
                print("Downloaded!")
                print("Asserting SHA256 Hash.")
                assert(sha == get_sha_of_file(tmp_path))
                print("SHA256 Hash is correct!")
                print("Extracting tarball...")
                with tarfile.open(tmp_path, 'r:gz') as tar:
                    members = tar.getmembers()
                    for member in tqdm(iterable=members, total=len(members)):
                        tar.extract(member=member,path=self.data_dir)
                print("Extracted!")
                if delete:
                    print("Deleting tarball...")
                    os.remove(tmp_path)
                
                progress_list.append(filename)
                with open(progress_file, "wb") as fp:
                    pickle.dump(progress_list, fp)



if __name__ == "__main__":

    SPLITS = ["train"]
    DATA_DIR = "/home/wph52/greenearthnet"

    # only the first dataset links 
    first_links = {key: [DOWNLOAD_LINKS[key][0]] for key in DOWNLOAD_LINKS.keys()}

    Downloader.get(data_dir=DATA_DIR, splits=SPLITS, links=first_links)