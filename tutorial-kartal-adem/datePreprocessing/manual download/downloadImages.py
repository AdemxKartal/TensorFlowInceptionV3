from six.moves import urllib
import tarfile
import os


BaseUrl = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102'
UrlDownloadImages = BaseUrl+'/102flower'
UrlLabelImages = BaseUrl+'/imagelabels.mat'
UrlSplitImages= BaseUrl+'/setid.mat'
LabelFile='imagelabels.mat'
SplitFile='setid.mat'
FileNameZipped = 'oxford_flowers102.tgz'
DownloadinDir = os.getcwd() +'/' + FileNameZipped
FileNameUnzipped= 'jpg'
DataSetDir= os.getcwd() +'/jpg'


def download_fromUrl(url,fileName):
    print('download from: ', url)
    urllib.request.urlretrieve(url, fileName)
    print('downloaded file: ',fileName)





def download_dataset():
    print('downloading flower images from: ', UrlDownloadImages)
    urllib.request.urlretrieve(UrlDownloadImages, FileNameZipped)

def extract_tgz_file():
    tar=tarfile.open(FileNameZipped, "r:gz")
    tar.extractall()
    tar.close()

def get_row_dataset():
    if not os.path.exists(FileNameZipped):
        print('start download dataset from : ', UrlDownloadImages)
        download_dataset()
        print('dataset downloaded in : ', DownloadinDir)
    else:
        print('data set already downloaded in : ',DownloadinDir)
    if not os.path.exists(FileNameUnzipped):
        print('start extracting ', FileNameZipped)
        extract_tgz_file()
        print(FileNameZipped, 'extracted in : ', DataSetDir)
    else:
        print('data set already extracted in: ', DataSetDir)


#get_row_dataset()