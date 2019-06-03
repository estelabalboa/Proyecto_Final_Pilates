from sys import argv
import urllib
from bs4 import BeautifulSoup
import datetime


def show_help():
    print
    'Insta Image Downloader'
    print
    ''
    print
    'Usage:'
    print
    'insta.py [OPTION] [URL]'
    print
    ''
    print
    'Options:'
    print
    '-u [Instagram URL]\tDownload single photo from Instagram URL'
    print
    '-f [File path]\t\tDownload Instagram photo(s) using file list'
    print
    '-h, --help\t\tShow this help message'
    print
    ''
    print
    'Example:'
    print
    'python insta.py -u https://instagram.com/p/xxxxx'
    print
    'python insta.py -f /home/username/filelist.txt'
    print
    ''
    exit()


def download_single_file(file_URL):
    print
    'Downloading image...'
    f = urllib.urlopen(file_URL)
    html_source = f.read()
    soup = BeautifulSoup(html_source, 'html.parser')
    meta_tag = soup.find_all('meta', {'property': 'og:image'})
    img_URL = meta_tag[0]['content']
    file_name = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + '.jpg'
    urllib.urlretrieve(img_URL, file_name)
    print
    'Done. Image saved to disk as ' + file_name


if __name__ == '__main__':
    if len(argv) == 1:
        show_help()

    if argv[1] in ('-h', '--help'):
        show_help()

    elif argv[1] == '-u':
        instagram_URL = argv[2]
        download_single_file(instagram_URL)

    elif argv[1] == '-f':
        filePath = argv[2]
        f = open(filePath)
        line = f.readline()
        while line:
            instagram_URL = line.rstrip('\n')
            download_single_file(instagram_URL)

            line = f.readline()
        f.close()
