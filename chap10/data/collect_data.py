import os
import re
import urllib
from multiprocessing import Process

SUPPORTED_FORMATS = ['jpg', 'png', 'jpeg']
URL_TEMPLATE = r'http://image.b***u.com/search/flip?tn=b***uimage&ie=utf-8&word={keyword}&pn={index}'

def download_images_from_b***u(dir_name, keyword, start_index, end_index):
    index = start_index
    while index < end_index:
        url = URL_TEMPLATE.format(keyword=keyword, index=index)
        try:
            html_text = urllib.urlopen(url).read().decode('utf-8', 'ignore')
            image_urls = re.findall(r'"objURL":"(.*?)"', html_text)
            if not image_urls:
                print('Cannot retrieve anymore image urls \nStopping ...'.format(url))
                break
        except IOError as e:
            print(e)
            print('Cannot open {}. \nStopping ...'.format(url))
            break

        downloaded_urls = []
        for url in image_urls:
            filename = url.split('/')[-1]
            ext = filename[filename.rfind('.')+1:]
            if ext.lower() not in SUPPORTED_FORMATS:
                index += 1
                continue
            filename = '{}/{:0>6d}.{}'.format(dir_name, index, ext)
            cmd = 'wget "{}" -t 3 -T 5 -O {}'.format(url, filename)
            os.system(cmd)
            
            if os.path.exists(filename) and os.path.getsize(filename) > 1024:
                index_url = '{:0>6d},{}'.format(index, url)
                downloaded_urls.append(index_url)
            else:
                os.system('rm {}'.format(filename))

            index += 1
            if index >= end_index:
                break

        with open('{}_urls.txt'.format(dir_name), 'a') as furls:
            urls_text = '{}\n'.format('\n'.join(downloaded_urls))
            if len(urls_text) > 11:
                furls.write(urls_text)

def download_images(keywords, num_per_kw, procs_per_kw):
    args_list = []
    for class_id, keyword in enumerate(keywords):
        dir_name = '{:0>3d}'.format(class_id)
        os.system('mkdir -p {}'.format(dir_name))
        num_per_proc = int(round(float(num_per_kw/procs_per_kw)))
        for i in range(procs_per_kw):
            start_index = i * num_per_proc
            end_index = start_index + num_per_proc - 1
            args_list.append((dir_name, keyword, start_index, end_index))

    processes = [Process(target=download_images_from_b***u, args=x) for x in args_list]

    print('Starting to download images with {} processes ...'.format(len(processes)))

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    print('Done!')

if __name__ == "__main__":
    with open('keywords.txt', 'rb') as f:
        foods = f.read().split()
    download_images(foods, 2000, 3)
