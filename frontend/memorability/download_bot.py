"""
	Bot to download video files from the urls in the database
"""
import os
import sys
import urllib.request

from tqdm import tqdm

URL_FILENAME = 'video_urls.csv'
DEV_URL_FILENAME = 'dev_video_urls.csv'
TEST_URL_FILENAME = 'test_urls.csv'


class DownloadBot:
	"""
		Automatically retrieve and download video files from a url 
		address.

	Args:
		urls: (n_samples,) list of url addresses
		save_dir: str, output directory to save files into

	Example:
		>>> bot = DownloadBot(['https://www.my_site.com/my_video.mp4'],
		>>>		save_dir='.video_dataset/')
		>>> bot.download()
	"""
	def __init__(self, urls, save_dir, **kwargs):
		self.idx_name = [x[0] for x in urls]
		self.urls = [x[1] for x in urls]
		self.save_dir = save_dir
		self.other = kwargs

		os.makedirs(save_dir, exist_ok=False)

	def download(self):
		"""
			Download videos
		"""
		for u, url in tqdm(enumerate(self.urls), total=len(self.urls)):
			name = '{:d}.mp4'.format(self.idx_name[u])
			path = os.path.join(self.save_dir, name)
			urllib.request.urlretrieve(url, path, **self.other)


def read_url_file(filename):
	with open(filename, 'r') as file:
		idx = []
		url = []
		file.readline()
		for line in file:
			p = line.split(',')
			idx.append(int(p[0]))
			url.append(p[1])
	return [(i, u) for i, u in zip(idx, url)]


if __name__ == "__main__":
	lists_root = sys.argv[1]	# Dir to CSV lists of videos
	download_dir = sys.argv[2]	# Download directory
	
	lists = [f for f in os.listdir(lists_root) if f[0] != '.']
	try:
		data = read_url_file(os.path.join(lists_root, URL_FILENAME))
		data.extend(read_url_file(os.path.join(lists_root,
			DEV_URL_FILENAME)))
		data.extend(read_url_file(os.path.join(lists_root,
			TEST_URL_FILENAME)))
	except OSError as e:
		print('List file {:s} not found within {:s}.\n{}'.format(
			URL_FILENAME, lists_root, e))
	
	# Download only if not already downloaded
	bot = DownloadBot(data, download_dir)
	bot.download()