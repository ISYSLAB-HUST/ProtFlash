from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name='ProtFlash',
      version='0.1.1',
      description='protein language model',
      long_description=long_description,
      long_description_content_type="text/markdown",
      keywords='protein language model',
      classifiers = [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
      url='http://github.com/isyslab-hust/ProtFlash',
      author='Lei Wang',
      author_email='wanglei94@hust.edu.cn',
      license='MIT',
      packages=['ProtFlash'],
      install_requires=[
        'einops',
        'torch',
        'numpy'
    ],
      include_package_data=True,
      zip_safe=False)