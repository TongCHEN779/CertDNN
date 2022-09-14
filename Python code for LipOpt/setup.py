from setuptools import setup


setup(name='lipsnet',
      version='0.1',
      description='Lipschitz constant estimation for neural networks',
      author='anon1',
      author_email='anon@anonym.com',
      license='MIT',
      packages=['polyopt'],
      install_requires=[
          'torch',
      ],
      zip_safe=False)

