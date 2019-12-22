from distutils.core import setup

setup(name='Road3D',
      version='1.0',
      description='Road3D Env setup',
      author='Greg Ward',
      author_email='gward@python.net',
      install_requires=['numpy==1.17.4',
                        'scipy==1.3.1',
                        'keras==2.2.4',
                        'pandas==0.25.1',
                        'tensorflow-gpu==1.13.1',
                        'opencv-python==4.1.1.26',
                        'matplotlib==3.1.1'],
     )