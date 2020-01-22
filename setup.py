from distutils.core import setup

setup(name='Road3D',
      version='1.0',
      description='Road3D Env setup',
      author='Greg Ward',
      author_email='gward@python.net',
      install_requires=['numpy==1.18.0',
                        'scipy==1.4.1',
                        'keras==2.2.4',
                        'pandas==0.25.1',
                        'tensorflow-gpu==2.0.0',
                        'opencv-contrib-python=4.1.2.30'
                        'matplotlib==3.1.2'],
     )
