from distutils.core import setup

setup(name='Road3D',
      version='1.0',
      description='Road3D Env setup',
      author='Greg Ward',
      author_email='gward@python.net',
      install_requires=['numpy==1.18.1',
                        'keras==2.3.1',
                        'tensorflow-gpu==1.15.0',
                        'opencv-python==4.1.2.30'
                        'matplotlib==3.1.2'],
     )
