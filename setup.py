from setuptools import setup

setup(
    name='portfolio_backtester',
    version='1.4',    
    description='Sleipnir Backtester',
    url='https://github.com/jeslied/portfolio_backtester',
    author='Jesper Liedholm',
    author_email='liedholm.dev@gmail.com',
    license='BSD 2-clause',
    packages=['portfolio_backtester'],
    install_requires=['pandas',
                      'numpy',                     
                      'matplotlib',
                      'tqdm'
                      ],

    classifiers=[
        'Intended Audience :: Developers',
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)