from setuptools import setup, find_packages

setup(
	name='pyNA',
	author="Laurens Jozef A. Voet",
	author_email="lvoet@mit.edu",
	description="pyNA (python Noise Assessment) computes aircraft noise on take-off trajectories.",
	version='1.0.0',
	packages=['pyNA'],
	url="https://github.mit.edu/lvoet/pyNA",
	include_package_data=True,
	package_data={'': ['data/*']},
	install_requires=[
		'numpy==1.21.6',
		'scipy==1.7.2',
		'openmdao==3.16.0',
		'dymos==1.4.0',
		'julia==0.5.6',
		'pandas==1.3.5',
		'matplotlib==3.5.3'
	],
	python_requires=">=3.7"
)