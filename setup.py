from setuptools import setup, find_packages

setup(
	name='pyNA',
	author="Laurens Jozef A. Voet",
	author_email="laurens.ja.voet@gmail.com",
	description="python Noise Assessment tool to assess aircraft noise during take-off and landing operations.",
	version='1.0.0',
	packages=['pyNA'],
	url="https://github.com/MIT-LAE/pyNA",
	include_package_data=True,
	package_data={'': ['data/*']},
	install_requires=[
		'dataclasses==0.6',
		'dymos>=1.4.0',
		'ipython==7.24.1',
		'json5==0.9.5',
		'julia==0.5.6',
		'matplotlib==3.2.2',
		'numpy>1.21.0',
		'openmdao>=3.16.0',
		'pandas==1.0.5',
		'scipy>1.7.0',
		'tqdm==4.47.0'
	],
	python_requires=">=3.7"
)