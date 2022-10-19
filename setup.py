#!/usr/bin/env python

from setuptools import setup

with open('README.md') as f:
	long_description = f.read()
	
reqs = open('requirements.txt').readlines()
	
setup(
	name='yadi',
	version='1.0.0',
	description='yadi: yet another dss interface',
	long_description_content_type='text/markdown',
	long_description=long_description,
	author=['Samuel Talkington','Jorge Fernandez','Amanda West','Alejandro Owen'],
	author_email='talkington@pm.me',
	url='https://github.com/samtalki/yadi/',
	packages = ['yadi'],
	include_package_data=True,
	setup_requires=reqs,
	install_requires=reqs,
	#entry_points={'console_scripts': 'mohca_cl = mohca_cl:init_cli'}
)
