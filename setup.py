from setuptools import setup, find_packages

setup(
    name='unionllm',
    version='0.1.26',
    license='MIT',
    description='A Python library for unified access to Chinese domestic large language models.',
    author='everfly',
    author_email='tagriver@gmail.com',
    url='https://github.com/EvalsOne/UnionLLM',
    packages=find_packages(exclude=['cookbook', 'tests', 'docs']),
    package_data={'': ['*.txt', '*.rst']},
    include_package_data=True,
    exclude_package_data={'': ['.DS_Store']},
    install_requires=[
        'openai',
        'pydantic',
        'zhipuai',
        'tenacity',
        'dashscope',
        'websocket_client',
        'requests',
        'litellm'
    ],
)