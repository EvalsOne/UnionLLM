from setuptools import setup, find_packages

setup(
    name='unionllm',
    version='0.1.4',
    packages=find_packages(),
    license='MIT',
    description='A Python library for unified access to Chinese domestic large language models.',
    author='everfly',
    author_email='tagriver@gmail.com',
    url='https://github.com/EvalsOne/UnionLLM',
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