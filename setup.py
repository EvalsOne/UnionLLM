from setuptools import setup, find_packages

setup(
    name='cnlitellm',
    version='0.2.2',
    packages=find_packages(),
    license='MIT',
    description='A Python library for unified access to Chinese domestic large language models.',
    author='everfly',
    author_email='tagriver@gmail.com',
    url='https://github.com/EvalsOne/CNLiteLLM',
    install_requires=[
        'openai',
        'pydantic',
        'zhipuai',
        'tenacity',
        'dashscope',
        'websocket_client',
        'requests'
    ],
)