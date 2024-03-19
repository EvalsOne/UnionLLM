from setuptools import setup, find_packages

setup(
    name='cnlitellm',
    version='0.1',
    packages=find_packages(),
    license='MIT',
    description='A Python library for unified access to Chinese domestic large language models.',
    author='everfly',
    author_email='tagriver@gmail.com',
    url='https://github.com/EvalsOne/CNLiteLLM',  # Optional
    install_requires=[
        'openai',
        'pydantic',
        'zhipuai'  # 假设'zhipuai'是一个有效的Python包名
    ],
)