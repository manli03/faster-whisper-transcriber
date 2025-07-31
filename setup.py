from setuptools import setup


def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    with open(filename, 'r') as f:
        lineiter = (line.strip() for line in f)
        return [line for line in lineiter if line and not line.startswith("#")]


setup(
    name='faster-whisper-transcriber',
    version='1.0.0',
    py_modules=['transcribe'],
    install_requires=parse_requirements('requirements.txt'),
    entry_points={
        'console_scripts': ['transcribe=transcribe:main'],
    }
)
