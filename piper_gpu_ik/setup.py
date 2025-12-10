from setuptools import setup

package_name = 'piper_gpu_ik'
python_package = 'piper_gpu_ik_py'

setup(
    name=package_name,
    version='0.0.1',
    packages=[python_package],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/srv', ['srv/BatchIk.srv']),
        ('share/' + package_name + '/launch', []),
        ('share/' + package_name + '/config', []),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='auto_generated',
    maintainer_email='user@example.com',
    description='GPU FABRIK IK server and pick/place coordinator for Piper',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gpu_fabrik_server = piper_gpu_ik_py.gpu_fabrik_server:main',
        ],
    },
)
