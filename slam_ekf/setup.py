from setuptools import setup, find_packages
import os

package_name = 'slam_ekf'

def package_files(data_files, directory_list):

    paths_dict = {}

    for directory in directory_list:

        for (path, directories, filenames) in os.walk(directory):

            for filename in filenames:

                file_path = os.path.join(path, filename)
                install_path = os.path.join('share', package_name, path)

                if install_path in paths_dict.keys():
                    paths_dict[install_path].append(file_path)

                else:
                    paths_dict[install_path] = [file_path]

    for key in paths_dict.keys():
        data_files.append((key, paths_dict[key]))

    return data_files

data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ]

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=package_files(data_files, ['model/', 'launch/', 'worlds/']),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ji',
    maintainer_email='liuji1031@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'slam_ekf=slam_ekf.slam_ekf_node:main'
        ],
    },
)
