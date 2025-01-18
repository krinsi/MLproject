from setuptools import find_packages,setup
from typing import List
import sys
import os

# Add the project root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    this func return list of requiremets
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n"," ") for req in requirements]
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
name='MLproject',
version='0.0.1',
author='krinsi',
author_email='krinsiradadiya004@gmail.com',
package=find_packages(),
install_requires=get_requirements('requirements.txt')
)