from setuptools import find_packages,setup
from typing import List

Hypen_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    this function will return a list of requirements
    '''
    
    requirements=[]

    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace('\n',"") for req in requirements]

        if Hypen_E_DOT in requirements:
            requirements.remove(Hypen_E_DOT)

    return requirements


setup(
name='ML-Credit_card_default_Prediction',
version='0.0.2',
author='San',
author_email='sanssarkar@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')
)